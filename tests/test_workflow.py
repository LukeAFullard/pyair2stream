"""
pyair2stream.Model: each step's settings, the files passed from step to step, and the refusals.
The runs themselves are replaced by a stand-in that records the settings file and writes the
files a real run would leave; examples 02, 03, 04 and 08 run the Model end to end.
"""

import json
import os
import types

import numpy as np
import pandas as pd
import pytest
import yaml

import pyair2stream
from pyair2stream import workflow

BASE = {
    "project_name": "test", "station_name": "River", "series": "c", "time_resolution": "1d",
    "version": 8, "integrator": "CRN", "objective_function": "NSE", "random_seed": 7,
    "parameter_bounds": {"min": [0] * 8, "max": [1] * 8},
    "uncertainty_options": {"noise_model": "ar1", "prediction_interval": 90},
    "cross_validation": {"threshold": 18},
    "paths": {"input_data": "cal.csv", "validation_data": "val.csv"},
}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """Replace pyair2stream.run: record each settings file read, write what a real run would."""
    monkeypatch.chdir(tmp_path)
    seen = []

    def fake_run(path, verbose=True):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        seen.append(cfg)
        out = cfg["paths"]["output_dir"]
        os.makedirs(out, exist_ok=True)
        mode = cfg["run_mode"]
        if mode in ("DE", "DE-MCMC", "PSO") and not (cfg.get("cross_validation") or {}).get("enabled"):
            json.dump({"qmedia": 1.25, "version": 8, "integrator": "CRN"},
                      open(os.path.join(out, "calibration_metadata.json"), "w"))
        if mode == "DE-MCMC":
            open(os.path.join(out, "MCMC_chain_River_c_1d.csv"), "w").write("a1\n0\n")
        if (cfg.get("cross_validation") or {}).get("enabled"):
            open(os.path.join(out, "cv_results.csv"), "w").write("fold\n2001\n")
            open(os.path.join(out, "cv_conformal_margins.csv"), "w").write("level\n90\n")
        if mode == "FORWARD":
            open(os.path.join(out, "Forward_Prediction_Envelopes_River_c_1d.csv"), "w").write("x\n")
            if (cfg.get("uncertainty_options") or {}).get("save_ensemble"):
                open(os.path.join(out, "Forward_Prediction_Ensemble_River_c_1d.npz"), "w").write("")
                open(os.path.join(out, "Forward_Prediction_Ensemble_River_c_1d_meta.json"), "w").write("{}")
        return types.SimpleNamespace(output_dir=out, settings=cfg)

    monkeypatch.setattr("pyair2stream.main.run", fake_run)
    return seen


def test_model_is_part_of_the_package():
    assert pyair2stream.Model is workflow.Model


def test_calibrate_check_and_predict_pass_their_files_on(runs):
    m = pyair2stream.Model(BASE, output_dir="out")
    m.calibrate()
    cal = runs[-1]
    assert cal["run_mode"] == "DE-MCMC" and "cross_validation" not in cal
    assert cal["paths"] == {"input_data": "cal.csv", "validation_data": "val.csv",
                            "output_dir": os.path.join("out", "calibration")}
    # Each step is run from its own settings file, which says how to run it again.
    text = open(m.settings_file("calibration")).read()
    assert "pyair2stream --config out/calibration.yaml" in text

    m.check(cross_validation={"season_months": [6, 7, 8]})
    check = runs[-1]
    assert check["run_mode"] == "DE"
    assert check["Qmedia"] == 1.25                                # the calibration's, not recomputed
    assert check["cross_validation"] == {"threshold": 18, "season_months": [6, 7, 8], "enabled": True}
    assert check["uncertainty_options"] == BASE["uncertainty_options"]
    assert "validation_data" not in check["paths"]

    m.predict("val.csv", name="baseline", conformal=True)
    fwd = runs[-1]
    assert fwd["run_mode"] == "FORWARD"
    assert "parameter_bounds" not in fwd and "cross_validation" not in fwd
    assert fwd["paths"]["calibration_metadata"] == os.path.join("out", "calibration", "calibration_metadata.json")
    opts = fwd["forward_options"]
    assert opts["mcmc_chain_path"] == os.path.join("out", "calibration", "MCMC_chain_River_c_1d.csv")
    assert opts["n_samples"] == 1000 and opts["random_seed"] == 7
    assert opts["conformal_margins"] == os.path.join("out", "check", "cv_conformal_margins.csv")
    assert fwd["uncertainty_options"]["save_ensemble"] is True
    assert fwd["uncertainty_options"]["noise_model"] == "ar1"


def test_paired_prediction_from_a_dataframe(runs):
    m = pyair2stream.Model(BASE, output_dir="out")
    m.calibrate()
    m.predict("val.csv", name="baseline")
    dates = pd.date_range("2010-01-01", periods=3)
    warmer = pd.DataFrame({"T_air": [1.0, 2.0, 3.0], "T_water": np.nan, "Discharge": 1.0}, index=dates)
    m.predict(warmer, name="warmer", paired_with="baseline")
    fwd = runs[-1]
    assert fwd["forward_options"]["reuse_sample_indices_from"] == os.path.join(
        "out", "baseline", "Forward_Prediction_Ensemble_River_c_1d_meta.json")
    assert "n_samples" not in fwd["forward_options"] and "random_seed" not in fwd["forward_options"]
    written = pd.read_csv(fwd["paths"]["input_data"])
    assert fwd["paths"]["input_data"] == os.path.join("out", "inputs", "warmer.csv")
    assert list(written.columns) == ["Date", "T_air", "T_water", "Discharge"]
    assert list(written.Date) == ["2010-01-01", "2010-01-02", "2010-01-03"]


def test_best_fit_only(runs):
    m = pyair2stream.Model({**BASE, "run_mode": "PSO"}, output_dir="out")
    m.calibrate(uncertainty=False)
    assert runs[-1]["run_mode"] == "PSO"
    m.predict("val.csv")                               # no MCMC sample: the best fit, no intervals
    assert "forward_options" not in runs[-1]
    with pytest.raises(ValueError, match="no MCMC sample"):
        m.predict("val.csv", name="p2", intervals=True)
    with pytest.raises(ValueError, match="need prediction intervals"):
        m.predict("val.csv", name="p3", conformal=True)


def test_refusals(runs):
    with pytest.raises(ValueError, match="input_data"):
        pyair2stream.Model({**BASE, "paths": {}})
    with pytest.raises(ValueError, match="describe the calibration"):
        pyair2stream.Model({**BASE, "run_mode": "FORWARD"})
    m = pyair2stream.Model(BASE, output_dir="out")
    with pytest.raises(ValueError, match="calibrate first"):
        m.check()
    with pytest.raises(ValueError, match="calibrate first"):
        m.predict("val.csv")
    m.calibrate()
    with pytest.raises(ValueError, match="check first"):
        m.predict("val.csv", conformal=True)
    with pytest.raises(ValueError, match="must be the same in every step"):
        m.predict("val.csv", changes={"version": 5})
    with pytest.raises(ValueError, match="must be the same in every step"):
        m.check(changes={"uncertainty_options": {"noise_model": "iid"}})
    with pytest.raises(ValueError, match="another step"):
        m.predict("val.csv", name="check")
    m.predict("val.csv", name="p")
    with pytest.raises(ValueError, match="already made"):
        m.predict("val.csv", name="p")
    with pytest.raises(ValueError, match="no prediction named"):
        m.predict("val.csv", name="q", paired_with="nothing")


def test_changes_apply_to_one_step(runs):
    m = pyair2stream.Model(BASE, output_dir="out")
    m.calibrate(changes={"optimization": {"mcmc_steps": 5000}})
    assert runs[-1]["optimization"] == {"mcmc_steps": 5000}
    m.predict("val.csv", changes={"forward_options": {"n_samples": 200}})
    assert runs[-1]["forward_options"]["n_samples"] == 200
    assert runs[-1]["forward_options"]["mcmc_chain_path"].endswith("MCMC_chain_River_c_1d.csv")
    assert "optimization" not in runs[-1]


def test_earlier_results_and_reading_them(runs, monkeypatch):
    first = pyair2stream.Model(BASE, output_dir="out")
    first.calibrate()
    first.check()
    first.predict("val.csv", name="baseline")
    # A later session: the same folders, without running the steps again.
    m = pyair2stream.Model(BASE, output_dir="later")
    m.use_calibration(os.path.join("out", "calibration")).use_check(os.path.join("out", "check"))
    m.use_prediction("baseline", os.path.join("out", "baseline"))
    m.predict("val.csv", name="again", paired_with="baseline", conformal=True)
    opts = runs[-1]["forward_options"]
    assert opts["mcmc_chain_path"] == os.path.join("out", "calibration", "MCMC_chain_River_c_1d.csv")
    assert opts["conformal_margins"] == os.path.join("out", "check", "cv_conformal_margins.csv")
    with pytest.raises(FileNotFoundError):
        m.use_calibration("out")

    calls = []
    monkeypatch.setattr("pyair2stream.scenario.load_ensemble", lambda p: calls.append(p) or "loaded")
    monkeypatch.setattr("pyair2stream.scenario.paired_difference_from_files", lambda a, b: (a, b))
    assert m.ensemble("baseline") == "loaded"
    assert calls == [os.path.join("out", "baseline", "Forward_Prediction_Ensemble_River_c_1d.npz")]
    a, b = m.difference("again", "baseline")
    assert a.startswith(os.path.join("later", "again")) and b.startswith(os.path.join("out", "baseline"))


def test_settings_file_and_numpy_values(runs):
    settings = {**BASE, "Tice_cover": np.float64(0.5), "parameter_bounds": {"min": np.zeros(8), "max": np.ones(8)}}
    path = "settings.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({**BASE, "paths": {"input_data": "cal.csv", "output_dir": "from_file"}}, f)
    assert pyair2stream.Model(path).output_dir == "from_file"
    m = pyair2stream.Model(settings, output_dir="out")
    m.calibrate()
    assert runs[-1]["Tice_cover"] == 0.5 and runs[-1]["parameter_bounds"]["max"] == [1.0] * 8
