"""Warnings for physically implausible daily simulations (model.check_daily_plausibility)."""
import numpy as np
import pytest
import yaml

from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, call_model, check_daily_plausibility, funcobj, statis

# Version 5 on the Mentue: the published parameters, and a parameter set that weekly-scored
# calibrations with the authors' bounds ended on in validation V4 (case J): negative a2 and a3,
# a daily simulation that swings between 0 °C and high values whose weekly means still fit.
SENSIBLE = [3.0752551, 0.66800995, 1.0101169, 0.0, 0.0, 1.6162276, 0.58313392, 0.0]
ZIGZAG = [-4.451, -1.822, -4.412, 0.0, 0.0, 1.920, 0.144, 0.0]


def _data(tmp_path, time_resolution="1w"):
    cfg = {"station_name": "S", "series": "c", "version": 5, "run_mode": "DE", "integrator": "CRN",
           "time_resolution": time_resolution, "random_seed": 1,
           "paths": {"input_data": "data/switzerland/MAH_2369_calibration.csv", "output_dir": str(tmp_path / "o")}}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    data = read_calibration(str(path))
    read_Tseries(data, "c")
    aggregation(data)
    statis(data)
    return data


def test_sensible_parameters_pass_quietly(tmp_path, capsys):
    data = _data(tmp_path)
    data.par[:] = SENSIBLE
    call_model(data)
    out = check_daily_plausibility(data)
    assert out["n_negative_B"] == 0 and out["min_B"] > 0
    assert out["change_corr"] > 0.3
    assert "Warning" not in capsys.readouterr().out


def test_a_zigzag_fit_with_negative_relaxation_is_flagged(tmp_path, capsys):
    data = _data(tmp_path)
    data.par[:] = ZIGZAG
    call_model(data)
    out = check_daily_plausibility(data)
    assert out["n_negative_B"] > 0 and out["min_B"] < 0
    assert out["change_corr"] < -0.9
    text = capsys.readouterr().out
    assert "relaxation rate B is negative" in text
    assert "zigzags from one day to the next" in text
    # The zigzag hides in weekly means: its weekly score is not far from the sensible one's.
    weekly_zigzag = funcobj(data)
    data.par[:] = SENSIBLE
    call_model(data)
    assert weekly_zigzag > 0.9 and funcobj(data) > 0.9


def test_segment_boundaries_are_not_paired(tmp_path):
    data = _data(tmp_path, "1d")
    data.par[:] = SENSIBLE
    call_model(data)
    whole = check_daily_plausibility(data)["change_corr"]
    # Gap-tolerant segments: the change across a boundary is not paired with its neighbours.
    data.gap_tolerant = True
    mid = data.n_tot // 2
    data.segments = [(0, mid), (mid + 1, data.n_tot - 1)]
    data.Twat_mod = data.Twat_mod.copy()
    data.Twat_mod[mid + 1:] += 50.0                 # a jump between segments only
    assert check_daily_plausibility(data)["change_corr"] == pytest.approx(whole, abs=0.01)


def test_forward_run_reports_a_zigzag_calibration(tmp_path, capsys):
    from pyair2stream.main import forward
    data = _data(tmp_path)
    data.par[:] = ZIGZAG
    data.par_best = np.array(ZIGZAG, dtype=float)
    call_model(data)
    data.finalfit = funcobj(data)
    forward(data)
    assert "zigzags from one day to the next" in capsys.readouterr().out
