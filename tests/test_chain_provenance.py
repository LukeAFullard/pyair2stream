"""A FORWARD run refuses a DE-MCMC chain fitted with another model version, integrator,
Qmedia, Tice_cover or min_theta_floor (recorded in the chain's _meta.json): the parameters mean something only
with the settings they were fitted with (docs/METHODS.md §13)."""
import json

import numpy as np
import pandas as pd
import pytest

from pyair2stream.optimization import forward_mode
from tests.test_report04_uncertainty_and_mcmc import _build_calibration_data


def _forward(tmp_path, meta):
    data = _build_calibration_data(str(tmp_path / "f"))
    data.Twat_obs[:] = -999.0
    data.runmode = "FORWARD"
    chain_path = str(tmp_path / "chain.csv")
    pd.DataFrame(np.random.default_rng(0).random((10, 8)) * 0.5,
                 columns=[f"par_{i + 1}" for i in range(8)]).to_csv(chain_path, index=False)
    with open(chain_path.replace(".csv", "_meta.json"), "w") as f:
        json.dump({"rho": 0.8, "sigma": 0.5, **meta}, f)
    data.forward_options = {"enable_prediction_intervals": True, "mcmc_chain_path": chain_path,
                            "n_samples": 5, "random_seed": 42}
    data.uncertainty_options = {"noise_model": "ar1", "ar1_rho": None, "max_divergent_fraction": 1.0}
    forward_mode(data)


MATCHING = {"version": 8, "integrator": "RK4", "qmedia": 10.0,     # as _build_calibration_data
            "Tice_cover": 0.0, "min_theta_floor": None}


@pytest.mark.parametrize("qmedia", [10.0, 10.004])     # typed with fewer digits: the same Qmedia
def test_matching_chain_is_accepted(tmp_path, capsys, qmedia):
    _forward(tmp_path, {**MATCHING, "qmedia": qmedia})
    assert "cannot be checked" not in capsys.readouterr().out


@pytest.mark.parametrize("key,value,word", [("version", 7, "model version"), ("integrator", "CRN", "integrator"),
                                            ("qmedia", 10.02, "Qmedia"), ("Tice_cover", 4.0, "Tice_cover"),
                                            ("min_theta_floor", 1e-6, "min_theta_floor")])
def test_mismatched_chain_is_refused(tmp_path, key, value, word):
    with pytest.raises(ValueError, match=word):
        _forward(tmp_path, {**MATCHING, key: value})


def test_old_chain_without_the_record_gives_a_note(tmp_path, capsys):
    _forward(tmp_path, {})
    assert "cannot be checked" in capsys.readouterr().out


def test_chain_without_the_settings_record_gives_a_note(tmp_path, capsys):
    _forward(tmp_path, {"version": 8, "integrator": "RK4", "qmedia": 10.0})
    assert "Tice_cover and min_theta_floor" in capsys.readouterr().out
