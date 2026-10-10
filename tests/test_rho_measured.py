"""
Sparse water-temperature measurements (every other day, weekly) give no pairs of
consecutive measured days, so the error persistence rho cannot be measured and falls back
to 0. The run records this (`rho_measured` in the chain's and the FORWARD run's
_meta.json) and summary.md says so in its uncertainty table: the parameter ranges and the
ranges of multi-day quantities are then too narrow.
"""

import json
import os
import unittest

import numpy as np
import pandas as pd

from pyair2stream.config import CommonData
from pyair2stream.results import write_summary
from pyair2stream.uncertainty import estimate_ar1_rho, rho_measured
import test_optimization


def _series(n=400, every=1):
    rng = np.random.default_rng(0)
    obs = np.full(n, -999.0)
    obs[::every] = 10.0 + rng.normal(size=len(obs[::every]))
    mod = np.where(obs != -999.0, obs + rng.normal(size=n), 0.0)
    return mod, obs, np.ones(n, dtype=bool), [(0, n - 1)]


def test_daily_measurements_measure_rho_and_sparse_ones_do_not():
    mod, obs, mask, seg = _series(every=1)
    assert rho_measured(obs, mask, seg)
    for every in (2, 7):
        mod, obs, mask, seg = _series(every=every)
        assert not rho_measured(obs, mask, seg)
        assert estimate_ar1_rho(mod, obs, mask, seg) == 0.0      # the fallback it flags


def test_the_count_stays_within_segments_and_needs_30_pairs():
    obs = np.full(100, -999.0)
    obs[0:31] = 1.0                                               # 30 consecutive pairs
    mask = np.ones(100, dtype=bool)
    assert rho_measured(obs, mask, [(0, 99)])
    assert not rho_measured(obs, mask, [(0, 15), (16, 99)])       # split: 15 + 14 pairs
    mask[5] = False                                               # an unscored day breaks two pairs
    assert not rho_measured(obs, mask, [(0, 99)])


def _summary_with(tmp_path, meta):
    data = CommonData()
    data.station, data.series, data.time_res, data.runmode = "s", "x", "1d", "FORWARD"
    data.version, data.mod_num, data.folder = 5, "CRN", str(tmp_path)
    data.forward_options = {"enable_prediction_intervals": True}
    data.uncertainty_options = {}
    with open(tmp_path / "Forward_Prediction_Ensemble_s_x_1d_meta.json", "w") as f:
        json.dump(meta, f)
    return open(write_summary(data, {}, None, []), encoding="utf-8").read()


def test_the_summary_flags_an_unmeasured_rho(tmp_path):
    text = _summary_with(tmp_path, {"residual_sigma": 0.8, "rho": 0.0, "rho_measured": False})
    assert "| Error persistence ρ | 0.000: **not measured**" in text and "too narrow" in text


def test_the_summary_is_unchanged_when_rho_was_measured_or_not_recorded(tmp_path):
    for meta in ({"rho": 0.86, "rho_measured": True}, {"rho": 0.86}):
        text = _summary_with(tmp_path, meta)
        assert "| Error persistence ρ | 0.860 |" in text and "not measured" not in text


class TestForwardRecordsWhetherRhoWasMeasured(unittest.TestCase):
    # The small model set-up of test_optimization (not inherited, so its tests do not run twice).
    setUp = test_optimization.TestOptimization.setUp
    tearDown = test_optimization.TestOptimization.tearDown

    def test_forward_meta(self):
        from pyair2stream.optimization import forward_mode
        rng = np.random.default_rng(123)
        chain_path = os.path.join(self.data.folder, "dummy_chain.csv")
        pd.DataFrame(rng.random((10, 8)), columns=[f"par_{i+1}" for i in range(8)]).to_csv(chain_path, index=False)
        self.data.forward_options = {'enable_prediction_intervals': True, 'mcmc_chain_path': chain_path,
                                     'residual_sigma': 1.0, 'n_samples': 5, 'random_seed': 42}
        meta_path = os.path.join(self.data.folder, "Forward_Prediction_Ensemble_test_station_test_series_1d_meta.json")
        sidecar_path = chain_path.replace('.csv', '_meta.json')

        def run(sidecar=None, **unc):
            if sidecar is None:
                if os.path.exists(sidecar_path):
                    os.remove(sidecar_path)
            else:
                with open(sidecar_path, 'w') as f:
                    json.dump(sidecar, f)
            self.data.uncertainty_options = {"noise_model": "ar1", "ar1_rho": None, **unc}
            forward_mode(self.data)
            return json.load(open(meta_path))["rho_measured"]

        assert run({"rho": 0.0, "rho_measured": False, "noise_model_used_for_this_run": "ar1"}) is False
        assert run({"rho": 0.5, "rho_measured": True, "noise_model_used_for_this_run": "ar1"}) is True
        assert run({"rho": 0.5, "noise_model_used_for_this_run": "ar1"}) is True     # chains before 0.5.1
        assert run({"rho": 0.0, "rho_measured": False, "noise_model_used_for_this_run": "ar1"},
                   ar1_rho=0.8) is True                                               # set by the user
        assert run(None) is False        # own residuals: 10 measured days, 9 pairs
        assert run({"rho": 0.5, "noise_model_used_for_this_run": "iid"}) is None
