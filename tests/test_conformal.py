"""Split conformal margins of the prediction intervals (docs/METHODS.md §13)."""

import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from pyair2stream import scenario
from pyair2stream.config import CommonData
from pyair2stream.cross_validation import FoldResult, check_interval_coverage, conformal_margin, conformal_margins
from pyair2stream.main import CONFORMAL_FILE, write_conformal_margins
from pyair2stream.optimization import forward_mode
from pyair2stream.uncertainty import generate_ar1_noise

from test_ensemble_provenance import _build_forward_data, _write_chain_csv


def _folds(n_years, true_sigma, model_sigma=0.6, rho=0.8, seed=0):
    """Held-out years whose errors are AR(1) of size `true_sigma`, predicted with an error
    model of size `model_sigma`."""
    rng = np.random.default_rng(seed)
    folds = []
    for k in range(n_years):
        dates = pd.date_range(f"{2000 + k}-01-01", f"{2000 + k}-12-31")
        sim = 10 - 8 * np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 20) / 365)
        obs = sim + generate_ar1_noise(len(dates), true_sigma, rho, [(0, len(dates) - 1)], rng)
        obs[rng.random(len(dates)) < 0.05] = -999.0
        folds.append(FoldResult(fold_id=k, label=str(2000 + k), held_out_start=dates[0], held_out_end=dates[-1],
                                n_obs_held_out=int((obs != -999).sum()), par_best=np.zeros(8), nse=np.nan,
                                kge=np.nan, rmse=np.nan, obs_held_out=obs, sim_held_out=sim, dates_held_out=dates,
                                sigma=model_sigma, rho=rho, years_held_out=dates.year.to_numpy()))
    return folds


def test_every_held_out_year_has_the_same_weight():
    # One year of 1000 values inside (score -1), one of 10 values outside (score 2): with equal
    # weights per year, 60% of the weight is reached only at the second year's scores.
    scores = [np.full(1000, -1.0), np.full(10, 2.0)]
    assert conformal_margin(scores, 50) == -1.0
    assert conformal_margin(scores, 60) == 2.0
    assert conformal_margin([np.array([])], 90) != conformal_margin([np.array([])], 90)   # NaN


def test_margins_are_near_zero_when_the_error_model_is_right():
    table = conformal_margins(_folds(30, true_sigma=0.6), levels=[90], seed=1, n_simulations=400)
    assert set(table.window_days) == {1, 7, 30}
    days = table[table.window_days == 1].iloc[0]
    assert abs(days.margin) < 0.1                       # sigma 0.6: the 90% range is about ±1 degC
    assert days.inside_before == pytest.approx(0.9, abs=0.02)
    assert days.inside_after == pytest.approx(0.9, abs=0.02)


def test_margins_restore_coverage_when_the_intervals_are_too_narrow():
    folds = _folds(30, true_sigma=1.2)
    table = conformal_margins(folds, levels=[80, 90], seed=1, n_simulations=400)
    for r in table.itertuples():
        assert r.margin > 0
        assert r.inside_before < r.level / 100 - 0.1
        assert r.inside_after == pytest.approx(r.level / 100, abs=0.03)
    # The scores come from the same simulations as check_interval_coverage.
    coverage = check_interval_coverage(folds, levels=[90], seed=1, n_simulations=400)
    days = table[(table.window_days == 1) & (table.level == 90)].iloc[0]
    assert days.inside_before == pytest.approx(coverage["daily inside"].iloc[0])


def test_no_margins_from_fewer_than_three_years():
    assert conformal_margins(_folds(2, true_sigma=0.6), levels=[90], seed=1, n_simulations=50).empty


def test_widen_range():
    lo, hi = scenario.widen_range([1.0, 5.0], [3.0, 9.0], 0.5, floor=0.8)
    np.testing.assert_allclose(lo, [0.8, 4.5])
    np.testing.assert_allclose(hi, [3.5, 9.5])
    lo, hi = scenario.widen_range([1.0, 5.0], [3.0, 9.0], -1.5)   # narrowed, never past the midpoint
    np.testing.assert_allclose(lo, [2.0, 6.5])
    np.testing.assert_allclose(hi, [2.0, 7.5])
    ens = np.random.default_rng(0).normal(10, 1, (2000, 3))
    lo, hi = scenario.conformal_range(ens, 90, 0.3)
    base_lo, base_hi = scenario.central_range(ens, 90, axis=0)
    np.testing.assert_allclose(lo, base_lo - 0.3)
    np.testing.assert_allclose(hi, base_hi + 0.3)


def _data_for_margins(folder):
    data = CommonData()
    data.folder, data.station, data.series, data.version = folder, "test_station", "test_series", 8
    data.random_seed = 1
    data.uncertainty_options = {"noise_model": "ar1", "rho_timescale": "weekly", "prediction_interval": 90.0}
    return data


def test_the_file_written_by_the_cross_validation(tmp_path):
    data = _data_for_margins(str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        write_conformal_margins(data, _folds(5, true_sigma=1.2), 90.0)
    table = scenario.read_conformal_margins(os.path.join(str(tmp_path), CONFORMAL_FILE))
    assert list(table.columns) == list(scenario.CONFORMAL_COLUMNS)
    assert set(table.level) == {50, 80, 90, 95}
    assert (table.station == "test_station_test_series").all() and (table.noise_model == "ar1").all()
    m7 = scenario.conformal_margin(os.path.join(str(tmp_path), CONFORMAL_FILE), 90, window_days=7)
    assert m7 == float(table[(table.window_days == 7) & (table.level == 90)].margin.iloc[0])
    with pytest.raises(ValueError, match="No conformal margin for 14-day values"):
        scenario.conformal_margin(table, 90, window_days=14)


class TestForward:
    def _setup(self, tmp_path, margin=0.5, **row):
        folder = str(tmp_path / "fwd")
        data = _build_forward_data(folder)
        chain = os.path.join(folder, "chain.csv")
        _write_chain_csv(chain, n_rows=20)
        with open(chain.replace(".csv", "_meta.json"), "w") as f:
            json.dump({"sigma": 0.8, "rho": 0.5, "noise_model_used_for_this_run": "ar1",
                       "rho_timescale": "weekly"}, f)
        base = {"station": "test_station_test_series", "version": 8, "window_days": 1, "level": 90.0,
                "margin": margin, "held_out_years": 6, "values": 2000, "inside_before": 0.85,
                "inside_after": 0.9, "noise_model": "ar1", "rho_timescale": "weekly"}
        path = str(tmp_path / CONFORMAL_FILE)
        pd.DataFrame([{**base, **row}]).to_csv(path, index=False)
        data.forward_options = {"enable_prediction_intervals": True, "mcmc_chain_path": chain, "n_samples": 20,
                                "random_seed": 1, "conformal_margins": path}
        data.uncertainty_options = {"noise_model": "ar1", "prediction_interval": 90.0}
        return data, folder

    def test_the_envelope_gets_the_widened_interval(self, tmp_path):
        data, folder = self._setup(tmp_path, margin=0.5)
        data.Tice_cover = np.float64(0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            forward_mode(data)
        env = pd.read_csv(os.path.join(folder, "Forward_Prediction_Envelopes_test_station_test_series_1d.csv"))
        np.testing.assert_allclose(env.Twat_mod_upper_conformal, env.Twat_mod_upper + 0.5)
        np.testing.assert_allclose(env.Twat_mod_lower_conformal, np.maximum(env.Twat_mod_lower - 0.5, 0.0))
        meta = json.load(open(os.path.join(folder, "Forward_Prediction_Ensemble_test_station_test_series_1d_meta.json")))
        assert meta["conformal_margins"]["margin"] == 0.5
        assert len(meta["conformal_margins"]["sha256"]) == 64

    @pytest.mark.parametrize("row, message", [({"version": 5}, "model version 5"),
                                              ({"noise_model": "iid"}, "noise_model 'iid'"),
                                              ({"rho_timescale": "daily"}, "rho_timescale 'daily'"),
                                              ({"level": 80.0}, "no margin for single days at the 90%")])
    def test_a_file_for_another_model_is_refused(self, tmp_path, row, message):
        data, _ = self._setup(tmp_path, **row)
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match=message):
            forward_mode(data)

    def test_an_overridden_error_model_is_refused(self, tmp_path):
        data, _ = self._setup(tmp_path)
        data.forward_options["residual_sigma"] = 1.0
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match="overrides the error model"):
            forward_mode(data)
