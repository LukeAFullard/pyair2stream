"""
Acceptance tests for docs/audit/04_uncertainty_and_mcmc.md.

Covers: the AR(1) vs iid MCMC likelihood (Defect A / 4.1), `save_ensemble` raw
output (Defect B / 4.2), `pyair2stream/scenario.py` helpers, the `forward_mode`
sigma sidecar fallback (Defect C / 4.3), and reflected walker initialisation on a
bound (Defect D / 4.4).
"""

import os
import shutil
import numpy as np
import pandas as pd
import scipy.signal
import pytest

from pyair2stream.config import CommonData, PI
from pyair2stream.optimization import (
    _iid_log_likelihood,
    _ar1_log_likelihood,
    _reflected_walker_init,
    DE_MCMC_mode,
    forward_mode,
)
from pyair2stream.uncertainty import build_ar1_runs
from pyair2stream import scenario


# ---------------------------------------------------------------------------
# 4.1 -- AR(1) vs iid likelihood posterior width
# ---------------------------------------------------------------------------

def _profile_log_L_iid(mu, y):
    mod = np.full_like(y, mu)
    return _iid_log_likelihood(mod, y)


def _profile_log_L_ar1(mu, y, rho, runs):
    residuals = np.full_like(y, mu) - y
    return _ar1_log_likelihood(residuals, rho, runs)


def _half_width(log_L_fn, ref, search_span):
    """Distance from `ref` at which `log_L_fn` drops by 0.5 from `log_L_fn(ref)`."""
    peak = log_L_fn(ref)

    def f(delta):
        return log_L_fn(ref + delta) - (peak - 0.5)

    lo, hi = 0.0, search_span
    assert f(hi) < 0, "search_span too small to bracket the half-width"
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_ar1_likelihood_recovers_posterior_width_vs_iid():
    """
    Synthetic AR(1) residuals with known rho: the AR(1) concentrated likelihood's
    curvature-implied width for the mean should track the TRUE sampling variability
    of the mean estimator (within 1.5x), while the iid likelihood's width
    understates it by more than 2x -- docs/audit/04, acceptance criterion 1.
    """
    rho_true = 0.9
    sigma_true = 1.0
    N = 2000
    mu_true = 5.0

    def make_series(rng_local):
        eps = rng_local.standard_normal(N)
        epsilon = np.empty(N)
        epsilon[0] = sigma_true * eps[0]
        epsilon[1:] = sigma_true * np.sqrt(1 - rho_true ** 2) * eps[1:]
        e = scipy.signal.lfilter([1.0], [1.0, -rho_true], epsilon)
        return mu_true + e

    rng = np.random.default_rng(12345)

    # "Truth": empirical sampling SD of the sample-mean estimator across many
    # independent realizations of the actual AR(1) data-generating process.
    n_reps = 400
    mu_hats = np.array([make_series(rng).mean() for _ in range(n_reps)])
    true_sd = mu_hats.std(ddof=1)

    # One representative dataset to build the profile likelihoods on.
    y = make_series(rng)
    mu_hat = y.mean()
    runs = build_ar1_runs(np.ones(N, dtype=bool), [(0, N - 1)])

    search_span = 20.0 / np.sqrt(N)
    width_iid = _half_width(lambda m: _profile_log_L_iid(m, y), mu_hat, search_span)
    width_ar1 = _half_width(lambda m: _profile_log_L_ar1(m, y, rho_true, runs), mu_hat, search_span)

    assert width_ar1 <= 1.5 * true_sd
    assert width_ar1 >= true_sd / 1.5
    assert width_iid < true_sd / 2.0


# ---------------------------------------------------------------------------
# Shared fixture builder for the DE_MCMC_mode / forward_mode tests below
# ---------------------------------------------------------------------------

def _build_calibration_data(folder):
    data = CommonData()
    data.n_tot = 365 + 40
    data.date = np.zeros((data.n_tot, 3), dtype=np.int32)
    dates = pd.date_range(start='2000-01-01', periods=data.n_tot, freq='D')
    data.date[:, 0] = dates.year
    data.date[:, 1] = dates.month
    data.date[:, 2] = dates.day

    data.tt = np.zeros(data.n_tot, dtype=np.float64)
    data.Tair = np.zeros(data.n_tot, dtype=np.float64)
    data.Q = np.zeros(data.n_tot, dtype=np.float64)
    data.Twat_obs = np.full(data.n_tot, -999.0, dtype=np.float64)
    data.Twat_mod = np.zeros(data.n_tot, dtype=np.float64)
    data.Twat_obs_agg = np.full(data.n_tot, -999.0, dtype=np.float64)
    data.Twat_mod_agg = np.full(data.n_tot, -999.0, dtype=np.float64)

    data.version = 8
    data.mod_num = 'RK4'
    data.time_res = '1d'
    data.fun_obj = 'RMS'
    data.runmode = 'DE-MCMC'
    data.station = 'test_station'
    data.series = 'test_series'
    data.Qmedia = np.float64(10.0)
    data.Tice_cover = np.float64(0.0)

    rng = np.random.default_rng(1)
    for i in range(data.n_tot):
        data.tt[i] = np.float64(i / 365.0)
        data.Tair[i] = 15.0 + 10.0 * np.sin(2.0 * PI * data.tt[i])
        data.Q[i] = 10.0 + 5.0 * np.cos(2.0 * PI * data.tt[i])
        if i >= 365:
            data.Twat_obs[i] = 12.0 + 8.0 * np.sin(2.0 * PI * data.tt[i]) + 0.1 * rng.standard_normal()

    n_dat = data.n_tot - 365
    data.n_dat = n_dat
    data.I_inf = np.zeros((n_dat, 3), dtype=np.int32)
    data.I_pos = np.zeros(n_dat, dtype=np.int32)
    for k, i in enumerate(range(365, data.n_tot)):
        data.I_inf[k, 0] = k
        data.I_inf[k, 1] = k
        data.I_inf[k, 2] = i
        data.I_pos[k] = i
        data.Twat_obs_agg[i] = data.Twat_obs[i]

    data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
    data.parmin = np.zeros(8, dtype=np.float64)
    data.parmax = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float64)
    data.flag_par = np.ones(8, dtype=np.bool_)

    data.mineff_index = -1e30
    data.folder = folder
    os.makedirs(folder, exist_ok=True)
    return data


# ---------------------------------------------------------------------------
# 4.2 -- save_ensemble reproduces the envelope percentiles
# ---------------------------------------------------------------------------

def test_save_ensemble_matches_envelope_percentiles():
    folder = 'test_report04_save_ensemble_output'
    try:
        data = _build_calibration_data(folder)
        data.n_particles = 2
        data.n_run = 1
        data.mcmc_walkers = 16
        data.mcmc_steps = 10
        data.uncertainty_options = {"noise_model": "iid", "ar1_rho": None, "save_ensemble": True}

        DE_MCMC_mode(data, seed=7)

        env_path = os.path.join(folder, "MCMC_envelopes_test_station_test_series_1d.csv")
        ensemble_path = os.path.join(folder, "MCMC_ensemble_test_station_test_series_1d.npz")
        assert os.path.exists(env_path)
        assert os.path.exists(ensemble_path)

        env_df = pd.read_csv(env_path)
        ensemble, dates = scenario.load_ensemble(ensemble_path)

        assert ensemble.shape[0] > 0
        assert ensemble.shape[1] == len(env_df)

        p_lower = np.percentile(ensemble, 5.0, axis=0)
        p_50 = np.percentile(ensemble, 50.0, axis=0)
        p_upper = np.percentile(ensemble, 95.0, axis=0)

        np.testing.assert_allclose(p_lower, env_df['Twat_mod_lower'].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(p_50, env_df['Twat_mod_p50'].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(p_upper, env_df['Twat_mod_upper'].to_numpy(), atol=1e-9)
    finally:
        if os.path.exists(folder):
            shutil.rmtree(folder)


# ---------------------------------------------------------------------------
# scenario.py helpers
# ---------------------------------------------------------------------------

def test_paired_difference_zero_for_identical_ensembles():
    ens = np.random.default_rng(0).normal(size=(20, 50))
    diff = scenario.paired_difference(ens, ens.copy())
    np.testing.assert_array_equal(diff, np.zeros_like(ens))


def test_paired_difference_shape_mismatch_raises():
    ens_a = np.zeros((5, 10))
    ens_b = np.zeros((5, 11))
    with pytest.raises(ValueError):
        scenario.paired_difference(ens_a, ens_b)


def test_aggregate_and_exceedance():
    dates = pd.date_range('2020-01-01', periods=14, freq='D')
    ensemble = np.tile(np.arange(14, dtype=np.float64), (3, 1))

    agg = scenario.aggregate(ensemble, dates, how='mean', freq='7D')
    assert agg.shape == (3, 2)
    np.testing.assert_allclose(agg[0], [np.mean(np.arange(7)), np.mean(np.arange(7, 14))])

    counts = scenario.exceedance(ensemble, threshold=6.0, consecutive_days=1)
    # days 7..13 (7 days) exceed 6.0 in each row
    np.testing.assert_array_equal(counts, np.full(3, 7))


# ---------------------------------------------------------------------------
# 4.3 -- forward_mode sigma sidecar fallback / hard failure
# ---------------------------------------------------------------------------

def test_forward_mode_raises_when_sigma_unavailable():
    folder = 'test_report04_forward_sigma_output'
    try:
        data = _build_calibration_data(folder)
        data.Twat_obs[:] = -999.0  # pure projection, no observations
        data.runmode = 'FORWARD'

        chain_path = os.path.join(folder, "dummy_chain.csv")
        pd.DataFrame(np.random.rand(10, 8), columns=[f"par_{i+1}" for i in range(8)]).to_csv(chain_path, index=False)
        # Deliberately no _meta.json sidecar and no residual_sigma override.

        data.forward_options = {
            'enable_prediction_intervals': True,
            'mcmc_chain_path': chain_path,
            'n_samples': 5,
            'random_seed': 42,
        }
        data.uncertainty_options = {"noise_model": "iid", "ar1_rho": None}

        with pytest.raises(ValueError):
            forward_mode(data)
    finally:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def test_forward_mode_uses_sigma_from_sidecar():
    folder = 'test_report04_forward_sigma_sidecar_output'
    try:
        data = _build_calibration_data(folder)
        data.Twat_obs[:] = -999.0
        data.runmode = 'FORWARD'

        chain_path = os.path.join(folder, "dummy_chain.csv")
        pd.DataFrame(np.random.rand(10, 8), columns=[f"par_{i+1}" for i in range(8)]).to_csv(chain_path, index=False)
        sidecar_path = chain_path.replace('.csv', '_meta.json')
        import json
        with open(sidecar_path, 'w') as f:
            json.dump({"rho": 0.0, "sigma": 0.5}, f)

        data.forward_options = {
            'enable_prediction_intervals': True,
            'mcmc_chain_path': chain_path,
            'n_samples': 5,
            'random_seed': 42,
        }
        data.uncertainty_options = {"noise_model": "iid", "ar1_rho": None, "max_divergent_fraction": 1.0}

        forward_mode(data)  # should not raise

        env_path = os.path.join(folder, "Forward_Prediction_Envelopes_test_station_test_series_1d.csv")
        assert os.path.exists(env_path)
    finally:
        if os.path.exists(folder):
            shutil.rmtree(folder)


# ---------------------------------------------------------------------------
# 4.4 -- reflected walker initialisation retains spread on a bound
# ---------------------------------------------------------------------------

def test_reflected_walker_init_retains_spread_on_bound():
    rng = np.random.default_rng(3)
    ndim = 3
    lo = np.zeros(ndim)
    hi = np.array([1.0, 2.0, 10.0])
    # The DE optimum sits exactly on the lower bound in every dimension.
    initial = lo.copy()
    scale = 1e-3 * (hi - lo)

    p0 = _reflected_walker_init(initial, scale, lo, hi, nwalkers=32, rng=rng)

    assert p0.shape == (32, ndim)
    assert np.all(p0 >= lo - 1e-12)
    assert np.all(p0 <= hi + 1e-12)
    assert np.all(np.var(p0, axis=0) > 0.0)


def test_reflected_walker_init_raises_on_zero_scale():
    rng = np.random.default_rng(3)
    initial = np.zeros(2)
    scale = np.zeros(2)  # degenerate: no spread requested at all
    lo = np.zeros(2)
    hi = np.ones(2)

    with pytest.raises(ValueError):
        _reflected_walker_init(initial, scale, lo, hi, nwalkers=10, rng=rng)
