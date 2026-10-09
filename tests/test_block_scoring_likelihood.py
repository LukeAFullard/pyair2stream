"""The least-squares likelihood with weekly or monthly scoring (time_resolution 'Nw' or '1m')."""
import json

import numpy as np
import pytest

from pyair2stream.optimization import _least_squares_log_likelihood
from pyair2stream.uncertainty import (mean_error_variance_factor, scored_error_variance_factor, scoring_block_days,
                                      weekly_mean_correlation)


def _ar1_paths(n_paths, n, rho, rng):
    e = rng.standard_normal((n_paths, n))
    x = np.empty((n_paths, n))
    x[:, 0] = e[:, 0]
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
    return x


def test_block_days():
    assert scoring_block_days("1d") == 1
    assert scoring_block_days("1w") == 7
    assert scoring_block_days("2w") == 14
    assert scoring_block_days("12w") == 84
    assert scoring_block_days("1m") == 30
    for bad in ("daily", "", "0w", "w", "2m", "1y"):
        with pytest.raises(ValueError):
            scoring_block_days(bad)


def test_daily_factor_is_the_usual_effective_sample_size():
    for rho in (0.0, 0.3, 0.8, 0.99):
        assert mean_error_variance_factor(rho, 1) == pytest.approx((1 + rho) / (1 - rho))
    assert mean_error_variance_factor(0.0, 7) == 1.0
    assert mean_error_variance_factor(-0.2, 7) == 1.0
    # Longer blocks: less correlated from block to block, so a smaller factor.
    f = [mean_error_variance_factor(0.9, m) for m in (1, 7, 14, 30)]
    assert all(a > b > 1.0 for a, b in zip(f, f[1:]))


def test_block_correlations_decay_as_stated():
    # Block means k >= 1 apart: correlation r_b * rho**(m * (k - 1)).
    rng = np.random.default_rng(3)
    rho, m = 0.9, 7
    x = _ar1_paths(1, 400_000, rho, rng)[0]
    b = x[: len(x) // m * m].reshape(-1, m).mean(axis=1)
    r_b = weekly_mean_correlation(rho, m)
    for k in (1, 2, 3):
        assert np.corrcoef(b[:-k], b[k:])[0, 1] == pytest.approx(r_b * rho ** (m * (k - 1)), abs=0.015)


@pytest.mark.parametrize("rho,m", [(0.7, 7), (0.9, 7), (0.95, 7), (0.9, 30)])
def test_factor_matches_the_variance_of_a_mean_of_block_errors(rho, m):
    rng = np.random.default_rng(11)
    n_blocks, n_paths = 80, 3000
    b = _ar1_paths(n_paths, n_blocks * m, rho, rng).reshape(n_paths, n_blocks, m).mean(axis=2)
    measured = b.mean(axis=1).var() / (b.var(axis=1, ddof=1).mean() / n_blocks)
    # Finite n: the sample variance of correlated block means is biased low by about 2*factor/n.
    assert measured == pytest.approx(mean_error_variance_factor(rho, m), rel=0.12)


def test_least_squares_likelihood_counts_blocks_with_the_block_factor():
    rng = np.random.default_rng(5)
    residuals = np.full(700, -999.0)
    idx = np.arange(3, 700, 7)                      # one scored value per week
    residuals[idx] = rng.normal(0, 0.4, len(idx))
    runs = [np.array([i]) for i in idx]
    n, sse = len(idx), float(np.sum(residuals[idx] ** 2))
    for rho in (0.0, 0.6, 0.93):
        factor = scored_error_variance_factor(idx, rho, 7)
        assert factor == pytest.approx(mean_error_variance_factor(rho, 7), rel=0.02)    # 100 consecutive weeks
        expected = -0.5 * n / factor * np.log(sse / n)
        assert _least_squares_log_likelihood(residuals, rho, runs, 7) == pytest.approx(expected)
    # Daily scoring: n_eff = n / factor, close to n (1 - rho) / (1 + rho) for consecutive days.
    e = rng.normal(0, 0.4, 500)
    runs = [np.arange(500)]
    rho = 0.8
    factor = scored_error_variance_factor(np.arange(500), rho)
    assert factor == pytest.approx((1 + rho) / (1 - rho), rel=0.01)
    assert _least_squares_log_likelihood(e, rho, runs) == pytest.approx(-0.5 * 500 / factor * np.log(np.sum(e ** 2) / 500))


def _run_weekly(tmp_path, likelihood):
    import yaml
    from pyair2stream.io import read_calibration, read_Tseries
    from pyair2stream.model import aggregation, statis
    from pyair2stream.optimization import DE_MCMC_mode
    cfg = {"station_name": "S", "series": "c", "version": 3, "run_mode": "DE-MCMC", "random_seed": 1,
           "time_resolution": "1w", "prc": 0.6,
           "optimization": {"n_run": 30, "n_particles": 10, "mcmc_walkers": 8, "mcmc_steps": 2000},
           "uncertainty_options": {"likelihood": likelihood, "strict_convergence": False},
           "parameter_bounds": {"min": [0, 0, 0, -1, 0, 0, 0, -1], "max": [15, 1.5, 5, 1, 20, 10, 1, 5]},
           "paths": {"input_data": "data/switzerland/MAH_2369_calibration.csv", "output_dir": str(tmp_path / "o")}}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    data = read_calibration(str(path))
    read_Tseries(data, "c")
    aggregation(data)
    statis(data)
    DE_MCMC_mode(data, seed=1)
    return json.load(open(tmp_path / "o" / "MCMC_chain_S_c_1w_meta.json"))


def test_weekly_scoring_records_the_block_factor(tmp_path, capsys):
    meta = _run_weekly(tmp_path, "least_squares")
    assert meta["scoring_block_days"] == 7
    # From the spacing of the scored weeks (some weeks are not scored with prc 0.6): near the
    # factor for consecutive weeks.
    assert meta["likelihood_variance_factor"] == pytest.approx(mean_error_variance_factor(meta["rho"], 7), rel=0.15)
    # Much smaller than the daily factor for the same rho, which would over-widen the posterior.
    assert meta["likelihood_variance_factor"] < 0.5 * (1 + meta["rho"]) / (1 - meta["rho"])
    assert "exact AR(1) likelihood treats" not in capsys.readouterr().out


def test_weekly_scoring_with_the_exact_likelihood_warns(tmp_path, capsys):
    meta = _run_weekly(tmp_path, "exact")
    assert meta["scoring_block_days"] == 7
    assert meta["likelihood_variance_factor"] is None
    assert "exact AR(1) likelihood treats the scored errors as independent" in capsys.readouterr().out
