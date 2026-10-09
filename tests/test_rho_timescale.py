"""rho estimated from week-to-week persistence (uncertainty_options.rho_timescale)."""
import numpy as np
import pytest

from pyair2stream.uncertainty import (estimate_ar1_rho, estimate_ar1_rho_weekly, estimate_rho,
                                      weekly_mean_correlation, MIN_WEEK_PAIRS_FOR_RHO_ESTIMATE)


def _ar1(n, rho, rng, sigma=0.5):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = sigma * e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + sigma * np.sqrt(1 - rho ** 2) * e[t]
    return x


def _estimate(err, method, mask=None, segments=None):
    n = len(err)
    obs = np.full(n, 10.0)
    mask = np.ones(n, bool) if mask is None else mask
    segments = [(0, n - 1)] if segments is None else segments
    return method(obs + err, obs, mask, segments)


def test_weekly_mean_correlation_matches_simulation():
    rng = np.random.default_rng(1)
    x = _ar1(350_000, 0.8, rng)
    m = x[: len(x) // 7 * 7].reshape(-1, 7).mean(axis=1)
    assert abs(np.corrcoef(m[:-1], m[1:])[0, 1] - weekly_mean_correlation(0.8)) < 0.01
    assert weekly_mean_correlation(0.0) == 0.0
    values = [weekly_mean_correlation(r) for r in np.linspace(0, 0.99, 50)]
    assert np.all(np.diff(values) > 0)


def test_weekly_and_daily_agree_on_ar1_errors():
    # Eight years of AR(1) errors, as in the validation's synthetic data: both estimate the same rho.
    est_w, est_d = [], []
    for seed in range(20):
        x = _ar1(2922, 0.7, np.random.default_rng(seed))
        est_w.append(_estimate(x, estimate_ar1_rho_weekly))
        est_d.append(_estimate(x, estimate_ar1_rho))
    assert abs(np.mean(est_w) - 0.7) < 0.03
    assert abs(np.mean(est_d) - 0.7) < 0.02
    assert np.std(est_w) < 0.06


def test_weekly_sees_a_slow_component_that_daily_misses():
    # Fast (2-day) plus slow (25-day) errors, the structure measured on real rivers.
    rng = np.random.default_rng(3)
    x = np.sqrt(0.6) * _ar1(20_000, 0.55, rng) + np.sqrt(0.4) * _ar1(20_000, 0.96, rng)
    daily, weekly = _estimate(x, estimate_ar1_rho), _estimate(x, estimate_ar1_rho_weekly)
    assert daily < 0.85 < weekly
    # The weekly rho reproduces the correlation between each 7-day mean and the one a week later.
    m = np.convolve(x, np.ones(7) / 7, mode="valid")
    assert abs(weekly_mean_correlation(weekly) - np.corrcoef(m[:-7], m[7:])[0, 1]) < 1e-6


def test_incomplete_weeks_and_segment_boundaries_are_not_paired():
    rng = np.random.default_rng(4)
    x = _ar1(2922, 0.7, rng)
    mask = np.ones(len(x), bool)
    mask[::10] = False                          # a missing day in most weeks
    full = _estimate(x, estimate_ar1_rho_weekly)
    gappy = _estimate(x, estimate_ar1_rho_weekly, mask=mask)
    assert 0.4 < gappy < 0.95 and gappy != full
    # Two segments: a week never pairs with one in the other segment.
    segs = [(0, 1460), (1461, 2921)]
    assert 0.5 < _estimate(x, estimate_ar1_rho_weekly, segments=segs) < 0.9


def test_falls_back_to_daily_with_too_few_weeks():
    x = _ar1(7 * (MIN_WEEK_PAIRS_FOR_RHO_ESTIMATE - 2), 0.6, np.random.default_rng(5))
    assert _estimate(x, estimate_ar1_rho_weekly) == _estimate(x, estimate_ar1_rho)


def test_limits():
    alternating = np.repeat(np.tile([1.0, -1.0], 300), 7)       # weekly means alternate: r < 0
    assert _estimate(alternating, estimate_ar1_rho_weekly) == 0.0
    trend = np.linspace(-1, 1, 3000)                             # near-perfect persistence
    assert _estimate(trend, estimate_ar1_rho_weekly) == pytest.approx(0.99)


def test_weekly_option_is_never_below_the_daily_estimate():
    # The week-to-week estimate alone is imprecise for weakly correlated errors; the 'weekly'
    # option takes the larger of the two, so it never gives less persistence than 'daily'.
    for seed in range(30):
        rng = np.random.default_rng(100 + seed)
        x = _ar1(2922, [0.0, 0.3, 0.7][seed % 3], rng)
        obs = np.full(len(x), 10.0)
        args = (obs + x, obs, np.ones(len(x), bool), [(0, len(x) - 1)])
        assert estimate_rho(*args, timescale="weekly") >= estimate_rho(*args, timescale="daily")


def test_estimate_rho_dispatch():
    x = _ar1(2922, 0.7, np.random.default_rng(7))
    obs = np.full(len(x), 10.0)
    args = (obs + x, obs, np.ones(len(x), bool), [(0, len(x) - 1)])
    assert estimate_rho(*args, timescale="daily") == estimate_ar1_rho(*args)
    assert estimate_rho(*args, timescale="weekly") == max(estimate_ar1_rho(*args), estimate_ar1_rho_weekly(*args))
    with pytest.raises(ValueError):
        estimate_rho(*args, timescale="monthly")


def test_config_option(tmp_path):
    import yaml
    from pyair2stream.io import read_calibration
    from pyair2stream.config import DEFAULT_RHO_TIMESCALE
    assert DEFAULT_RHO_TIMESCALE == "weekly"
    base = {"station_name": "S", "series": "c", "version": 5, "run_mode": "DE",
            "paths": {"input_data": "data/switzerland/MAH_2369_calibration.csv", "output_dir": str(tmp_path / "o")}}
    for value, ok in ((None, "weekly"), ("daily", "daily"), ("weekly", "weekly"), ("monthly", None)):
        cfg = dict(base)
        if value is not None:
            cfg["uncertainty_options"] = {"rho_timescale": value}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(cfg))
        if ok is None:
            with pytest.raises(ValueError, match="rho_timescale"):
                read_calibration(str(path))
        else:
            assert read_calibration(str(path)).uncertainty_options["rho_timescale"] == ok


def test_empty_config_sections_use_the_defaults(tmp_path):
    """A section whose every line is commented out is null in YAML; it means the defaults."""
    from pyair2stream.io import read_calibration
    path = tmp_path / "c.yaml"
    path.write_text("station_name: S\nseries: c\nversion: 5\nrun_mode: DE\n"
                    "paths:\n  input_data: data/switzerland/MAH_2369_calibration.csv\n"
                    f"  output_dir: {tmp_path / 'o'}\n"
                    "uncertainty_options:\n#  rho_timescale: daily\n"
                    "optimization:\nforward_options:\ncross_validation:\nparameter_bounds:\n")
    data = read_calibration(str(path))
    assert data.uncertainty_options["rho_timescale"] == "weekly"
    assert data.uncertainty_options["likelihood"] == "least_squares"


def test_exact_likelihood_uses_the_day_to_day_rho(tmp_path):
    """With likelihood 'exact', the likelihood removes the day-to-day correlation with the
    lag-1 rho; rho_timescale sets the rho of the prediction noise (recorded as 'rho')."""
    import json
    import yaml
    from pyair2stream.io import read_calibration, read_Tseries
    from pyair2stream.model import aggregation, statis
    from pyair2stream.optimization import DE_MCMC_mode
    cfg = {"station_name": "S", "series": "c", "version": 3, "run_mode": "DE-MCMC", "random_seed": 1,
           "optimization": {"n_run": 30, "n_particles": 10, "mcmc_walkers": 8, "mcmc_steps": 2000},
           "uncertainty_options": {"likelihood": "exact", "rho_timescale": "weekly", "strict_convergence": False},
           "parameter_bounds": {"min": [-5, -5, -5, -1, 0, 0, 0, -1], "max": [15, 1.5, 5, 1, 20, 10, 1, 5]},
           "paths": {"input_data": "data/switzerland/MAH_2369_calibration.csv", "output_dir": str(tmp_path / "o")}}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    data = read_calibration(str(path))
    read_Tseries(data, "c")
    aggregation(data)
    statis(data)
    DE_MCMC_mode(data, seed=1)
    meta = json.load(open(tmp_path / "o" / "MCMC_chain_S_c_1d_meta.json"))
    assert meta["rho_timescale"] == "weekly"
    assert meta["rho_likelihood"] <= meta["rho"]
    assert 0.5 < meta["rho_likelihood"] < 0.9


# --- Edge cases ------------------------------------------------------------------------------

def test_windows_never_cross_segment_boundaries():
    # Two segments whose errors have opposite fixed offsets. A window straddling the boundary
    # would see a jump; the estimate must equal that of the two segments estimated separately.
    rng = np.random.default_rng(10)
    a, b = _ar1(1500, 0.7, rng) + 0.5, _ar1(1500, 0.7, rng) - 0.5
    x = np.concatenate([a, b])
    obs = np.full(len(x), 10.0)
    joint = estimate_ar1_rho_weekly(obs + x, obs, np.ones(len(x), bool), [(0, 1499), (1500, 2999)])
    # Same data with a 40-day hole between the parts, as one segment: no window can span the hole.
    x2 = np.concatenate([a, np.zeros(40), b])
    obs2 = np.full(len(x2), 10.0)
    obs2[1500:1540] = -999.0
    holed = estimate_ar1_rho_weekly(obs2 + x2, obs2, np.ones(len(x2), bool), [(0, len(x2) - 1)])
    assert joint == pytest.approx(holed, abs=1e-12)


def test_short_segments_and_unscored_days_are_ignored():
    rng = np.random.default_rng(11)
    x = _ar1(2922, 0.7, rng)
    obs = np.full(len(x), 10.0)
    full = estimate_ar1_rho_weekly(obs + x, obs, np.ones(len(x), bool), [(0, 2921)])
    # Adding 13-day segments (too short to hold a pair of windows) changes nothing.
    segs = [(0, 2921), (2922, 2934)]
    x3 = np.concatenate([x, 5 * np.ones(13)])
    obs3 = np.full(len(x3), 10.0)
    assert estimate_ar1_rho_weekly(obs3 + x3, obs3, np.ones(len(x3), bool), segs) == pytest.approx(full)
    # Days marked unscored (eval_mask False) are treated like missing days.
    mask = np.ones(len(x), bool)
    mask[100:130] = False
    obs4 = obs.copy()
    obs4[100:130] = -999.0
    assert (estimate_ar1_rho_weekly(obs + x, obs, mask, [(0, 2921)])
            == pytest.approx(estimate_ar1_rho_weekly(obs4 + x, obs4, np.ones(len(x), bool), [(0, 2921)])))


def test_constant_errors_give_zero():
    n = 1000
    obs = np.full(n, 10.0)
    args = (obs + 0.3, obs, np.ones(n, bool), [(0, n - 1)])
    assert estimate_ar1_rho_weekly(*args) == 0.0
    assert estimate_rho(*args, timescale="weekly") == estimate_rho(*args, timescale="daily") == 0.0


def test_fallback_threshold_is_exact():
    from pyair2stream.uncertainty import WEEK
    # One segment of n days holds n - 2*WEEK + 1 pairs of windows a week apart.
    need = WEEK * MIN_WEEK_PAIRS_FOR_RHO_ESTIMATE
    rng = np.random.default_rng(12)
    for n, falls_back in ((need + 2 * WEEK - 1, False), (need + 2 * WEEK - 2, True)):
        x = 0.5 * np.repeat(rng.standard_normal(n // 7 + 1), 7)[:n] + 0.1 * rng.standard_normal(n)
        obs = np.full(n, 10.0)
        args = (obs + x, obs, np.ones(n, bool), [(0, n - 1)])
        weekly, daily = estimate_ar1_rho_weekly(*args), estimate_ar1_rho(*args)
        assert (weekly == daily) == falls_back


def test_estimates_are_deterministic():
    x = _ar1(2922, 0.75, np.random.default_rng(13))
    obs = np.full(len(x), 10.0)
    args = (obs + x, obs, np.ones(len(x), bool), [(0, len(x) - 1)])
    assert estimate_rho(*args, timescale="weekly") == estimate_rho(*args, timescale="weekly")


def test_weekly_resolution_calibration_records_rho_from_daily_errors(tmp_path):
    """With weekly scoring, rho is still estimated from the daily errors at the chosen time
    scale: it sets the daily prediction noise and, through the correlation of block means,
    the effective sample size of the least-squares likelihood."""
    import json
    import yaml
    from pyair2stream.io import read_calibration, read_Tseries
    from pyair2stream.model import aggregation, statis
    from pyair2stream.optimization import DE_MCMC_mode
    cfg = {"station_name": "S", "series": "c", "version": 3, "run_mode": "DE-MCMC", "random_seed": 1,
           "time_resolution": "1w", "prc": 0.6,
           "optimization": {"n_run": 30, "n_particles": 10, "mcmc_walkers": 8, "mcmc_steps": 2000},
           "uncertainty_options": {"strict_convergence": False},
           # Physically sensible bounds: with weekly scoring and a very short search, negative
           # relaxation rates can give a day-to-day zigzag whose weekly means still fit.
           "parameter_bounds": {"min": [0, 0, 0, -1, 0, 0, 0, -1], "max": [15, 1.5, 5, 1, 20, 10, 1, 5]},
           "paths": {"input_data": "data/switzerland/MAH_2369_calibration.csv", "output_dir": str(tmp_path / "o")}}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    data = read_calibration(str(path))
    read_Tseries(data, "c")
    aggregation(data)
    statis(data)
    DE_MCMC_mode(data, seed=1)
    meta = json.load(open(tmp_path / "o" / "MCMC_chain_S_c_1w_meta.json"))
    assert meta["rho_timescale"] == "weekly"
    assert 0.6 < meta["rho"] < 0.99


def test_rho_at_its_limit_warns(capsys):
    """Errors that drift with the seasons (a systematic error) push rho to its cap; say so,
    as a printed warning, so it reaches summary.md and RunResult.messages."""
    n = 4 * 365
    err = 0.8 * np.sin(2 * np.pi * np.arange(n) / 365.0)
    assert _estimate(err, lambda *a: estimate_rho(*a, "weekly")) == 0.99
    out = capsys.readouterr().out
    assert "Warning: rho reached its upper limit" in out and "bias_by_month" in out
    _estimate(_ar1(n, 0.7, np.random.default_rng(2)), lambda *a: estimate_rho(*a, "weekly"))
    assert "upper limit" not in capsys.readouterr().out


@pytest.mark.parametrize("sidecar_extra,option,expect_note", [
    ({}, "weekly", True),                              # a 0.4.1 chain (consecutive days) under the new default
    ({}, "daily", False),
    ({"rho_timescale": "weekly"}, "weekly", False),
    ({"rho_timescale": "weekly"}, "daily", True),
])
def test_forward_says_when_the_chains_rho_is_from_another_time_scale(tmp_path, capsys, sidecar_extra, option,
                                                                    expect_note):
    import json
    import pandas as pd
    from tests.test_report04_uncertainty_and_mcmc import _build_calibration_data
    from pyair2stream.optimization import forward_mode
    data = _build_calibration_data(str(tmp_path / "f"))
    data.Twat_obs[:] = -999.0
    data.runmode = "FORWARD"
    chain_path = str(tmp_path / "chain.csv")
    pd.DataFrame(np.random.default_rng(0).random((10, 8)) * 0.5,
                 columns=[f"par_{i + 1}" for i in range(8)]).to_csv(chain_path, index=False)
    with open(chain_path.replace(".csv", "_meta.json"), "w") as f:
        json.dump({"rho": 0.8, "sigma": 0.5, **sidecar_extra}, f)
    data.forward_options = {"enable_prediction_intervals": True, "mcmc_chain_path": chain_path,
                            "n_samples": 5, "random_seed": 42}
    data.uncertainty_options = {"noise_model": "ar1", "ar1_rho": None, "rho_timescale": option,
                                "max_divergent_fraction": 1.0}
    forward_mode(data)
    out = capsys.readouterr().out
    assert "Using rho=0.8000 carried from calibration run" in out
    assert ("was estimated at the" in out) == expect_note
