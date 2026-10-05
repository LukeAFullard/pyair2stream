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
    # The weekly rho reproduces the week-to-week correlation of the errors.
    m = x[: len(x) // 7 * 7].reshape(-1, 7).mean(axis=1)
    assert abs(weekly_mean_correlation(weekly) - np.corrcoef(m[:-1], m[1:])[0, 1]) < 1e-6


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
