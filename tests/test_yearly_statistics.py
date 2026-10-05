"""
Yearly statistics of saved simulations (scenario.year_statistics), the cross-validated
check of their predicted ranges (cross_validation.check_yearly_statistics) and the
correction for a bias found by that check (scenario.correct_statistic).
docs/METHODS.md §11 and §13.
"""

import numpy as np
import pandas as pd
import pytest

from pyair2stream import scenario
from pyair2stream.cross_validation import FoldResult, check_yearly_statistics, warmest_months
from pyair2stream.uncertainty import generate_ar1_noise


def test_year_statistics_by_hand():
    dates = pd.date_range("2001-01-01", "2002-12-31")
    x = np.zeros(len(dates))
    i = dates.get_loc(pd.Timestamp("2001-07-10"))
    x[i:i + 7] = [10, 11, 12, 13, 14, 15, 16]       # 7-day mean 13, highest day 16
    x[dates.get_loc(pd.Timestamp("2002-08-01"))] = 30
    x[dates.get_loc(pd.Timestamp("2002-08-02"))] = np.nan
    out = scenario.year_statistics(x, dates, threshold=12.5)
    assert out[2001]["highest daily mean"][0] == 16
    assert out[2001]["highest 7-day mean"][0] == pytest.approx(13.0)
    assert out[2001]["days above threshold"][0] == 4
    assert out[2002]["highest daily mean"][0] == 30
    # Windows containing the missing day are not used; others still are.
    assert out[2002]["highest 7-day mean"][0] == pytest.approx(30 / 7)
    assert out[2002]["days above threshold"][0] == 1


def test_year_statistics_matches_pandas_rolling_per_member():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-01-01", periods=800)
    ens = rng.normal(15, 3, (5, len(dates)))
    ens[:, rng.random(len(dates)) < 0.1] = np.nan
    out = scenario.year_statistics(ens, dates, threshold=18.0)
    for k in range(5):
        s = pd.Series(ens[k], index=dates)
        for year in (2005, 2006, 2007):
            y = s[s.index.year == year]
            assert out[year]["highest daily mean"][k] == pytest.approx(y.max())
            week = pd.Series(y.to_numpy()).rolling(7).mean()
            assert out[year]["highest 7-day mean"][k] == pytest.approx(week.max())
            assert out[year]["days above threshold"][k] == (y > 18).sum()


def test_year_labels_can_be_water_years():
    dates = pd.date_range("2001-01-01", "2002-12-31")
    labels = np.where(dates.month >= 10, dates.year + 1, dates.year)
    out = scenario.year_statistics(np.arange(len(dates), dtype=float), dates, 1e9, years=labels)
    assert sorted(out) == [2001, 2002, 2003]
    assert out[2002]["highest daily mean"][0] == dates.get_loc(pd.Timestamp("2002-09-30"))


def test_pit_splits_ties_and_is_uniform_when_right():
    rng = np.random.default_rng(1)
    assert scenario.pit(np.array([1, 2, 2, 3]), 0, rng) == 0
    assert scenario.pit(np.array([1, 2, 2, 3]), 4, rng) == 1
    p = [scenario.pit(np.array([1, 2, 2, 3]), 2, rng) for _ in range(2000)]
    assert 0.25 <= min(p) and max(p) <= 0.75 and np.mean(p) == pytest.approx(0.5, abs=0.02)
    sims = rng.normal(size=1000)
    pits = np.array([scenario.pit(sims, v, rng) for v in rng.normal(size=4000)])
    assert np.mean((pits >= 0.05) & (pits <= 0.95)) == pytest.approx(0.9, abs=0.02)


def test_correct_statistic_shifts_by_the_mean_deviation_with_its_uncertainty():
    values = np.zeros(200000)
    dev = np.array([-1.0, -0.5, -0.8, -0.7, -1.0])
    out = scenario.correct_statistic(values, dev, seed=3)
    assert out.shape == values.shape
    assert np.median(out) == pytest.approx(dev.mean(), abs=0.01)
    # The spread is the standard error of the mean, from a t distribution with n - 1 df.
    from scipy.stats import t
    se = dev.std(ddof=1) / np.sqrt(len(dev))
    assert np.percentile(out, 95) - np.median(out) == pytest.approx(t.ppf(0.95, 4) * se, rel=0.03)
    # Same seed, same result; deviations that are NaN are ignored.
    np.testing.assert_array_equal(out, scenario.correct_statistic(values, np.append(dev, np.nan), seed=3))
    with pytest.raises(ValueError, match="at least 3"):
        scenario.correct_statistic(values, [0.1, 0.2])


def _folds(n_years, offset=0.0, sigma=0.6, rho=0.8, seed=0):
    """Held-out years whose measurements are the simulation plus AR(1) error of the stated
    size (plus a constant offset): the check's error model is then exactly right."""
    rng = np.random.default_rng(seed)
    folds = []
    for k in range(n_years):
        dates = pd.date_range(f"{2000 + k}-01-01", f"{2000 + k}-12-31")
        doy = dates.dayofyear.to_numpy()
        sim = 10 - 8 * np.cos(2 * np.pi * (doy - 20) / 365) + rng.normal(0, 1, len(dates))
        obs = sim + generate_ar1_noise(len(dates), sigma, rho, [(0, len(dates) - 1)], rng) + offset
        obs[rng.random(len(dates)) < 0.05] = -999.0          # a few unmeasured days
        folds.append(FoldResult(fold_id=k, label=str(2000 + k), held_out_start=dates[0],
                                held_out_end=dates[-1], n_obs_held_out=int((obs != -999).sum()),
                                par_best=np.zeros(8), nse=np.nan, kge=np.nan, rmse=np.nan,
                                obs_held_out=obs, sim_held_out=sim, dates_held_out=dates,
                                sigma=sigma, rho=rho, years_held_out=dates.year.to_numpy()))
    return folds


def test_check_holds_when_the_error_model_is_right():
    per_year, summary = check_yearly_statistics(_folds(60), threshold=16.0, n_simulations=400, seed=1)
    assert set(per_year.statistic) == set(scenario.YEARLY_STATISTICS)
    assert (per_year.groupby("statistic").size() == 60).all()
    for r in summary.itertuples():
        assert r.n_years == 60
        assert r.expected_inside_90_low <= r.share_inside_90 <= r.expected_inside_90_high
        assert r.mean_deviation_ci95_lower < 0 < r.mean_deviation_ci95_upper or r.statistic == "days above threshold"
    assert per_year.season_months.iloc[0] == "6 7 8 9"          # the four warmest months by default


def test_check_finds_a_bias_and_the_correction_removes_it():
    per_year, summary, sims = check_yearly_statistics(_folds(30, offset=1.0), threshold=16.0, n_simulations=400,
                                                      seed=2, return_simulations=True)
    peak = summary.set_index("statistic").loc["highest 7-day mean"]
    assert peak.mean_deviation_ci95_lower > 0.5 and peak.share_inside_90 < 0.8
    # Nested correction: each year corrected with the deviations of the other years.
    sub = per_year[per_year.statistic == "highest 7-day mean"].set_index("year")
    rng = np.random.default_rng(0)
    inside = []
    for year in sub.index:
        others = sub.deviation.drop(year)
        corrected = scenario.correct_statistic(sims[(year, "highest 7-day mean")], others, seed=year)
        p = scenario.pit(corrected, sub.loc[year, "measured"], rng)
        inside.append(0.05 <= p <= 0.95)
    assert np.mean(inside) >= 0.8


def test_years_without_enough_of_the_season_measured_are_skipped():
    folds = _folds(2)
    summer = np.isin(folds[0].dates_held_out.month, (6, 7, 8, 9))
    folds[0].obs_held_out[summer & (np.arange(len(summer)) % 3 == 0)] = -999.0   # a third of summer missing
    per_year, _ = check_yearly_statistics(folds, threshold=16.0, season_months=[6, 7, 8, 9],
                                          n_simulations=50, seed=0)
    assert sorted(per_year.year.unique()) == [2001]


def test_warmest_months():
    dates = pd.date_range("2000-01-01", "2001-12-31")
    temps = -np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 30) / 365)    # warmest at the end of July
    assert warmest_months(dates, temps) == [6, 7, 8, 9]
    assert warmest_months(dates, -temps) == [1, 2, 3, 12]                   # southern hemisphere


def test_season_months_and_threshold_config(tmp_path):
    import yaml
    from pyair2stream.io import read_calibration
    cfg = {"run_mode": "DE", "paths": {"output_dir": str(tmp_path)},
           "cross_validation": {"enabled": True, "threshold": 18, "season_months": [7, 6, 8]}}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    cv = read_calibration(str(path)).cross_validation
    assert cv.threshold == 18.0 and cv.season_months == [6, 7, 8]
    cfg["cross_validation"]["season_months"] = [0, 13]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="season_months"):
        read_calibration(str(path))
