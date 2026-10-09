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
    out = scenario.year_statistics(ens, dates, threshold=18.0, partial_years="keep")
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
    assert sorted(out) == [2002]                     # Jan-Sep 2001 and Oct-Dec 2002 are partial
    assert out[2002]["highest daily mean"][0] == dates.get_loc(pd.Timestamp("2002-09-30"))
    kept = scenario.year_statistics(np.arange(len(dates), dtype=float), dates, 1e9, years=labels,
                                    partial_years="keep")
    assert sorted(kept) == [2001, 2002, 2003]


def test_partial_years_are_left_out_with_a_warning(capsys):
    dates = pd.date_range("2003-07-19", "2006-03-03")         # starts and ends part-way through a year
    x = np.where(dates.month == 7, 20.0, 5.0)
    out = scenario.year_statistics(x, dates, threshold=18.0)
    assert sorted(out) == [2004, 2005]
    assert "year(s) 2003, 2006 are only partly covered" in capsys.readouterr().out
    assert sorted(scenario.year_statistics(x, dates, 18.0, partial_years="keep")) == [2003, 2004, 2005, 2006]
    whole = pd.date_range("2004-01-01", "2005-12-31")
    assert scenario.partial_year_labels(whole, whole.year) == []
    # A noleap series (no 29 February) is still whole; a 360-day year is not a calendar year.
    noleap = whole[~((whole.month == 2) & (whole.day == 29))]
    assert scenario.partial_year_labels(noleap, noleap.year) == []
    one = pd.date_range("2004-10-01", periods=365)
    assert scenario.partial_year_labels(one, np.full(365, 2005), calendar_years=False) == []
    assert scenario.partial_year_labels(one[:200], np.full(200, 2005), calendar_years=False) == [2005]
    with pytest.raises(ValueError):
        scenario.year_statistics(x, dates, 18.0, partial_years="drop")


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


# --- Any level, not only 90% ---------------------------------------------------------------

def test_central_range_and_inside_range_at_any_level():
    x = np.arange(1001, dtype=float)                   # 0..1000: percentiles are the values themselves
    assert scenario.central_range(x, 90) == pytest.approx((50.0, 950.0))
    assert scenario.central_range(x, 95) == pytest.approx((25.0, 975.0))
    assert scenario.central_range(x, 50) == pytest.approx((250.0, 750.0))
    assert list(scenario.inside_range([0.02, 0.03, 0.5, 0.97, 0.98], 95)) == [False, True, True, True, False]
    with pytest.raises(ValueError):
        scenario.central_range(x, 100)


@pytest.mark.parametrize("level", [80.0, 95.0, 97.5])
def test_check_reports_the_chosen_level(level):
    per_year, summary = check_yearly_statistics(_folds(40), threshold=16.0, level=level, n_simulations=400, seed=4)
    assert (per_year.level == level).all()
    # The per-year range is the central range of the simulations, and 'inside' follows from the PIT.
    assert ((per_year.lower <= per_year["median"]) & (per_year["median"] <= per_year.upper)).all()
    assert (per_year.inside == scenario.inside_range(per_year.pit, level)).all()
    for lev in (50, 80, 90, 95, level):                # always these, and the chosen one
        col = f"share_inside_{lev:g}"
        assert col in summary.columns
        # With the right error model, every level holds within the range expected by chance.
        for r in summary.to_dict("records"):
            assert r[f"expected_inside_{lev:g}_low"] <= r[col] <= r[f"expected_inside_{lev:g}_high"]


def test_parameter_interval_sets_the_jackknife_level(tmp_path):
    import yaml
    from pyair2stream.io import read_calibration
    from pyair2stream.cross_validation import summarize
    cfg = {"run_mode": "DE", "paths": {"output_dir": str(tmp_path)},
           "uncertainty_options": {"parameter_interval": 95}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    assert read_calibration(str(tmp_path / "c.yaml")).uncertainty_options["parameter_interval"] == 95.0
    folds = _folds(4)
    for k, f in enumerate(folds):
        f.par_best = np.full(8, 1.0 + 0.1 * k)
    table = summarize(folds, n_blocks=5, level=0.95)
    assert {"jackknife_95_lower", "jackknife_95_upper"} <= set(table.fold)
    wide = table.set_index("fold")
    narrow = summarize(folds, n_blocks=5, level=0.90).set_index("fold")
    assert wide.loc["jackknife_95_upper", "p1"] > narrow.loc["jackknife_90_upper", "p1"]
    cfg["uncertainty_options"]["parameter_interval"] = 100
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="parameter_interval"):
        read_calibration(str(tmp_path / "c.yaml"))


def test_parameter_summary_interval_follows_the_level_and_the_zero_test_stays_at_95():
    from pyair2stream.post_processing import parameter_summary
    rng = np.random.default_rng(0)
    chain = pd.DataFrame({"par_1": rng.normal(1.8, 1.0, 200000),       # 0 at about the 3.6th percentile
                          "par_2": rng.normal(0.0, 1.0, 200000)})
    t90 = parameter_summary(chain, 90).set_index("Parameter")
    t50 = parameter_summary(chain, 50).set_index("Parameter")
    assert t90.loc["par_2", "90%_CI_Upper"] == pytest.approx(1.645, abs=0.02)
    assert t50.loc["par_2", "50%_CI_Upper"] == pytest.approx(0.674, abs=0.02)
    # par_1: the 90% interval excludes 0 but the 95% one does not, so it is not "significant".
    assert t90.loc["par_1", "90%_CI_Lower"] > 0
    assert not t90.loc["par_1", "Significantly_Diff_From_Zero"]
    assert t50.loc["par_1", "Significantly_Diff_From_Zero"] == t90.loc["par_1", "Significantly_Diff_From_Zero"]


def test_interval_coverage_holds_at_every_level_when_the_error_model_is_right():
    from pyair2stream.cross_validation import check_interval_coverage
    cov = check_interval_coverage(_folds(40), extra_level=97.5, n_simulations=500, seed=5).set_index("level")
    assert list(cov.index) == [50.0, 80.0, 90.0, 95.0, 97.5]
    for level, r in cov.iterrows():
        assert abs(r["daily inside"] - level / 100) < 0.02
        assert abs(r["7-day inside"] - level / 100) < 0.04
    assert cov["days"].iloc[0] == sum(int((f.obs_held_out != -999).sum()) for f in _folds(40))


def test_the_check_leaves_out_a_year_the_held_out_dates_cover_only_in_part():
    # A last year ending on 5 July: its 35 summer days are all measured, but the summer is
    # mostly outside the record, so the year is not judged on it.
    folds = _folds(6)
    last = folds[-1]
    keep = last.dates_held_out <= pd.Timestamp(f"{last.label}-07-05")
    folds[-1] = FoldResult(fold_id=last.fold_id, label=last.label, held_out_start=last.held_out_start,
                           held_out_end=last.dates_held_out[keep][-1], n_obs_held_out=int(keep.sum()),
                           par_best=last.par_best, nse=np.nan, kge=np.nan, rmse=np.nan,
                           obs_held_out=last.sim_held_out[keep].copy(),       # every day measured
                           sim_held_out=last.sim_held_out[keep], dates_held_out=last.dates_held_out[keep],
                           sigma=last.sigma, rho=last.rho, years_held_out=last.years_held_out[keep])
    per_year, _ = check_yearly_statistics(folds, threshold=16.0, season_months=[6, 7, 8, 9],
                                          n_simulations=50, seed=1)
    assert sorted(set(per_year.year)) == [2000, 2001, 2002, 2003, 2004]
