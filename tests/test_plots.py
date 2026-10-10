"""
pyair2stream.plots: the prediction range, the change between scenarios and a yearly statistic
against a limit.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from pyair2stream import plots, scenario  # noqa: E402


def _probabilities(ax):
    return [t for t in ax.texts if t.get_gid() == "probability"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _ensemble(n=200, days=731, seed=0):    # 2020 (a leap year) and 2021
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    base = 10 + 8 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 110) / 365)
    return base + rng.normal(0, 1, (n, days)), dates


def test_prediction_range_takes_the_range_after_averaging_each_simulation():
    ens, dates = _ensemble()
    ax = plots.prediction_range(ens, dates, window=7, level=80)
    week = pd.DataFrame(ens.T).rolling(7).mean().to_numpy().T        # each simulation's 7-day means
    median = ax.get_lines()[0].get_ydata()
    assert np.all(np.isnan(median[:6]))
    assert np.allclose(median[6:], np.median(week[:, 6:], axis=0))
    band = ax.collections[0].get_paths()[0].vertices[:, 1]
    lo, hi = scenario.central_range(week[:, 6:], 80, axis=0)
    assert np.isclose(band.min(), lo.min()) and np.isclose(band.max(), hi.max())
    # The range of 7-day means is narrower than the 7-day mean of the daily range.
    lo_d, hi_d = scenario.central_range(ens, 80, axis=0)
    daily_band = pd.Series(hi_d - lo_d).rolling(7).mean().to_numpy()[6:]
    assert np.median(hi - lo) < 0.6 * np.median(daily_band)


def test_prediction_range_measured_series_by_date_and_limit():
    ens, dates = _ensemble()
    obs = pd.Series(ens[0], index=dates).drop(dates[100:110])          # ten days not measured
    ax = plots.prediction_range(ens, dates, measured=obs, limit=17)
    measured = [ln for ln in ax.get_lines() if ln.get_label() == "measured"][0].get_ydata()
    assert np.isnan(measured[100:110]).all() and np.allclose(measured[:100], ens[0, :100])
    assert any(np.allclose(ln.get_ydata(), 17) for ln in ax.get_lines())
    with pytest.raises(ValueError):
        plots.prediction_range(ens, dates[:-1])
    with pytest.raises(ValueError):
        plots.prediction_range(ens, dates, measured=ens[0, :-1])


def test_change_by_month_and_day_for_one_or_several_scenarios():
    ens, dates = _ensemble()
    diff = np.tile(np.where(dates.month.isin([6, 7, 8]), 1.5, 0.5), (len(ens), 1))
    ax = plots.change({"A": diff, "B": diff + 0.2}, dates, by="month", reference=2, reference_label="air")
    dots = [ln for ln in ax.get_lines() if ln.get_marker() in plots.MARKERS]
    assert [ln.get_label() for ln in dots] == ["A", "B"]
    assert [ln.get_marker() for ln in dots] == ["o", "s"]       # told apart by shape, not only by colour
    assert np.allclose(dots[0].get_ydata(), [1.5 if m in (6, 7, 8) else 0.5 for m in range(1, 13)])
    assert np.allclose(dots[1].get_ydata(), dots[0].get_ydata() + 0.2)
    assert [t.get_text() for t in ax.get_xticklabels()][:3] == ["Jan", "Feb", "Mar"]

    ax = plots.change(diff, dates, by="year")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["2020", "2021"]
    # A year the dates cover only in part is left out of the yearly changes.
    ax = plots.change(diff[:, 100:], dates[100:], by="year")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["2021"]
    ax = plots.change(diff, dates, by="day")
    assert np.allclose(ax.get_lines()[-1].get_ydata(), diff[0])
    with pytest.raises(ValueError):
        plots.change(diff, dates, by="week")


def test_yearly_statistic_from_year_statistics_with_probabilities_and_measured():
    ens, dates = _ensemble()
    stats = scenario.year_statistics(ens, dates, threshold=15)
    measured = scenario.year_statistics(ens[0], dates, threshold=15)
    ax = plots.yearly_statistic(stats, limit=18.5, measured=measured)
    expected = [np.mean(stats[y]["highest 7-day mean"] > 18.5) for y in (2020, 2021)]
    assert [float(t.get_text()) for t in _probabilities(ax)] == pytest.approx(expected, abs=0.005)
    marks = [ln for ln in ax.get_lines() if ln.get_label() == "measured"][0].get_ydata()
    assert np.allclose(marks, [measured[y]["highest 7-day mean"][0] for y in (2020, 2021)])

    days = plots.yearly_statistic(stats, statistic="days above threshold", limit=60)
    assert days.get_ylabel().endswith("(days)")
    with pytest.raises(KeyError):
        plots.yearly_statistic(stats, statistic="lowest daily mean")
    with pytest.raises(ValueError):
        plots.yearly_statistic({})


def test_yearly_statistic_several_scenarios_and_plain_arrays():
    ens, dates = _ensemble()
    stats = scenario.year_statistics(ens, dates, threshold=15)
    warmer = {y: v["highest 7-day mean"] + 1 for y, v in stats.items()}       # {year: values}
    ax = plots.yearly_statistic({"now": stats, "warmer": warmer}, limit=19)
    dots = [ln for ln in ax.get_lines() if ln.get_marker() in plots.MARKERS]
    assert len(dots) == 4 and len(_probabilities(ax)) == 4
    assert sorted(ln.get_marker() for ln in dots) == ["o", "o", "s", "s"]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["now", "warmer", "limit (19 °C)"]
    # Many bars: probabilities are not written unless asked for.
    many = {y: np.random.default_rng(y).normal(20, 1, 100) for y in range(1990, 2020)}
    assert len(_probabilities(plots.yearly_statistic(many, limit=20))) == 0
    assert len(_probabilities(plots.yearly_statistic(many, limit=20, show_probability=True))) == 30


def test_plots_draw_on_a_given_axes_and_leave_matplotlib_settings_alone():
    ens, dates = _ensemble()
    before = dict(matplotlib.rcParams)
    fig, axes = plt.subplots(1, 2)
    assert plots.prediction_range(ens, dates, ax=axes[0]) is axes[0]
    assert plots.change(ens - ens.mean(), dates, ax=axes[1], by="day") is axes[1]
    assert len(fig.axes) == 2
    assert dict(matplotlib.rcParams) == before


def test_prediction_range_conformal_margin_and_measured_outside():
    ens, dates = _ensemble()
    obs = ens[0] + np.where(np.arange(len(dates)) % 10 == 0, 3.0, 0.0)       # every tenth day well off
    ax = plots.prediction_range(ens, dates, measured=obs, margin=0.5, floor=0.0, outside=True)
    lo, hi = scenario.conformal_range(ens, 90, 0.5, axis=0, floor=0.0)
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 2
    assert np.allclose(dashed[0].get_ydata(), lo) and np.allclose(dashed[1].get_ydata(), hi)
    groups = {ln.get_label(): ln for ln in ax.get_lines() if ln.get_label().startswith("measured")}
    assert set(groups) == {"measured, inside", "measured, inside only with the margin", "measured, outside both"}
    assert len({ln.get_marker() for ln in groups.values()}) == 3                # three shapes, not three colours
    plain_lo, plain_hi = scenario.central_range(ens, 90, axis=0)
    out = (obs < lo) | (obs > hi)
    assert len(groups["measured, outside both"].get_xdata()) == out.sum()
    only = ~out & ((obs < plain_lo) | (obs > plain_hi))
    assert len(groups["measured, inside only with the margin"].get_xdata()) == only.sum()
    assert sum(len(ln.get_xdata()) for ln in groups.values()) == len(dates)
    # Without a margin: inside and outside only.
    ax = plots.prediction_range(ens, dates, measured=obs, outside=True)
    labels = {ln.get_label() for ln in ax.get_lines() if ln.get_label().startswith("measured")}
    assert labels == {"measured, inside", "measured, outside"}


def _margins_table():
    rows = []
    for w, shift in ((1, -0.03), (7, -0.05), (30, -0.08)):
        for level in (50, 80, 90, 95):
            rows.append({"window_days": w, "level": level, "margin": 0.1, "inside_before": level / 100 + shift,
                         "inside_after": level / 100 - 0.005})
    return pd.DataFrame(rows)


def test_coverage_draws_each_window_without_and_with_the_margin():
    table = _margins_table()
    ax = plots.coverage(table)
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    assert np.allclose(lines["30-day means, as computed"].get_ydata(), [42, 72, 82, 87])
    assert np.allclose(lines["days, with conformal margin"].get_ydata(), [49.5, 79.5, 89.5, 94.5])
    assert np.allclose(lines["holds as stated"].get_xdata(), lines["holds as stated"].get_ydata())
    shapes = {name.split(",")[0]: ln.get_marker() for name, ln in lines.items() if "," in name}
    assert len(set(shapes.values())) == 3
    # Without inside_after, one line per window.
    ax = plots.coverage(table.drop(columns="inside_after"))
    assert sorted(ln.get_label() for ln in ax.get_lines()) == ["30-day means", "7-day means", "days",
                                                               "holds as stated"]
    with pytest.raises(ValueError):
        plots.coverage(table.drop(columns="inside_before"))


def _cv_results(par4=(0.05, -0.04, 0.02)):
    folds = {"fold": ["2001", "2002", "2003"], "n_obs_held_out": [365] * 3, "NSE": [0.9, 0.8, 0.95],
             "KGE": [0.9, 0.8, 0.9], "RMSE": [0.5, 0.7, 0.4],
             "p1": [1.0, 1.1, 0.9], "p2": [2.0, 2.0, 2.2], "p3": [0.0, 0.0, 0.0], "p4": list(par4)}
    t = pd.DataFrame(folds)
    from pyair2stream.cross_validation import jackknife_rows
    jk = jackknife_rows(t[["p1", "p2", "p3", "p4"]].to_numpy(), n_blocks=3)
    summary = pd.DataFrame([{"fold": "mean"}, {"fold": "std"},
                            {"fold": "pooled", "NSE": 0.88, "KGE": 0.87, "RMSE": 0.55}, *jk])
    return pd.concat([t, summary], ignore_index=True)


def test_cv_by_fold_bars_and_pooled_score():
    t = _cv_results()
    ax = plots.cv_by_fold(t)
    assert [p.get_height() for p in ax.patches] == pytest.approx([0.5, 0.7, 0.4])
    assert [x.get_text() for x in ax.get_xticklabels()] == ["2001", "2002", "2003"]
    assert any(np.allclose(ln.get_ydata(), 0.55) for ln in ax.get_lines())
    # Two runs side by side, told apart by hatching as well as colour.
    ax = plots.cv_by_fold({"A": t, "B": t.assign(RMSE=t.RMSE + 0.1)}, metric="RMSE")
    assert len(ax.patches) == 6 and len({p.get_hatch() for p in ax.patches}) == 2
    with pytest.raises(ValueError):
        plots.cv_by_fold(t, metric="MAE")


def test_cv_parameters_as_percent_of_the_mean_with_jackknife_intervals():
    t = _cv_results()
    ax = plots.cv_parameters(t)
    # a3 is fixed at zero in every fold (not active), a4's interval includes zero: neither is drawn.
    assert [x.get_text() for x in ax.get_yticklabels()] == ["a2", "a1"]
    assert "a4" in ax.get_title(loc="left") and "a3" not in ax.get_title(loc="left")
    a1 = ax.collections[0].get_offsets()[:, 0]
    assert np.allclose(np.sort(a1), [-10, 0, 10])
    band = [ln for ln in ax.get_lines() if ln.get_label().endswith("jackknife interval")][0].get_xdata()
    lo, hi = (float(t.loc[t.fold == f"jackknife_90_{k}", "p1"].iloc[0]) for k in ("lower", "upper"))
    assert np.allclose(band, [100 * (lo - 1) / 1, 100 * (hi - 1) / 1])
    # Combinations of parameters get intervals the same way.
    ax = plots.cv_parameters(t, combinations={"a2/a1": lambda p: p["a2"] / p["a1"]}, n_blocks=3,
                             labels={"a1": "a1 constant"})
    assert [x.get_text() for x in ax.get_yticklabels()] == ["a2/a1", "a2", "a1 constant"]
    with pytest.raises(ValueError):
        plots.cv_parameters(t, combinations={"a2/a1": lambda p: p["a2"] / p["a1"]})
    with pytest.raises(ValueError):
        plots.cv_parameters(t[t.fold.isin(["2001", "mean"])])


def test_a_run_s_own_figures_leave_the_callers_matplotlib_settings_alone(monkeypatch):
    from pyair2stream import post_processing
    seen = {}
    monkeypatch.setattr(post_processing, "_post_process",
                        lambda data, toll: seen.update(family=list(matplotlib.rcParams["font.family"])))
    before = list(matplotlib.rcParams["font.family"])
    post_processing.post_process(object())
    assert seen["family"] == ["serif"]                       # the run's figures use its fonts ...
    assert list(matplotlib.rcParams["font.family"]) == before  # ... and the caller's settings come back
