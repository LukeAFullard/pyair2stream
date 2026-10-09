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


def _ensemble(n=200, days=730, seed=0):
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
    dots = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    assert [ln.get_label() for ln in dots] == ["A", "B"]
    assert np.allclose(dots[0].get_ydata(), [1.5 if m in (6, 7, 8) else 0.5 for m in range(1, 13)])
    assert np.allclose(dots[1].get_ydata(), dots[0].get_ydata() + 0.2)
    assert [t.get_text() for t in ax.get_xticklabels()][:3] == ["Jan", "Feb", "Mar"]

    ax = plots.change(diff, dates, by="year")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["2020", "2021"]
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
    dots = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    assert len(dots) == 4 and len(_probabilities(ax)) == 4
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
