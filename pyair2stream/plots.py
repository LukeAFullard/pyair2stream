"""
Ready-made figures for simulation ensembles: the prediction range over time, the change
between two scenarios, and a yearly statistic against a limit.

Each function draws on a matplotlib Axes (a new figure if `ax` is not given) and returns
that Axes. Save the figure with `ax.figure.savefig("name.png")`. They take the arrays that
`scenario.load_ensemble`, `scenario.paired_difference_from_files` and
`scenario.year_statistics` return (USER_GUIDE.md §12).

The ranges are always taken across the simulations after any averaging, simulation by
simulation: the range of 7-day means is not the 7-day mean of the daily range.
"""

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .scenario import central_range

# Colours in a fixed order, checked for colour-vision deficiency; text and reference lines in grey.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def prediction_range(ensemble, dates, level: float = 90, window: int = 1, measured=None, limit: float = None,
                     ax=None, color: str = None, label: str = None):
    """
    The median of the simulations and their central `level`% range, day by day.

    Parameters
    ----------
    ensemble : ndarray, shape (n_simulations, n_days)
        For example from `scenario.load_ensemble`.
    dates : array-like of dates, length n_days
    level : float
        Width of the range in per cent (90: from the 5th to the 95th percentile).
    window : int
        1: daily values. Above 1: each simulation's mean over the day and the `window - 1`
        days before it (7 for 7-day means), before the range is taken.
    measured : pandas.Series indexed by date, or array of length n_days, optional
        Measured water temperature, drawn as points (averaged over the same window; a
        window with a missing day is left out).
    limit : float, optional
        A temperature limit, drawn as a dashed line.
    ax : matplotlib Axes, optional
    color, label : str, optional
        Colour and legend name of this ensemble (to draw several on one Axes).

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.dates as mdates
    ax = _axes(ax, (9, 3.6))
    dates = pd.DatetimeIndex(dates)
    ens = _moving_mean(np.atleast_2d(np.asarray(ensemble, dtype=np.float64)), window)
    if ens.shape[1] != len(dates):
        raise ValueError(f"ensemble has {ens.shape[1]} days but dates has {len(dates)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # days without a value (start of a window)
        lo, hi = central_range(ens, level, axis=0)
        med = np.nanmedian(ens, axis=0)
    color = color or PALETTE[0]
    name = f"{label}: " if label else ""
    ax.fill_between(dates, lo, hi, color=color, alpha=0.25, lw=0, label=f"{name}{level:g}% range")
    ax.plot(dates, med, color=color, lw=1.2, label=f"{name}median")
    if measured is not None:
        obs = _moving_mean(np.atleast_2d(_on_dates(measured, dates)), window)[0]
        ax.plot(dates, obs, ".", color=INK, ms=3, label="measured")
    if limit is not None:
        _reference(ax, limit, f"limit {limit:g} °C")
    what = "Water temperature" if window == 1 else f"{window}-day mean water temperature"
    ax.set_ylabel(f"{what} (°C)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    _legend_below(ax)
    return _style(ax)


def change(differences, dates, by: str = "month", level: float = 90, reference: float = None,
           reference_label: str = None, ax=None, colors=None):
    """
    The change between two scenarios, from paired simulations.

    Parameters
    ----------
    differences : ndarray, shape (n_simulations, n_days), or dict {name: ndarray}
        Scenario minus baseline, simulation by simulation, from
        `scenario.paired_difference_from_files`. A dict draws several scenarios.
    dates : array-like of dates, length n_days
    by : "day", "month" or "year"
        "day": the median change and its range on each day. "month": each simulation's
        mean change in each calendar month (all years together), as a median and a
        range bar. "year": the same for each year; a year the dates cover only in part
        (at the start or end) is left out, with a warning, as its mean would describe
        only those days.
    level : float
        Width of the range in per cent.
    reference : float, optional
        A value to mark, for example the change made to the air temperature.
    reference_label : str, optional
    ax : matplotlib Axes, optional
    colors : list or dict of colours, optional

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.dates as mdates
    if by not in ("day", "month", "year"):
        raise ValueError(f"by must be 'day', 'month' or 'year', got {by!r}")
    series = differences if isinstance(differences, Mapping) else {None: differences}
    dates = pd.DatetimeIndex(dates)
    ax = _axes(ax, (9, 3.6) if by == "day" else (8, 3.6))
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    colors = _colors(colors, list(series))
    n = len(series)
    for k, (name, diff) in enumerate(series.items()):
        diff = np.atleast_2d(np.asarray(diff, dtype=np.float64))
        if diff.shape[1] != len(dates):
            raise ValueError(f"differences has {diff.shape[1]} days but dates has {len(dates)}")
        c = colors[name]
        if by == "day":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = central_range(diff, level, axis=0)
                med = np.nanmedian(diff, axis=0)
            prefix = f"{name}: " if name is not None else ""
            ax.fill_between(dates, lo, hi, color=c, alpha=0.25, lw=0, label=f"{prefix}{level:g}% range")
            ax.plot(dates, med, color=c, lw=1.2, label=f"{prefix}median")
            continue
        keys = dates.month if by == "month" else dates.year
        groups = sorted(set(keys))
        if by == "year":
            from .scenario import partial_year_labels
            partial = partial_year_labels(dates, dates.year)
            if partial and k == 0:
                print(f"Warning: year(s) {', '.join(str(int(y)) for y in partial)} are only partly covered by the "
                      "dates, so they are left out of the yearly changes.")
            groups = [g for g in groups if g not in partial]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.array([np.nanmean(diff[:, keys == g], axis=1) for g in groups])     # (groups, simulations)
            lo, hi = central_range(means, level, axis=1)
            med = np.nanmedian(means, axis=1)
        x = np.arange(len(groups)) + (k - (n - 1) / 2) * min(0.24, 0.8 / n)
        ax.vlines(x, lo, hi, color=c, lw=4, alpha=0.45)
        ax.plot(x, med, "o", color=c, ms=5, label=name if name is not None else "median")
        if k == 0:
            ax.set_xticks(np.arange(len(groups)),
                          [MONTHS[g - 1] for g in groups] if by == "month" else [str(g) for g in groups])
    if reference is not None:
        _reference(ax, reference, reference_label)
    if by == "day":
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_ylabel("Change in water temperature (°C)")
    else:
        ax.set_ylabel(f"Mean change in {by} (°C)")
        ax.set_title(f"Dot: median of the simulations; bar: {level:g}% range", fontsize=9, color=INK2, loc="left")
    if by == "day" or n > 1 or reference_label:
        _legend_below(ax)
    return _style(ax)


def yearly_statistic(values, statistic: str = "highest 7-day mean", limit: float = None, measured=None,
                     level: float = 90, ax=None, colors=None, show_probability: bool = None):
    """
    A yearly statistic in every simulation, year by year, against a limit.

    Each year shows the median of the simulations (dot) and their central `level`% range
    (bar). With a `limit`, the share of simulations above it, the chance that the limit was
    (or will be) exceeded, is written above each bar.

    Parameters
    ----------
    values : dict
        {year: ndarray of the statistic in each simulation}; or the output of
        `scenario.year_statistics` (then `statistic` picks the statistic); or
        {name: either of these} to compare several scenarios, or corrected and
        uncorrected values (`scenario.correct_statistic`).
    statistic : str
        Name of the statistic, when `values` holds several (`scenario.YEARLY_STATISTICS`).
    limit : float, optional
        Drawn as a dashed line; the probabilities count simulations strictly above it.
    measured : dict, optional
        {year: measured value}, or `scenario.year_statistics` of the measured series.
    level : float
        Width of the range in per cent.
    ax : matplotlib Axes, optional
    colors : list or dict of colours, optional
    show_probability : bool, optional
        Write the probabilities above the bars (default: with a limit and at most 24 bars).

    Returns
    -------
    matplotlib Axes
    """
    series = _by_name(values, statistic)
    unit = "days" if "days" in statistic else "°C"
    years = sorted({y for v in series.values() for y in v})
    n = len(series)
    if show_probability is None:
        show_probability = limit is not None and n * len(years) <= 24
    ax = _axes(ax, (max(4.5, 1.1 + 0.55 * n * len(years)), 3.8))
    colors = _colors(colors, list(series))
    step = min(0.28, 0.8 / n)
    tops = []
    for k, (name, by_year) in enumerate(series.items()):
        c = colors[name]
        labelled = name is None
        for i, year in enumerate(years):
            if year not in by_year:
                continue
            v = by_year[year][np.isfinite(by_year[year])]
            if not len(v):
                continue
            x = i + (k - (n - 1) / 2) * step
            lo, hi = central_range(v, level)
            ax.vlines(x, lo, hi, color=c, lw=4, alpha=0.45)
            ax.plot(x, np.median(v), "o", color=c, ms=5, label=None if labelled else name)
            labelled = True
            tops.append(hi)
            if show_probability and limit is not None:
                ax.annotate(f"{np.mean(v > limit):.2f}", (x, hi), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color=INK2, gid="probability")
    if measured is not None:
        meas = _measured_by_year(measured, statistic)
        xs = [i for i, y in enumerate(years) if y in meas and np.isfinite(meas[y])]
        ax.plot(xs, [meas[years[i]] for i in xs], "_", color=INK, ms=16, mew=2, label="measured")
    if limit is not None:
        _reference(ax, limit, f"limit ({limit:g} {unit})")
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_xlim(-0.6, len(years) - 0.4)
    if show_probability and tops:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, max(tops) + 0.12 * (hi - lo)))
    ax.set_ylabel(f"{statistic[0].upper()}{statistic[1:]} ({unit})")
    title = f"Dot: median of the simulations; bar: {level:g}% range"
    if show_probability and limit is not None:
        title += f"; number: chance above {limit:g}"
    ax.set_title(title, fontsize=9, color=INK2, loc="left")
    if n > 1 or measured is not None or limit is not None:
        _legend_below(ax)
    return _style(ax)


# --- Helpers --------------------------------------------------------------------------------------

def _axes(ax, size):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=size)
    return ax


def _style(ax):
    """Recessive axes and grid, set on this Axes only (the user's matplotlib settings are left alone)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK2)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return ax


def _legend_below(ax):
    """The legend under the plot, where it cannot cover the data."""
    entries = len(ax.get_legend_handles_labels()[1])
    if entries:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, frameon=False, ncol=min(3, entries))


def _reference(ax, value, label):
    """A dashed horizontal line (a limit, a reference value), named in the legend."""
    ax.axhline(value, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1, label=label)


def _colors(colors, names):
    if isinstance(colors, Mapping):
        return {n: colors.get(n, PALETTE[i % len(PALETTE)]) for i, n in enumerate(names)}
    colors = list(colors) if colors is not None else list(PALETTE)
    return {n: colors[i % len(colors)] for i, n in enumerate(names)}


def _moving_mean(x, window):
    """Each row's mean over the day and the `window - 1` days before it (NaN where any is missing)."""
    if window <= 1:
        return x
    return pd.DataFrame(x.T).rolling(window, min_periods=window).mean().to_numpy().T


def _on_dates(measured, dates):
    if isinstance(measured, pd.Series):
        idx = pd.DatetimeIndex(measured.index)
        return pd.Series(measured.to_numpy(np.float64), index=idx).reindex(dates).to_numpy()
    m = np.asarray(measured, dtype=np.float64)
    if m.shape != (len(dates),):
        raise ValueError(f"measured has shape {m.shape}; expected ({len(dates)},) or a Series indexed by date")
    return m


def _is_year(key):
    return isinstance(key, (int, np.integer)) and not isinstance(key, bool)


def _by_year(by_year, statistic):
    out = {}
    for year, v in by_year.items():
        if isinstance(v, Mapping):
            if statistic not in v:
                raise KeyError(f"statistic {statistic!r} not found; available: {', '.join(map(str, v))}")
            v = v[statistic]
        out[int(year)] = np.asarray(v, dtype=np.float64).ravel()
    return out


def _by_name(values, statistic):
    """{name: {year: values}}; name None for a single set."""
    if not isinstance(values, Mapping) or not values:
        raise ValueError("values must be a non-empty dict: {year: values}, scenario.year_statistics output, "
                         "or {name: either}")
    if all(_is_year(k) for k in values):
        return {None: _by_year(values, statistic)}
    return {name: _by_year(v, statistic) for name, v in values.items()}


def _measured_by_year(measured, statistic):
    return {y: float(v[0]) if len(v) else np.nan for y, v in _by_year(measured, statistic).items()}
