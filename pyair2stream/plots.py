"""
Ready-made figures.

For simulation ensembles: the prediction range over time (`prediction_range`), the change
between two scenarios (`change`), and a yearly statistic against a limit
(`yearly_statistic`). They take the arrays that `scenario.load_ensemble`,
`scenario.paired_difference_from_files` and `scenario.year_statistics` return
(USER_GUIDE.md §12).

For a cross-validation: how often the prediction intervals held (`coverage`), the score
on each held-out year (`cv_by_fold`), and the parameters fitted without each year
(`cv_parameters`). They take its files `cv_conformal_margins.csv` and `cv_results.csv`; a
cross-validation run draws all three itself.

Each function draws on a matplotlib Axes (a new figure if `ax` is not given) and returns
that Axes. Save the figure with `ax.figure.savefig("name.png")`.

The ranges are always taken across the simulations after any averaging, simulation by
simulation: the range of 7-day means is not the 7-day mean of the daily range.
"""

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .scenario import central_range, widen_range

# Colours in a fixed order, checked for colour-vision deficiency; text and reference lines in grey.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MARKERS = ("o", "s", "^", "D", "v")          # with the colours, so series differ in shape too
LINESTYLES = ("-", (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)))
HATCHES = (None, "///", "...", "xxx", "\\\\\\")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def prediction_range(ensemble, dates, level: float = 90, window: int = 1, measured=None, limit: float = None,
                     ax=None, color: str = None, label: str = None, margin: float = None, floor: float = None,
                     outside: bool = False):
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
    margin : float, optional
        A conformal margin (°C) for this level and window (`scenario.conformal_margin`): the
        range widened by it (`scenario.conformal_range`) is drawn as dashed lines.
    floor : float, optional
        The lowest value the widened range may take: `Tice_cover` (0 °C by default).
    outside : bool
        Mark the measured values that fall outside the range (crosses). With a `margin`,
        those inside only the widened range are marked too (open squares).

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
    if margin is not None:
        wlo, whi = widen_range(lo, hi, margin, floor=floor)
        ax.plot(dates, wlo, color=color, lw=0.9, ls="--", label=f"{name}with conformal margin ({margin:+.2f} °C)")
        ax.plot(dates, whi, color=color, lw=0.9, ls="--")
    if measured is not None:
        obs = _moving_mean(np.atleast_2d(_on_dates(measured, dates)), window)[0]
        if not outside:
            ax.plot(dates, obs, ".", color=INK, ms=3, label="measured")
        else:
            with np.errstate(invalid="ignore"):
                inside = (obs >= lo) & (obs <= hi)
                widened = (obs >= wlo) & (obs <= whi) if margin is not None else inside
            # Each group has its own shape as well as its own colour, so it can be told apart without colour.
            groups = [(inside, dict(marker=".", color=INK, ms=3), "measured, inside"),
                      (widened & ~inside, dict(marker="s", color=PALETTE[3], mfc="white", mew=1.1, ms=4),
                       "measured, inside only with the margin"),
                      (np.isfinite(obs) & ~widened, dict(marker="x", color=PALETTE[1], mew=1.3, ms=5),
                       "measured, outside both" if margin is not None else "measured, outside")]
            for mask, style, text in groups:
                if (margin is None and "only" in text) or not mask.any():
                    continue
                ax.plot(dates[mask], obs[mask], ls="none", label=text, **style)
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
            ax.plot(dates, med, color=c, lw=1.2, ls=LINESTYLES[k % len(LINESTYLES)], label=f"{prefix}median")
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
        ax.plot(x, med, MARKERS[k % len(MARKERS)], color=c, ms=5, label=name if name is not None else "median")
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
        ax.set_title(f"Marker: median of the simulations; bar: {level:g}% range", fontsize=9, color=INK2, loc="left")
    if by == "day" or n > 1 or reference_label:
        _legend_below(ax)
    return _style(ax)


def yearly_statistic(values, statistic: str = "highest 7-day mean", limit: float = None, measured=None,
                     level: float = 90, ax=None, colors=None, show_probability: bool = None):
    """
    A yearly statistic in every simulation, year by year, against a limit.

    Each year shows the median of the simulations (a marker; one shape per set of values) and their central `level`% range
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
            ax.plot(x, np.median(v), MARKERS[k % len(MARKERS)], color=c, ms=5, label=None if labelled else name)
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
    title = f"Marker: median of the simulations; bar: {level:g}% range"
    if show_probability and limit is not None:
        title += f"; number: chance above {limit:g}"
    ax.set_title(title, fontsize=9, color=INK2, loc="left")
    if n > 1 or measured is not None or limit is not None:
        _legend_below(ax)
    return _style(ax)


# --- Cross-validation -----------------------------------------------------------------------------

def coverage(margins, ax=None, colors=None):
    """
    How often the prediction intervals held in the held-out years, at each stated level.

    One line (and marker shape) for single days and one for each averaging window (7-day and 30-day means): the
    share of held-out measured values inside the interval, against the level it claims. On
    the dotted diagonal, an interval holds as often as it claims; below it, too rarely (too
    narrow). Open markers: the interval as computed. Filled markers: with the conformal
    margin, each held-out year judged with a margin set from the other years only, so no
    year helps to set its own margin (docs/METHODS.md §13).

    No band for chance is drawn: the values of a year are not independent of each other
    (errors last several days), so a binomial band would be far too narrow.

    Parameters
    ----------
    margins : pandas.DataFrame or str
        `cv_conformal_margins.csv` (its path or `scenario.read_conformal_margins`), or any
        table with the columns `window_days`, `level` (%), `inside_before` (a share, 0-1)
        and, optionally, `inside_after`.
    ax : matplotlib Axes, optional
    colors : list or dict of colours (by window_days), optional

    Returns
    -------
    matplotlib Axes
    """
    table = pd.read_csv(margins) if isinstance(margins, str) else pd.DataFrame(margins)
    missing = [c for c in ("window_days", "level", "inside_before") if c not in table.columns]
    if missing or table.empty:
        raise ValueError(f"margins needs the columns window_days, level and inside_before "
                         f"({'missing ' + ', '.join(missing) if missing else 'no rows'})")
    ax = _axes(ax, (5.2, 4.6))
    windows = sorted(int(w) for w in table.window_days.unique())
    colors = _colors(colors, windows)
    after = "inside_after" in table.columns and table.inside_after.notna().any()
    for k, w in enumerate(windows):
        rows = table[table.window_days == w].sort_values("level")
        name = _window_name(w)
        c, m = colors[w], MARKERS[k % len(MARKERS)]       # shape as well as colour tells the windows apart
        ax.plot(rows.level, 100 * rows.inside_before, marker=m, ls=(0, (3, 2)) if after else "-", color=c, ms=5,
                mfc="white" if after else c, lw=1, label=f"{name}, as computed" if after else name)
        if after:
            ax.plot(rows.level, 100 * rows.inside_after, marker=m, ls="-", color=c, ms=5, lw=1.2,
                    label=f"{name}, with conformal margin")
    lo = max(0.0, 10 * np.floor((min(table.level.min(), 100 * table.inside_before.min()) - 5) / 10))
    ax.plot([lo, 100], [lo, 100], color=INK2, lw=0.8, ls=":", zorder=1, label="holds as stated")
    ax.set(xlim=(lo, 100), ylim=(lo, 100), xlabel="Stated level of the interval (%)",
           ylabel="Held-out values inside it (%)")
    ax.set_aspect("equal")
    ax.set_title("Below the diagonal: the interval held too rarely", fontsize=9, color=INK2, loc="left")
    _legend_below(ax, xlabel=True, ncol=2)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    return _style(ax)


def cv_by_fold(results, metric: str = "RMSE", ax=None, colors=None):
    """
    The score of each held-out year (or block) of a cross-validation.

    Each bar is the score on a year the calibration did not see. The dashed line is the
    score of all held-out days together (`pooled`). A year far worse than the others is
    one the model does not represent well.

    Parameters
    ----------
    results : pandas.DataFrame or str, or dict {name: either}
        `cv_results.csv` (its path or the table). A dict compares several runs, for example
        two model versions, year by year.
    metric : "RMSE", "NSE" or "KGE"
    ax : matplotlib Axes, optional
    colors : list or dict of colours, optional

    Returns
    -------
    matplotlib Axes
    """
    if metric not in ("RMSE", "NSE", "KGE"):
        raise ValueError(f"metric must be 'RMSE', 'NSE' or 'KGE', got {metric!r}")
    tables = {name: _cv_table(t) for name, t in (results.items() if isinstance(results, Mapping)
                                                   else [(None, results)])}
    folds = []
    for t in tables.values():
        folds += [f for f in _fold_rows(t).fold if f not in folds]
    n = len(tables)
    ax = _axes(ax, (max(6.0, 1.5 + 0.5 * n * len(folds)), 3.4))
    colors = _colors(colors, list(tables))
    width = 0.8 / n
    for k, (name, t) in enumerate(tables.items()):
        rows = _fold_rows(t).set_index("fold")
        x = np.array([i for i, f in enumerate(folds) if f in rows.index]) + (k - (n - 1) / 2) * width
        ax.bar(x, [float(rows.loc[f, metric]) for f in folds if f in rows.index], width * 0.95,
               color=colors[name], hatch=HATCHES[k % len(HATCHES)], edgecolor="white", lw=0, label=name)
        pooled = t[t.fold == "pooled"]
        if len(pooled) and np.isfinite(float(pooled[metric].iloc[0])):
            ax.axhline(float(pooled[metric].iloc[0]), color=INK2 if n == 1 else colors[name], lw=1,
                       ls=LINESTYLES[1 + k % (len(LINESTYLES) - 1)], zorder=3,
                       label=f"{name + ': ' if name is not None else ''}all held-out days together")
    ax.set_xticks(range(len(folds)), folds)
    if len(folds) > 12:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    unit = " (°C)" if metric == "RMSE" else ""
    better = "lower is better" if metric == "RMSE" else "higher is better; 1 is perfect"
    ax.set(xlabel="Held out", ylabel=f"{metric} on the held-out year{unit}")
    ax.set_title(f"Each bar: the score on a year the calibration did not see ({better})", fontsize=9,
                 color=INK2, loc="left")
    _legend_below(ax, xlabel=True, ncol=2)
    return _style(ax)


def cv_parameters(results, combinations=None, n_blocks: int = None, labels=None, ax=None, color: str = None):
    """
    How firmly the data fix each parameter: the value fitted without each held-out year,
    and the jackknife interval, as a difference from the mean of the folds in per cent of
    its value.

    Dots far apart, or a wide interval, mean the data fix that parameter poorly. A
    parameter whose interval includes zero is not drawn (a percentage of a value that may
    be zero means nothing); a note under the axes gives its interval instead.

    Parameters
    ----------
    results : pandas.DataFrame or str
        `cv_results.csv` (its path or the table). Its `jackknife_<level>_lower/upper` rows
        give the intervals.
    combinations : dict {name: function}, optional
        Quantities made from the parameters, for example
        `{"a2/a3": lambda p: p["a2"] / p["a3"]}`; `p` holds each parameter's fold values.
        Parameters that trade off against each other can make a quantity much better fixed
        than either of them. Their intervals are computed the same way, so `n_blocks`, the
        number of years (blocks) in the whole record, is needed.
    n_blocks : int, optional
    labels : dict {name: text}, optional
        Longer names for the rows.
    ax : matplotlib Axes, optional
    color : str, optional

    Returns
    -------
    matplotlib Axes
    """
    from .cross_validation import jackknife_rows
    t = _cv_table(results)
    folds = _fold_rows(t)
    if len(folds) < 2:
        raise ValueError("cv_parameters needs at least two held-out folds")
    lower = [f for f in t.fold if str(f).startswith("jackknife_") and str(f).endswith("_lower")]
    level = float(lower[0].split("_")[1]) if lower else None
    cols = [c for c in t.columns if c[:1] == "p" and c[1:].isdigit() and np.nanmax(np.abs(
        folds[c].astype(float))) > 0]
    values = {f"a{c[1:]}": folds[c].to_numpy(float) for c in cols}
    rows = []
    for c in cols:
        lo = hi = np.nan
        if level is not None:
            lo = float(t.loc[t.fold == f"jackknife_{level:g}_lower", c].iloc[0])
            hi = float(t.loc[t.fold == f"jackknife_{level:g}_upper", c].iloc[0])
        rows.append((f"a{c[1:]}", values[f"a{c[1:]}"], lo, hi))
    if combinations:
        if n_blocks is None or level is None:
            raise ValueError("combinations need n_blocks (the number of years in the whole record) and the "
                             "jackknife rows of cv_results.csv")
        for name, fun in combinations.items():
            x = np.asarray(fun(values), dtype=np.float64)
            jk = jackknife_rows(x[:, None], n_blocks, level=level / 100)
            rows.append((name, x, jk[1]["p1"], jk[2]["p1"]))
    color = color or PALETTE[0]
    shown, skipped = [], []
    for name, x, lo, hi in rows:
        centre = x.mean()
        if centre == 0 or (np.isfinite(lo) and lo <= 0 <= hi):
            skipped.append(f"{name}: folds {x.min():.3g} to {x.max():.3g}"
                           + (f", interval {lo:.3g} to {hi:.3g}" if np.isfinite(lo) else ""))
            continue
        scale = 100 / abs(centre)
        shown.append((name, (x - centre) * scale, (lo - centre) * scale, (hi - centre) * scale))
    ax = _axes(ax, (8, 1.4 + 0.42 * max(len(shown), 1)))
    labels = labels or {}
    for k, (name, dev, lo, hi) in enumerate(shown):
        y = len(shown) - 1 - k
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=color, alpha=0.3, lw=7, solid_capstyle="round",
                    label=f"{level:g}% jackknife interval" if k == 0 else None)
        ax.scatter(dev, np.full(len(dev), y), s=30, color=color, edgecolor="white", linewidth=1.1, zorder=3,
                   label=f"fitted without one year ({len(dev)} folds)" if k == 0 else None)
    reach = max([5.0] + [np.nanmax(np.abs(np.r_[d, lo, hi])) for _, d, lo, hi in shown])
    ax.set_xlim(-1.1 * reach, 1.1 * reach)
    ax.set_yticks(range(len(shown)), [labels.get(name, name) for name, *_ in shown][::-1])
    ax.set_ylim(-0.7, len(shown) - 0.3)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_xlabel("Difference from the mean of the folds (% of its value)")
    title = "Close to 0: the data fix the value firmly"
    if skipped:
        title += "\nNot drawn, as the interval includes zero: " + "; ".join(skipped)
    ax.set_title(title, fontsize=9, color=INK2, loc="left")
    _legend_below(ax, xlabel=True, ncol=2)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.grid(False, axis="y")
    return ax



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


def _legend_below(ax, xlabel: bool = False, ncol: int = 3):
    """The legend under the plot, where it cannot cover the data (and under the x-axis label, if there is one)."""
    entries = len(ax.get_legend_handles_labels()[1])
    if entries:
        below = 0.1
        if xlabel:
            height = ax.get_position().height * ax.figure.get_figheight()      # inches
            below = max(0.1, 0.62 / height)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -below), fontsize=8, frameon=False,
                  ncol=min(ncol, entries))


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


def _window_name(days):
    return "days" if int(days) == 1 else f"{int(days)}-day means"


SUMMARY_FOLDS = ("mean", "std", "pooled")


def _cv_table(results):
    t = pd.read_csv(results) if isinstance(results, str) else pd.DataFrame(results)
    if "fold" not in t.columns:
        raise ValueError("results must be cv_results.csv (its path or the table): no 'fold' column")
    return t.assign(fold=t.fold.astype(str))


def _fold_rows(t):
    """The rows of the held-out folds (not the mean, std, pooled and jackknife rows)."""
    return t[~t.fold.isin(SUMMARY_FOLDS) & ~t.fold.str.startswith("jackknife")]
