"""
V17 - Is the model better than a simple regression?

A regression of water temperature on air temperature takes minutes to fit and needs no special
software. air2stream is only worth its extra effort if it predicts years it was not calibrated on
clearly better. Both are fitted on the same calibration years and predict the same later years,
on the three Swiss rivers and the 23 British Columbia rivers of Callahan & Moore (2025).
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from common import (DE_SETTINGS, RIVERS, WORK, Result, Section, Timer, calibrate, daily, mean_discharge, quiet,
                    river_csv, simulate, plot_style, save_figure, BLUE, ORANGE, INK2)

VERSIONS_TESTED = (5, 8)
MAX_AVERAGE_DAYS = 60           # the averaged-air regressions choose their averaging period up to this
HOT_SHARE = 0.10                # the hottest 10% of held-out days, by air temperature
PEAK_MIN_MEASURED = 0.8         # a year's peak is compared if this share of its June-September days is measured
SHARE_REQUIRED = 0.75           # criterion: version 8 beats every regression on this share of rivers
HEAT_DOME = ("2021-06-25", "2021-07-02")
QUICK_BC = ("07EA004", "08GA077")

REGRESSIONS = ("air, same day", "air, averaged", "air averaged, S-curve", "air averaged and discharge")
METHODS = ("version 8", "version 5") + REGRESSIONS + ("day-of-year average",)
METRICS = {"daily RMSE": "Daily RMSE (°C)", "7-day RMSE": "7-day mean RMSE (°C)",
           "yearly peak error": "Error in the yearly peak (°C)", "hot-day RMSE": "RMSE on the hottest 10% of days (°C)",
           "heat-dome RMSE": "RMSE in the 2021 heat dome (°C)"}


# --- Simple alternatives, fitted on the calibration years only ----------------------------------

def _averaged(tair, days):
    """Mean air temperature of the day and the `days` - 1 days before it."""
    return pd.Series(tair).rolling(days, min_periods=1).mean().to_numpy()


def _s_curve(x, mu, alpha, beta, gamma):
    """Logistic relation of Mohseni et al. (1998): water temperature levels off at low and high air
    temperatures (mu: lowest, alpha: highest, beta: air temperature at the steepest point)."""
    return mu + (alpha - mu) / (1 + np.exp(gamma * (beta - x)))


def _fit_line(x, y):
    """Least squares on the columns of x (an intercept is added)."""
    a = np.column_stack([np.ones(len(y))] + list(x))
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return coef


def _predict_line(coef, x):
    return coef[0] + sum(c * xi for c, xi in zip(coef[1:], x))


def _fit_s_curve(x, y):
    p0 = (max(float(np.min(y)), 0.0), float(np.max(y)), float(np.median(x)), 0.2)
    bounds = ([-5, 0, -20, 0.01], [20, 40, 40, 2])
    try:
        p, _ = curve_fit(_s_curve, x, y, p0=np.clip(p0, *bounds), bounds=bounds, maxfev=20000)
        return p
    except RuntimeError:
        return None


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def regressions(cal, held):
    """Predictions of the held-out rows from regressions fitted on the measured calibration days.
    `cal` and `held` have columns Date, T_air, T_water, Discharge. The averaging period of the
    averaged-air regressions is the one that fits the calibration years best."""
    ok = cal.T_water.notna().to_numpy()
    y = cal.T_water.to_numpy(float)[ok]
    out, chosen = {}, {}

    coef = _fit_line([cal.T_air.to_numpy(float)[ok]], y)
    out["air, same day"] = _predict_line(coef, [held.T_air.to_numpy(float)])

    def best(fit, predict):
        """Fit for every averaging period; keep the one with the lowest calibration RMSE."""
        scores = {}
        for days in range(1, MAX_AVERAGE_DAYS + 1):
            xc = _averaged(cal.T_air.to_numpy(float), days)
            p = fit(xc[ok], days)
            if p is not None:
                scores[days] = (_rmse(predict(p, xc[ok], days), y), p)
        days = min(scores, key=lambda d: scores[d][0])
        return days, scores[days][1]

    days, coef = best(lambda x, d: _fit_line([x], y), lambda p, x, d: _predict_line(p, [x]))
    out["air, averaged"] = _predict_line(coef, [_averaged(held.T_air.to_numpy(float), days)])
    chosen["air, averaged"] = days

    days, p = best(lambda x, d: _fit_s_curve(x, y), lambda p, x, d: _s_curve(x, *p))
    out["air averaged, S-curve"] = _s_curve(_averaged(held.T_air.to_numpy(float), days), *p)
    chosen["air averaged, S-curve"] = days

    q_floor = float(cal.Discharge[cal.Discharge > 0].min())
    log_q = lambda q: np.log(np.maximum(q.to_numpy(float), q_floor))
    lq = log_q(cal.Discharge)[ok]
    days, coef = best(lambda x, d: _fit_line([x, lq], y), lambda p, x, d: _predict_line(p, [x, lq]))
    out["air averaged and discharge"] = _predict_line(
        coef, [_averaged(held.T_air.to_numpy(float), days), log_q(held.Discharge)])
    chosen["air averaged and discharge"] = days

    doy_c = np.minimum(cal.Date.dt.dayofyear, 365)
    clim = cal.groupby(doy_c).T_water.mean().reindex(range(1, 366))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True, min_periods=1).mean()
    out["day-of-year average"] = clim.iloc[15:-15].loc[np.minimum(held.Date.dt.dayofyear, 365)].to_numpy()
    return out, chosen


# --- Scores on the held-out years ---------------------------------------------------------------

def scores(held, preds, heat_dome):
    """Each method's errors on the held-out measured days."""
    obs = held.T_water.to_numpy(float)
    measured = np.isfinite(obs)
    tair = held.T_air.to_numpy(float)
    hot = measured & (tair >= np.quantile(tair[measured], 1 - HOT_SHARE))
    week_obs = pd.Series(obs).rolling(7, center=True).mean().to_numpy()      # NaN unless all 7 days measured
    years = held.Date.dt.year.to_numpy()
    summer = held.Date.dt.month.between(6, 9).to_numpy()
    peak_years = [y for y in np.unique(years) if measured[(years == y) & summer].mean() >= PEAK_MIN_MEASURED]
    dome = (held.Date >= HEAT_DOME[0]).to_numpy() & (held.Date <= HEAT_DOME[1]).to_numpy() & measured
    out = {}
    for name, pred in preds.items():
        pred = np.asarray(pred, float)
        week_pred = pd.Series(pred).rolling(7, center=True).mean().to_numpy()
        w = np.isfinite(week_obs) & np.isfinite(week_pred)
        peaks = [np.max(pred[(years == y) & measured]) - np.max(obs[(years == y) & measured]) for y in peak_years]
        row = {"daily RMSE": _rmse(pred[measured], obs[measured]),
               "bias": float(np.mean(pred[measured] - obs[measured])),
               "7-day RMSE": _rmse(week_pred[w], week_obs[w]),
               "yearly peak error": float(np.mean(np.abs(peaks))) if peaks else np.nan,
               "hot-day RMSE": _rmse(pred[hot], obs[hot])}
        if heat_dome:
            row["heat-dome RMSE"] = _rmse(pred[dome], obs[dome]) if dome.sum() >= 5 else np.nan
        out[name] = row
    return out


# --- Jobs ---------------------------------------------------------------------------------------

def _bc_files(st):
    """The station's calibration years (from the first 1 January, as V15) and held-out years as input files."""
    from v15_british_columbia import _load, _write
    ts, _, _ = _load()
    g = ts[ts.station_number == st].sort_values("date").reset_index(drop=True)
    cal = g[g.period == "calibration"].reset_index(drop=True)
    first = cal.date.iloc[0]
    start = None if (first.month == 1 and first.day == 1) else f"{first.year + 1}-01-01"
    cal_path, _ = _write(st, "calibration", cal, start=start, tag="_v17")
    val_path, _ = _write(st, "validation", g[g.period == "validation"].reset_index(drop=True), tag="_v17")
    return cal_path, val_path


def _files(args):
    """The input files of one river (the BC files are written here, once, before the calibrations read them)."""
    dataset, st = args
    if dataset == "Switzerland":
        return river_csv(st, "calibration"), river_csv(st, "validation")
    return _bc_files(st)


def _model(args):
    """DE calibration of one version on the calibration years; prediction of the held-out years with
    the calibration years' mean discharge."""
    dataset, st, version, quick, (cal_path, val_path) = args
    tag = f"v17_{st}_{version}"
    q = mean_discharge(cal_path)
    settings = {"n_run": 40, "n_particles": 8} if quick else dict(DE_SETTINGS)
    with quiet():
        d = calibrate(cal_path, version, objective="NSE", name=tag, Qmedia=q, optimization=settings)
        _, sim = daily(simulate(val_path, version, d.par_best, "CRN", q, name=tag + "_pred"))
    return dataset, st, version, sim


def _stations(quick):
    from v15_british_columbia import _load
    swiss = [("Switzerland", st) for st in (("MAH_2369",) if quick else RIVERS)]
    bc = QUICK_BC if quick else sorted(_load()[0].station_number.unique())
    return swiss + [("British Columbia", st) for st in bc]


def run(ctx) -> Result:
    res = Result(
        code="V17", title="Is the model better than a simple regression?",
        question="Does air2stream predict years it was not calibrated on better than regressions of water "
                 "temperature on air temperature, fitted on the same years?",
        method="Three Swiss rivers (calibration and validation files) and the 23 British Columbia rivers of "
               "Callahan & Moore (2025) (calibration years from the first 1 January to 2020; held-out years "
               "2021-2022, which include the June 2021 heat dome). air2stream versions 5 and 8: DE, NSE, "
               "Crank-Nicolson, the authors' parameter ranges, Qmedia from the calibration years. Four "
               "regressions, fitted by least squares on the measured calibration days: (1) on the day's air "
               "temperature; (2) on air temperature averaged over the day and the days before it, the "
               f"number of days (1 to {MAX_AVERAGE_DAYS}) chosen on the calibration years; (3) the S-curve "
               "(logistic) relation of Mohseni et al. (1998) on the averaged air temperature; (4) on the "
               "averaged air temperature and the logarithm of discharge. The day-of-year average of the "
               "calibration years is shown for scale. Every method predicts the held-out years, scored on "
               "their measured days: daily RMSE, RMSE of 7-day means, the mean error in each year's highest "
               f"temperature (years with at least {PEAK_MIN_MEASURED:.0%} of June-September measured), RMSE on "
               f"the {HOT_SHARE:.0%} of days with the hottest air, and, in British Columbia, RMSE in the heat "
               "dome (25 June to 2 July 2021).",
        criterion=f"In each data set, version 8's daily RMSE on the held-out years is lower than that of the best "
                  f"regression on that river (the best on the held-out years) on at least {SHARE_REQUIRED:.0%} of "
                  "the rivers.")
    stations = _stations(ctx.quick)
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            files = dict(zip(stations, ex.map(_files, stations)))
            sims = list(ex.map(_model, [(ds, st, v, ctx.quick, files[(ds, st)])
                                        for ds, st in stations for v in VERSIONS_TESTED]))
    res.seconds = t.seconds
    sims = {(ds, st, v): sim for ds, st, v, sim in sims}

    rows, chosen_rows = [], []
    for ds, st in stations:
        cal_path, val_path = files[(ds, st)]
        cal = pd.read_csv(cal_path, parse_dates=["Date"])
        held = pd.read_csv(val_path, parse_dates=["Date"])
        preds, chosen = regressions(cal, held)
        preds = {f"version {v}": sims[(ds, st, v)] for v in VERSIONS_TESTED} | preds
        river = RIVERS.get(st, st)
        for method, row in scores(held, preds, heat_dome=ds == "British Columbia").items():
            rows.append({"data set": ds, "river": river, "method": method, **row})
        chosen_rows.append({"data set": ds, "river": river, **{f"days averaged: {k}": v for k, v in chosen.items()}})
    df = pd.DataFrame(rows)

    wide = df.pivot_table(index=["data set", "river"], columns="method", values="daily RMSE")
    wide["best regression"] = wide[list(REGRESSIONS)].min(axis=1)
    wide["which"] = wide[list(REGRESSIONS)].idxmin(axis=1)
    wide["version 8 better by (°C)"] = wide["best regression"] - wide["version 8"]
    share = (wide["version 8 better by (°C)"] > 0).groupby(level="data set").mean()
    res.passed = bool((share >= SHARE_REQUIRED).all())

    parts = []
    for ds in ("Switzerland", "British Columbia"):
        if ds not in share.index:
            continue
        w = wide.loc[ds]
        parts.append(f"{ds}: version 8 beat every regression on {int((w['version 8 better by (°C)'] > 0).sum())} "
                     f"of {len(w)} rivers; median daily RMSE {w['version 8'].median():.2f} °C against "
                     f"{w['best regression'].median():.2f} °C for the best regression")
    res.summary = "; ".join(parts) + "."

    # Medians over the rivers of each data set, and the number of rivers where version 8 does better.
    table = []
    for ds in df["data set"].unique():
        g = df[df["data set"] == ds]
        v8 = g[g.method == "version 8"].set_index("river")
        for method in METHODS:
            m = g[g.method == method].set_index("river")
            row = {"data set": ds, "method": method}
            for key in METRICS:
                if key in m and m[key].notna().any():
                    row[f"{key}, median (°C)"] = m[key].median()
                    if method != "version 8":
                        both = m[key].notna() & v8[key].notna()
                        row[f"{key}: version 8 better"] = f"{int((v8[key][both] < m[key][both]).sum())} of {int(both.sum())}"
            row["bias, median (°C)"] = m["bias"].median()
            table.append(row)
    table = pd.DataFrame(table)

    figs = [_fig_scatter(wide), _fig_metrics(df)]
    res.sections.append(Section(
        "Version 8 against the best regression, river by river",
        "The best regression is the one with the lowest error on the held-out years: a choice nobody could make "
        "before seeing those years, so the comparison favours the regressions.",
        figures=[figs[0]], tables=[("Daily RMSE on the held-out years (°C)", _rounded(wide.reset_index()))]))
    res.sections.append(Section(
        "Every method and every measure",
        "Medians over the rivers of each data set. \"version 8 better\" counts the rivers on which version 8 had "
        "the smaller error.",
        figures=[figs[1]], tables=[("Medians over the rivers", _rounded(table))]))
    res.sections.append(Section(
        "Averaging period chosen by the regressions",
        "Days of air temperature averaged by each averaged-air regression, chosen on the calibration years.",
        tables=[("Days averaged", pd.DataFrame(chosen_rows))]))
    res.sections.append(Section("Every river and method", "", tables=[("Every river and method", _rounded(df))]))
    which = wide["which"].value_counts()
    worst = int((wide[list(REGRESSIONS)].idxmax(axis=1) == "air, same day").sum())
    res.notes.append(
        f"The S-curve regression on averaged air temperature came closest on {int(which.get('air averaged, S-curve', 0))} "
        f"of {len(wide)} rivers. It levels off near 0 °C, where a river stays all winter however cold the air, "
        "and the averaging gives it some memory of earlier days. A straight line on the day's air temperature has "
        f"neither, and was the worst regression on {worst} of {len(wide)} rivers. air2stream carries the water temperature "
        "from one day to the next and lets the river's response change with discharge and season.")
    parts = []
    for key, label in (("yearly peak error", "each year's highest temperature"),
                       ("hot-day RMSE", f"the {HOT_SHARE:.0%} of days with the hottest air"),
                       ("heat-dome RMSE", "the 2021 heat dome")):
        for ds in ("British Columbia", "Switzerland"):
            won, n, v8, best = _beats(df, key, ds)
            if n:
                parts.append(f"{label} in {ds}: {won} of {n} rivers (median error {v8:.2f} °C against {best:.2f} °C)")
    res.notes.append(
        "On the extremes, air2stream's lead is smaller and less consistent. Rivers on which version 8 did better than "
        "every regression (compared with the regression closest on each river): " + "; ".join(parts) + ". For a "
        "question about peaks or hot spells, compare the model with a regression on your own held-out years, and "
        "check the yearly statistics by cross-validation (USER_GUIDE.md §13).")
    return res


def _beats(df, key, ds):
    """Rivers of a data set on which version 8's `key` was below every regression's, the number of rivers
    scored, and the medians of version 8's and of the closest regression's `key`."""
    g = df[df["data set"] == ds]
    if key not in g or g[key].isna().all():
        return 0, 0, np.nan, np.nan
    w = g.pivot(index="river", columns="method", values=key)
    best = w[list(REGRESSIONS)].min(axis=1)
    ok = w["version 8"].notna() & best.notna()
    return (int((w["version 8"][ok] < best[ok]).sum()), int(ok.sum()), float(w["version 8"][ok].median()),
            float(best[ok].median()))


def _rounded(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(3)
    return out


def _fig_scatter(wide):
    import matplotlib.pyplot as plt
    plot_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for ds, colour, marker in (("British Columbia", BLUE, "o"), ("Switzerland", ORANGE, "s")):
        if ds not in wide.index.get_level_values(0):
            continue
        w = wide.loc[ds]
        ax.plot(w["version 8"], w["best regression"], marker, color=colour, ms=6, mfc="none", mew=1.4,
                label=ds, ls="none")
        if ds == "Switzerland":
            for river, r in w.iterrows():
                ax.annotate(river, (r["version 8"], r["best regression"]), xytext=(5, -3),
                            textcoords="offset points", fontsize=7.5, color=INK2)
    lo = 0.9 * min(wide["version 8"].min(), wide["best regression"].min())
    hi = 1.05 * max(wide["version 8"].max(), wide["best regression"].max())
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.annotate("same error", (hi, hi), xytext=(-3, -12), textcoords="offset points", ha="right", fontsize=7.5,
                color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("air2stream version 8: daily RMSE (°C)")
    ax.set_ylabel("Best regression: daily RMSE (°C)")
    ax.set_title("Held-out years, one point per river.\nAbove the line: air2stream predicted better", fontsize=9.5)
    ax.legend(loc="lower right")
    return (save_figure(fig, "V17_regression.png"),
            "Daily RMSE on the held-out years: air2stream version 8 (across) against the regression that did best "
            "on that river (up). Points above the dashed line: air2stream predicted better.")


def _fig_metrics(df):
    import matplotlib.pyplot as plt
    plot_style()
    datasets = [ds for ds in ("British Columbia", "Switzerland") if ds in set(df["data set"])]
    keys = [k for k in METRICS if k in df and df[k].notna().any()]
    fig, axes = plt.subplots(len(datasets), len(keys), figsize=(2.6 * len(keys), 2.9 * len(datasets)),
                             squeeze=False, sharey=True)
    for i, ds in enumerate(datasets):
        g = df[df["data set"] == ds]
        for j, key in enumerate(keys):
            ax = axes[i][j]
            if not g[key].notna().any():
                ax.set_axis_off()
                continue
            for k, method in enumerate(METHODS):
                vals = g[g.method == method][key].dropna()
                colour = BLUE if method == "version 8" else ORANGE if method == "version 5" else INK2
                if len(vals) > 1:
                    ax.plot([vals.quantile(0.25), vals.quantile(0.75)], [k, k], color=colour, lw=1.2, alpha=0.6)
                ax.plot(vals.median(), k, "o", color=colour, ms=5)
            ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
            ax.set_title(f"{METRICS[key]}", fontsize=8.5)
            ax.set_xlim(left=0)
        axes[i][0].set_ylabel(ds)
    axes[0][0].set_ylim(len(METHODS) - 0.5, -0.5)        # first method at the top (the axes share y)
    fig.suptitle("Held-out years: median over the rivers (line: middle half of the rivers)", fontsize=10)
    fig.tight_layout()
    return (save_figure(fig, "V17_measures.png"),
            "Median error of each method over the rivers of each data set (dot) and the range of the middle half "
            "of the rivers (line). Blue: air2stream version 8; orange: version 5; grey: the regressions and the "
            "day-of-year average, all fitted on the same calibration years.")
