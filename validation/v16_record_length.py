"""
V16 - How many years of data does a calibration need?

A user planning measurements, or holding a short record, needs to know how long a record
must be before the calibrated model predicts other years about as well as it can. Each
Swiss river is calibrated on 1, 2, 3, 5 and 10 consecutive years of its calibration period
(several placements of each), and on the whole period, and every calibration predicts the
same later years (the validation file).
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (RIVERS, WORK, Result, Section, Timer, calibrate, daily, mean_discharge, metrics, river_csv,
                    simulate, plot_style, save_figure, BLUE, ORANGE, INK2, LIGHT_GREY)
from v5_real_rivers import _benchmarks

VERSIONS_TESTED = (5, 8)
LENGTHS = (1, 2, 3, 5, 10)        # years of data; the whole calibration period is added
MAX_PLACEMENTS = 6                # windows of each length, spread over the period
MIN_MEASURED = 0.8                # a year can be in a window if this share of its days has water temperature
TOL_MEDIAN = 0.05                 # °C: criterion for 3 or more years, version 8


def usable_years(st):
    cal = pd.read_csv(river_csv(st, "calibration"), parse_dates=["Date"])
    share = cal.groupby(cal.Date.dt.year).T_water.apply(lambda s: s.notna().mean())
    return [int(y) for y in share.index if share[y] >= MIN_MEASURED]


def windows(years, length):
    """Up to MAX_PLACEMENTS windows of `length` consecutive usable years, spread evenly."""
    runs = [years[i:i + length] for i in range(len(years) - length + 1)
            if years[i + length - 1] - years[i] == length - 1]
    if len(runs) <= MAX_PLACEMENTS:
        return runs
    pick = np.unique(np.round(np.linspace(0, len(runs) - 1, MAX_PLACEMENTS)).astype(int))
    return [runs[i] for i in pick]


def jobs(quick):
    out = []
    rivers = ("MAH_2369",) if quick else tuple(RIVERS)
    versions = (8,) if quick else VERSIONS_TESTED
    for st in rivers:
        years = usable_years(st)
        for version in versions:
            for length in ((1, 3) if quick else LENGTHS):
                if length >= len(years):
                    continue
                for w in (windows(years, length)[:2] if quick else windows(years, length)):
                    out.append((st, version, length, tuple(w)))
            out.append((st, version, "all", None))
    return out


def _fit(args):
    """Calibrate on the window's years only (as a user holding just those years would), with the
    window's own mean discharge, and predict the validation years."""
    st, version, length, years = args
    tag = f"v16_{st}_{version}_{length}_{years[0] if years else 'all'}"
    folder = os.path.join(WORK, tag)
    os.makedirs(folder, exist_ok=True)
    cal_path = river_csv(st, "calibration")
    if years is not None:
        cal = pd.read_csv(cal_path, parse_dates=["Date"])
        window = cal[cal.Date.dt.year.isin(years)]
        cal_path = os.path.join(folder, "window.csv")
        window.assign(Date=window.Date.dt.strftime("%Y-%m-%d")).to_csv(cal_path, index=False)
    qmedia = mean_discharge(cal_path)
    d = calibrate(cal_path, version, objective="NSE", name=tag, Qmedia=qmedia)
    val_path = river_csv(st, "validation")
    obs, sim = daily(simulate(val_path, version, d.par_best, "CRN", qmedia, name=tag + "_pred"))
    m = metrics(obs, sim)
    simple = {name: metrics(pd.read_csv(val_path).T_water.to_numpy(float), pred)["RMSE"]
              for name, pred in _benchmarks(cal_path, val_path).items()}
    return {"river": RIVERS[st], "version": version, "years of data": length,
            "calibration years": f"{years[0]}-{years[-1]}" if years else "all",
            "test RMSE (°C)": m["RMSE"], "test bias (°C)": m["bias"],
            "best simple alternative RMSE (°C)": min(simple.values())}


def run(ctx) -> Result:
    res = Result(
        code="V16", title="How many years of data a calibration needs",
        question="How long must a record be before the calibrated model predicts other years about as well as "
                 "a calibration on a long record does?",
        method="For each Swiss river, the years of the calibration file with at least "
               f"{MIN_MEASURED:.0%} of their days measured are cut into windows of {', '.join(map(str, LENGTHS))} "
               f"consecutive years (up to {MAX_PLACEMENTS} placements of each length, spread over the period). "
               "Each window alone is the calibration file, as for a user holding only those years: DE, NSE, "
               "Crank-Nicolson, the authors' parameter ranges, Qmedia from the window. Each calibration "
               "predicts the same later years, the validation file, and is compared with the calibration on the "
               "whole calibration file. Versions 5 and 8. The simple alternatives of V5 (day-of-year average, "
               "air-temperature regression), fitted on the same window, give the scale.",
        criterion=f"For version 8, with 3 or more years of data, the median RMSE on the later years is within "
                  f"{TOL_MEDIAN} °C of the whole-period calibration's, on every river.")
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            rows = list(ex.map(_fit, jobs(ctx.quick)))
    res.seconds = t.seconds
    df = pd.DataFrame(rows)
    full = df[df["years of data"] == "all"].set_index(["river", "version"])["test RMSE (°C)"]
    df["change against the whole period (°C)"] = [r["test RMSE (°C)"] - full[(r["river"], r["version"])]
                                                  for _, r in df.iterrows()]

    def agg(g):
        return pd.Series({"placements": len(g), "median RMSE (°C)": g["test RMSE (°C)"].median(),
                          "worst RMSE (°C)": g["test RMSE (°C)"].max(),
                          "median change (°C)": g["change against the whole period (°C)"].median(),
                          "worst change (°C)": g["change against the whole period (°C)"].max(),
                          "simple alternative, median (°C)": g["best simple alternative RMSE (°C)"].median()})
    order = {length: i for i, length in enumerate(LENGTHS + ("all",))}
    summary = (df.groupby(["river", "version", "years of data"], sort=False).apply(agg).reset_index()
               .sort_values(["river", "version", "years of data"], key=lambda s: s.map(order) if s.name ==
                            "years of data" else s))
    v8 = summary[(summary.version == 8) & summary["years of data"].map(lambda x: x == "all" or x >= 3)
                 & (summary["years of data"] != "all")]
    res.passed = bool((v8["median change (°C)"] <= TOL_MEDIAN).all()) if len(v8) else None

    parts = []
    for river in summary.river.unique():
        s = summary[(summary.river == river) & (summary.version == 8)].set_index("years of data")
        if 1 in s.index and "all" in s.index:
            three = f"; 3 years {s.loc[3, 'median RMSE (°C)']:.2f}" if 3 in s.index else ""
            parts.append(f"{river} 1 year {s.loc[1, 'median RMSE (°C)']:.2f} (worst "
                         f"{s.loc[1, 'worst RMSE (°C)']:.2f}){three}; whole period {s.loc['all', 'median RMSE (°C)']:.2f}")
    res.summary = ("Version 8, median RMSE on the later years: " + " · ".join(parts) + " °C. "
                   f"With 3 or more years, version 8's median was within "
                   f"{v8['median change (°C)'].max():+.3f} °C of the whole period's (worst single placement "
                   f"{v8['worst change (°C)'].max():+.2f} °C)." if len(v8) else "")
    res.notes.append("A single year can be enough when it is typical, but the worst single years cost much more "
                     "than the median: an unusual year (a heatwave, a wet summer) teaches the model the wrong "
                     "balance between air temperature and discharge. More years mostly protect against that. "
                     "The later years tested here are similar to the calibration years; predicting conditions "
                     "outside the calibrated range (V10) needs those conditions, or a range close to them, in the "
                     "record.")
    fig = _figure(df, summary)
    shown = summary.copy()
    for c in shown.columns:
        if "(°C)" in c:
            shown[c] = shown[c].round(3)
    res.sections.append(Section(
        "Prediction error against the length of the record",
        "Each point is one calibration on a window of consecutive years; the line joins the medians. Every "
        "calibration predicts the same later years.",
        figures=[fig], tables=[("By river, version and years of data", shown)]))
    detail = df.copy()
    for c in detail.columns:
        if "(°C)" in c:
            detail[c] = detail[c].round(3)
    res.sections.append(Section("Every calibration", "", tables=[("Every calibration", detail)]))
    return res


def _figure(df, summary):
    import matplotlib.pyplot as plt
    plot_style()
    rivers = list(summary.river.unique())
    fig, axes = plt.subplots(1, len(rivers), figsize=(4.0 * len(rivers), 3.8), sharey=True, squeeze=False)
    labels = [str(x) for x in LENGTHS] + ["whole period"]
    for ax, river in zip(axes[0], rivers):
        for version, colour, dx in ((5, ORANGE, -0.08), (8, BLUE, 0.08)):
            sub = df[(df.river == river) & (df.version == version)]
            if sub.empty:
                continue
            xs, med = [], []
            for i, length in enumerate(LENGTHS + ("all",)):
                vals = sub[sub["years of data"] == length]["test RMSE (°C)"]
                if vals.empty:
                    continue
                ax.plot(np.full(len(vals), i + dx), vals, "o", color=colour, ms=3.5, alpha=0.5, mfc="none")
                xs.append(i + dx)
                med.append(vals.median())
            ax.plot(xs, med, "-", color=colour, lw=1.6, label=f"version {version}")
        simple = summary[(summary.river == river) & (summary.version == 8)]
        if not simple.empty:
            ax.axhline(simple["simple alternative, median (°C)"].median(), color=LIGHT_GREY, lw=1, ls="--")
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("Years of data used for calibration")
        ax.set_title(river, fontsize=10)
    axes[0][0].set_ylabel("RMSE on the later years (°C)")
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Shorter records: how much worse the calibrated model predicts other years", fontsize=10)
    fig.tight_layout()
    return (save_figure(fig, "V16_record_length.png"),
            "Each circle is one calibration on a window of consecutive years; the line joins the medians "
            "(blue: version 8; orange: version 5). Grey dashed: the best simple alternative, fitted on the same "
            "years (median). Every calibration predicts the same later years.")
