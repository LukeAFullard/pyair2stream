"""
V9 - Probabilities that a limit was exceeded.

A compliance question is usually about a statistic of a year: its highest daily
mean, its highest 7-day mean, or the number of days above a threshold. The
package answers it from the saved simulations (docs/METHODS.md §13): the
statistic is computed in each simulated series, the probability of exceedance
is the share of series above the limit, and a range is given by percentiles.
This check tests those answers. (A) On synthetic data with a known truth, the
stated probabilities must come true as often as they say. (B) On real rivers,
the probabilities for years not used for calibration must beat the simple
alternative of how often the limit was exceeded in past years.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DE_SETTINGS, RIVERS, WORK, Result, Section, Timer, load, mean_discharge,
                    published_params, quiet, river_csv, plot_style, save_figure, BLUE, ORANGE, AQUA, INK2,
                    LIGHT_GREY)
from v3_recovery import noise, truth_series

STATS = ("highest daily mean", "highest 7-day mean", "days above threshold")
STAT_COLOUR = dict(zip(STATS, (BLUE, ORANGE, AQUA)))
RANGES = (0.5, 0.9)                     # central ranges whose coverage is tested
STATED = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # stated probabilities in the reliability table
REPLICATES = {5: 30, 8: 12}             # synthetic data sets per model version (as V4)
REAL_VERSIONS = (5, 8)
N_SAMPLES = 1000
WARM_SEASON = (6, 7, 8, 9)
MIN_WARM_SEASON_OBSERVED = 0.8          # a year is used if at least this share of June-September is observed
WARM_QUANTILE = 0.9                     # 'days above threshold': threshold = this quantile of calibration days
LIMIT_QUANTILES = (0.25, 0.5, 0.75)     # real rivers: limits at these quantiles of the calibration years
COVERAGE_CONFIDENCE = 0.99              # accepted coverage: central 99% binomial range around the nominal share


# --- Statistics of a year, in every simulation and in the measurements --------------------

def year_statistics(ens, dates, obs, warm):
    """For each year with enough of its warm season observed: each statistic in every simulated
    series (array) and in the measurements (number). The simulations are restricted to the days
    that were measured, so both sides are computed over the same days."""
    dates = pd.DatetimeIndex(dates)
    obs = np.asarray(obs, float)
    from pyair2stream import scenario
    out = []
    for year in sorted(set(dates.year)):
        iy = np.asarray(dates.year == year)
        warm_season = iy & np.asarray(dates.month.isin(WARM_SEASON))
        if np.isfinite(obs[warm_season]).mean() < MIN_WARM_SEASON_OBSERVED:
            continue
        seen = iy & np.isfinite(obs)
        # 7-day moving means within the year; a window counts only if all its 7 days were measured.
        week_obs = pd.Series(obs[iy]).rolling(7).mean().to_numpy()
        week_sim = pd.DataFrame(ens[:, iy].T).rolling(7).mean().to_numpy()
        full = np.isfinite(week_obs)
        out.append((year, {
            "highest daily mean": (ens[:, seen].max(axis=1), float(obs[seen].max())),
            "highest 7-day mean": (week_sim[full].max(axis=0), float(week_obs[full].max())),
            "days above threshold": (scenario.exceedance(ens[:, seen], warm).astype(float),
                                     float((obs[seen] > warm).sum())),
        }))
    return out


def pit(sims, value, rng):
    """Share of simulations below the measured value; ties (day counts) are split at random, the
    standard treatment for counts. If the probabilities are right, this is uniform on 0-1."""
    sims = np.asarray(sims)
    return (np.sum(sims < value) + rng.random() * np.sum(sims == value)) / len(sims)


def coverage_band(n, level):
    from scipy.stats import binom
    lo, hi = binom.interval(COVERAGE_CONFIDENCE, n, level)
    return lo / n, hi / n


def _forward_ensemble(version, par_best, chain, csv, qmedia, folder, seed):
    """FORWARD run with prediction intervals and the saved simulations, as a user would."""
    from pyair2stream import scenario
    from pyair2stream.optimization import forward_mode
    fcfg = {"version": version, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": qmedia,
            "parameters_forward": [float(x) for x in par_best],
            "uncertainty_options": {"noise_model": "ar1", "save_ensemble": True},
            "forward_options": {"enable_prediction_intervals": True, "mcmc_chain_path": chain,
                                "n_samples": N_SAMPLES, "random_seed": seed},
            "paths": {"input_data": csv, "output_dir": os.path.join(folder, "fwd")}}
    fdata = load(fcfg, os.path.basename(folder) + "_fwd")
    with quiet():
        forward_mode(fdata)
    return scenario.load_ensemble(os.path.join(folder, "fwd", "Forward_Prediction_Ensemble_S_c_1d.npz"))


def _calibrate_mcmc(version, csv, qmedia, folder, tag, seed):
    """DE-MCMC with the default error model and likelihood; None if it did not converge."""
    from pyair2stream.optimization import DE_MCMC_mode
    out = os.path.join(folder, "out")
    cfg = {"version": version, "integrator": "CRN", "run_mode": "DE-MCMC", "objective_function": "NSE",
           "random_seed": seed, "Qmedia": qmedia, "parameter_bounds": AUTHORS_BOUNDS,
           "optimization": {**DE_SETTINGS, "mcmc_walkers": 32, "mcmc_steps": 20000},
           "paths": {"input_data": csv, "output_dir": out}}
    data = load(cfg, tag)
    try:
        with quiet():
            DE_MCMC_mode(data, seed=seed)
    except RuntimeError:
        return None
    return data, os.path.join(out, "MCMC_chain_S_c_1d.csv")


# --- Part A: synthetic data with a known truth ------------------------------------------------

def _synthetic(args):
    version, r = args
    tag = f"v9_{version}_{r}"
    folder = os.path.join(WORK, tag)
    os.makedirs(folder, exist_ok=True)
    q_cal = mean_discharge(river_csv("MAH_2369", "calibration"))
    par_true = published_params(version, "MAH_2369")
    src_c, truth_c = truth_series(version, par_true, "calibration", q_cal, tag=tag)
    src_v, truth_v = truth_series(version, par_true, "validation", q_cal, tag=tag)
    rng = np.random.default_rng(90000 + 100 * version + r)
    cal_csv, val_csv = os.path.join(folder, "cal.csv"), os.path.join(folder, "val.csv")
    obs_c = np.round(truth_c + noise(len(truth_c), "ar1", rng), 3)
    obs_v = np.round(truth_v + noise(len(truth_v), "ar1", rng), 3)
    src_c.assign(T_water=obs_c).to_csv(cal_csv, index=False)
    src_v.assign(T_water=obs_v).to_csv(val_csv, index=False)
    fit = _calibrate_mcmc(version, cal_csv, q_cal, folder, tag, seed=r + 1)
    if fit is None:
        return {"version": version, "replicate": r, "converged": False, "years": []}
    data, chain = fit
    ens, dates = _forward_ensemble(version, data.par_best, chain, val_csv, q_cal, folder, seed=r + 1)
    warm = float(np.quantile(truth_c, WARM_QUANTILE))
    prng = np.random.default_rng(r)
    years = []
    for year, stats in year_statistics(ens, dates, obs_v, warm):
        for name, (sims, value) in stats.items():
            years.append({"statistic": name, "year": year, "pit": pit(sims, value, prng)})
    return {"version": version, "replicate": r, "converged": True, "years": years}


# --- Part B: real rivers, years not used for calibration --------------------------------------

def _real(args):
    st, version = args
    tag = f"v9r_{st}_{version}"
    folder = os.path.join(WORK, tag)
    os.makedirs(folder, exist_ok=True)
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    q_cal = mean_discharge(cal)
    fit = _calibrate_mcmc(version, cal, q_cal, folder, tag, seed=1)
    if fit is None:
        return {"river": RIVERS[st], "version": version, "converged": False, "rows": []}
    data, chain = fit
    ens, dates = _forward_ensemble(version, data.par_best, chain, val, q_cal, folder, seed=1)
    cal_df = pd.read_csv(cal, parse_dates=["Date"])
    obs_c = cal_df.T_water.to_numpy(float)
    warm = float(np.nanquantile(obs_c, WARM_QUANTILE))
    # The same statistics in the measured calibration years: the past-years alternative, and the limits.
    past = year_statistics(np.zeros((1, len(obs_c))), cal_df.Date, obs_c, warm)
    past_values = {name: np.array([stats[name][1] for _, stats in past]) for name in STATS}
    limits = {name: np.quantile(v, LIMIT_QUANTILES) for name, v in past_values.items()}
    obs_v = pd.read_csv(val).T_water.to_numpy(float)
    prng = np.random.default_rng(7)
    rows = []
    for year, stats in year_statistics(ens, dates, obs_v, warm):
        for name, (sims, value) in stats.items():
            row = {"river": RIVERS[st], "version": version, "year": year, "statistic": name,
                   "threshold (°C)": round(warm, 2) if name == "days above threshold" else "",
                   "measured": value, "median": float(np.median(sims)),
                   "pit": pit(sims, value, prng)}
            for level in RANGES:
                lo, hi = np.percentile(sims, [50 - 50 * level, 50 + 50 * level])
                row[f"{level:.0%} range"] = (float(lo), float(hi))
            row["events"] = [(float(lim), float(np.mean(sims > lim)), float(np.mean(past_values[name] > lim)),
                              float(value > lim)) for lim in limits[name]]
            rows.append(row)
    return {"river": RIVERS[st], "version": version, "converged": True, "rows": rows,
            "past years": {name: len(v) for name, v in past_values.items()}}


# --- The check --------------------------------------------------------------------------------

def run(ctx) -> Result:
    res = Result(
        code="V9", title="Probabilities that a limit was exceeded",
        question="When the package says there is a given chance that a yearly statistic (the highest daily "
                 "mean, the highest 7-day mean, or the number of days above a threshold) exceeded a limit, "
                 "does that happen as often as it says? And on real rivers, are these probabilities better "
                 "than going by how often the limit was exceeded in past years?",
        method="Probabilities are computed as docs/METHODS.md §13 describes: DE-MCMC with the default error "
               "model and likelihood on the calibration years, a FORWARD run of 1000 simulations of the later "
               "years (save_ensemble), each statistic computed per year in every simulated series (7-day "
               "moving means with pandas rolling, day counts with scenario.exceedance), and the probability of "
               "exceedance taken as the share of series above the limit. Every statistic is computed in the "
               "simulations over exactly the days that were measured, and a year is used only if at least 80% "
               "of its June-September days were measured. 'Days above threshold' counts days above the 90th "
               "percentile of the calibration years' daily temperatures. "
               "(A) Synthetic data as in V4 (Mentue forcing, the published version 5 and 8 parameters as the "
               "truth, AR(1) noise with sd 0.5 °C and lag-1 correlation 0.7; 30 and 12 data sets, each with "
               "three later years). For each year and statistic, the measured value's position among the "
               "simulations is recorded (the share of simulations below it; ties in day counts split at "
               "random). If the probabilities are right, a limit with a stated chance p of being exceeded is "
               "exceeded in a share p of cases, and the central 50% and 90% ranges contain the measured value "
               "50% and 90% of the time. "
               "(B) The three Swiss rivers, versions 5 and 8, calibrated on the calibration years and "
               "predicting each later year. Limits are set at the 25th, 50th and 75th percentiles of the "
               "statistic over the calibration years. The package's probabilities are scored against what "
               "was measured with the Brier score (mean squared difference between the probability and the "
               "outcome, 1 or 0), and compared with the past-years alternative: the share of calibration "
               "years in which the limit was exceeded.",
        criterion=f"(A) For each version and statistic, the share of measured values inside the central 50% and "
                  f"90% ranges lies within the range expected by chance around 50% and 90% (central "
                  f"{COVERAGE_CONFIDENCE:.0%} binomial range for the number of years tested). (B) For each "
                  f"version and statistic, the package's Brier score is lower than the past-years "
                  f"alternative's (Brier skill score above 0).")
    reps = {5: 3} if ctx.quick else REPLICATES
    jobs_a = [(v, r) for v, n in reps.items() for r in range(n)]
    stations = ["MAH_2369"] if ctx.quick else list(RIVERS)
    versions_b = (5,) if ctx.quick else REAL_VERSIONS
    jobs_b = [(st, v) for st in stations for v in versions_b]
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            fut_b = [ex.submit(_real, j) for j in jobs_b]
            out_a = list(ex.map(_synthetic, jobs_a))
            out_b = [f.result() for f in fut_b]
    res.seconds = t.seconds

    # Part A
    a = pd.DataFrame([dict(version=o["version"], replicate=o["replicate"], **y)
                      for o in out_a if o["converged"] for y in o["years"]])
    n_conv = {v: sum(o["converged"] for o in out_a if o["version"] == v) for v in reps}
    rows_a, rel_rows, ok_a = [], [], True
    for (v, name), g in a.groupby(["version", "statistic"], sort=False):
        u = g.pit.to_numpy()
        row = {"version": v, "statistic": name, "data sets (converged)": f"{n_conv[v]} of {reps[v]}",
               "years tested": len(u)}
        for level in RANGES:
            inside = float(np.mean(np.abs(u - 0.5) <= level / 2))
            lo, hi = coverage_band(len(u), level)
            row[f"inside {level:.0%} range"] = inside
            row[f"accepted ({level:.0%} range)"] = f"{lo:.0%}-{hi:.0%}"
            ok_a &= lo <= inside <= hi
        rows_a.append(row)
        rel = {"version": v, "statistic": name}
        for p in STATED:
            rel[f"stated {p:.0%}"] = f"{np.mean(u > 1 - p):.0%}"
        rel_rows.append(rel)
    ok_a &= all(n_conv[v] == reps[v] for v in reps)
    table_a = pd.DataFrame(rows_a)
    table_rel = pd.DataFrame(rel_rows)

    # Part B
    b_rows = [r for o in out_b if o["converged"] for r in o["rows"]]
    b = pd.DataFrame(b_rows)
    brier_rows, ok_b = [], all(o["converged"] for o in out_b)
    for (v, name), g in b.groupby(["version", "statistic"], sort=False):
        ev = np.array([e for evs in g.events for e in evs])          # limit, p model, p past, outcome
        bs_model = float(np.mean((ev[:, 1] - ev[:, 3]) ** 2))
        bs_past = float(np.mean((ev[:, 2] - ev[:, 3]) ** 2))
        skill = 1 - bs_model / bs_past if bs_past > 0 else np.nan
        ok_b &= bool(np.isfinite(skill) and skill > 0)
        inside = {level: float(np.mean(np.abs(g.pit - 0.5) <= level / 2)) for level in RANGES}
        brier_rows.append({"version": v, "statistic": name, "river-years": len(g), "limits tested": len(ev),
                           "limits exceeded": int(ev[:, 3].sum()),
                           "Brier score, pyair2stream": round(bs_model, 3),
                           "Brier score, past years": round(bs_past, 3), "Brier skill score": round(skill, 2),
                           "inside 50% range": inside[0.5], "inside 90% range": inside[0.9]})
    table_brier = pd.DataFrame(brier_rows)
    by_river = []
    if len(b):
        bb = b.assign(inside90=b.pit.sub(0.5).abs() <= 0.45, error=b.measured - b["median"])
        for (v, river, name), g in bb.groupby(["version", "river", "statistic"], sort=False):
            by_river.append({"version": v, "river": river, "statistic": name, "years": len(g),
                             "inside 90% range": f"{int(g.inside90.sum())} of {len(g)}",
                             "measured minus predicted median, mean": round(float(g.error.mean()), 2),
                             "correlation, predicted median vs measured":
                                 round(float(np.corrcoef(g["median"], g.measured)[0, 1]), 2) if len(g) >= 5 else ""})
    table_river = pd.DataFrame(by_river)
    res.passed = bool(ok_a and ok_b)

    res.summary = (f"(A) Synthetic data: the 90% ranges contained the measured yearly statistic "
                   f"{table_a['inside 90% range'].min():.0%}-{table_a['inside 90% range'].max():.0%} of the time "
                   f"across versions and statistics, and the 50% ranges {table_a['inside 50% range'].min():.0%}-"
                   f"{table_a['inside 50% range'].max():.0%}"
                   f"{'' if ok_a else ' (outside the accepted range in some cases, see table)'}.")
    if len(table_brier):
        res.summary += (f" (B) Real rivers, years not used for calibration: Brier skill score against the "
                        f"past-years alternative {table_brier['Brier skill score'].min():.2f} to "
                        f"{table_brier['Brier skill score'].max():.2f} (above 0: better); the 90% ranges "
                        f"contained the measured statistic in {table_brier['inside 90% range'].min():.0%}-"
                        f"{table_brier['inside 90% range'].max():.0%} of river-years.")
    for t in (table_a, table_brier):
        for col in [c for c in t.columns if c.startswith("inside ")]:
            t[col] = t[col].map(lambda x: f"{x:.0%}")

    res.sections.append(Section(
        "A. Synthetic data: do the stated probabilities come true?",
        "Each line shows, for one statistic, how often a limit was exceeded when the package gave it a "
        "stated chance of being exceeded. On the diagonal, the probabilities are right. The table below the "
        "figure gives the same numbers; the one above it, how often the central ranges contained the measured "
        "value.",
        figures=[_fig_reliability(a, reps)],
        tables=[("A. Measured yearly statistic inside the central ranges (synthetic data)", table_a),
                ("A. How often a limit was exceeded, by the chance the package stated (synthetic data)", table_rel)]))
    if len(b):
        shown = b.drop(columns=["events", "pit"]).copy()
        for col in ("measured", "median"):
            shown[col] = shown[col].round(2)
        for level in RANGES:
            shown[f"{level:.0%} range"] = shown[f"{level:.0%} range"].map(lambda x: f"{x[0]:.2f} to {x[1]:.2f}")
        res.sections.append(Section(
            "B. Real rivers: years not used for calibration",
            "Each version was calibrated on a river's calibration years and predicted its later years. The "
            "figure shows, for version 8, the predicted ranges of each year's statistics and what was measured. "
            "The Brier score measures how close the probabilities were to what happened (0 is perfect); the "
            "past-years alternative gives each limit the share of calibration years in which it was exceeded.",
            figures=[_fig_real(b, v) for v in sorted(set(b.version), reverse=True)],
            tables=[("B. Probabilities scored against the past-years alternative (Brier score)", table_brier),
                    ("B. By river: how often the 90% range held, and the mean error of the predicted median",
                     table_river),
                    ("B. Each river-year: predicted median and ranges, and the measured value", shown)]))
    res.notes.append(
        "Part A tests the calculation of probabilities and ranges from the saved simulations, where the model "
        "and its error model are exactly right. Part B tests the whole approach on real rivers, where they are "
        "not, with few years per river: its coverage numbers are uncertain by several percentage points.")
    if ok_b and len(table_brier):
        res.notes.append(
            "On real rivers the package's probabilities were closer to what happened than the past-years "
            "alternative for every statistic: they use the year's own weather and flow, which past years cannot.")
    if len(table_brier):
        worse = table_brier[table_brier["Brier skill score"] <= 0]
        if len(worse):
            res.notes.append(
                "Where the package did not beat the past-years alternative: " + "; ".join(
                    f"version {r.version}, {r.statistic} (Brier skill score {r['Brier skill score']:.2f})"
                    for _, r in worse.iterrows()) + ". See the table by river.")
        # The river with the most years shows whether a version follows the year-to-year changes.
        river = b.groupby("river").year.nunique().idxmax()
        parts = []
        for v in sorted(set(b.version)):
            g = b[(b.river == river) & (b.version == v) & (b.statistic == "highest 7-day mean")]
            if len(g) >= 5:
                parts.append(f"version {v}: predicted median {g['median'].min():.2f}-{g['median'].max():.2f} °C, "
                             f"correlation with the measured peak {np.corrcoef(g['median'], g.measured)[0, 1]:.2f}")
        if parts:
            g = b[(b.river == river) & (b.statistic == "highest 7-day mean")]
            res.notes.append(
                f"Year-to-year changes: on the {river} the measured highest 7-day mean ranged from "
                f"{g.measured.min():.2f} to {g.measured.max():.2f} °C over the years tested ({'; '.join(parts)}). "
                f"A version that cannot follow these changes gives probabilities no better than past years; "
                f"compare versions by cross-validation before relying on one (example 06).")
        cov = b.assign(inside90=b.pit.sub(0.5).abs() <= 0.45).groupby(["version", "statistic"]).inside90.mean()
        res.notes.append(
            f"On real rivers the 90% ranges of yearly statistics contained the measured value in "
            f"{cov.min():.0%}-{cov.max():.0%} of river-years, so they are too narrow, while on synthetic data "
            f"(part A) they hold. The error model (AR(1), the same all year) describes day-to-day model errors; "
            f"on real rivers the model can also be off by a similar amount for a whole summer (see the mean "
            f"errors by river), which widens the true uncertainty of a yearly peak. Treat probabilities for "
            f"yearly statistics on real rivers as approximate, and check them on your own validation years.")
    return res


def _fig_reliability(a, reps):
    import matplotlib.pyplot as plt
    plot_style()
    versions = sorted(reps)
    fig, axes = plt.subplots(1, len(versions), figsize=(4.2 * len(versions) + 0.6, 4.0), squeeze=False)
    grid = np.linspace(0.05, 0.95, 19)
    for ax, v in zip(axes[0], versions):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for name in STATS:
            u = a[(a.version == v) & (a.statistic == name)].pit.to_numpy()
            if not len(u):
                continue
            ax.plot(grid, [np.mean(u > 1 - p) for p in grid], color=STAT_COLOUR[name], marker="o", ms=3.5,
                    lw=1.3, label=name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Chance of exceeding the limit, stated by the package")
        ax.set_title(f"Version {v} ({len(a[(a.version == v)]) // len(STATS)} years tested)")
    axes[0][0].set_ylabel("Share of cases in which the limit was exceeded")
    axes[0][0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Synthetic data: stated chances against how often the limit was exceeded", y=1.02)
    return (save_figure(fig, "V9_reliability.png"),
            "For limits set at every level of the package's predicted distribution, the share of cases in which "
            "the measured statistic exceeded the limit, against the chance the package stated. Points on the "
            "dashed diagonal mean the stated chances are right.")


def _fig_real(b, version):
    import matplotlib.pyplot as plt
    plot_style()
    g = b[b.version == version]
    keys = list(dict.fromkeys(zip(g.river, g.year)))
    fig, axes = plt.subplots(1, len(STATS), figsize=(11, 0.32 * len(keys) + 1.6), sharey=True)
    for ax, name in zip(axes, STATS):
        ax.axvline(0, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for i, (river, year) in enumerate(keys):
            r = g[(g.river == river) & (g.year == year) & (g.statistic == name)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = len(keys) - 1 - i
            m = r["median"]
            ax.plot([r["90% range"][0] - m, r["90% range"][1] - m], [y, y], color=LIGHT_GREY, lw=5,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r["50% range"][0] - m, r["50% range"][1] - m], [y, y], color=BLUE, lw=5,
                    solid_capstyle="butt", alpha=0.6, zorder=3)
            inside = r["90% range"][0] <= r["measured"] <= r["90% range"][1]
            ax.scatter(r["measured"] - m, y, s=22, color="#0b0b0b" if inside else ORANGE, zorder=4)
        unit = "days" if name == "days above threshold" else "°C"
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(f"relative to the predicted median ({unit})", fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(keys))[::-1], [f"{river} {year}" for river, year in keys], fontsize=7.5)
    from matplotlib.lines import Line2D
    axes[-1].legend(handles=[Line2D([0], [0], color=LIGHT_GREY, lw=5, label="90% range"),
                             Line2D([0], [0], color=BLUE, lw=5, alpha=0.6, label="50% range"),
                             Line2D([0], [0], marker="o", lw=0, color="#0b0b0b", label="measured"),
                             Line2D([0], [0], marker="o", lw=0, color=ORANGE, label="measured, outside the 90% range")],
                    loc="upper center", bbox_to_anchor=(-0.7, -0.08), ncol=4, fontsize=7.5)
    fig.suptitle(f"Real rivers, version {version}: predicted ranges for years not used for calibration", y=1.0)
    return (save_figure(fig, f"V9_real_rivers_v{version}.png"),
            f"Version {version}: the predicted 50% and 90% ranges of each year's statistics (from 1000 "
            f"simulations, over the days that were measured) and the measured value, all relative to the "
            f"predicted median (dashed line). 'Days above threshold' uses each river's own threshold (90th "
            f"percentile of its calibration-year temperatures).")
