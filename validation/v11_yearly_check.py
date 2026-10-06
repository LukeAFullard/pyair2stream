"""
V11 - Cross-validated check and correction of yearly statistics.

A compliance question is about a statistic of a year (its highest daily mean, its
highest 7-day mean, the number of days above a threshold). V9 found that on real
rivers the predicted ranges of these statistics can be off, while daily and 7-day
ranges hold. This check measures, on many more years, how often the ranges hold in
years the model was not calibrated on, finds why when they do not, and tests the
correction the package offers (docs/METHODS.md §11, §13).
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import binom

from common import (AUTHORS_BOUNDS, DE_SETTINGS, RIVERS, WORK, Result, Section, Timer, load, quiet, river_csv,
                    plot_style, save_figure, BLUE, ORANGE, INK2, LIGHT_GREY)

VERSIONS = (5, 8)
LEVELS = (0.5, 0.9)
COVERAGE_CONFIDENCE = 0.99          # accepted coverage: central 99% binomial range, as V9


def _full_record(st: str, write: bool = False) -> str:
    """The calibration and validation files of a river joined into one record. Written once, by
    the parent process before the workers start (`write=True`), so no worker reads a half-written
    file."""
    path = os.path.join(WORK, f"v11_{st}_full.csv")
    if write:
        os.makedirs(WORK, exist_ok=True)
        tmp = path + ".tmp"
        pd.concat([pd.read_csv(river_csv(st, "calibration")),
                   pd.read_csv(river_csv(st, "validation"))]).to_csv(tmp, index=False)
        os.replace(tmp, path)
    return path


def _job(args):
    """Leave-one-year-out cross-validation of one river and version with the package, its check of
    yearly statistics, and the nested correction of every held-out year."""
    st, version, quick = args
    from pyair2stream import scenario
    from pyair2stream.cross_validation import check_yearly_statistics, cross_validate
    csv = _full_record(st)
    q = pd.read_csv(csv).Discharge
    settings = {"n_run": 40, "n_particles": 8} if quick else dict(DE_SETTINGS)
    cfg = {"version": version, "integrator": "CRN", "run_mode": "DE", "objective_function": "NSE",
           "random_seed": 1, "optimization": settings, "parameter_bounds": AUTHORS_BOUNDS,
           "Qmedia": float(q[q > 0].mean()), "paths": {"input_data": csv},
           "cross_validation": {"enabled": True, "unit": "year", "min_train_years": 0, "skip_first_year": True}}
    data = load(cfg, f"v11_{st}_{version}")
    if quick:   # a few folds only
        years = sorted(set(data.date[365:, 0]))
        data.cross_validation.min_train_years = max(0, len(years) - 7)
    with quiet():
        _, folds = cross_validate(data, "DE", return_folds=True)
        per_year, summary, sims = check_yearly_statistics(folds, seed=1, return_simulations=True)
    # Nested correction: each year is corrected with the deviations of the other years only.
    rng = np.random.default_rng(11)
    rows = []
    for r in per_year.itertuples():
        others = per_year[(per_year.statistic == r.statistic) & (per_year.year != r.year)].deviation
        corrected = scenario.correct_statistic(sims[(r.year, r.statistic)], others, seed=int(r.year))
        rows.append({"river": RIVERS[st], "version": version, "year": r.year, "statistic": r.statistic,
                     "measured": r.measured, "median": r.median, "deviation": r.deviation, "pit": r.pit,
                     "pit_corrected": scenario.pit(corrected, r.measured, rng),
                     "threshold": r.threshold, "season": r.season_months})
    return rows


def _inside(pits, level):
    return float(np.mean(np.abs(np.asarray(pits) - 0.5) <= level / 2))


def _accepted(n, level):
    lo, hi = binom.interval(COVERAGE_CONFIDENCE, n, level)
    return lo / n, hi / n


def run(ctx) -> Result:
    from pyair2stream.scenario import YEARLY_STATISTICS as STATS
    res = Result(
        code="V11", title="Cross-validated check and correction of yearly statistics",
        question="Over many years not used for calibration, how often do the predicted ranges of yearly "
                 "statistics (highest daily mean, highest 7-day mean, days above a threshold) contain the "
                 "measured value? When they do not, why? And does the package's correction, estimated from "
                 "other held-out years only, make the stated ranges hold?",
        method="For each Swiss river, the calibration and validation files are joined into one record "
               "(Mentue 2002-2012, Dischmabach 2003-2012, Rhône 1984-2013), and the package's leave-one-year-"
               "out cross-validation (run_mode DE, every year but the first held out once, Qmedia fixed) is "
               "run for versions 5 and 8. Its check of yearly statistics (cross_validation."
               "check_yearly_statistics, the outputs cv_yearly_statistics*.csv) makes 1000 simulations of each "
               "held-out year from that fold's calibrated parameters plus AR(1) error with the sigma and rho "
               "of its training years, computes each statistic over the days that were measured, and records "
               "where the measured value falls among the simulations (the PIT; ties split at random). A year "
               "counts if at least 80% of June-September was measured; the threshold is the 90th percentile of "
               "the measured temperatures. Then each held-out year is corrected with scenario."
               "correct_statistic, using the deviations (measured minus predicted median) of the other held-"
               "out years of the same river and version only, and its PIT is recorded again. Parameter "
               "uncertainty is not included (one parameter set per fold), so the ranges are slightly narrower "
               "than a FORWARD run from a DE-MCMC chain gives.",
        criterion=f"After correction, for each version and statistic, pooled over the three rivers, the share "
                  f"of held-out years whose measured value lies inside the central 90% range is within the "
                  f"range expected by chance (central {COVERAGE_CONFIDENCE:.0%} binomial range). The uncorrected "
                  f"ranges and the 50% ranges are reported, not judged.")
    stations = ["MAH_2369"] if ctx.quick else list(RIVERS)
    versions = (8,) if ctx.quick else VERSIONS
    jobs = [(st, v, ctx.quick) for st in stations for v in versions]
    jobs.sort(key=lambda j: j[0] != "SIO_2011")         # the longest record first
    for st in stations:
        _full_record(st, write=True)
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            rows = [r for out in ex.map(_job, jobs) for r in out]
    res.seconds = t.seconds
    df = pd.DataFrame(rows)

    # Pooled over rivers: uncorrected and corrected coverage.
    pooled, ok = [], True
    for (v, name), g in df.groupby(["version", "statistic"], sort=True):
        n = len(g)
        row = {"version": v, "statistic": name, "years": n}
        for level in LEVELS:
            lo, hi = _accepted(n, level)
            row[f"inside {level:.0%} range"] = _inside(g.pit, level)
            row[f"inside {level:.0%} range, corrected"] = _inside(g.pit_corrected, level)
            row[f"accepted ({level:.0%})"] = f"{lo:.0%}-{hi:.0%}"
            if level == 0.9:
                ok &= lo <= row[f"inside {level:.0%} range, corrected"] <= hi
        row["mean PIT"] = round(float(g.pit.mean()), 2)
        row["mean PIT, corrected"] = round(float(g.pit_corrected.mean()), 2)
        pooled.append(row)
    pooled = pd.DataFrame(pooled)
    res.passed = bool(ok)

    # By river: the bias of the predicted median (deviation) with its 95% confidence interval.
    from scipy.stats import t as student_t
    by_river = []
    for (v, river, name), g in df.groupby(["version", "river", "statistic"], sort=True):
        d = g.deviation.to_numpy()
        half = student_t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        unit = "days" if name == "days above threshold" else "°C"
        by_river.append({"version": v, "river": river, "statistic": name, "years": len(d),
                         "measured minus predicted median, mean": f"{d.mean():+.2f} {unit}",
                         "95% CI": f"{d.mean() - half:+.2f} to {d.mean() + half:+.2f}",
                         "bias": "yes" if abs(d.mean()) > half else "no",
                         "inside 90% range": f"{int(np.sum(np.abs(g.pit - 0.5) <= 0.45))} of {len(d)}",
                         "inside 90% range, corrected": f"{int(np.sum(np.abs(g.pit_corrected - 0.5) <= 0.45))} of {len(d)}"})
    by_river = pd.DataFrame(by_river)

    v8 = pooled[pooled.version == 8] if (pooled.version == 8).any() else pooled
    res.summary = (
        f"Over {int(df.groupby(['river', 'version']).year.nunique().groupby('version').sum().max())} held-out "
        f"river-years per version, the uncorrected 90% ranges held in "
        f"{pooled['inside 90% range'].min():.0%}-{pooled['inside 90% range'].max():.0%} of years "
        f"(version 8: {v8['inside 90% range'].min():.0%}-{v8['inside 90% range'].max():.0%}); after the "
        f"cross-validated correction, in {pooled['inside 90% range, corrected'].min():.0%}-"
        f"{pooled['inside 90% range, corrected'].max():.0%} (version 8: "
        f"{v8['inside 90% range, corrected'].min():.0%}-{v8['inside 90% range, corrected'].max():.0%}), and the "
        f"mean PIT moved from {pooled['mean PIT'].min():.2f}-{pooled['mean PIT'].max():.2f} to "
        f"{pooled['mean PIT, corrected'].min():.2f}-{pooled['mean PIT, corrected'].max():.2f} (0.5: unbiased). "
        f"{int((by_river.bias == 'yes').sum())} of {len(by_river)} river, version and statistic combinations "
        f"had a bias whose 95% confidence interval excludes zero.")
    shown = pooled.copy()
    for c in [c for c in shown.columns if c.startswith("inside ")]:
        shown[c] = shown[c].map(lambda x: f"{x:.0%}")
    res.sections.append(Section(
        "Coverage of the predicted ranges, before and after the correction",
        "Each held-out year of each river gives one value per statistic. The uncorrected ranges are those a "
        "FORWARD run would give from the fold's calibration; the corrected ones shift each year's simulations by "
        "the mean deviation of the other held-out years, with its uncertainty.",
        figures=[_fig(pooled)],
        tables=[("Pooled over the three rivers", shown),
                ("By river: bias of the predicted median (measured minus median) and coverage", by_river)]))
    res.notes.append(
        "Why the uncorrected ranges can fail: the model's error on the hottest days is not its typical error. "
        "Where the simulated peak is systematically too high or too low (see the table by river), adding random "
        "error of the typical size cannot move the predicted range to the measured value. This is a property of "
        "the fitted model at a site, not of the calculation, which V9 part A shows is right when the model is "
        "right. The check finds it from the data available, and the correction removes the average bias, with "
        "its uncertainty, using years other than the one predicted.")
    res.notes.append(
        "The check needs years: with n held-out years the coverage it reports is uncertain by about "
        "±sqrt(0.09/n) (one standard deviation: ±0.13 for 5 years, ±0.07 for 20), and the correction is uncertain by its standard error, "
        "which the corrected range includes. Daily and 7-day ranges are not affected: V5 and V10 show they hold.")
    return res


def _fig(pooled):
    import matplotlib.pyplot as plt
    plot_style()
    versions = sorted(set(pooled.version))
    fig, axes = plt.subplots(1, len(versions), figsize=(4.6 * len(versions), 3.4), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], versions):
        g = pooled[pooled.version == v].reset_index(drop=True)
        x = np.arange(len(g))
        lo, hi = _accepted(int(g.years.iloc[0]), 0.9)
        ax.axhspan(lo, hi, color=LIGHT_GREY, alpha=0.35, lw=0, zorder=0)
        ax.axhline(0.9, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.bar(x - 0.18, g["inside 90% range"], 0.34, color=ORANGE, label="uncorrected", zorder=2)
        ax.bar(x + 0.18, g["inside 90% range, corrected"], 0.34, color=BLUE, label="corrected", zorder=2)
        ax.set_xticks(x, [s.replace("days above", "days above\n").replace("highest ", "highest\n")
                          for s in g.statistic], fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f"Version {v} ({int(g.years.iloc[0])} held-out river-years)", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Held-out years inside the 90% range")
    axes[0][0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return (save_figure(fig, "V11_coverage.png"),
            "Share of held-out years whose measured statistic lies inside the predicted 90% range, before "
            "(orange) and after (blue) the cross-validated correction. Dashed: 90%; shaded: the range expected "
            "by chance for this number of years.")
