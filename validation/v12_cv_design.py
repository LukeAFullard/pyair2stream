"""
V12 - Cross-validation design: what is hidden, and from which years.

The package's leave-one-year-out cross-validation hides a held-out year's measured water
temperatures, the quantity predicted, and keeps its air temperature and discharge, the model's
inputs: the year is then predicted from its own inputs, as in any use of the model and as in
the textbook definition of cross-validation. This check measures whether three alternatives
would change what the cross-validation reports: hiding the inputs too, leaving a buffer of
unused days around the year, or calibrating on earlier years only (docs/METHODS.md §11).
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DE_SETTINGS, RIVERS, Result, Section, Timer, load, quiet, plot_style,
                    reference_line, save_figure, BLUE, ORANGE, AQUA, INK2, LIGHT_GREY)

VERSIONS = (5, 8)
BUFFER_DAYS = 60            # variant D: days hidden on each side of the held-out year
MIN_PAST_YEARS = 4          # variant C: years before the first year predicted
TOLERANCE_RMSE = 0.05       # degC: change in a year's RMSE accepted as "no change"
TOLERANCE_COVERAGE = 0.01   # change in the share of days inside the 90% interval
TOLERANCE_DEVIATION = {"highest daily mean": 0.05, "highest 7-day mean": 0.05, "days above threshold": 0.5}
VARIANTS = {
    "A": "default: the year's water temperatures hidden",
    "A2": "default, another optimizer seed",
    "B0": "gap-tolerant mode, nothing else hidden",
    "B": "inputs also hidden during calibration",
    "D": f"{BUFFER_DAYS}-day buffer around the year",
    "C": "forward only: calibrated on earlier years",
}


def _folds(data, variant):
    """(label, rows scored, rows whose water temperature is hidden, rows whose inputs are hidden)."""
    from pyair2stream.cross_validation import CVConfig, assign_year_groups, build_folds
    base = build_folds(data, CVConfig(unit="year", min_train_years=0, skip_first_year=True))
    years = sorted(int(y) for y in np.unique(assign_year_groups(data, 1)) if y != -999)
    out = []
    for label, idx in base:
        if variant in ("A", "A2", "B0"):
            out.append((label, idx, idx, None))
        elif variant == "B":
            out.append((label, idx, idx, idx))
        elif variant == "D":
            lo, hi = max(365, idx[0] - BUFFER_DAYS), min(data.n_tot - 1, idx[-1] + BUFFER_DAYS)
            out.append((label, idx, np.arange(lo, hi + 1), None))
        elif variant == "C" and years.index(int(label)) >= MIN_PAST_YEARS:
            out.append((label, idx, np.arange(idx[0], data.n_tot), None))
    return out


def _cross_validate(data, folds):
    """The package's fold loop (cross_validation.run_leave_one_year_out_cv), with the rows hidden
    from calibration and the rows scored given separately."""
    from pyair2stream.cross_validation import (FoldResult, MISSING_DATA_SENTINEL, _compute_fold_metrics,
                                               _fold_error_model, _run_optimizer, assign_year_groups)
    from pyair2stream.io import compute_doy_climatology, compute_qmedia
    from pyair2stream.model import aggregation, call_model, detect_segments, statis
    results = []
    for label, score_idx, twat_idx, input_idx in folds:
        twat0, tair0, q0 = data.Twat_obs.copy(), data.Tair.copy(), data.Q.copy()
        try:
            data.Twat_obs[twat_idx] = MISSING_DATA_SENTINEL
            if input_idx is not None:
                data.Tair[input_idx] = MISSING_DATA_SENTINEL
                data.Q[input_idx] = MISSING_DATA_SENTINEL
            compute_qmedia(data)                      # Qmedia is fixed in the config
            if data.gap_tolerant:
                compute_doy_climatology(data)
                data.segments = None
                detect_segments(data)
            aggregation(data)
            statis(data)
            _run_optimizer(data, "DE", None)
            data.par[:] = data.par_best[:]
            data.Tair[:], data.Q[:] = tair0, q0       # inputs restored to predict the year
            if data.gap_tolerant:
                data.segments = None
                detect_segments(data)
            call_model(data)
            sigma, rho = _fold_error_model(data)
            scored = data.eval_mask[score_idx] if data.eval_mask is not None else True
            obs = np.where(scored, twat0[score_idx], MISSING_DATA_SENTINEL)
            sim = data.Twat_mod[score_idx].copy()
            nse, kge, rmse = _compute_fold_metrics(obs, sim, MISSING_DATA_SENTINEL)
            d = data.date[score_idx]
            results.append(FoldResult(
                fold_id=len(results), label=label,
                held_out_start=pd.Timestamp(*d[0]), held_out_end=pd.Timestamp(*d[-1]),
                n_obs_held_out=int(np.sum(obs != MISSING_DATA_SENTINEL)), par_best=data.par_best.copy(),
                nse=nse, kge=kge, rmse=rmse, obs_held_out=obs, sim_held_out=sim,
                dates_held_out=pd.DatetimeIndex(pd.to_datetime({"year": d[:, 0], "month": d[:, 1], "day": d[:, 2]})),
                sigma=sigma, rho=rho, years_held_out=assign_year_groups(data, 1)[score_idx]))
        finally:
            data.Twat_obs[:], data.Tair[:], data.Q[:] = twat0, tair0, q0
    return results


def _job(args):
    st, version, variant, quick = args
    from v11_yearly_check import _full_record
    from pyair2stream.cross_validation import check_interval_coverage, check_yearly_statistics, warmest_months
    csv = _full_record(st)
    full = pd.read_csv(csv)
    q = full.Discharge
    settings = {"n_run": 40, "n_particles": 8} if quick else dict(DE_SETTINGS)
    cfg = {"version": version, "integrator": "CRN", "run_mode": "DE", "objective_function": "NSE",
           "random_seed": 2 if variant == "A2" else 1, "optimization": settings,
           "parameter_bounds": AUTHORS_BOUNDS, "Qmedia": float(q[q > 0].mean()),
           "paths": {"input_data": csv}, "gap_tolerant": variant in ("B0", "B")}
    data = load(cfg, f"v12_{st}_{version}_{variant}")
    folds = _folds(data, variant)
    if quick:
        folds = folds[-3:]
    # The same threshold and season for every variant: from the whole measured record.
    tw = full.T_water.replace(-999.0, np.nan)
    threshold = float(np.nanquantile(tw, 0.9))
    season = warmest_months(pd.to_datetime(full.Date), tw.values)
    with quiet():
        res = _cross_validate(data, folds)
        per_year, _ = check_yearly_statistics(res, threshold=threshold, season_months=season, seed=1)
        coverage = [check_interval_coverage([r], levels=[90.0], seed=1).iloc[0] for r in res]
    tag = {"river": RIVERS[st], "version": version, "variant": variant}
    years = []
    for r, c in zip(res, coverage):
        o = np.where(r.obs_held_out == -999.0, np.nan, r.obs_held_out)
        years.append({**tag, "year": int(r.label), "rmse": r.rmse, "bias": float(np.nanmean(r.sim_held_out - o)),
                      "n": int(np.isfinite(o).sum()), "sigma": r.sigma, "rho": r.rho,
                      "daily inside": c["daily inside"], "days": c["days"],
                      "7-day inside": c["7-day inside"], "7-day means": c["7-day means"],
                      **{f"p{i + 1}": v for i, v in enumerate(r.par_best)}})
    stats = per_year[["year", "statistic", "measured", "median", "deviation", "inside"]].assign(**tag)
    return years, stats.to_dict("records")


def _pooled(g):
    return pd.Series({"RMSE": float(np.sqrt(np.sum(g.rmse ** 2 * g.n) / g.n.sum())),
                      "days inside 90%": float(np.sum(g["daily inside"] * g["days"]) / g["days"].sum()),
                      "7-day means inside 90%": float(np.sum(g["7-day inside"] * g["7-day means"]) / g["7-day means"].sum())})


def _jackknife_se(g, pcols):
    n = len(g) + 1                       # blocks in the record: the first year is never held out
    p = g[pcols].to_numpy()
    return np.sqrt((n - 1) / len(g) * np.sum((p - p.mean(axis=0)) ** 2, axis=0))


def run(ctx) -> Result:
    from v11_yearly_check import _full_record
    from pyair2stream.scenario import YEARLY_STATISTICS as STATS
    res = Result(
        code="V12", title="Cross-validation design: what is hidden, and from which years",
        question="Cross-validation hides a held-out year's measured water temperatures but keeps its air "
                 "temperature and discharge, the model's inputs, from which the year is predicted. Would it "
                 "report anything different if the inputs were hidden from the calibration too, if a buffer of "
                 "days around the year were also left out, or if only earlier years were used for calibration?",
        method="For each Swiss river (calibration and validation files joined: Mentue 2002-2012, Dischmabach "
               "2003-2012, Rhône 1984-2013) and versions 5 and 8, leave-one-year-out cross-validation is run "
               "with the settings of V11 (DE, authors' bounds, Qmedia fixed) in six variants. (A) The package "
               "default: the year's water temperatures are hidden; the model runs through the year with its "
               "inputs during calibration. (A2) As A with another optimizer seed: the change chance alone "
               "makes. (B0) Gap-tolerant mode with nothing else hidden. (B) Gap-tolerant mode with the year's "
               "air temperature and discharge also hidden during calibration: the record stops before the year "
               "and restarts after it, as if the year did not exist. "
               f"(D) As A, plus the water temperatures of the {BUFFER_DAYS} days on each side of the year "
               f"(h-block cross-validation). (C) Forward only: the year and every later year hidden, so the "
               f"model is calibrated on earlier years only (at least {MIN_PAST_YEARS}). In every variant the "
               "year is then predicted from its own inputs, and scored on the days measured: RMSE, the share of "
               "days and 7-day means inside the 90% interval (from the fold's sigma and rho), and the three "
               "yearly statistics of V11 (threshold: 90th percentile of the river's measured temperatures; "
               "season: its four warmest months).",
        criterion=f"Hiding the inputs (B) and the buffer (D) each change no held-out year's RMSE by more than "
                  f"{TOLERANCE_RMSE} °C (below the accuracy of most temperature loggers), the pooled share of "
                  f"days inside the 90% interval by no more than {TOLERANCE_COVERAGE:.0%} (one percentage "
                  f"point), and the mean error of each yearly statistic by no more than 0.05 °C (0.5 days for "
                  f"days above the threshold). B0 gives the same results as A. Forward-only (C) answers a "
                  f"different question (forecasting) and is reported, not judged.")
    stations = ["MAH_2369"] if ctx.quick else list(RIVERS)
    versions = (8,) if ctx.quick else VERSIONS
    jobs = [(st, v, var, ctx.quick) for st in stations for v in versions for var in VARIANTS]
    jobs.sort(key=lambda j: j[0] != "SIO_2011")         # the longest record first
    for st in stations:
        _full_record(st, write=True)
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            outs = list(ex.map(_job, jobs))
    res.seconds = t.seconds
    years = pd.DataFrame([r for y, _ in outs for r in y])
    stats = pd.DataFrame([r for _, s in outs for r in s])
    key = ["river", "version", "year"]
    pcols = [c for c in years.columns if c.startswith("p") and c[1:].isdigit()]
    a = years[years.variant == "A"].set_index(key)
    sa = stats[stats.variant == "A"].set_index(key + ["statistic"])

    # Each variant against A on the same years.
    rows, ok = [], True
    for v, label in VARIANTS.items():
        b = years[years.variant == v].set_index(key)
        common = a.index.intersection(b.index)
        d_rmse = (b.loc[common, "rmse"] - a.loc[common, "rmse"]).abs()
        pa, pb = _pooled(a.loc[common]), _pooled(b.loc[common])
        sb = stats[stats.variant == v].set_index(key + ["statistic"])
        sc = sa.index.intersection(sb.index)
        row = {"variant": f"{v}: {label}", "years": len(common), "RMSE (°C)": round(pb.RMSE, 3),
               "RMSE of A, same years": round(pa.RMSE, 3), "largest change in a year's RMSE": round(d_rmse.max(), 3),
               "days inside 90%": f"{pb['days inside 90%']:.1%}", "A, same years": f"{pa['days inside 90%']:.1%}",
               "7-day means inside 90%": f"{pb['7-day means inside 90%']:.1%}"}
        for name in STATS:
            g_b = sb.loc[sc].xs(name, level="statistic")
            g_a = sa.loc[sc].xs(name, level="statistic")
            unit = "days" if name == "days above threshold" else "°C"
            row[f"{name}: inside 90% range"] = f"{g_b.inside.mean():.0%} (A {g_a.inside.mean():.0%})"
            row[f"{name}: mean error"] = (f"{g_b.deviation.mean():+.2f} (A {g_a.deviation.mean():+.2f}) {unit}")
            if v in ("B", "D") and abs(g_b.deviation.mean() - g_a.deviation.mean()) > TOLERANCE_DEVIATION[name]:
                ok = False
        if v in ("B", "D"):
            ok &= bool(d_rmse.max() <= TOLERANCE_RMSE)
            ok &= bool(abs(pb["days inside 90%"] - pa["days inside 90%"]) <= TOLERANCE_COVERAGE)
        if v == "B0":
            ok &= bool(np.allclose(b.loc[common, "rmse"], a.loc[common, "rmse"]))
        rows.append(row)
    table = pd.DataFrame(rows)
    res.passed = bool(ok)

    # By river and version: RMSE and daily coverage, A against B and against C (C on its own years).
    by_river = []
    for (river, version), g in years.groupby(["river", "version"], sort=True):
        ga = g[g.variant == "A"].set_index("year")
        gc = g[g.variant == "C"].set_index("year")
        gb = g[g.variant == "B"].set_index("year")
        pa, pb = _pooled(ga), _pooled(gb)
        pac, pc = _pooled(ga.loc[gc.index]), _pooled(gc)
        by_river.append({"river": river, "version": version, "years": len(ga),
                         "RMSE A": round(pa.RMSE, 3), "RMSE B (inputs hidden)": round(pb.RMSE, 3),
                         "years forward": len(gc), "RMSE A, same years": round(pac.RMSE, 3),
                         "RMSE C (forward)": round(pc.RMSE, 3),
                         "days inside 90%: A, same years": f"{pac['days inside 90%']:.1%}",
                         "days inside 90%: C": f"{pc['days inside 90%']:.1%}",
                         "sigma: A, same years": round(float(ga.loc[gc.index, "sigma"].mean()), 3),
                         "sigma: C": round(float(gc.sigma.mean()), 3)})
    by_river = pd.DataFrame(by_river)

    # Parameters: change per fold in units of A's jackknife standard error, and the jackknife SE itself.
    par_rows = []
    for (river, version), g in years.groupby(["river", "version"], sort=True):
        ga = g[g.variant == "A"].set_index("year")
        se_a = _jackknife_se(ga, pcols)
        se_a[se_a == 0] = np.nan
        row = {"river": river, "version": version, "folds": len(ga)}
        for v in ("A2", "B"):
            gv = g[g.variant == v].set_index("year").loc[ga.index]
            z = np.nanmax(np.abs((gv[pcols].to_numpy() - ga[pcols].to_numpy()) / se_a), axis=1)
            ratio = _jackknife_se(gv, pcols) / se_a
            row[f"{v}: folds with a parameter moved > 2 SE"] = int(np.sum(z > 2))
            row[f"{v}: jackknife SE / A's, median (range)"] = (f"{np.nanmedian(ratio):.2f} "
                                                              f"({np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f})")
        par_rows.append(row)
    par_table = pd.DataFrame(par_rows)

    b_row, d_row, c_row = (table.iloc[list(VARIANTS).index(v)] for v in ("B", "D", "C"))
    largest = "largest change in a year's RMSE"
    seed_change = table.iloc[list(VARIANTS).index("A2")][largest]
    # Forward only against A on the same years: daily coverage and the training-year sigma.
    a_same = years[years.variant == "A"].set_index(key)
    c_years = years[years.variant == "C"].set_index(key)
    a_same = a_same.loc[a_same.index.intersection(c_years.index)]
    cov_a, cov_c = _pooled(a_same)["days inside 90%"], _pooled(c_years)["days inside 90%"]
    sig_a, sig_c = float(a_same.sigma.mean()), float(c_years.sigma.mean())
    res.summary = (
        f"Over {int(b_row.years)} held-out river-years, hiding the inputs during calibration changed no year's "
        f"RMSE by more than {b_row[largest]:.3f} °C (another optimizer seed alone: {seed_change:.3f} °C), and "
        f"the share of days inside the 90% interval from {b_row['A, same years']} to {b_row['days inside 90%']}; "
        f"a {BUFFER_DAYS}-day buffer changed no year's RMSE by more than {d_row[largest]:.3f} °C. Calibrating on "
        f"earlier years only gave, on its {int(c_row.years)} years, an RMSE of {c_row['RMSE (°C)']:.3f} °C "
        f"against {c_row['RMSE of A, same years']:.3f} °C, 90% intervals that held on {cov_c:.1%} of days "
        f"against {cov_a:.1%}, and a training-year sigma of {sig_c:.2f} °C against {sig_a:.2f} °C.")
    res.sections.append(Section(
        "Each variant against the default",
        "Every variant is compared with A on the same held-out years. A2 shows how much the results move by "
        "chance (another optimizer seed). For the yearly statistics, 'mean error' is measured minus predicted "
        "median, averaged over years.",
        figures=[_fig(years)],
        tables=[("Pooled over rivers and versions", table),
                ("By river and version", by_river)]))
    res.sections.append(Section(
        "Parameters and their jackknife intervals",
        "Change of each fold's parameters in units of A's jackknife standard error, and the jackknife standard "
        "error of each variant relative to A's (median, and range over parameters).",
        tables=[("By river and version", par_table)]))
    if res.passed:
        res.notes.append(
            "Hiding the inputs changed no result by more than the tolerance, as expected: the model never uses "
            "measured water temperature, only air temperature and discharge, so hiding a year's water "
            "temperatures removes all of that year's information about the answer. The inputs are needed to "
            "predict the year, in cross-validation as in any use of the model; during calibration they only "
            "shape the first days to weeks of the following year. Nor did the buffer: whole years are long "
            "compared with the time the model's errors persist.")
    else:
        res.notes.append(
            "Hiding the inputs or the buffer changed the results by more than the tolerance (see the tables): "
            "the cross-validation then depends on its design, and its results should be reported with both.")
    if cov_c < cov_a:
        res.notes.append(
            f"Calibrating on earlier years only is the test for forecasting into a changing future. Here its 90% "
            f"intervals held {100 * (cov_a - cov_c):.1f} percentage points less often than the default's on the "
            f"same years" + (f", because the error size estimated from fewer, earlier years ({sig_c:.2f} °C) was "
                             f"smaller than that from all other years ({sig_a:.2f} °C)" if sig_c < sig_a else "")
            + ". V5 (the later years of each record) tests the same situation with DE-MCMC. For predictions of "
            "future years, quote V5 and V10 as well as the cross-validation.")
    moved = int(sum(r["A2: folds with a parameter moved > 2 SE"] for r in par_rows))
    if moved:
        res.notes.append(
            f"With another optimizer seed alone, {moved} folds ended on parameters more than 2 jackknife standard "
            "errors away, with almost the same fit: parameters that trade off (equifinality) make single folds "
            "unstable, and the jackknife standard errors inherit this (see the table). Read the jackknife "
            "intervals of poorly determined parameters as indicative, and check them with a second seed.")
    return res


def _fig(years):
    """Left: change in each held-out year's RMSE against A. Right: days inside the 90% interval, A and C."""
    import matplotlib.pyplot as plt
    plot_style()
    key = ["river", "version", "year"]
    a = years[years.variant == "A"].set_index(key)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    show = [("A2", "another seed"), ("B", "inputs hidden"), ("D", f"{BUFFER_DAYS}-day buffer"),
            ("C", "forward only")]
    rng = np.random.default_rng(0)
    for i, (v, label) in enumerate(show):
        b = years[years.variant == v].set_index(key)
        common = a.index.intersection(b.index)
        d = (b.loc[common, "rmse"] - a.loc[common, "rmse"]).to_numpy()
        ax1.scatter(d, i + rng.uniform(-0.18, 0.18, len(d)), s=10, color=BLUE, alpha=0.7, lw=0)
    ax1.axvspan(-TOLERANCE_RMSE, TOLERANCE_RMSE, color=LIGHT_GREY, alpha=0.3, lw=0, zorder=0)
    reference_line(ax1, 0, axis="x")
    ax1.set_yticks(range(len(show)), [s[1] for s in show])
    ax1.invert_yaxis()
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("Change in the held-out year's RMSE against the default (°C)")
    ax1.set_title("Each held-out year", fontsize=9)

    groups = []
    for (river, version), g in years.groupby(["river", "version"], sort=True):
        ga = g[g.variant == "A"].set_index("year")
        gc = g[g.variant == "C"].set_index("year")
        if len(gc):
            groups.append((f"{river}\nv{version}", _pooled(ga.loc[gc.index])["days inside 90%"],
                           _pooled(gc)["days inside 90%"]))
    x = np.arange(len(groups))
    ax2.bar(x - 0.18, [g[1] * 100 for g in groups], 0.34, color=BLUE, label="default, same years")
    ax2.bar(x + 0.18, [g[2] * 100 for g in groups], 0.34, color=ORANGE, label="forward only")
    reference_line(ax2, 90, "90%")
    ax2.set_xticks(x, [g[0] for g in groups], fontsize=7.5)
    ax2.set_ylim(80, 95)
    ax2.set_ylabel("Days inside the 90% interval (%)")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="lower left", fontsize=7.5)
    ax2.set_title("Calibrating on earlier years only", fontsize=9)
    fig.tight_layout()
    return (save_figure(fig, "V12_variants.png"),
            "Left: change in each held-out year's RMSE when the optimizer seed changes, when the year's inputs "
            f"are hidden during calibration, with a {BUFFER_DAYS}-day buffer, and when only earlier years are "
            f"used (shaded: ±{TOLERANCE_RMSE} °C). Right: share of held-out days inside the 90% interval, default "
            "against forward only, on the same years.")
