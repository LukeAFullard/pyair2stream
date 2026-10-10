"""
V18 - Prediction intervals on independent rivers (British Columbia).

The error model's persistence rule (rho from week-to-week persistence, the default) was chosen and
first tested on the three Swiss rivers (V4, V5, V9, V11). This check tests it, and the whole chain of
prediction intervals, on 23 rivers it was not developed on: the British Columbia streams of Callahan and
Moore (2025; data/british_columbia/), in (A) the later years 2021-2022, never used for calibration, and
(B) every year of the calibration period held out in turn by cross-validation. Each part is run with the
default rho and with rho from consecutive days, so the rule is compared with the alternative it replaced.
The pass criterion was fixed in this file before the check was first run.
"""

import dataclasses
import glob
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import binom

from common import (LEVELS, WORK, Result, Section, Timer, accepted_real, coverage_by_level, load, quiet,
                    plot_style, save_figure, BLUE, ORANGE, INK2, LIGHT_GREY)
from v15_british_columbia import (HEAT_DOME, MCMC_STEPS, MCMC_WALKERS, QUICK_STATIONS, _calibration_config, _load,
                                  _write)

RHO = ("weekly", "daily")                       # uncertainty_options.rho_timescale; "weekly" is the default
WINDOWS = {"days": 1, "7-day means": 7, "30-day means": 30}
JUDGED = (50, 80, 90, 95)                       # levels judged (99%: reported; V5, V11 show it is too narrow)
MULTI_DAY_RANGE = (0.85, 0.95)                  # accepted share of 7- and 30-day means inside the 90% interval
YEARLY_JUDGED = (0.8, 0.9, 0.95)                # yearly statistics, corrected, as V11
BINOMIAL_CONFIDENCE = 0.99                      # range of coverage expected by chance, as V11
SUMMER = (6, 7, 8)
CHECK_SIMULATIONS = 1000
CACHE = os.path.join(WORK, "v18_cache")


def _code_fingerprint():
    """A hash of the package's and the validation suite's code. Saved results are reused only by
    exactly the same code."""
    from common import REPO
    h = hashlib.sha256()
    for pattern in (os.path.join(REPO, "pyair2stream", "**", "*.py"), os.path.join(REPO, "validation", "*.py")):
        for name in sorted(glob.glob(pattern, recursive=True)):
            h.update(os.path.relpath(name, REPO).encode())
            with open(name, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def _resumable(job):
    """Run `(function, args)`, or reuse its result saved by an earlier, interrupted run of the same code.
    Returns (result, reused)."""
    fn, args = job
    key = "_".join(str(a) for a in args)
    path = os.path.join(CACHE, f"{fn.__name__}_{key}_{_code_fingerprint()}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), True
    out = fn(args)
    os.makedirs(CACHE, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(path + ".tmp", path)
    return out, False


def _rolling(x, window):
    """Moving means of `window` days along the last axis (NaN unless every day is present)."""
    if window == 1:
        return np.asarray(x, float)
    x = np.atleast_2d(np.asarray(x, float))
    out = pd.DataFrame(x.T).rolling(window).mean().to_numpy().T
    return out


def _counts(ens, obs, levels=LEVELS):
    """{level: (inside, n)} for the finite values of `obs` and the central ranges of `ens`."""
    ok = np.isfinite(obs) & np.all(np.isfinite(ens), axis=0)
    share = coverage_by_level(ens[:, ok], obs[ok], levels) if ok.any() else {lev: np.nan for lev in levels}
    n = int(ok.sum())
    return {lev: (0 if n == 0 else int(round(share[lev] * n)), n) for lev in levels}


def _error_diagnostics(err, sigma, rho):
    """How the errors of a period compare with the error model (sigma, rho) it was predicted with: their
    RMSE, and their mean in units of the standard deviation the error model gives a mean of that many
    days (z; about 1 in size if the error model describes how errors persist over the period)."""
    from pyair2stream.uncertainty import scored_error_variance_factor
    err = np.asarray(err, float)
    pos = np.flatnonzero(np.isfinite(err))
    if len(pos) < 2:
        return None
    e = err[pos]
    sd_mean = sigma * np.sqrt(scored_error_variance_factor(pos, rho) / len(pos))
    return {"n": len(pos), "sigma": float(sigma), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mean": float(np.mean(e)), "sd_mean": float(sd_mean), "z": float(np.mean(e) / sd_mean)}


# --- (A) Later years: DE-MCMC on the calibration years, FORWARD on 2021-2022 -------------------

def _later_years(args):
    st, rho, quick = args
    from pyair2stream import scenario
    from pyair2stream.optimization import DE_MCMC_mode, forward_mode
    tag = f"v18_{st}_{rho}"
    cfg = _calibration_config(st, tag, quick, run_mode="DE-MCMC",
                              uncertainty_options={"noise_model": "ar1", "likelihood": "least_squares",
                                                   "rho_timescale": rho, "strict_convergence": False})
    cfg["optimization"].update({"mcmc_walkers": 16 if quick else MCMC_WALKERS,
                                "mcmc_steps": 600 if quick else MCMC_STEPS})
    data = load(cfg, tag)
    with quiet():
        DE_MCMC_mode(data, seed=1)
    out = os.path.join(WORK, tag, "out")
    meta = json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))
    ts, _, _ = _load()
    val = ts[(ts.station_number == st) & (ts.period == "validation")].sort_values("date").reset_index(drop=True)
    path_v, val = _write(st, "validation", val, tag=f"18_{rho}")
    fcfg = {"version": 8, "integrator": "CRN", "run_mode": "FORWARD", "objective_function": "RMS",
            "Qmedia": cfg["Qmedia"], "parameters_forward": [float(x) for x in data.par_best],
            "uncertainty_options": {"noise_model": "ar1", "save_ensemble": True},
            "forward_options": {"enable_prediction_intervals": True,
                                "mcmc_chain_path": os.path.join(out, "MCMC_chain_S_c_1d.csv"),
                                "n_samples": 100 if quick else 1000, "random_seed": 1},
            "paths": {"input_data": path_v, "output_dir": os.path.join(WORK, tag, "fwd")}}
    fdata = load(fcfg, tag + "_fwd")
    with quiet():
        forward_mode(fdata)
    ens, dates = scenario.load_ensemble(os.path.join(WORK, tag, "fwd", "Forward_Prediction_Ensemble_S_c_1d.npz"))
    obs = val.tw_obs.to_numpy(dtype=float)
    if ens.shape[1] != len(obs):
        raise RuntimeError(f"{st}: ensemble of {ens.shape[1]} days for {len(obs)} validation days")
    dates = pd.DatetimeIndex(dates)
    summer = dates.month.isin(SUMMER)
    heat = (dates >= pd.Timestamp(HEAT_DOME[0])) & (dates <= pd.Timestamp(HEAT_DOME[1]))
    counts = {what: _counts(_rolling(ens, w), _rolling(obs, w)[0] if w > 1 else obs)
              for what, w in WINDOWS.items()}
    counts["summer days"] = _counts(ens[:, summer], obs[summer])
    counts["heat-dome days"] = _counts(ens[:, heat], obs[heat])
    ok = np.isfinite(obs)
    best = np.asarray(fdata.Twat_mod[365:], float)
    diag = _error_diagnostics(np.where(ok, best - obs, np.nan), float(meta["sigma"]), float(meta["rho"]))
    lo, hi = np.percentile(ens[:, ok], [5, 95], axis=0)
    median = np.median(ens[:, ok], axis=0)
    return {"station": st, "rho_timescale": rho, "converged": bool(meta["converged"]),
            "steps": int(meta["steps_run"]), "rho": float(meta["rho"]), "sigma": float(meta["sigma"]),
            "counts": counts, "diagnostics": diag, "width": float(np.mean(hi - lo)), "centre bias": float(np.mean(median - obs[ok]))}


# --- (B) Held-out years of the calibration period: leave-one-year-out cross-validation ----------

def _cv_config(st, quick):
    tag = f"v18_cv_{st}"
    cfg = _calibration_config(st, tag, quick, run_mode="DE", uncertainty_options={"noise_model": "ar1"},
                              cross_validation={"enabled": True, "unit": "year", "min_train_years": 0,
                                                "skip_first_year": True})
    if quick:       # the last three years only
        n_years = len(set(pd.read_csv(cfg["paths"]["input_data"]).Date.str[:4]))
        cfg["cross_validation"]["min_train_years"] = max(0, n_years - 4)
    return cfg, tag


def _fold_rhos(cfg, tag, folds):
    """Each fold's sigma and rho from its training years, with rho at each time scale, computed as the
    package's cross-validation computes them (cross_validation._fold_error_model on the record with the
    fold's measurements hidden). The weekly values must reproduce those the cross-validation stored."""
    from pyair2stream.cross_validation import _fold_error_model, _mask_fold, _restore_fold, build_folds
    from pyair2stream.model import aggregation, call_model, statis
    data = load(cfg, tag + "_rho")
    built = build_folds(data, data.cross_validation)
    if [label for label, _ in built] != [f.label for f in folds]:
        raise RuntimeError("folds differ from those of the cross-validation")
    out = {rho: [] for rho in RHO}
    for (label, idx), f in zip(built, folds):
        saved = _mask_fold(data, idx)
        try:
            aggregation(data)
            statis(data)
            data.par[:] = f.par_best[:]
            call_model(data)
            for rho in RHO:
                data.uncertainty_options = {**(data.uncertainty_options or {}), "rho_timescale": rho}
                sigma, r = _fold_error_model(data)
                if rho == "weekly" and not (np.isclose(sigma, f.sigma, rtol=1e-12, atol=1e-12)
                                            and np.isclose(r, f.rho, rtol=1e-12, atol=1e-12)):
                    raise RuntimeError(f"fold {label}: recomputed sigma/rho {sigma}/{r} differ from the "
                                       f"cross-validation's {f.sigma}/{f.rho}")
                out[rho].append(dataclasses.replace(f, sigma=sigma, rho=r))
        finally:
            _restore_fold(data, idx, *saved)
    return out


def _window_coverage(folds, window, levels=LEVELS, seed=1):
    """As cross_validation.check_interval_coverage, for moving means of `window` days: {level: (inside, n)}."""
    from pyair2stream.cross_validation import _fold_ensemble
    totals = {lev: [0, 0] for lev in levels}
    for r in folds:
        rng = np.random.default_rng([int(seed), int(r.fold_id), 7])
        obs, ens = _fold_ensemble(r, "ar1", CHECK_SIMULATIONS, rng)
        for lev, (k, n) in _counts(_rolling(ens, window), _rolling(obs, window)[0] if window > 1 else obs,
                                   levels).items():
            totals[lev][0] += k
            totals[lev][1] += n
    return {lev: tuple(v) for lev, v in totals.items()}


def _held_out_years(args):
    st, quick = args
    from pyair2stream import scenario
    from pyair2stream.cross_validation import check_yearly_statistics, cross_validate
    cfg, tag = _cv_config(st, quick)
    data = load(cfg, tag)
    with quiet():
        _, folds = cross_validate(data, "DE", return_folds=True)
        by_rho = _fold_rhos(cfg, tag, folds)
    out = {"station": st, "folds": len(folds), "counts": {}, "years": [], "rho": {},
           "folds raised": int(sum(w.rho > d.rho for w, d in zip(by_rho["weekly"], by_rho["daily"])))}
    out["diagnostics"] = []
    for f in by_rho["weekly"]:
        obs = np.where(f.obs_held_out == -999.0, np.nan, f.obs_held_out)
        sim = np.where(f.sim_held_out == -999.0, np.nan, f.sim_held_out)
        d = _error_diagnostics(sim - obs, f.sigma, f.rho)
        if d is not None:
            out["diagnostics"].append({"fold": f.label, **d})
    for rho, fl in by_rho.items():
        out["rho"][rho] = float(np.mean([f.rho for f in fl]))
        out["counts"][rho] = {what: _window_coverage(fl, w) for what, w in WINDOWS.items()}
        with quiet():
            per_year, _, sims = check_yearly_statistics(fl, seed=1, return_simulations=True)
        rng = np.random.default_rng(18)
        for r in per_year.itertuples():
            others = per_year[(per_year.statistic == r.statistic) & (per_year.year != r.year)].deviation
            if len(others) < 3:       # correct_statistic needs three other years
                continue
            corrected = scenario.correct_statistic(sims[(r.year, r.statistic)], others, seed=int(r.year))
            out["years"].append({"station": st, "rho_timescale": rho, "year": r.year, "statistic": r.statistic,
                                 "pit": r.pit, "pit_corrected": scenario.pit(corrected, r.measured, rng)})
    return out


# --- Report -----------------------------------------------------------------------------------

def _pool(rows, key):
    """Pooled share inside, per level, from a list of {level: (inside, n)}."""
    k = {lev: sum(r[key][lev][0] for r in rows) for lev in LEVELS}
    n = {lev: sum(r[key][lev][1] for r in rows) for lev in LEVELS}
    return {lev: (k[lev] / n[lev] if n[lev] else np.nan) for lev in LEVELS}, n[LEVELS[0]]


def _inside(pits, level):
    return float(np.mean(np.abs(np.asarray(pits, float) - 0.5) <= level / 2))


def run(ctx) -> Result:
    res = Result(
        code="V18", title="Prediction intervals on independent rivers (British Columbia)",
        question="The rule that sets the error model's persistence (rho from week-to-week persistence, the "
                 "default) was chosen and first tested on the three Swiss rivers. On 23 rivers it was not "
                 "developed on, do the prediction intervals hold at their stated levels, for days, 7-day "
                 "means and 30-day means, in later years never used for calibration and in years held out by "
                 "cross-validation? Do the corrected ranges of yearly statistics hold? And does the default "
                 "rho do better over several weeks than rho from consecutive days?",
        method="The 23 British Columbia streams of Callahan and Moore (2025; data/british_columbia/), version 8 "
               "(the version they published), each station's calibration years from the first 1 January, the "
               "RMS objective, Crank-Nicolson, the authors' parameter ranges and the calibration Qmedia, as in "
               "V15. (A) Later years: DE-MCMC with the default least-squares likelihood "
               f"({MCMC_WALKERS} walkers, run until converged, at most {MCMC_STEPS} steps, as V15) on the "
               "calibration years, then a FORWARD run of 1000 series from the chain for 1 January 2021 to 31 "
               "October 2022 (including the June 2021 heat dome), with the chain's sigma and rho. The share of "
               "measured days inside the central 50, 80, 90, 95 and 99% ranges is recorded, and the same for "
               "7-day and 30-day moving means (computed in each series; a mean counts if all its days were "
               "measured), for summer (June-August) and for the heat-dome days. (B) Held-out years: the "
               "package's leave-one-year-out cross-validation (run_mode DE, every year but the first held out "
               "once). Each held-out year gets 1000 simulations from its fold's parameters plus AR(1) error "
               "with the sigma and rho of the fold's training years (cross_validation.check_interval_coverage "
               "and check_yearly_statistics; parameter uncertainty not included), scored as in (A); the "
               "yearly statistics (highest daily mean, highest 7-day mean, days above the 90th percentile, in "
               "the four warmest months) are corrected with scenario.correct_statistic from the other held-"
               "out years of the same station only, as in V11. Both parts are run with rho_timescale "
               "'weekly' (the default) and 'daily' (rho from consecutive days). In part B the folds' parameters "
               "are the same for both; only each fold's rho differs, recomputed from its training years with "
               "the package's own function and checked to reproduce the cross-validation's weekly value "
               "exactly. Shares are pooled over stations, weighted by the number of values.",
        criterion=f"Fixed before the check was first run. With the default rho: (A) for the stations whose "
                  f"DE-MCMC converged, pooled, the share of 2021-2022 days inside the interval is within the "
                  f"accepted range at {', '.join(f'{x}%' for x in JUDGED)} (a miss rate between half and 1.5 "
                  f"times the stated one, as V5), and the share of 7-day and of 30-day means inside the 90% "
                  f"interval is between {MULTI_DAY_RANGE[0]:.0%} and {MULTI_DAY_RANGE[1]:.0%}. (B) Pooled over "
                  f"every held-out year, days, 7-day and 30-day means inside the "
                  f"{', '.join(f'{x}%' for x in JUDGED[1:])} intervals are within the accepted range, and for "
                  f"each yearly statistic the corrected share of years inside the central "
                  f"{', '.join(f'{x:.0%}' for x in YEARLY_JUDGED)} ranges is within the central "
                  f"{BINOMIAL_CONFIDENCE:.0%} binomial range (as V11). (C) For 30-day means, in both parts, "
                  f"the share inside the 90% interval is not further from 90% with the default rho than "
                  f"with rho from consecutive days (the default takes the larger of the two estimates, so where "
                  f"the day-to-day one is larger they are the same). The 99% intervals (shown too narrow in V5 and V11), single stations, "
                  f"unconverged stations, summer, the heat dome and the uncorrected yearly statistics are "
                  f"reported, not judged.")
    _, par_table, _ = _load()
    stations = list(QUICK_STATIONS) if ctx.quick else list(par_table.index)
    jobs_a = [(_later_years, (st, rho, ctx.quick)) for st in stations for rho in RHO]
    jobs_b = [(_held_out_years, (st, ctx.quick)) for st in stations]
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            fut_a = [ex.submit(_resumable, j) for j in jobs_a]       # the longest jobs first
            fut_b = [ex.submit(_resumable, j) for j in jobs_b]
            runs_a = [f.result() for f in fut_a]
            runs_b = [f.result() for f in fut_b]
    res.seconds = t.seconds
    reused = sum(was for _, was in runs_a + runs_b)
    if reused:
        res.notes.append(f"{reused} of {len(runs_a) + len(runs_b)} station jobs were reused from an earlier, "
                         "interrupted run of exactly the same code; the run time covers only this run.")
    a = [r for r, _ in runs_a]
    b = [r for r, _ in runs_b]
    converged = {st for st in stations if all(r["converged"] for r in a if r["station"] == st)}
    # Short quick-run chains never converge: the quick run pools every station, to exercise the code.
    pooled_stations = set(stations) if ctx.quick else converged

    # (A) pooled over converged stations, and by station
    pooled_a = []
    for rho in RHO:
        rows = [r for r in a if r["rho_timescale"] == rho and r["station"] in pooled_stations]
        for what in (*WINDOWS, "summer days", "heat-dome days"):
            if not rows:
                continue
            share, n = _pool([r["counts"] for r in rows], what)
            pooled_a.append({"rho": rho, "values": what, "n": n, **{f"{lev}%": share[lev] for lev in LEVELS}})
    pooled_a = pd.DataFrame(pooled_a)

    def _get(df, rho, what, lev):
        g = df[(df.rho == rho) & (df["values"] == what)]
        return float(g[f"{lev}%"].iloc[0]) if len(g) else np.nan

    fails = []
    for lev in JUDGED:
        x = _get(pooled_a, "weekly", "days", lev)
        lo, hi = accepted_real(lev)
        if not (lo <= x <= hi):
            fails.append(f"(A) days, {lev}% interval: {x:.1%} (accepted {lo:.1%}-{hi:.1%})")
    for what in ("7-day means", "30-day means"):
        x = _get(pooled_a, "weekly", what, 90)
        if not (MULTI_DAY_RANGE[0] <= x <= MULTI_DAY_RANGE[1]):
            fails.append(f"(A) {what}, 90% interval: {x:.1%}")

    # (B) pooled over every held-out year
    pooled_b = []
    for rho in RHO:
        for what in WINDOWS:
            share, n = _pool([r["counts"][rho] for r in b], what)
            pooled_b.append({"rho": rho, "values": what, "n": n, **{f"{lev}%": share[lev] for lev in LEVELS}})
    pooled_b = pd.DataFrame(pooled_b)
    for what in WINDOWS:
        for lev in JUDGED[1:]:
            x = _get(pooled_b, "weekly", what, lev)
            lo, hi = accepted_real(lev)
            if not (lo <= x <= hi):
                fails.append(f"(B) {what}, {lev}% interval: {x:.1%} (accepted {lo:.1%}-{hi:.1%})")
    years = pd.DataFrame([y for r in b for y in r["years"]])
    yearly = []
    for (rho, name), g in (years.groupby(["rho_timescale", "statistic"], sort=False) if len(years) else []):
        n = len(g)
        row = {"rho": rho, "statistic": name, "years": n}
        for level in YEARLY_JUDGED:
            lo, hi = binom.interval(BINOMIAL_CONFIDENCE, n, level)
            row[f"{level:.0%}, uncorrected"] = _inside(g.pit, level)
            row[f"{level:.0%}, corrected"] = _inside(g.pit_corrected, level)
            row[f"{level:.0%}, expected by chance"] = f"{lo / n:.0%}-{hi / n:.0%}"
            if rho == "weekly" and not (lo / n <= row[f"{level:.0%}, corrected"] <= hi / n):
                fails.append(f"(B) {name}, corrected {level:.0%} range: {row[f'{level:.0%}, corrected']:.0%} "
                             f"(expected by chance {lo / n:.0%}-{hi / n:.0%})")
        row["mean PIT"] = round(float(g.pit.mean()), 2)
        row["mean PIT, corrected"] = round(float(g.pit_corrected.mean()), 2)
        yearly.append(row)
    yearly = pd.DataFrame(yearly)

    # (C) the default rho against rho from consecutive days, 30-day means
    for part, df in (("A", pooled_a), ("B", pooled_b)):
        w, d = _get(df, "weekly", "30-day means", 90), _get(df, "daily", "30-day means", 90)
        if not abs(w - 0.9) <= abs(d - 0.9):
            fails.append(f"(C) part {part}: 30-day means inside the 90% interval {w:.1%} with the default rho, "
                         f"{d:.1%} with rho from consecutive days")
    # The quick run has too few stations and years to judge coverage: it only requires every job to run.
    res.passed = True if ctx.quick else not fails

    # Summary
    def _row(df, rho, what):
        return ", ".join(f"{lev}% {_get(df, rho, what, lev):.1%}" for lev in LEVELS)
    n_conv = len(converged)
    rho_of = {(r["station"], r["rho_timescale"]): r["rho"] for r in a}
    raised_a = sum(rho_of[(st, "weekly")] > rho_of[(st, "daily")] for st in stations)
    raised_b = sum(r["folds raised"] for r in b)
    res.notes.append(
        f"The default rho is the larger of the day-to-day and week-to-week estimates. It was above the day-to-day "
        f"value at {raised_a} of {len(stations)} stations' calibrations (part A) and in {raised_b} of "
        f"{sum(r['folds'] for r in b)} folds (part B); elsewhere the two error models are the same. The errors "
        f"of these rivers persist more from one day to the next than those of the Swiss rivers (part A: "
        f"day-to-day rho {min(rho_of[(st, 'daily')] for st in stations):.2f}-"
        f"{max(rho_of[(st, 'daily')] for st in stations):.2f}, against 0.70-0.86 there), so the rule changes "
        f"less here.")
    res.summary = (
        f"(A) DE-MCMC converged with both rho at {n_conv} of {len(stations)} stations. In 2021-2022, pooled over "
        f"those stations, the default intervals contained {_row(pooled_a, 'weekly', 'days')} of measured days; "
        f"90% intervals contained {_get(pooled_a, 'weekly', '7-day means', 90):.1%} of 7-day means and "
        f"{_get(pooled_a, 'weekly', '30-day means', 90):.1%} of 30-day means (rho from consecutive days: "
        f"{_get(pooled_a, 'daily', '7-day means', 90):.1%} and {_get(pooled_a, 'daily', '30-day means', 90):.1%}). "
        f"(B) Over {int(sum(r['folds'] for r in b))} held-out years, the default intervals contained "
        f"{_row(pooled_b, 'weekly', 'days')} of days; 90% intervals contained "
        f"{_get(pooled_b, 'weekly', '7-day means', 90):.1%} of 7-day means and "
        f"{_get(pooled_b, 'weekly', '30-day means', 90):.1%} of 30-day means (rho from consecutive days: "
        f"{_get(pooled_b, 'daily', '7-day means', 90):.1%} and {_get(pooled_b, 'daily', '30-day means', 90):.1%}). ")
    if len(yearly):
        y = yearly[yearly.rho == "weekly"]
        res.summary += (f"Corrected 90% ranges of yearly statistics held in "
                        f"{y['90%, corrected'].min():.0%}-{y['90%, corrected'].max():.0%} of held-out years "
                        f"(uncorrected {y['90%, uncorrected'].min():.0%}-{y['90%, uncorrected'].max():.0%}).")
    if ctx.quick:
        res.summary += " Quick run: coverage reported, not judged."
    elif fails:
        res.summary += " Criteria not met: " + "; ".join(fails) + "."

    # Tables
    def _fmt(df):
        df = df.copy()
        for c in df.columns:
            if (c.endswith("%") or ", uncorrected" in c or ", corrected" in c) and not c.startswith("mean PIT"):
                df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
        return df
    by_station = []
    for r in a:
        row = {"station": r["station"], "rho": r["rho_timescale"],
               "converged (steps)": f"{'yes' if r['converged'] else 'NO'} ({r['steps']})",
               "rho value": round(r["rho"], 3), "sigma (°C)": round(r["sigma"], 3)}
        for what in (*WINDOWS, "summer days", "heat-dome days"):
            k, n = r["counts"][what][90]
            row[f"{what} inside 90%"] = f"{k / n:.0%} of {n}" if n else ""
        row["mean 90% width (°C)"] = round(r["width"], 2)
        row["median minus measured (°C)"] = round(r["centre bias"], 2)
        by_station.append(row)
    by_station_b = []
    for r in b:
        row = {"station": r["station"], "held-out years": r["folds"]}
        for rho in RHO:
            row[f"mean rho ({rho})"] = round(r["rho"][rho], 3)
            for what in WINDOWS:
                k, n = r["counts"][rho][what][90]
                row[f"{what} inside 90% ({rho})"] = f"{k / n:.0%}" if n else ""
        by_station_b.append(row)
    accepted = pd.DataFrame([{"level": f"{x}%", "accepted share inside": "{:.1%}-{:.1%}".format(*accepted_real(x)),
                              "judged": "yes" if x in JUDGED else "no (reported)"} for x in LEVELS])
    res.sections.append(Section(
        "A. Later years (2021-2022), from DE-MCMC on the calibration years",
        "Share of measured values inside the central range at each level, pooled over the stations whose DE-MCMC "
        "converged with both rho, and the 90% coverage of each station. 7-day and 30-day means are moving means "
        "computed in each simulated series, counted where every day was measured. 2021 contains the June heat "
        "dome, the hottest week on record at many of these stations.",
        figures=[_fig_levels(pooled_a, pooled_b), _fig_stations(a, b, pooled_stations)],
        tables=[("A. Pooled over converged stations", _fmt(pooled_a)),
                ("A. By station, 90% intervals", pd.DataFrame(by_station)),
                ("Accepted coverage (miss rate between half and 1.5 times the stated one)", accepted)]))
    res.sections.append(Section(
        "B. Held-out years of the calibration period (leave-one-year-out cross-validation)",
        "The same shares for every year held out in turn, from each fold's parameters and its training years' "
        "sigma and rho (no parameter uncertainty, so slightly narrower than a FORWARD run's), and the yearly "
        "statistics before and after the correction estimated from the other held-out years of the station.",
        tables=[("B. Pooled over every held-out year", _fmt(pooled_b)),
                ("B. Yearly statistics, share of held-out years inside the central range", _fmt(yearly)),
                ("B. By station, 90% intervals", pd.DataFrame(by_station_b))]))

    # Why intervals miss: held-out errors against the error model they were predicted with
    from scipy.stats import norm
    diag_a = [r["diagnostics"] for r in a if r["rho_timescale"] == "weekly" and r["station"] in pooled_stations
              and r["diagnostics"]]
    diag_b = [d for r in b for d in r["diagnostics"]]

    def _ratio(ds):
        return float(np.sqrt(sum(d["rmse"] ** 2 * d["n"] for d in ds) / sum(d["sigma"] ** 2 * d["n"] for d in ds)))

    def _rms_z(ds):
        return float(np.sqrt(np.mean([d["z"] ** 2 for d in ds])))
    diag_rows = []
    for part, ds in (("A. later years (2021-2022), converged stations", diag_a), ("B. held-out years", diag_b)):
        if ds:
            r_ = _ratio(ds)
            diag_rows.append({"part": part, "periods": len(ds), "held-out RMSE / sigma": round(r_, 2),
                              "90% coverage this ratio alone gives": f"{2 * norm.cdf(norm.ppf(0.95) / r_) - 1:.1%}",
                              "size of each period's mean error / what the error model allows (RMS of z)":
                                  round(_rms_z(ds), 2),
                              "periods with |z| > 2": f"{sum(abs(d['z']) > 2 for d in ds)} of {len(ds)} "
                                                      f"(error model: about 5%)"})
    diag_station = []
    for r in b:
        ds = r["diagnostics"]
        if ds:
            diag_station.append({"station": r["station"], "held-out years": len(ds),
                                 "training sigma (°C)": round(float(np.mean([d["sigma"] for d in ds])), 3),
                                 "held-out RMSE (°C)": round(float(np.sqrt(np.mean([d["rmse"] ** 2 for d in ds]))), 3),
                                 "held-out RMSE / sigma": round(_ratio(ds), 2),
                                 "yearly mean errors (°C)": " ".join(f"{d['mean']:+.2f}" for d in ds),
                                 "SD the error model gives a yearly mean (°C)":
                                     round(float(np.mean([d["sd_mean"] for d in ds])), 2),
                                 "RMS of z": round(_rms_z(ds), 2)})
    if diag_rows:
        res.sections.append(Section(
            "C. Why intervals miss: the errors of new periods against the error model",
            "For each predicted period (part A: 2021-2022 at each station; part B: each held-out year), the RMSE of "
            "the best fit against the measurements, divided by the sigma the period was predicted with (from the "
            "calibration or training years), and the period's mean error divided by the standard deviation the "
            "error model (sigma, rho) gives the mean of that many days (z). If the error model described the new "
            "periods, the ratio would be about 1, and z would be about 1 in size (|z| > 2 in about 5% of periods). "
            "Added after the first full run to explain its result; not part of the criterion.",
            tables=[("C. Pooled", pd.DataFrame(diag_rows)), ("C. By station, held-out years", pd.DataFrame(diag_station))]))
        if diag_b:
            ratio_b, z_b = _ratio(diag_b), _rms_z(diag_b)
            big = sum(abs(d["z"]) > 2 for d in diag_b)
            text = (f"Section C compares the errors of new periods with the error model. In the held-out years the daily errors "
                    f"were {ratio_b:.2f} times the size of the training years' residuals, which alone brings a 90% "
                    f"interval down to about {2 * norm.cdf(norm.ppf(0.95) / ratio_b) - 1:.0%}; and the size of each "
                    f"year's mean error was {z_b:.1f} times what the error model allows a yearly mean ({big} of "
                    f"{len(diag_b)} years with |z| > 2, against about 5% expected).")
            if z_b > 1.5:
                text += (" The model's error shifts from one year to the next by more than errors that persist for "
                         "weeks produce. A single rho, however it is chosen, cannot describe that, and sigma "
                         "estimated from the calibration years does not include it.")
            text += " On the Swiss rivers, on which the error model was developed, held-out daily 90% intervals held on 90% of days (V11)."
            res.notes.append(text)

    w_a, d_a = _get(pooled_a, "weekly", "30-day means", 90), _get(pooled_a, "daily", "30-day means", 90)
    w_b, d_b = _get(pooled_b, "weekly", "30-day means", 90), _get(pooled_b, "daily", "30-day means", 90)
    res.notes.append(
        f"These rivers played no part in choosing the error model. Over 30 days, 90% intervals held for "
        f"{w_a:.0%} (later years) and {w_b:.0%} (held-out years) of means with the default rho, against "
        f"{d_a:.0%} and {d_b:.0%} with rho from consecutive days. Daily values depend little on rho, because "
        "sigma sets their spread; means over several weeks depend on how long errors persist, which is what the "
        "default rho is chosen to describe.")
    if n_conv < len(stations) and not ctx.quick:
        res.notes.append(
            f"At {len(stations) - n_conv} stations the DE-MCMC did not converge within the step limit with at "
            "least one rho (V15 found the same: the data there do not pin down all eight parameters). A user would "
            "be told not to use those intervals; they are shown in the table by station but left out of the "
            "pooled shares. Part B needs no MCMC and includes every station.")
    res.notes.append(
        "Part A tests later years, as a prediction would be used; with the June 2021 heat dome, it also includes "
        "days warmer than any in the calibration years at many stations (V10). Part B tests many more years, each predicted from "
        "the others. Neither can test the parameter intervals themselves, which needs a known truth (V4).")
    return res


def _fig_levels(pa, pb):
    """Stated level against achieved coverage, both parts, both rho, for days and multi-day means."""
    import matplotlib.pyplot as plt
    plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), sharex=True, sharey=True)
    grid = np.linspace(45, 100, 200)
    for i, (part, df) in enumerate((("Later years (2021-2022)", pa), ("Held-out years", pb))):
        for j, what in enumerate(WINDOWS):
            ax = axes[i, j]
            ax.fill_between(grid, 100 * (1 - 1.5 * (1 - grid / 100)), 100 * (1 - 0.5 * (1 - grid / 100)),
                            color=LIGHT_GREY, alpha=0.35, lw=0, zorder=0)
            ax.plot([45, 100], [45, 100], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
            for rho, colour, label in (("weekly", BLUE, "rho from weekly persistence (default)"),
                                       ("daily", ORANGE, "rho from consecutive days")):
                g = df[(df.rho == rho) & (df["values"] == what)]
                if len(g):
                    ax.plot(list(LEVELS), [100 * float(g[f"{x}%"].iloc[0]) for x in LEVELS], marker="o", ms=3.5,
                            lw=1.2, color=colour, label=label)
            ax.set_xlim(45, 100)
            ax.set_ylim(45, 100)
            ax.set_aspect("equal")
            ax.set_title(f"{part}: {what}", fontsize=9)
            if i == 1:
                ax.set_xlabel("Stated level (%)")
        axes[i, 0].set_ylabel("Measured values inside (%)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return (save_figure(fig, "V18_levels.png"),
            "British Columbia, 23 rivers not used to develop the error model: stated level against the share of "
            "measured values inside the interval, for days and for 7-day and 30-day means, with rho from weekly "
            "persistence (the default) and from consecutive days. Top: later years 2021-2022 (stations whose MCMC "
            "converged). Bottom: years held out by cross-validation. Dashed: stated = achieved; shaded: the "
            "accepted range.")


def _fig_stations(a, b, converged):
    """90% coverage of 30-day means by station, default against daily rho."""
    import matplotlib.pyplot as plt
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, title, pairs in (
            (axes[0], "Later years (2021-2022)",
             [(st, *[next(r["counts"]["30-day means"][90] for r in a if r["station"] == st and r["rho_timescale"] == rho)
                     for rho in RHO]) for st in sorted({r["station"] for r in a}) if st in converged]),
            (axes[1], "Held-out years",
             [(r["station"], r["counts"]["weekly"]["30-day means"][90], r["counts"]["daily"]["30-day means"][90])
              for r in b])):
        x = np.arange(len(pairs))
        for k, (colour, label) in enumerate(((BLUE, "default rho"), (ORANGE, "rho from consecutive days"))):
            y = [100 * p[1 + k][0] / p[1 + k][1] if p[1 + k][1] else np.nan for p in pairs]
            ax.scatter(x, y, s=16, color=colour, label=label, zorder=3)
        ax.axhline(90, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.axhspan(100 * MULTI_DAY_RANGE[0], 100 * MULTI_DAY_RANGE[1], color=LIGHT_GREY, alpha=0.35, lw=0, zorder=0)
        ax.set_xticks(x, [p[0] for p in pairs], rotation=90, fontsize=6.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("30-day means inside the 90% interval (%)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return (save_figure(fig, "V18_stations.png"),
            "Each station's share of 30-day means inside the 90% interval, with the default rho and with rho from "
            "consecutive days (shaded: 85-95%). Single stations have few independent months, so they scatter; "
            "the pooled shares are judged.")
