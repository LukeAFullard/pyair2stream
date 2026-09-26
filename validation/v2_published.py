"""
V2 - Reproduces published results (Piccolroaz et al., 2016).

Part A: the published parameters of every air2stream version, run through
pyair2stream, must give the published calibration and validation RMSE.
Part B: calibrating from scratch with DE must fit at least as well as the
published calibration.
Part C: the recalibrated parameters are compared with the published ones, and
the original Fortran program scores both sets.
Part D: the original calibration method, run as distributed (the Fortran
program's PSO with the authors' example settings), and pyair2stream's PSO with
the same settings.
Part E: the same with the RK4 scheme instead of Crank-Nicolson.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import (REPO, AUTHORS_BOUNDS, RIVERS, VERSIONS, Result, Timer, calibrate, load, mean_discharge,
                    params_at_bounds, published_params, published_rmse, quiet, river_csv, simulate)

TOL_A = 0.001    # °C; the published RMSE are rounded to 3 decimals
TOL_B = 0.002    # °C; DE may not beat the published fit by more than rounding
MATCH = 0.01     # parameters "match" when each is within 1% of its range of the published value
TOL_C = 0.01     # °C; where parameters differ, both sets must predict the validation years this closely
TOL_REF = 1e-4   # °C; the original program's RMSE of a parameter set must equal pyair2stream's this closely
RANGE = np.array(AUTHORS_BOUNDS["max"], float) - np.array(AUTHORS_BOUNDS["min"], float)

# Part D. The calibration settings distributed with the original code and its Swiss example
# (fortran/upstream: input.txt, PSO.txt, Switzerland/parameters.txt): PSO with 500 particles and
# 500 iterations, c1 = c2 = 2, inertia 0.9 -> 0.4, RMSE, Crank-Nicolson, the bounds used here.
PSO_SETTINGS = {"n_particles": 500, "n_run": 500, "c1": 2.0, "c2": 2.0, "wmax": 0.9, "wmin": 0.4}
PSO_RUNS = 3            # the original program seeds its random numbers from the clock: every run differs
PACKAGE_PSO_STATIONS = ("MAH_2369",)   # one river, to keep the run time down
UPSTREAM = os.path.join(REPO, "fortran", "upstream")


def _run_original(st, v, integ, runmode, par=None):
    """Run the original Fortran program on its own copy of the Swiss example data, set up as
    distributed; returns the lines of its result file (parameters, then -RMSE)."""
    sys.path.insert(0, REPO)
    from tests.fortran_runner import _build_fortran_binary
    air, water = st.split("_")
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(_build_fortran_binary(), os.path.join(d, "air2stream"))
        os.makedirs(os.path.join(d, "Switzerland"))
        for f in (f"{st}_cc.txt", f"{st}_cv.txt", "parameters.txt"):
            shutil.copy(os.path.join(UPSTREAM, "Switzerland", f), os.path.join(d, "Switzerland", f))
        if par is not None:
            with open(os.path.join(d, "Switzerland", "parameters_forward.txt"), "w") as f:
                f.write(" ".join(f"{x:.12f}" for x in par) + "\n")
        with open(os.path.join(d, "input.txt"), "w") as f:
            f.write(f"! Main input\nSwitzerland\n{air}\n{water}\nc\n1d\n{v}\n0\nRMS\n{integ}\n{runmode}\n0.60\n"
                    f"{PSO_SETTINGS['n_run']}\n0\n")
        with open(os.path.join(d, "PSO.txt"), "w") as f:
            f.write(f"! PSO parameters\n{PSO_SETTINGS['n_particles']}\n{PSO_SETTINGS['c1']} {PSO_SETTINGS['c2']}\n"
                    f"{PSO_SETTINGS['wmax']} {PSO_SETTINGS['wmin']}\n")
        subprocess.run(["./air2stream"], cwd=d, input="go\n", capture_output=True, text=True, check=True)
        return open(os.path.join(d, "Switzerland", f"output_{v}", f"1_{runmode}_RMS_{st}_c_1d.out")).read().split("\n")


def _original_pso(args):
    """One calibration by the original Fortran program, set up exactly as its Swiss example
    (optionally with another integrator, e.g. RK4 as listed in the program's readme)."""
    st, v, run, integ = (*args, "CRN")[:4]
    lines = _run_original(st, v, integ, "PSO")
    return {"method": "original program (Fortran PSO)", "integrator": integ, "station": st, "version": v,
            "run": run, "par": [float(x) for x in lines[0].split()], "calibration RMSE": -float(lines[1])}


def _original_rmse(args):
    """Calibration RMSE of a parameter set, computed by the original program (FORWARD mode, CRN)."""
    st, v, par = args
    from pyair2stream.config import ACTIVE_PARAMS
    p = np.zeros(8)
    p[list(ACTIVE_PARAMS[v])] = np.asarray(par, float)[list(ACTIVE_PARAMS[v])]
    return -float(_run_original(st, v, "CRN", "FORWARD", p)[1])


def _referee(ctx, rows_c):
    """Part C: the original program scores the published and the recalibrated parameters."""
    from pyair2stream.config import ACTIVE_PARAMS
    inv = {river: st for st, river in RIVERS.items()}
    sets = []
    for r in rows_c:
        for source in ("published", "recalibrated"):
            par = np.zeros(8)
            for name, x in r[source].items():
                par[int(name[1]) - 1] = x
            sets.append((inv[r["river"]], r["version"], par))
    with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
        scores = list(ex.map(_original_rmse, sets))
    rows = []
    for i, r in enumerate(rows_c):
        f_pub, f_rec = scores[2 * i], scores[2 * i + 1]
        gap = f_pub - f_rec
        rows.append({"river": r["river"], "version": r["version"], "parameters match": r["match"],
                     "original program: published parameters": round(f_pub, 5),
                     "original program: recalibrated parameters": round(f_rec, 5),
                     "pyair2stream: recalibrated parameters": r["calibration RMSE, recalibrated"],
                     "largest difference, pyair2stream vs original program": round(max(
                         abs(f_pub - r["calibration RMSE, published"]),
                         abs(f_rec - r["calibration RMSE, recalibrated"])), 6),
                     "better fit, by the original program": ("tie" if abs(gap) <= TOL_REF
                                                             else ("recalibrated" if gap > 0 else "published"))})
    return pd.DataFrame(rows)


def _package_pso(args):
    """One calibration by pyair2stream's PSO with the same settings."""
    st, v, run, integ = (*args, "CRN")[:4]
    from pyair2stream.optimization import PSO_mode
    cfg = {"version": v, "integrator": integ, "run_mode": "PSO", "objective_function": "RMS", "random_seed": run,
           "parameter_bounds": AUTHORS_BOUNDS, "optimization": dict(PSO_SETTINGS),
           "paths": {"input_data": river_csv(st, "calibration")}}
    data = load(cfg, f"v2pso_{st}_{v}_{run}_{integ}")
    with quiet():
        PSO_mode(data, seed=run)
    return {"method": "pyair2stream PSO", "integrator": integ, "station": st, "version": v, "run": run,
            "par": [float(x) for x in data.par_best], "calibration RMSE": float(-data.finalfit)}


def _part_d(ctx, de_par):
    """Tables for part D. `de_par` maps (station, version) to the DE recalibration."""
    from pyair2stream.config import ACTIVE_PARAMS
    from v1_fortran import fortran_available
    jobs = [(st, v, run) for st in RIVERS for v in VERSIONS for run in range(1, PSO_RUNS + 1)]
    with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
        package = ex.map(_package_pso, [j for j in jobs if j[0] in PACKAGE_PSO_STATIONS])
        runs = list(ex.map(_original_pso, jobs)) if fortran_available() else []
        runs += list(package)
    summary, values = [], []
    for st, river in RIVERS.items():
        for v in VERSIONS:
            act = list(ACTIVE_PARAMS[v])
            pub = np.array(published_params(v, st), float)
            here = [r for r in runs if r["station"] == st and r["version"] == v]
            row = {"river": river, "version": v, "published parameters": round(de_par[(st, v)][2], 4)}
            for method, short in (("original program (Fortran PSO)", "original program"),
                                  ("pyair2stream PSO", "pyair2stream PSO")):
                rs = [r for r in here if r["method"] == method]
                if not rs:
                    continue
                diffs = [100 * np.max(np.abs(np.array(r["par"])[act] - pub[act]) / RANGE[act]) for r in rs]
                row[f"{short}: RMSE of each run"] = ", ".join(f"{r['calibration RMSE']:.4f}" for r in rs)
                row[f"{short}: runs reproducing the published parameters"] = f"{sum(d <= 100 * MATCH for d in diffs)} of {len(rs)}"
            row["pyair2stream DE"] = round(de_par[(st, v)][1], 4)
            summary.append(row)
            if any(100 * np.max(np.abs(np.array(r["par"])[act] - pub[act]) / RANGE[act]) > 100 * MATCH for r in here):
                sets = [("published", pub, de_par[(st, v)][2])]
                sets += [(f"{r['method']}, run {r['run']}", np.array(r["par"]), r["calibration RMSE"]) for r in here]
                sets.append(("pyair2stream DE", np.array(de_par[(st, v)][0]), de_par[(st, v)][1]))
                for name, par, rmse in sets:
                    values.append({"river": river, "version": v, "parameters": name,
                                   **{f"a{j + 1}": (f"{par[j]:.3f}" if j in act else "") for j in range(8)},
                                   "calibration RMSE": f"{rmse:.4f}"})
    return pd.DataFrame(summary), pd.DataFrame(values)


def _rmse_function(csv, v, qmedia):
    """RMSE of any parameter set on `csv` (the data are loaded once)."""
    from pyair2stream.model import call_model
    data = load({"version": v, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": float(qmedia),
                 "parameters_forward": [0.0] * 8, "paths": {"input_data": csv}}, "v2c")
    obs = data.Twat_obs[365:]
    m = obs != -999.0

    def rmse(par):
        data.par[:] = par
        with quiet():
            call_model(data)
        return float(np.sqrt(np.mean((data.Twat_mod[365:][m] - obs[m]) ** 2)))
    return rmse


def _compare_parameters(st, v, pub, de):
    """Part C for one river and version."""
    from pyair2stream.config import ACTIVE_PARAMS
    act = list(ACTIVE_PARAMS[v])
    pub, de = np.asarray(pub, float), np.asarray(de, float)
    diff = float(np.max(np.abs(pub[act] - de[act]) / RANGE[act]))
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    q = mean_discharge(cal)
    f_cal, f_val = _rmse_function(cal, v, q), _rmse_function(val, v, q)
    row = {"river": RIVERS[st], "version": v,
           "published": {f"a{j + 1}": float(pub[j]) for j in act},
           "recalibrated": {f"a{j + 1}": float(de[j]) for j in act},
           "largest difference (% of range)": round(100 * diff, 2), "match": diff <= MATCH,
           "calibration RMSE, published": round(f_cal(pub), 5), "calibration RMSE, recalibrated": round(f_cal(de), 5)}
    if diff > MATCH:
        # Does a local search started from the published parameters improve their fit?
        def obj(x):
            p = pub.copy()
            p[act] = x
            return f_cal(p)
        lo, hi = np.array(AUTHORS_BOUNDS["min"], float)[act], np.array(AUTHORS_BOUNDS["max"], float)[act]
        local = minimize(obj, pub[act], method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": 100})
        # Is the recalibrated answer repeatable? Recalibrate with a different random seed.
        again = np.asarray(calibrate(cal, v, objective="RMS", seed=2, name="v2cal2").par_best, float)
        row.update({"local search from published": round(float(local.fun), 5),
                    "seed 2 vs seed 1 (% of range)": round(100 * float(np.max(np.abs(again[act] - de[act])
                                                                               / RANGE[act])), 2),
                    "validation RMSE, published": round(f_val(pub), 4),
                    "validation RMSE, recalibrated": round(f_val(de), 4)})
    return row


def _rmse(data) -> float:
    return float(-data.finalfit) if hasattr(data, "finalfit") else np.nan


def _sim_rmse(csv, v, par, qmedia) -> float:
    d = simulate(csv, v, par, "CRN", qmedia, objective="RMS", name="v2")
    obs, sim = d.Twat_obs[365:], d.Twat_mod[365:]
    m = obs != -999.0
    return float(np.sqrt(np.mean((sim[m] - obs[m]) ** 2)))


# Part E: the same with the RK4 scheme instead of Crank-Nicolson. To keep the run time down,
# pyair2stream's PSO is run only for the Mentue's versions 7 and 8, where the parameters trade off.
RK4_PACKAGE_CASES = (("MAH_2369", 7), ("MAH_2369", 8))


def _score_with(args):
    """RMSE of the published parameters when simulated with CRN and with RK4."""
    st, v = args
    cal = river_csv(st, "calibration")
    out = {}
    for integ in ("CRN", "RK4"):
        d = simulate(cal, v, published_params(v, st), integ, mean_discharge(cal), objective="RMS", name=f"v2e_{st}_{v}")
        obs, sim = d.Twat_obs[365:], d.Twat_mod[365:]
        m = obs != -999.0
        ok = np.all(np.isfinite(sim)) and np.max(np.abs(sim)) < 100
        out[integ] = float(np.sqrt(np.mean((sim[m] - obs[m]) ** 2))) if ok else None
    return st, v, out


def _de_rk4(args):
    st, v = args
    d = calibrate(river_csv(st, "calibration"), v, objective="RMS", integrator="RK4", name=f"v2e_de_{st}_{v}")
    return st, v, list(d.par_best), _rmse(d)


def _part_e(ctx):
    from pyair2stream.config import ACTIVE_PARAMS
    from v1_fortran import fortran_available
    cases = [(st, v) for st in RIVERS for v in VERSIONS]
    with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
        package = ex.map(_package_pso, [(st, v, run, "RK4") for st, v in RK4_PACKAGE_CASES
                                        for run in range(1, PSO_RUNS + 1)])
        original = (ex.map(_original_pso, [(st, v, run, "RK4") for st, v in cases for run in range(1, PSO_RUNS + 1)])
                    if fortran_available() else [])
        scores = {(st, v): out for st, v, out in ex.map(_score_with, cases)}
        de = {(st, v): (par, rmse) for st, v, par, rmse in ex.map(_de_rk4, cases)}
        runs = list(original) + list(package)
    fmt = lambda x: "diverged" if x is None else f"{x:.4f}"
    summary, values = [], []
    for st, v in cases:
        act = list(ACTIVE_PARAMS[v])
        pub = np.array(published_params(v, st), float)
        diff = lambda par: 100 * float(np.max(np.abs(np.array(par)[act] - pub[act]) / RANGE[act]))
        row = {"river": RIVERS[st], "version": v,
               "published parameters, RMSE with CRN": fmt(scores[(st, v)]["CRN"]),
               "published parameters, RMSE with RK4": fmt(scores[(st, v)]["RK4"]),
               "DE with RK4: RMSE": round(de[(st, v)][1], 4),
               "DE with RK4: largest difference from published (% of range)": round(diff(de[(st, v)][0]), 1)}
        here = [r for r in runs if r["station"] == st and r["version"] == v]
        for method, short in (("original program (Fortran PSO)", "original program, PSO with RK4"),
                              ("pyair2stream PSO", "pyair2stream PSO with RK4")):
            rs = [r for r in here if r["method"] == method]
            if rs:
                row[f"{short}: RMSE of each run"] = ", ".join(f"{r['calibration RMSE']:.4f}" for r in rs)
                row[f"{short}: runs reproducing the published parameters"] = \
                    f"{sum(diff(r['par']) <= 100 * MATCH for r in rs)} of {len(rs)}"
        summary.append(row)
        if st == "MAH_2369":
            sets = [("published", pub, scores[(st, v)]["CRN"]), ("DE with RK4", np.array(de[(st, v)][0]), de[(st, v)][1])]
            sets += [(f"{r['method']} with RK4, run {r['run']}", np.array(r["par"]), r["calibration RMSE"]) for r in here]
            for name, par, rmse in sets:
                values.append({"river": RIVERS[st], "version": v, "parameters": name,
                               **{f"a{j + 1}": (f"{par[j]:.3f}" if j in act else "") for j in range(8)},
                               "calibration RMSE": fmt(rmse), "largest difference from published (% of range)":
                                   "" if name == "published" else round(diff(par), 1)})
    return pd.DataFrame(summary), pd.DataFrame(values)


def run(ctx) -> Result:
    res = Result(
        code="V2", title="Reproduces published results",
        question="Given the published parameters, does pyair2stream reproduce the published model errors "
                 "for every model version? Does its own calibration fit at least as well, and find the published "
                 "parameters?",
        method="Piccolroaz et al. (2016) calibrated every air2stream version (by RMSE, with the "
               "Crank-Nicolson scheme) on three Swiss rivers and published the parameters and the RMSE for "
               "the calibration and validation periods (data/switzerland/published/). (A) Each published "
               "parameter set is simulated with CRN and its RMSE computed. For the validation period the "
               "original program recomputed Qmedia from the validation data, so that is done here too; "
               "pyair2stream by default keeps the calibration Qmedia (reported separately). "
               "(B) Each version is recalibrated from scratch by DE (RMS objective, CRN, the authors' "
               "parameter ranges). (C) The recalibrated parameters are compared with the published ones. "
               "Where they differ, a local search (L-BFGS-B) is started from the published parameters to "
               "see whether their fit can be improved, and both parameter sets predict the validation "
               "years (with the calibration Qmedia). As an independent referee, the original Fortran program "
               "(FORWARD mode, with its own copy of the data) computes the calibration RMSE of both parameter "
               "sets. (D) The paper calibrated with Particle Swarm Optimisation "
               "(PSO). The original Fortran program is run in calibration mode exactly as distributed with its "
               "Swiss example (PSO, 500 particles, 500 iterations, c1 = c2 = 2, inertia 0.9 to 0.4, RMSE, "
               f"Crank-Nicolson, the same parameter ranges; the paper itself does not state the swarm size or "
               f"iterations), {PSO_RUNS} times per river and version, since it seeds its random numbers from "
               f"the clock. pyair2stream's PSO is run {PSO_RUNS} times with the same settings on the Mentue "
               f"only, to keep the run time down. (E) The same with the RK4 scheme instead of "
               f"Crank-Nicolson (the original program's readme lists RK4 in its example): the published "
               f"parameters are simulated with both schemes, each version is recalibrated by DE with RK4, the "
               f"original program's PSO is run {PSO_RUNS} times per case with RK4, and pyair2stream's PSO with "
               f"RK4 for the Mentue's versions 7 and 8.",
        criterion=f"(A) all 30 published RMSE reproduced to within {TOL_A} °C. "
                  f"(B) DE calibration RMSE no worse than published + {TOL_B} °C in all 15 cases. "
                  f"(C) Wherever a recalibrated parameter differs from the published value by more than "
                  f"{MATCH:.0%} of its range, the two parameter sets predict the validation years with RMSE "
                  f"within {TOL_C} °C of each other; and, where gfortran is available, the original program "
                  f"computes the same calibration RMSE as pyair2stream for both parameter sets, to within "
                  f"{TOL_REF} °C. (D) and (E) are descriptive.")
    rows_a, rows_b, rows_c = [], [], []
    de_par = {}
    with Timer() as t:
        for st, river in RIVERS.items():
            cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
            q_cal, q_val = mean_discharge(cal), mean_discharge(val)
            for v in VERSIONS:
                par = published_params(v, st)
                rc = _sim_rmse(cal, v, par, q_cal)
                rv = _sim_rmse(val, v, par, q_val)
                rv_fixed = _sim_rmse(val, v, par, q_cal)
                pc, pv = published_rmse(v, st, "calibration"), published_rmse(v, st, "validation")
                rows_a.append({"river": river, "version": v, "published cal": pc, "pyair2stream cal": round(rc, 4),
                               "published val": pv, "pyair2stream val": round(rv, 4),
                               "val, calibration Qmedia": round(rv_fixed, 4),
                               "max |diff|": round(max(abs(rc - pc), abs(rv - pv)), 4)})
                if ctx.quick and not (st == "MAH_2369" and v in (3, 8)):
                    continue
                d = calibrate(cal, v, objective="RMS", name="v2cal")
                rows_b.append({"river": river, "version": v, "published cal RMSE": pc,
                               "DE cal RMSE": round(_rmse(d), 4),
                               "difference": round(_rmse(d) - pc, 4),
                               "parameters at a bound": ", ".join(params_at_bounds(d.par_best, v)) or "none"})
                rows_c.append(_compare_parameters(st, v, par, d.par_best))
                de_par[(st, v)] = (list(d.par_best), _rmse(d), rows_c[-1]["calibration RMSE, published"])
        from v1_fortran import fortran_available
        referee = _referee(ctx, rows_c) if fortran_available() else None
        d_summary, d_values = (None, None) if ctx.quick else _part_d(ctx, de_par)
        e_summary, e_values = (None, None) if ctx.quick else _part_e(ctx)
    a, b, c = pd.DataFrame(rows_a), pd.DataFrame(rows_b), pd.DataFrame(rows_c)
    # The parameter values themselves: published and recalibrated, one column per parameter.
    names = [f"a{j}" for j in range(1, 9)]
    values = []
    for r in rows_c:
        for source in ("published", "recalibrated"):
            values.append({"river": r["river"], "version": r["version"], "parameters": source,
                           **{n: (f"{r[source][n]:.3f}" if n in r[source] else "") for n in names},
                           "same as published": ("" if source == "published"
                                                 else ("yes" if r["match"] else "NO"))})
    values = pd.DataFrame(values)
    c = c.drop(columns=["published", "recalibrated"])
    ok_a = bool((a["max |diff|"] <= TOL_A).all())
    ok_b = bool((b["difference"] <= TOL_B).all())
    differ = c[~c["match"]]
    ok_c = bool(differ.empty or ((differ["validation RMSE, published"]
                                  - differ["validation RMSE, recalibrated"]).abs() <= TOL_C).all())
    if referee is not None:
        ok_c &= bool((referee["largest difference, pyair2stream vs original program"] <= TOL_REF).all())
    res.passed = ok_a and ok_b and ok_c
    res.seconds = t.seconds
    res.summary = (f"(A) {'All' if ok_a else 'Not all'} {2 * len(a)} published RMSE values reproduced "
                   f"(largest difference {a['max |diff|'].max():.4f} °C). "
                   f"(B) DE matched or improved on the published calibration in "
                   f"{int((b['difference'] <= TOL_B).sum())} of {len(b)} cases "
                   f"(mean difference {b['difference'].mean():+.3f} °C). "
                   f"(C) The recalibrated parameters match the published ones in {int(c['match'].sum())} of "
                   f"{len(c)} cases.")
    if len(differ):
        better = differ["calibration RMSE, published"] - differ["calibration RMSE, recalibrated"]
        res.summary += (f" Where they differ, the fits are within {better.abs().max():.3f} °C of each other and "
                        f"the predictions for the validation years within "
                        f"{(differ['validation RMSE, published'] - differ['validation RMSE, recalibrated']).abs().max():.3f} °C.")
        res.notes.append(
            "Where the parameters differ (" + ", ".join(f"{r.river} version {r.version}" for r in differ.itertuples())
            + "), the best fit lies in a long, flat valley along which the parameters trade off against each "
            "other: many combinations fit almost equally well, and an optimiser can stop at different points in "
            "it. For versions 7 and 8 the recalibration found a better fit than the published parameters, the "
            "same with a different random seed, and a local search started from the published parameters moves "
            "towards it: the "
            "published parameters were not at the best fit of the calibration data. The two sets predict the "
            "validation years equally well. Individual parameter values of these versions should therefore not "
            "be interpreted on their own (see also V4).")
    if referee is not None:
        worst = referee["largest difference, pyair2stream vs original program"].max()
        res.summary += (f" The original program, scoring both sets itself, gives the same RMSE as pyair2stream "
                        f"(largest difference {worst:.6f} °C)" if worst <= TOL_REF else
                        f" The original program's RMSE differs from pyair2stream's by up to {worst:.6f} °C "
                        f"(see table)")
        ref = referee[~referee["parameters match"]]
        if len(ref):
            verdict = ref["better fit, by the original program"].value_counts()
            res.summary += (f"; where the parameters differ, it finds the recalibrated parameters the better fit "
                            f"in {verdict.get('recalibrated', 0)} of {len(ref)} cases")
            if verdict.get("tie", 0):
                res.summary += f", a tie (within {TOL_REF} °C) in {verdict['tie']}"
            if verdict.get("published", 0):
                res.summary += f", and a worse fit in {verdict['published']}"
            res.summary += "."
            if worst <= TOL_REF and not verdict.get("published", 0):
                res.notes.append(
                    "Bug or optimiser? The table 'C. The original program as referee' separates the two. A bug in "
                    "the model, the data handling or the error calculation would show up in part A, where no "
                    "optimiser is involved; none does. A bug that made a calibration only look better would be "
                    "exposed by an independent scorer: the original program, given the recalibrated parameters, "
                    "computes the same RMSE as pyair2stream and agrees that they fit better than the published "
                    f"ones, or equally well to within {TOL_REF} °C. The differences in the parameters are therefore "
                    "where the optimisers stopped along a "
                    "flat valley, not a difference in how the model or its error is computed.")
        else:
            res.summary += "."
    worst = (a["val, calibration Qmedia"] - a["published val"]).abs().max()
    res.notes.append(f"With pyair2stream's default of keeping the calibration Qmedia for the validation period, "
                     f"validation RMSE differs from the published value by at most {worst:.3f} °C (only "
                     f"versions 4, 7 and 8 use discharge).")
    if (b["parameters at a bound"] != "none").any():
        res.notes.append("Some DE optima lie on the authors' parameter bounds (last column): the best fit "
                         "would lie outside those ranges. The fit is still valid within the stated bounds.")
    res.tables += [("A. Published parameters: RMSE (°C)", a), ("B. DE recalibration: calibration RMSE (°C)", b),
                   ("C. Recalibrated vs published parameters: fit and predictions", c),
                   ("C. Parameter values, published and recalibrated (blank: not used by the version; "
                    "'NO': a parameter differs by more than 1% of its range)", values)]
    if referee is not None:
        res.tables += [("C. The original program as referee: calibration RMSE (°C) of the published and the "
                        "recalibrated parameters, computed by the original Fortran program from its own copy "
                        "of the data", referee.assign(**{
                            col: referee[col].map(lambda x, n=n: f"{x:.{n}f}")
                            for col, n in (("original program: published parameters", 5),
                                           ("original program: recalibrated parameters", 5),
                                           ("pyair2stream: recalibrated parameters", 5),
                                           ("largest difference, pyair2stream vs original program", 6))}))]
    if d_summary is not None:
        res.tables += [(f"D. Calibration by the original method (PSO, authors' example settings): calibration "
                        f"RMSE (°C) of each run, and runs whose parameters are within {MATCH:.0%} of their range "
                        f"of the published ones", d_summary)]
        if len(d_values):
            res.tables += [("D. Parameter values of every run, where any run differs from the published "
                            "parameters", d_values)]
        orig = [c for c in d_summary.columns if c.startswith("original program: runs")]
        if orig:
            n_all = d_summary[orig[0]].str.startswith(f"{PSO_RUNS} of").sum()
            res.summary += (f" (D) The original program, run as distributed, reproduced the published parameters "
                            f"in all {PSO_RUNS} runs in {n_all} of {len(d_summary)} cases; elsewhere its own runs "
                            f"differ from each other.")
            res.notes.append(
                "Part D answers whether the published results can be reproduced with the original method and "
                "settings. Where the best fit is sharp, every run of the original program returns the published "
                "parameters. Where it is flat (the parameters trade off), the original program does not reproduce "
                "itself: each run stops at a different point along the valley, with a different fit, because PSO "
                "runs a fixed number of iterations and is seeded from the clock. The published values are one "
                "such run, and cannot be reproduced exactly by anyone, including with the original program. "
                "pyair2stream's PSO behaves in the same way; its DE calibration (part B) finds a better fit than "
                "these runs, repeatably.")
    if e_summary is not None:
        res.tables += [("E. With the RK4 scheme: the published parameters scored with each scheme, and "
                        "recalibration with RK4 (RMSE in °C)", e_summary),
                       ("E. Mentue: parameter values from calibration with RK4", e_values)]
        crn = e_summary["published parameters, RMSE with CRN"]
        rk4 = e_summary["published parameters, RMSE with RK4"]
        n_div = int((rk4 == "diverged").sum())
        close = e_summary[e_summary["DE with RK4: largest difference from published (% of range)"] <= 100 * MATCH]
        res.summary += (f" (E) The published parameters are Crank-Nicolson parameters: with RK4 they diverge in "
                        f"{n_div} of {len(e_summary)} cases and give a different error in the rest. Calibrating "
                        f"with RK4 returns parameters within {MATCH:.0%} of range of the published ones in "
                        f"{len(close)} of {len(e_summary)} cases.")
        res.notes.append(
            "Part E: the published errors are reproduced exactly with Crank-Nicolson (part A), the scheme the "
            "paper states, and not with RK4. Where the water temperature responds slowly (the Mentue, versions "
            "3-5) RK4 and Crank-Nicolson behave alike, so calibrating with RK4 gives parameters close to the "
            "published ones, though with a different error. Where it responds faster, RK4 becomes unstable "
            "for the published parameters (it needs the relaxation rate below 2.785 per day), and calibration "
            "with RK4 is forced to different parameters. Parameters are specific to the scheme they were "
            "calibrated with (V6).")
    res.figure_data = a
    return res
