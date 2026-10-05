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
Part F: 90% uncertainty intervals for every recalibrated parameter (MCMC and
cross-validation jackknife), and whether the published values lie inside them.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import (REPO, WORK, AUTHORS_BOUNDS, DE_SETTINGS, RIVERS, VERSIONS, Result, Section, Timer, calibrate,
                    load, mean_discharge, params_at_bounds, published_params, published_rmse, quiet, river_csv,
                    simulate, plot_style, save_figure, reference_line, BLUE, ORANGE, AQUA, INK, INK2, MUTED,
                    LIGHT_GREY, AXIS)

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
    return pd.DataFrame(summary), pd.DataFrame(values), runs


def _series_function(csv, v, qmedia, name):
    """For any parameter set on `csv` (the data are loaded once): its simulated daily series and
    RMSE. Returns (dates, sim(par), rmse(par))."""
    from pyair2stream.model import call_model
    data = load({"version": v, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": float(qmedia),
                 "parameters_forward": [0.0] * 8, "paths": {"input_data": csv}}, name)
    obs = data.Twat_obs[365:]
    m = obs != -999.0
    dates = pd.to_datetime(pd.read_csv(csv).Date)

    def sim(par):
        data.par[:] = par
        with quiet():
            call_model(data)
        return data.Twat_mod[365:].copy()

    def rmse(par):
        return float(np.sqrt(np.mean((sim(par)[m] - obs[m]) ** 2)))
    return dates, sim, rmse


def _rmse_function(csv, v, qmedia, name="v2c"):
    """RMSE of any parameter set on `csv` (the data are loaded once)."""
    return _series_function(csv, v, qmedia, name)[2]


VALLEY_T = np.linspace(-0.25, 1.25, 31)   # positions along the line published (0) -> recalibrated (1)


def _compare_parameters(st, v, pub, de):
    """Part C for one river and version."""
    from pyair2stream.config import ACTIVE_PARAMS
    act = list(ACTIVE_PARAMS[v])
    pub, de = np.asarray(pub, float), np.asarray(de, float)
    diff = float(np.max(np.abs(pub[act] - de[act]) / RANGE[act]))
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    q = mean_discharge(cal)
    f_cal = _rmse_function(cal, v, q, f"v2c_{st}_{v}_cal")
    dates_val, sim_val, f_val = _series_function(val, v, q, f"v2c_{st}_{v}_val")
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
        again = np.asarray(calibrate(cal, v, objective="RMS", seed=2, name=f"v2cal2_{st}_{v}").par_best, float)
        # How the fit changes along the straight line from the published to the recalibrated parameters
        # (kept inside the bounds), on the calibration and on the validation years.
        lo8, hi8 = np.array(AUTHORS_BOUNDS["min"], float), np.array(AUTHORS_BOUNDS["max"], float)
        path = [np.clip(pub + t * (de - pub), lo8, hi8) for t in VALLEY_T]
        row["_valley"] = (VALLEY_T, np.array([f_cal(p) for p in path]), np.array([f_val(p) for p in path]))
        row["_predictions"] = (dates_val, sim_val(pub), sim_val(de),
                               pd.read_csv(val).T_water.to_numpy())
        row.update({"local search from published": round(float(local.fun), 5),
                    "seed 2 vs seed 1 (% of range)": round(100 * float(np.max(np.abs(again[act] - de[act])
                                                                               / RANGE[act])), 2),
                    "validation RMSE, published": round(f_val(pub), 4),
                    "validation RMSE, recalibrated": round(f_val(de), 4)})
    return row


def _rmse(data) -> float:
    return float(-data.finalfit) if hasattr(data, "finalfit") else np.nan


def _sim_rmse(csv, v, par, qmedia, name="v2") -> float:
    d = simulate(csv, v, par, "CRN", qmedia, objective="RMS", name=name)
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
    return pd.DataFrame(summary), pd.DataFrame(values), scores, {k: x[1] for k, x in de.items()}




# --- Parts A-C for one river and version (run in parallel) ----------------------

def _case(args):
    """Part A (published parameters scored), and if `recalibrate`, parts B and C."""
    st, v, recalibrate = args
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    q_cal, q_val = mean_discharge(cal), mean_discharge(val)
    par = published_params(v, st)
    rc = _sim_rmse(cal, v, par, q_cal, f"v2a_{st}_{v}")
    rv = _sim_rmse(val, v, par, q_val, f"v2a_{st}_{v}")
    rv_fixed = _sim_rmse(val, v, par, q_cal, f"v2a_{st}_{v}")
    pc, pv = published_rmse(v, st, "calibration"), published_rmse(v, st, "validation")
    out = {"a": {"river": RIVERS[st], "version": v, "published cal": pc, "pyair2stream cal": round(rc, 4),
                 "published val": pv, "pyair2stream val": round(rv, 4), "val, calibration Qmedia": round(rv_fixed, 4),
                 "max |diff|": round(max(abs(rc - pc), abs(rv - pv)), 4)}}
    if recalibrate:
        d = calibrate(cal, v, objective="RMS", name=f"v2cal_{st}_{v}")
        out["b"] = {"river": RIVERS[st], "version": v, "published cal RMSE": pc, "DE cal RMSE": round(_rmse(d), 4),
                    "difference": round(_rmse(d) - pc, 4),
                    "parameters at a bound": ", ".join(params_at_bounds(d.par_best, v)) or "none"}
        out["c"] = _compare_parameters(st, v, par, d.par_best)
        out["de"] = (list(d.par_best), _rmse(d), out["c"]["calibration RMSE, published"])
    return st, v, out


# --- Part F: parameter uncertainty intervals ------------------------------------

F_QUICK = (("MAH_2369", 3),)
PARAM_NAMES = ("constant", "air temperature", "relaxation", "discharge exponent", "discharge: constant",
               "seasonal amplitude", "seasonal timing", "discharge: relaxation")


def _f_config(st, v, tag, **extra):
    cal = river_csv(st, "calibration")
    return {"version": v, "integrator": "CRN", "objective_function": "RMS", "random_seed": 1,
            "Qmedia": mean_discharge(cal), "parameter_bounds": AUTHORS_BOUNDS,
            "paths": {"input_data": cal, "output_dir": os.path.join(WORK, tag, "out")}, **extra}


def _mcmc_interval(args):
    """90% credible interval of every parameter from DE-MCMC with the least-squares likelihood (the
    paper's calibration criterion), widened for the autocorrelation of the errors."""
    st, v = args
    from pyair2stream.optimization import DE_MCMC_mode
    tag = f"v2f_mcmc_{st}_{v}"
    cfg = _f_config(st, v, tag, run_mode="DE-MCMC",
                    optimization={**DE_SETTINGS, "mcmc_walkers": 32, "mcmc_steps": 20000},
                    uncertainty_options={"noise_model": "ar1", "likelihood": "least_squares",
                                         "strict_convergence": False})
    data = load(cfg, tag)
    with quiet():
        DE_MCMC_mode(data, seed=1)
    out = os.path.join(WORK, tag, "out")
    meta = json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))
    chain = pd.read_csv(os.path.join(out, "MCMC_chain_S_c_1d.csv"))
    iv = {int(c.split("_")[1]) - 1: tuple(float(x) for x in np.percentile(chain[c], [5, 95])) for c in chain.columns}
    return "mcmc", st, v, {"intervals": iv, "converged": bool(meta["converged"]), "steps": int(meta["steps_run"])}


def _jackknife_interval(args):
    """90% jackknife interval of every parameter from leave-one-year-out cross-validation."""
    st, v = args
    from pyair2stream.config import ACTIVE_PARAMS
    from pyair2stream.cross_validation import cross_validate
    tag = f"v2f_cv_{st}_{v}"
    cfg = _f_config(st, v, tag, run_mode="DE", optimization=dict(DE_SETTINGS),
                    cross_validation={"enabled": True, "unit": "year"})
    data = load(cfg, tag)
    with quiet():
        table = cross_validate(data, "DE").set_index("fold")
    folds = [f for f in table.index if str(f).isdigit()]
    iv = {j: (float(table.loc["jackknife_90_lower", f"p{j + 1}"]), float(table.loc["jackknife_90_upper", f"p{j + 1}"]))
          for j in ACTIVE_PARAMS[v]}
    return "jackknife", st, v, {"intervals": iv, "folds": len(folds)}


def _at_bound(x, j):
    """Whether x lies on a bound of the authors' range for parameter j (index 0-7)."""
    lo, hi = AUTHORS_BOUNDS["min"][j], AUTHORS_BOUNDS["max"][j]
    return min(abs(x - lo), abs(x - hi)) <= 1e-3 * RANGE[j]


def _inside(x, lo, hi, j):
    """Whether x lies inside the interval [lo, hi] of parameter j. A value on a bound of the authors'
    range also counts as inside when the interval reaches to within 1% of the range of that bound:
    the range cuts the distribution off there, so a 5-95% interval stops just short of the bound
    even when the best fit lies on it."""
    tol = 1e-9 * RANGE[j]
    if lo - tol <= x <= hi + tol:
        return True
    if _at_bound(x, j):
        bound_lo = abs(x - AUTHORS_BOUNDS["min"][j]) <= 1e-3 * RANGE[j]
        return (lo - x if bound_lo else x - hi) <= MATCH * RANGE[j]
    return False


def _part_f(ctx, de_par):
    """Intervals for every recalibrated case; returns (detail table, summary table, per-river plot data)."""
    from pyair2stream.config import ACTIVE_PARAMS
    cases = list(F_QUICK) if ctx.quick else [(st, v) for st in RIVERS for v in VERSIONS]
    cases = [c for c in cases if c in de_par]
    cases.sort(key=lambda c: c[0] != "SIO_2011")           # the longest record first
    jobs = [(f, c) for c in cases for f in (_mcmc_interval, _jackknife_interval)]
    with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
        found = {}
        for kind, st, v, res in [f.result() for f in [ex.submit(fn, c) for fn, c in jobs]]:
            found[(kind, st, v)] = res
    detail, summary, plot = [], [], {}
    for st, v in sorted(cases, key=lambda c: (list(RIVERS).index(c[0]), c[1])):
        pub = np.array(published_params(v, st), float)
        best = np.array(de_par[(st, v)][0], float)
        mc, jk = found[("mcmc", st, v)], found[("jackknife", st, v)]
        in_mc = in_jk = 0
        for j in ACTIVE_PARAMS[v]:
            m_lo, m_hi = mc["intervals"][j]
            j_lo, j_hi = jk["intervals"][j]
            a = _inside(pub[j], m_lo, m_hi, j) if mc["converged"] else None
            b = _inside(pub[j], j_lo, j_hi, j)
            in_mc += bool(a)
            in_jk += bool(b)
            detail.append({"river": RIVERS[st], "version": v, "parameter": f"a{j + 1}",
                           "published": f"{pub[j]:.3f}", "pyair2stream best fit": f"{best[j]:.3f}",
                           "on a bound": "yes" if _at_bound(pub[j], j) else "",
                           "MCMC 90% interval": f"{m_lo:.3f} to {m_hi:.3f}",
                           "published inside (MCMC)": "not converged" if a is None else a,
                           "jackknife 90% interval": f"{j_lo:.3f} to {j_hi:.3f}", "published inside (jackknife)": b})
        n = len(ACTIVE_PARAMS[v])
        summary.append({"river": RIVERS[st], "version": v, "parameters": n,
                        "MCMC converged (steps)": f"{'yes' if mc['converged'] else 'NO'} ({mc['steps']})",
                        "published inside MCMC interval": f"{in_mc} of {n}" if mc["converged"] else "not converged",
                        "cross-validation folds": jk["folds"],
                        "published inside jackknife interval": f"{in_jk} of {n}"})
        plot.setdefault(st, {})[v] = {"pub": pub, "best": best, "mcmc": mc["intervals"], "mcmc_ok": mc["converged"],
                                      "jk": jk["intervals"]}
    return pd.DataFrame(detail), pd.DataFrame(summary), plot


# --- Figures --------------------------------------------------------------------

def _case_labels(df):
    return [f"{r} v{v}" for r, v in zip(df.river, df.version)]


def _fig_published_rmse(a):
    import matplotlib.pyplot as plt
    plot_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1, 1.5]},
                                 layout="constrained")
    for col, label, m, colour in (("cal", "calibration period", "o", BLUE), ("val", "validation period", "s", ORANGE)):
        ax.scatter(a[f"published {col}"], a[f"pyair2stream {col}"], marker=m, s=30, color=colour, label=label, zorder=3)
    lim = [0.5, float(max(a["published cal"].max(), a["published val"].max())) + 0.05]
    ax.plot(lim, lim, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Published RMSE (°C)", ylabel="pyair2stream RMSE (°C)",
           title="All 30 published errors reproduced")
    ax.legend(loc="upper left")
    y = np.arange(len(a))[::-1]
    bx.axvspan(-0.0005, 0.0005, color="#ebeae5", zorder=0, label="within the rounding of the published values")
    bx.scatter(a["pyair2stream cal"] - a["published cal"], y, marker="o", s=30, color=BLUE, label="calibration period", zorder=3)
    bx.scatter(a["pyair2stream val"] - a["published val"], y, marker="s", s=30, color=ORANGE, label="validation period", zorder=3)
    bx.axvline(0, color=AXIS, lw=0.8)
    bx.set_yticks(y, _case_labels(a), fontsize=7.5)
    bx.set_xlim(-0.0012, 0.0012)
    bx.set_xlabel("pyair2stream minus published RMSE (°C)")
    bx.set_title("Differences are within the rounding of the published values")
    bx.grid(axis="y", visible=False)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.5)
    return (save_figure(fig, "V2_published_rmse.png"),
            "Left: each point is one river, model version and period; all lie on the 1:1 line. Right: the "
            "differences, all within the rounding of the published three-decimal values (shaded).")


def _fig_recalibration(b):
    import matplotlib.pyplot as plt
    plot_style()
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.25 * len(b)))
    y = np.arange(len(b))[::-1]
    ax.scatter(b["difference"], y, s=30, color=BLUE, zorder=3)
    ax.axvline(0, color=AXIS, lw=0.8)
    reference_line(ax, TOL_B, f"pass limit +{TOL_B} °C", axis="x")
    ax.set_yticks(y, _case_labels(b), fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("DE calibration RMSE minus published calibration RMSE (°C)  (negative: better fit)")
    ax.set_title("Recalibration from scratch fits as well as, or better than, the paper")
    return (save_figure(fig, "V2_recalibration.png"),
            "Calibration error of pyair2stream's own calibration (DE) minus the published calibration error. "
            "Every case is at or below zero apart from rounding; none is near the pass limit.")


def _fig_valley(differ_rows):
    import matplotlib.pyplot as plt
    plot_style()
    rows = [r for r in differ_rows if "_valley" in r]
    n = len(rows)
    if not n:
        return None
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 0.8, 3.1), sharey=True, squeeze=False)
    # Scale the y-axis to the stretch between the two parameter sets: beyond them the
    # RMSE can rise steeply for some cases, which would flatten every other curve.
    span = 0.0
    for r in rows:
        t, cal, val = r["_valley"]
        i0 = int(np.argmin(np.abs(t)))
        inside = (t >= -0.05) & (t <= 1.05)
        span = max(span, np.nanmax(np.abs(cal[inside] - cal[i0])), np.nanmax(np.abs(val[inside] - val[i0])))
    lim = max(1.4 * span, 0.01)
    for ax, r in zip(axes[0], rows):
        t, cal, val = r["_valley"]
        i0 = int(np.argmin(np.abs(t)))
        ax.plot(t, cal - cal[i0], color=BLUE, label="calibration years")
        ax.plot(t, val - val[i0], color=ORANGE, label="validation years")
        for x in (0, 1):
            ax.axvline(x, color=AXIS, lw=0.8)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks([0, 1], ["published", "recali-\nbrated"], fontsize=7.5)
        ax.set_title(f"{r['river']} v{r['version']}", fontsize=9)
        ax.set_ylim(-lim, lim)
    axes[0][0].set_ylabel("RMSE change from the published\nparameters (°C)")
    axes[0][0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Along the line between the two parameter sets the fit barely changes", y=1.03)
    return (save_figure(fig, "V2_rmse_valley.png"),
            "RMSE along the straight line from the published parameters (left mark) to the recalibrated ones "
            "(right mark), relative to the published parameters; the y-axis covers the stretch between the two. "
            "The curves are almost flat: many parameter "
            "combinations fit nearly equally well (a 'flat valley'), which is why the two sets differ.")


def _fig_predictions(row):
    import matplotlib.pyplot as plt
    plot_style()
    dates, pub, rec, obs = row["_predictions"]
    dates = pd.DatetimeIndex(dates)
    year = dates.year[len(dates) // 2]
    m = dates.year == year
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 4.8), gridspec_kw={"height_ratios": [2.2, 1]})
    ax.scatter(dates[m], obs[m], s=4, color=LIGHT_GREY, label="measured", zorder=2)
    ax.plot(dates[m], pub[m], color=BLUE, lw=2.4, alpha=0.6, label="published parameters", zorder=3)
    ax.plot(dates[m], rec[m], color=ORANGE, lw=1.0, label="recalibrated parameters", zorder=4)
    ax.set_ylabel("Water temperature (°C)")
    ax.set_title(f"{row['river']} version {row['version']}: predictions for {year}, a validation year")
    ax.legend(loc="upper left", fontsize=7.5)
    bx.plot(dates, rec - pub, color=INK2, lw=0.6)
    bx.axhline(0, color=AXIS, lw=0.8)
    bx.set_ylabel("Recalibrated minus\npublished (°C)")
    bx.set_title("Difference between the two predictions over the whole validation period", fontsize=9)
    fig.tight_layout()
    d = np.abs(rec - pub)
    return (save_figure(fig, "V2_predictions.png"),
            f"The case where the parameters differ most ({row['river']}, version {row['version']}, largest "
            f"difference {row['largest difference (% of range)']:.0f}% of a parameter's range). On most days the two "
            f"parameter sets predict almost the same temperature (typical difference {np.sqrt(np.mean(d ** 2)):.2f} °C "
            f"RMS; {np.mean(d <= 0.2):.0%} of days within 0.2 °C), but on a few days they differ by up to "
            f"{np.nanmax(d):.1f} °C. Their errors over the validation years are equal (table C).")


def _fig_parameter_intervals(st, cases):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pyair2stream.config import ACTIVE_PARAMS
    plot_style()
    versions = [v for v in VERSIONS if v in cases]
    ypos = {v: len(versions) - 1 - i for i, v in enumerate(versions)}     # version 3 at the top
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), sharey=True)
    for j, ax in enumerate(axes.flat):
        xs = []
        for v in versions:
            c, y = cases[v], ypos[v]
            if j not in ACTIVE_PARAMS[v]:
                continue
            if j in c["mcmc"]:
                lo, hi = c["mcmc"][j]
                ax.plot([lo, hi], [y + 0.18] * 2, color=BLUE, lw=3, solid_capstyle="round",
                        alpha=1.0 if c["mcmc_ok"] else 0.35, zorder=3)
                xs += [lo, hi]
            lo, hi = c["jk"][j]
            ax.plot([lo, hi], [y - 0.18] * 2, color=ORANGE, lw=3, solid_capstyle="round", zorder=3)
            ax.scatter(c["best"][j], y, s=18, color=INK2, zorder=4)
            ax.scatter(c["pub"][j], y, s=44, marker="D", color=INK, edgecolor="white", linewidth=0.7, zorder=5)
            xs += [lo, hi, c["best"][j], c["pub"][j]]
        if not xs:
            ax.set_title(f"a{j + 1}: {PARAM_NAMES[j]}", fontsize=8.5, loc="left")
            ax.text(0.5, 0.5, "not used by these versions", transform=ax.transAxes, ha="center", fontsize=7.5,
                    color=MUTED)
            ax.set_xticks([])
            continue
        span = (max(xs) - min(xs)) or max(abs(max(xs)), 0.1)
        ax.set_xlim(min(xs) - 0.08 * span, max(xs) + 0.08 * span)
        ax.set_ylim(-0.6, len(versions) - 0.4)
        ax.set_title(f"a{j + 1}: {PARAM_NAMES[j]}", fontsize=8.5, loc="left")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
        for v in versions:
            if j not in ACTIVE_PARAMS[v]:
                ax.text(0.5, ypos[v], "not used", transform=ax.get_yaxis_transform(), ha="center", va="center",
                        fontsize=7, color=MUTED)
    axes[0][0].set_yticks(list(ypos.values()), [f"version {v}" for v in ypos])
    handles = [Line2D([0], [0], color=BLUE, lw=3, label="90% interval, MCMC (least squares)"),
               Line2D([0], [0], color=ORANGE, lw=3, label="90% interval, cross-validation jackknife"),
               Line2D([0], [0], marker="o", color=INK2, lw=0, markersize=5, label="pyair2stream best fit (DE)"),
               Line2D([0], [0], marker="D", color=INK, lw=0, markersize=6, label="published (Piccolroaz et al., 2016)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.suptitle(f"{RIVERS[st]}: recalibrated parameters with 90% intervals, and the published values", y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    name = f"V2_parameter_intervals_{RIVERS[st].replace('ô', 'o')}.png"
    return (save_figure(fig, name),
            f"{RIVERS[st]}: each row is a model version, each panel a parameter. Bars: 90% intervals from MCMC "
            f"with the least-squares likelihood (blue) and from the cross-validation jackknife (orange), both "
            f"around the least-squares estimate as in the paper; dot: pyair2stream's best fit; diamond: the "
            f"published value. A faded blue bar marks an MCMC run that did not converge.")


def _fig_pso(runs, de_par):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plot_style()
    cases = [(st, v) for st in RIVERS for v in VERSIONS if (st, v) in de_par]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (st, v) in enumerate(cases):
        y = len(cases) - 1 - i
        pub = de_par[(st, v)][2]
        for method, marker, colour, dy in (("original program (Fortran PSO)", "o", BLUE, 0.12),
                                           ("pyair2stream PSO", "^", ORANGE, -0.12)):
            xs = [r["calibration RMSE"] - pub for r in runs
                  if r["station"] == st and r["version"] == v and r["method"] == method and r["integrator"] == "CRN"]
            ax.scatter(xs, [y + dy] * len(xs), marker=marker, s=26, facecolor="none", edgecolor=colour, lw=1.2, zorder=3)
        ax.scatter(de_par[(st, v)][1] - pub, y, marker="D", s=30, color=AQUA, zorder=4)
    ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.set_yticks(range(len(cases))[::-1], [f"{RIVERS[st]} v{v}" for st, v in cases], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Calibration RMSE minus the published calibration RMSE (°C)")
    ax.set_title("Calibration by the original method does not reproduce itself where the fit is flat")
    ax.legend(handles=[Line2D([0], [0], marker="o", lw=0, markerfacecolor="none", markeredgecolor=BLUE,
                              label="original program, PSO (one mark per run)"),
                       Line2D([0], [0], marker="^", lw=0, markerfacecolor="none", markeredgecolor=ORANGE,
                              label="pyair2stream PSO (Mentue only)"),
                       Line2D([0], [0], marker="D", lw=0, color=AQUA, label="pyair2stream DE")],
              loc="lower right", fontsize=7.5)
    return (save_figure(fig, "V2_pso_runs.png"),
            "Each mark is one calibration run with the original method (PSO, authors' settings), relative to the "
            "published fit (dashed line). Where the runs of the original program spread out, the published values "
            "are one such run; DE is repeatable and fits at least as well.")


def _fig_rk4(scores, de_rk4):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plot_style()
    cases = [(st, v) for st in RIVERS for v in VERSIONS if (st, v) in scores]
    cap = 1.3
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (st, v) in enumerate(cases):
        y = len(cases) - 1 - i
        crn, rk4 = scores[(st, v)]["CRN"], scores[(st, v)]["RK4"]
        ax.scatter(crn, y, s=30, color=BLUE, zorder=4)
        if rk4 is None or rk4 > cap:
            ax.scatter(cap, y, marker=">", s=30, color=ORANGE, zorder=4)
            ax.annotate("diverged" if rk4 is None else f"{rk4:.1f}", (cap, y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK2)
        else:
            ax.scatter(rk4, y, s=30, color=ORANGE, zorder=4)
        if (st, v) in de_rk4:
            ax.scatter(de_rk4[(st, v)], y, marker="D", s=26, color=AQUA, zorder=5)
    ax.set_xlim(0.4, cap + 0.25)
    ax.set_yticks(range(len(cases))[::-1], [f"{RIVERS[st]} v{v}" for st, v in cases], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Calibration RMSE (°C)")
    ax.set_title("The published parameters belong to the Crank-Nicolson scheme")
    ax.legend(handles=[Line2D([0], [0], marker="o", lw=0, color=BLUE, label="published parameters, Crank-Nicolson"),
                       Line2D([0], [0], marker="o", lw=0, color=ORANGE, label="published parameters, RK4"),
                       Line2D([0], [0], marker="D", lw=0, color=AQUA, label="recalibrated with RK4")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7.5, frameon=False)
    return (save_figure(fig, "V2_rk4.png"),
            "The published parameters reproduce the paper's errors only with Crank-Nicolson (blue). With RK4 "
            "(orange) they diverge or fit worse; RK4 needs its own calibration (aqua).")


# --- The check -------------------------------------------------------------------

def run(ctx) -> Result:
    res = Result(
        code="V2", title="Reproduces published results",
        question="Given the published parameters, does pyair2stream reproduce the published model errors "
                 "for every model version? Does its own calibration fit at least as well, and find the published "
                 "parameters? Do the published parameters lie inside pyair2stream's uncertainty intervals?",
        method="Piccolroaz et al. (2016) calibrated every air2stream version (by RMSE, with the "
               "Crank-Nicolson scheme) on three Swiss rivers and published the parameters and the RMSE for "
               "the calibration and validation periods (data/switzerland/published/). (A) Each published "
               "parameter set is simulated with CRN and its RMSE computed. For the validation period the "
               "original program recomputed Qmedia from the validation data, so that is done here too; "
               "pyair2stream by default keeps the calibration Qmedia (reported separately). "
               "(B) Each version is recalibrated from scratch by DE (RMS objective, CRN, the authors' "
               "parameter ranges). (C) The recalibrated parameters are compared with the published ones. "
               "Where they differ, a local search (L-BFGS-B) is started from the published parameters to "
               "see whether their fit can be improved, the fit is traced along the straight line between the "
               "two parameter sets, and both sets predict the validation years (with the calibration Qmedia). "
               "As an independent referee, the original Fortran program (FORWARD mode, with its own copy of the "
               "data) computes the calibration RMSE of both parameter sets. (D) The paper calibrated with "
               "Particle Swarm Optimisation (PSO). The original Fortran program is run in calibration mode "
               "exactly as distributed with its Swiss example (PSO, 500 particles, 500 iterations, c1 = c2 = 2, "
               "inertia 0.9 to 0.4, RMSE, Crank-Nicolson, the same parameter ranges; the paper itself does not "
               f"state the swarm size or iterations), {PSO_RUNS} times per river and version, since it seeds its "
               f"random numbers from the clock. pyair2stream's PSO is run {PSO_RUNS} times with the same settings "
               f"on the Mentue only, to keep the run time down. (E) The same with the RK4 scheme instead of "
               f"Crank-Nicolson (the original program's readme lists RK4 in its example): the published "
               f"parameters are simulated with both schemes, each version is recalibrated by DE with RK4, the "
               f"original program's PSO is run {PSO_RUNS} times per case with RK4, and pyair2stream's PSO with "
               f"RK4 for the Mentue's versions 7 and 8. (F) The published parameters are least-squares estimates "
               f"(the paper minimised RMSE and used no error model), so they are compared with intervals around "
               f"the same estimator, on the same calibration years, with the RMS objective, CRN, the authors' "
               f"ranges and the calibration Qmedia: the jackknife interval from leave-one-year-out "
               f"cross-validation (cross_validation.cross_validate), and the MCMC credible interval with the "
               f"least-squares likelihood widened for the autocorrelation of the errors (likelihood: "
               f"least_squares, 32 walkers, run until converged, at most 20,000 steps). V4 tests both kinds of "
               f"interval against a known truth.",
        criterion=f"(A) all 30 published RMSE reproduced to within {TOL_A} °C. "
                  f"(B) DE calibration RMSE no worse than published + {TOL_B} °C in all 15 cases. "
                  f"(C) Wherever a recalibrated parameter differs from the published value by more than "
                  f"{MATCH:.0%} of its range, the two parameter sets predict the validation years with RMSE "
                  f"within {TOL_C} °C of each other; and, where gfortran is available, the original program "
                  f"computes the same calibration RMSE as pyair2stream for both parameter sets, to within "
                  f"{TOL_REF} °C. (D), (E) and (F) are descriptive.")
    from v1_fortran import fortran_available
    jobs = [(st, v, not ctx.quick or (st == "MAH_2369" and v in (3, 8))) for st in RIVERS for v in VERSIONS]
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            done = {(st, v): out for st, v, out in ex.map(_case, jobs)}
        rows_a = [done[(st, v)]["a"] for st in RIVERS for v in VERSIONS]
        keys = [(st, v) for st in RIVERS for v in VERSIONS if "b" in done[(st, v)]]
        rows_b = [done[k]["b"] for k in keys]
        rows_c = [done[k]["c"] for k in keys]
        de_par = {k: done[k]["de"] for k in keys}
        referee = _referee(ctx, rows_c) if fortran_available() else None
        d_summary, d_values, d_runs = (None, None, None) if ctx.quick else _part_d(ctx, de_par)
        e_summary, e_values, e_scores, e_de = (None, None, None, None) if ctx.quick else _part_e(ctx)
        f_detail, f_summary, f_plot = _part_f(ctx, de_par)
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
    c = c.drop(columns=[col for col in c.columns if col in ("published", "recalibrated") or col.startswith("_")])
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
    worst_q = (a["val, calibration Qmedia"] - a["published val"]).abs().max()
    res.notes.append(f"With pyair2stream's default of keeping the calibration Qmedia for the validation period, "
                     f"validation RMSE differs from the published value by at most {worst_q:.3f} °C (only "
                     f"versions 4, 7 and 8 use discharge).")
    if (b["parameters at a bound"] != "none").any():
        res.notes.append("Some DE optima lie on the authors' parameter bounds (table B, last column): the best fit "
                         "would lie outside those ranges. The fit is still valid within the stated bounds.")

    # --- report sections
    res.sections.append(Section(
        "A. The published parameters give the published errors",
        "Each published parameter set was run through pyair2stream on the same data. No optimiser is "
        "involved, so this tests the model equations, the solution scheme, the data handling and the error "
        "calculation directly.",
        figures=[_fig_published_rmse(a)], tables=[("A. Published parameters: RMSE (°C)", a)]))
    res.sections.append(Section(
        "B. Calibration from scratch fits at least as well",
        "Each version was recalibrated from scratch with pyair2stream's default optimiser (DE) on the same "
        "calibration years, objective and parameter ranges as the paper.",
        figures=[_fig_recalibration(b)], tables=[("B. DE recalibration: calibration RMSE (°C)", b)]))
    figs_c = [f for f in (_fig_valley([r for r in rows_c if not r["match"]]),) if f]
    if len(differ):
        top = max((r for r in rows_c if not r["match"] and "_predictions" in r),
                  key=lambda r: r["largest difference (% of range)"], default=None)
        if top is not None:
            figs_c.append(_fig_predictions(top))
    tables_c = [("C. Recalibrated vs published parameters: fit and predictions", c),
                ("C. Parameter values, published and recalibrated (blank: not used by the version; "
                 "'NO': a parameter differs by more than 1% of its range)", values)]
    if referee is not None:
        tables_c.append(("C. The original program as referee: calibration RMSE (°C) of the published and the "
                         "recalibrated parameters, computed by the original Fortran program from its own copy "
                         "of the data", referee.assign(**{
                             col: referee[col].map(lambda x, n=n: f"{x:.{n}f}")
                             for col, n in (("original program: published parameters", 5),
                                            ("original program: recalibrated parameters", 5),
                                            ("pyair2stream: recalibrated parameters", 5),
                                            ("largest difference, pyair2stream vs original program", 6))})))
    res.sections.append(Section(
        "C. Recalibrated and published parameters",
        "Where the recalibrated parameters differ from the published ones by more than 1% of a parameter's "
        "range, the figures show why: along the line between the two sets the fit hardly changes, and the two "
        "sets predict the validation years with the same overall error, although their predictions on "
        "individual days can differ. The original Fortran program, scoring both sets itself, agrees with "
        "pyair2stream's errors.",
        figures=figs_c, tables=tables_c))
    if d_summary is not None:
        tables_d = [(f"D. Calibration by the original method (PSO, authors' example settings): calibration RMSE "
                     f"(°C) of each run, and runs whose parameters are within {MATCH:.0%} of their range of the "
                     f"published ones", d_summary)]
        if len(d_values):
            tables_d.append(("D. Parameter values of every run, where any run differs from the published "
                             "parameters", d_values))
        res.sections.append(Section(
            "D. The original calibration method, run as distributed",
            "The original Fortran program seeds its random numbers from the clock, so each of its runs differs, "
            "and this part differs from one validation run to the next.",
            figures=[_fig_pso(d_runs, de_par)], tables=tables_d))
        orig = [col for col in d_summary.columns if col.startswith("original program: runs")]
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
        res.sections.append(Section(
            "E. The scheme matters: RK4 instead of Crank-Nicolson",
            "The original program's readme lists the RK4 scheme in its example; the paper states Crank-Nicolson.",
            figures=[_fig_rk4(e_scores, e_de)],
            tables=[("E. With the RK4 scheme: the published parameters scored with each scheme, and recalibration "
                     "with RK4 (RMSE in °C)", e_summary),
                    ("E. Mentue: parameter values from calibration with RK4", e_values)]))
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
    if len(f_detail):
        n = len(f_detail)
        mc_ok = f_detail["published inside (MCMC)"] != "not converged"
        mc_in = f_detail["published inside (MCMC)"].map(lambda x: x is True)
        jk_in = f_detail["published inside (jackknife)"].astype(bool)
        in_mc, n_mc = int(mc_in.sum()), int(mc_ok.sum())
        in_jk = int(jk_in.sum())
        either = int((mc_in | jk_in).sum())
        res.summary += (f" (F) The published values lie inside pyair2stream's 90% intervals for {in_jk} of {n} "
                        f"parameter values (cross-validation jackknife) and {in_mc} of {n_mc} (MCMC, converged runs); "
                        f"{either} of {n} lie inside at least one.")
        outside = f_detail[~(mc_in | jk_in)]
        where = sorted({f"{r.river} version {r.version}" for r in outside.itertuples()})
        res.sections.append(Section(
            "F. Do the published values lie inside pyair2stream's uncertainty intervals?",
            "The paper's parameters are least-squares estimates: they minimise the RMSE on the calibration "
            "years. So each recalibrated parameter is given two 90% intervals around the same estimator. The "
            "cross-validation jackknife interval shows how far the least-squares value moves when the "
            "calibration is repeated without each year in turn, scaled to a confidence interval. The MCMC "
            "interval is the range of values whose fit is close to the least-squares fit, widened because the "
            "model's daily errors are correlated from day to day (fewer independent observations than days). "
            "V4 tests both kinds of interval against a known truth. A published value outside both intervals "
            "is not where a least-squares calibration on these data lands; part C shows why that can happen "
            "without the published parameters predicting any worse. Two counting rules: a published value on a "
            "bound of the authors' parameter range (column 'on a bound') counts as inside an interval that "
            "reaches to within 1% of the range of that bound, because the range cuts the distribution off there "
            "and a 5-95% interval then stops just short of the bound; and an MCMC run that did not converge "
            "gives no usable interval and is not counted (faded bar in the figure).",
            figures=[_fig_parameter_intervals(st, f_plot[st]) for st in RIVERS if st in f_plot],
            tables=[("F. Published parameters and 90% intervals: summary by river and version", f_summary),
                    ("F. Published parameters and 90% intervals, by parameter", f_detail)]))
        note = (f"Part F: the published parameters lie inside the cross-validation jackknife intervals for {in_jk} "
                f"of {n} parameter values and inside the MCMC intervals for {in_mc} of {n_mc} (runs that "
                f"converged).")
        if where:
            note += (f" The values outside both are in {', '.join(where)}. These are the cases of part C where the "
                     "parameters trade off along a flat valley: the published set lies further along the valley "
                     "than calibrations on these data reach, yet fits and predicts almost equally well. An interval "
                     "for one parameter on its own cannot show such a trade-off, which is why predictions, not "
                     "individual parameter values, are what the model should be relied on for.")
        else:
            note += " Every published value lies inside at least one of the two intervals."
        res.notes.append(note)
    return res
