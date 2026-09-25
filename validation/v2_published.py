"""
V2 - Reproduces published results (Piccolroaz et al., 2016).

Part A: the published parameters of every air2stream version, run through
pyair2stream, must give the published calibration and validation RMSE.
Part B: calibrating from scratch with DE must fit at least as well as the
published calibration.
"""

import numpy as np
import pandas as pd

from common import (RIVERS, VERSIONS, Result, Timer, calibrate, mean_discharge, params_at_bounds,
                    published_params, published_rmse, river_csv, simulate)

TOL_A = 0.001    # °C; the published RMSE are rounded to 3 decimals
TOL_B = 0.002    # °C; DE may not beat the published fit by more than rounding


def _rmse(data) -> float:
    return float(-data.finalfit) if hasattr(data, "finalfit") else np.nan


def _sim_rmse(csv, v, par, qmedia) -> float:
    d = simulate(csv, v, par, "CRN", qmedia, objective="RMS", name="v2")
    obs, sim = d.Twat_obs[365:], d.Twat_mod[365:]
    m = obs != -999.0
    return float(np.sqrt(np.mean((sim[m] - obs[m]) ** 2)))


def run(ctx) -> Result:
    res = Result(
        code="V2", title="Reproduces published results",
        question="Given the published parameters, does pyair2stream reproduce the published model errors "
                 "for every model version? And does its own calibration fit at least as well?",
        method="Piccolroaz et al. (2016) calibrated every air2stream version (by RMSE, with the "
               "Crank-Nicolson scheme) on three Swiss rivers and published the parameters and the RMSE for "
               "the calibration and validation periods (data/switzerland/published/). (A) Each published "
               "parameter set is simulated with CRN and its RMSE computed. For the validation period the "
               "original program recomputed Qmedia from the validation data, so that is done here too; "
               "pyair2stream by default keeps the calibration Qmedia (reported separately). "
               "(B) Each version is recalibrated from scratch by DE (RMS objective, CRN, the authors' "
               "parameter ranges).",
        criterion=f"(A) all 30 published RMSE reproduced to within {TOL_A} °C. "
                  f"(B) DE calibration RMSE no worse than published + {TOL_B} °C in all 15 cases.")
    rows_a, rows_b = [], []
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
    a, b = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    ok_a = bool((a["max |diff|"] <= TOL_A).all())
    ok_b = bool((b["difference"] <= TOL_B).all())
    res.passed = ok_a and ok_b
    res.seconds = t.seconds
    res.summary = (f"(A) {'All' if ok_a else 'Not all'} {2 * len(a)} published RMSE values reproduced "
                   f"(largest difference {a['max |diff|'].max():.4f} °C). "
                   f"(B) DE matched or improved on the published calibration in "
                   f"{int((b['difference'] <= TOL_B).sum())} of {len(b)} cases "
                   f"(mean difference {b['difference'].mean():+.3f} °C).")
    worst = (a["val, calibration Qmedia"] - a["published val"]).abs().max()
    res.notes.append(f"With pyair2stream's default of keeping the calibration Qmedia for the validation period, "
                     f"validation RMSE differs from the published value by at most {worst:.3f} °C (only "
                     f"versions 4, 7 and 8 use discharge).")
    if (b["parameters at a bound"] != "none").any():
        res.notes.append("Some DE optima lie on the authors' parameter bounds (last column): the best fit "
                         "would lie outside those ranges. The fit is still valid within the stated bounds.")
    res.tables += [("A. Published parameters: RMSE (°C)", a), ("B. DE recalibration: calibration RMSE (°C)", b)]
    res.figure_data = a
    return res
