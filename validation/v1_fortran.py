"""
V1 - Same results as the original Fortran on real inputs.

The unit tests compare against the Fortran on a synthetic series with constant
discharge, which leaves the discharge terms (theta**a4, a5, a8) untested. This
check uses real, variable Swiss discharge and air temperature.
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd

from common import (REPO, WORK, VERSIONS, Result, published_params, river_csv, mean_discharge,
                    simulate, Timer)

INTEGRATORS = ("EUL", "RK2", "RK4", "CRN")     # the ones the Fortran implements
TOL = 1e-5                                     # °C; the Fortran prints 6 decimals
PLAUSIBLE = 100.0                              # beyond this a run has diverged


def fortran_available() -> bool:
    return shutil.which("gfortran") is not None and os.path.exists(
        os.path.join(REPO, "fortran", "upstream", "src", "AIR2STREAM_MAIN.f90"))


def run(ctx) -> Result:
    res = Result(
        code="V1", title="Same results as the original Fortran",
        question="Does pyair2stream compute the same daily water temperatures as the original Fortran "
                 "air2stream when both are given real, variable inputs?",
        method="The Fortran source (git submodule) is compiled with gfortran. For each model version and "
               "each integrator the Fortran implements, both programs simulate the Mentue calibration "
               "record (8 years of real air temperature and discharge) with the published parameters "
               "(Piccolroaz et al., 2016). Daily outputs are compared.",
        criterion=f"On every day where both stay below {PLAUSIBLE:.0f} °C, they differ by at most {TOL:g} °C; "
                  "and they diverge (exceed it) on exactly the same days.")
    if not fortran_available():
        res.summary = "Skipped: gfortran or the Fortran submodule is not available."
        return res

    sys.path.insert(0, REPO)
    from tests.fortran_runner import run_fortran_model

    with Timer() as t:
        src = pd.read_csv(river_csv("MAH_2369", "calibration"))
        n = len(src)
        # The Fortran runner writes its input starting 2000-01-01 with only the first
        # day's water temperature (4 °C); the Python run is given exactly the same.
        dates = pd.date_range("2000-01-01", periods=n, freq="D")
        csv = os.path.join(WORK, "v1_input.csv")
        os.makedirs(WORK, exist_ok=True)
        tw = np.full(n, np.nan)
        tw[0] = 4.0
        pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "T_air": src.T_air, "T_water": tw,
                      "Discharge": src.Discharge}).to_csv(csv, index=False)
        qmedia = mean_discharge(csv)
        rows = []
        for v in VERSIONS:
            par = published_params(v, "MAH_2369")
            for integ in INTEGRATORS:
                py = simulate(csv, v, par, integ, qmedia, name="v1").Twat_mod[365:]
                ft = run_fortran_model(v, integ, n, src.T_air.values, src.Discharge.values, par, qmedia, 4.0)
                py_ok = np.isfinite(py) & (np.abs(py) < PLAUSIBLE)
                ft_ok = np.isfinite(ft) & (np.abs(ft) < PLAUSIBLE)
                both = py_ok & ft_ok
                max_diff = float(np.max(np.abs(py[both] - ft[both]))) if both.any() else np.nan
                rows.append({"version": v, "integrator": integ, "days compared": int(both.sum()),
                             "days diverged (Python)": int((~py_ok).sum()),
                             "days diverged (Fortran)": int((~ft_ok).sum()),
                             "same divergence days": bool(np.array_equal(py_ok, ft_ok)),
                             "max |difference| (°C)": max_diff})
    df = pd.DataFrame(rows)
    ok = (df["same divergence days"] & ((df["max |difference| (°C)"] <= TOL) | (df["days compared"] == 0))).all()
    res.passed = bool(ok)
    res.seconds = t.seconds
    n_div = int((df["days diverged (Python)"] > 0).sum())
    res.summary = (f"All {len(df)} version × integrator cases agree to within "
                   f"{df['max |difference| (°C)'].max():.1e} °C" if ok else "Some cases disagree (see table).")
    if n_div:
        res.summary += (f"; in {n_div} cases (explicit integrators) both programs diverge on the same days "
                        "with these parameters, which were calibrated with CRN (see V6).")
    res.tables.append(("Python vs Fortran, Mentue 2002-2009, published parameters", df))
    return res
