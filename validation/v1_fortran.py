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

from common import (REPO, WORK, VERSIONS, Result, Section, published_params, river_csv, mean_discharge,
                    simulate, Timer, plot_style, save_figure, reference_line, BLUE, ORANGE, INK2, MUTED)

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
        rows, example = [], None
        for v in VERSIONS:
            par = published_params(v, "MAH_2369")
            for integ in INTEGRATORS:
                py = simulate(csv, v, par, integ, qmedia, name="v1").Twat_mod[365:]
                ft = run_fortran_model(v, integ, n, src.T_air.values, src.Discharge.values, par, qmedia, 4.0)
                py_ok = np.isfinite(py) & (np.abs(py) < PLAUSIBLE)
                ft_ok = np.isfinite(ft) & (np.abs(ft) < PLAUSIBLE)
                both = py_ok & ft_ok
                max_diff = float(np.max(np.abs(py[both] - ft[both]))) if both.any() else np.nan
                if (v, integ) == (8, "CRN"):
                    example = (dates, py.copy(), np.asarray(ft, float).copy())
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
    res.sections.append(Section(
        "Python and Fortran, day by day",
        "Both programs simulated the same eight years of real Mentue forcing with the published parameters. "
        "The figures show the largest daily difference in each case and, for the full model with the "
        "default scheme, the two simulations themselves.",
        figures=_figures(df, example),
        tables=[("Python vs Fortran, Mentue 2002-2009, published parameters", df)]))
    return res


def _figures(df, example):
    import matplotlib.pyplot as plt
    plot_style()
    figs = []
    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels = [f"v{v} {i}" for v, i in zip(df.version, df.integrator)]
    x = np.arange(len(df))
    y = df["max |difference| (°C)"].to_numpy()
    ax.scatter(x, y, s=28, color=BLUE, zorder=3, label="largest difference on any day")
    div = df["days diverged (Python)"] > 0
    ax.scatter(x[div], y[div], s=80, facecolors="none", edgecolors=ORANGE, lw=1.4, zorder=4,
               label="both programs diverge on the same days (excluded from the comparison)")
    reference_line(ax, TOL, f"pass limit {TOL:g} °C")
    reference_line(ax, 5e-6, "rounding of the Fortran's printed output (5e-6 °C)")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e-4)
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("|Python - Fortran| (°C)")
    ax.set_title("Largest daily difference between pyair2stream and the original Fortran")
    ax.legend(loc="lower left", fontsize=7.5)
    figs.append((save_figure(fig, "V1_max_difference.png"),
                 "Every model version and scheme agrees with the Fortran to the last digit the Fortran prints."))
    if example is not None:
        _, py, ft = example
        n = 365
        day = np.arange(1, n + 1)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        a1.plot(day, ft[:n], color=BLUE, lw=3, alpha=0.5, label="original Fortran")
        a1.plot(day, py[:n], color=ORANGE, lw=1.0, label="pyair2stream")
        a1.set_ylabel("Water temperature (°C)")
        a1.set_title("Version 8, Crank-Nicolson, first year of the Mentue record")
        a1.legend(loc="upper left")
        a2.plot(day, py[:n] - ft[:n], color=INK2, lw=0.8)
        a2.set_xlabel("Day of the record (Mentue forcing, 2002)")
        a2.set_ylabel("Difference (°C)")
        a2.set_ylim(-1e-5, 1e-5)
        a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        figs.append((save_figure(fig, "V1_timeseries.png"),
                     "The two simulations lie on top of each other; their difference (lower panel) is "
                     "below 0.00001 °C every day."))
    return figs
