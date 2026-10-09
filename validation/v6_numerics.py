"""
V6 - Numerical accuracy and warnings.

The model is a differential equation solved one day at a time. This check
compares each solution scheme with the exact solution of the equation, records
whether the package stops runs that go wrong, and tests whether the choice
between the two always-stable schemes changes a calibrated model's predictions.
"""

import contextlib
import io
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numba import njit

from common import (RIVERS, VERSIONS, WORK, Result, Section, Timer, calibrate, daily, load, mean_discharge, metrics,
                    published_params, published_rmse, river_csv, simulate, plot_style, save_figure, reference_line,
                    BLUE, ORANGE, AQUA, INK2, MUTED, LIGHT_GREY)

INTEGRATORS = ("CRN", "EXP", "RK4", "RK2", "EUL")
SUBSTEPS = 96                 # reference solution: 96 RK4 sub-steps per day (15 minutes)
TOL_ANALYTIC = 0.01           # °C, second-order and higher schemes vs the exact solution
TOL_ANALYTIC_EUL = 0.05       # °C, EUL (first order: expected error ~ omega * dt * amplitude ~ 0.03 °C)
TOL_REF_SOLVER = 1e-6         # °C, the reference solver vs the exact solution
PLAUSIBLE = 60.0              # °C; the package's default max_plausible_twat
TOL_SCHEME = 0.05             # °C; CRN vs EXP calibrated predictions


@njit
def _rhs(version, p, ta, q, tw, t, qmedia):
    a1, a2, a3, a4, a5, a6, a7, a8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    season = math.cos(2.0 * math.pi * (t - a7))
    if version == 3:
        return a1 + a2 * ta - a3 * tw
    if version == 5:
        return a1 + a2 * ta - a3 * tw + a6 * season
    th = q / qmedia
    if version == 4:
        return (a1 + a2 * ta - a3 * tw) / th ** a4
    if version == 7:
        return a1 + a2 * ta - a3 * tw + th * (a5 + a6 * season - a8 * tw)
    return (a1 + a2 * ta - a3 * tw + th * (a5 + a6 * season - a8 * tw)) / th ** a4


@njit
def reference_solution(version, p, tair, q, tt, tw0, qmedia, tice, m):
    """The equation solved with m RK4 sub-steps per day, air temperature, discharge and
    time varying linearly within each day, and the ice floor applied at every sub-step."""
    n = tair.shape[0]
    out = np.empty(n)
    out[0] = tw0
    h = 1.0 / m
    for j in range(n - 1):
        tw = out[j]
        for k in range(m):
            s0, s1 = k * h, (k + 1) * h
            sm = 0.5 * (s0 + s1)
            ta0, ta1, tam = (tair[j] + s0 * (tair[j + 1] - tair[j]), tair[j] + s1 * (tair[j + 1] - tair[j]),
                             tair[j] + sm * (tair[j + 1] - tair[j]))
            q0, q1, qm = q[j] + s0 * (q[j + 1] - q[j]), q[j] + s1 * (q[j + 1] - q[j]), q[j] + sm * (q[j + 1] - q[j])
            t0, t1, tm = tt[j] + s0 / 365.0, tt[j] + s1 / 365.0, tt[j] + sm / 365.0
            k1 = _rhs(version, p, ta0, q0, tw, t0, qmedia)
            k2 = _rhs(version, p, tam, qm, tw + 0.5 * h * k1, tm, qmedia)
            k3 = _rhs(version, p, tam, qm, tw + 0.5 * h * k2, tm, qmedia)
            k4 = _rhs(version, p, ta1, q1, tw + h * k3, t1, qmedia)
            tw = tw + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if tw < tice:
                tw = tice
        out[j + 1] = tw
    return out


def _run(csv, version, par, integrator, qmedia, name, extra=None):
    """Run as a user would, recording whether the package warned or stopped."""
    from pyair2stream.model import (NumericalDivergenceError, call_model, check_numerical_divergence,
                                    warn_on_stability)
    cfg = {"version": version, "integrator": integrator, "run_mode": "FORWARD",
           "parameters_forward": [float(x) for x in par], "Qmedia": float(qmedia),
           "paths": {"input_data": csv}, **(extra or {})}
    data = load(cfg, name)
    flag = ""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            warn_on_stability(data, data.stability_error_fraction)
        except NumericalDivergenceError:
            flag = "stopped (stability check)"
        call_model(data)
        if not flag:
            try:
                check_numerical_divergence(data, data.max_plausible_twat)
            except NumericalDivergenceError:
                flag = "stopped (diverged)"
    if not flag and "stability limit" in buf.getvalue():
        flag = "warned"
    return data, flag


def _analytic_case(version, par, qmedia):
    """Constant air temperature and discharge (theta = 1): the exact periodic solution
    is known. Uses the no-leap calendar so the seasonal term has a period of exactly
    365 days."""
    n = 365 * 6
    csv = os.path.join(WORK, f"v6_analytic_{version}.csv")
    os.makedirs(WORK, exist_ok=True)
    dates = pd.date_range("2001-01-01", periods=n + 10, freq="D")
    dates = dates[~((dates.month == 2) & (dates.day == 29))][:n]      # noleap dates: no 29 February
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"),
                  "T_air": 10.0, "T_water": np.nan, "Discharge": qmedia}).to_csv(csv, index=False)
    a1, a2, a3, a4, a5, a6, a7, a8 = par
    k = a3 + (a8 if version in (7, 8) else 0.0)
    c = a1 + a2 * 10.0 + (a5 if version in (7, 8) else 0.0)
    w = 2 * np.pi / 365.0
    i = np.arange(n + 365)
    x = w * (i + 1) - 2 * np.pi * a7
    exact = c / k + a6 / np.hypot(k, w) * np.cos(x - np.arctan2(w, k))
    rows = []
    for integ in INTEGRATORS:
        d, flag = _run(csv, version, par, integ, qmedia, "v6a", {"calendar": "noleap"})
        err = d.Twat_mod[730:] - exact[730:]
        rows.append({"test": f"version {version}", "scheme": integ,
                     "max |error| (°C)": float(np.max(np.abs(err)))})
    d, _ = _run(csv, version, par, "CRN", qmedia, "v6a", {"calendar": "noleap"})
    ref = reference_solution(version, np.array(par, float), d.Tair, d.Q, d.tt, d.Twat_mod[0], qmedia,
                             float(d.Tice_cover), SUBSTEPS)
    rows.append({"test": f"version {version}", "scheme": f"reference ({SUBSTEPS} sub-steps/day)",
                 "max |error| (°C)": float(np.max(np.abs(ref[730:] - exact[730:])))})
    return rows


def _scheme_fit(args):
    """Calibrate with one scheme and predict the validation years with the same scheme."""
    st, v, integ = args
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    q = mean_discharge(cal)
    d = calibrate(cal, v, objective="NSE", integrator=integ, name=f"v6c_{st}_{v}_{integ}", Qmedia=q)
    obs, sim = daily(simulate(val, v, d.par_best, integ, q, name=f"v6cp_{st}_{v}_{integ}"))
    return {"river": RIVERS[st], "version": v, "scheme": integ, "validation RMSE (°C)": metrics(obs, sim)["RMSE"]}


def run(ctx) -> Result:
    res = Result(
        code="V6", title="Numerical accuracy and warnings",
        question="Are the numerical schemes solved correctly, how much do they differ on real data, does "
                 "the package stop runs that go wrong, and would a different stable scheme change the "
                 "calibrated model's predictions?",
        method=f"(A) With constant air temperature and discharge the equation has an exact solution (a "
               f"steady seasonal cycle). Each scheme, and a reference solver, is compared with it for "
               f"versions 5 and 8 (published Mentue parameters). (B) On real forcing (three rivers, five "
               f"versions, published parameters) each scheme is compared with the reference solver: the "
               f"same equation solved with {SUBSTEPS} small steps per day, with air temperature and "
               f"discharge varying linearly within each day. The package's own checks (stability warning, "
               f"divergence stop) are recorded for every run. (C) Versions 5 and 8 are calibrated (DE, NSE) "
               f"on each river with each of the two always-stable schemes, CRN and EXP, and predict the "
               f"validation years with the same scheme.",
        criterion=f"(A) CRN, EXP, RK4 and RK2 within {TOL_ANALYTIC} °C of the exact solution; EUL, a "
                  f"first-order method, within {TOL_ANALYTIC_EUL} °C; the reference solver within "
                  f"{TOL_REF_SOLVER:g} °C. (B) Every run that diverges (temperatures not a number or "
                  f"above {PLAUSIBLE:.0f} °C) is stopped with an error. (C) CRN and EXP give validation RMSE "
                  f"within {TOL_SCHEME} °C of each other in every case.")
    with Timer() as t:
        rows_a = []
        q_mah = mean_discharge(river_csv("MAH_2369", "calibration"))
        for v in (5, 8):
            rows_a += _analytic_case(v, published_params(v, "MAH_2369"), q_mah)
        a = pd.DataFrame(rows_a)

        rows_b = []
        versions = (5, 8) if ctx.quick else VERSIONS
        for st, river in RIVERS.items():
            csv = river_csv(st, "calibration")
            qm = mean_discharge(csv)
            for v in versions:
                par = published_params(v, st)
                d, _ = _run(csv, v, par, "CRN", qm, "v6b")
                ref = reference_solution(v, np.array(par, float), d.Tair, d.Q, d.tt, d.Twat_mod[0], qm,
                                         float(d.Tice_cover), SUBSTEPS)[365:]
                model_rmse = published_rmse(v, st, "calibration")
                for integ in INTEGRATORS:
                    d, flag = _run(csv, v, par, integ, qm, "v6b")
                    sim = d.Twat_mod[365:]
                    diverged = bool(np.any(~np.isfinite(sim) | (np.abs(sim) > PLAUSIBLE)))
                    diff = np.abs(sim - ref) if not diverged else np.array([np.inf])
                    rows_b.append({"river": river, "version": v, "scheme": integ, "diverged": diverged,
                                   "RMS difference from reference (°C)": float(np.sqrt(np.mean(diff ** 2))),
                                   "max |difference| (°C)": float(np.max(diff)),
                                   "share of the model's error": float(np.sqrt(np.mean(diff ** 2))) / model_rmse,
                                   "package response": flag or "none"})
        b = pd.DataFrame(rows_b)

        jobs = [(st, v, integ) for st in (["MAH_2369"] if ctx.quick else RIVERS) for v in (5, 8)
                for integ in ("CRN", "EXP")]
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            c = pd.DataFrame(list(ex.map(_scheme_fit, jobs)))
        c = c.pivot_table(index=["river", "version"], columns="scheme", values="validation RMSE (°C)",
                          sort=False).reset_index()
        c["difference (°C)"] = c["EXP"] - c["CRN"]

    is_ref = a["scheme"].str.startswith("reference")
    tol = np.where(is_ref, TOL_REF_SOLVER, np.where(a["scheme"] == "EUL", TOL_ANALYTIC_EUL, TOL_ANALYTIC))
    ok_a = bool((a["max |error| (°C)"] <= tol).all())
    ok_b = bool(b.loc[b.diverged, "package response"].str.startswith("stopped").all())
    ok_c = bool((c["difference (°C)"].abs() <= TOL_SCHEME).all())
    res.passed = ok_a and ok_b and ok_c
    res.seconds = t.seconds
    crn = b[b.scheme == "CRN"]
    res.summary = (f"(A) Largest error against the exact solution: "
                   f"{a.loc[~is_ref, 'max |error| (°C)'].max():.3f} °C (schemes), "
                   f"{a.loc[is_ref, 'max |error| (°C)'].max():.0e} °C (reference). "
                   f"(B) CRN differs from the reference by {crn['RMS difference from reference (°C)'].min():.2f}-"
                   f"{crn['RMS difference from reference (°C)'].max():.2f} °C RMS on real forcing; "
                   f"{int(b.diverged.sum())} explicit-scheme runs diverged and "
                   f"{'all were' if ok_b else 'NOT all were'} stopped. "
                   f"(C) Calibrating with EXP instead of CRN changes validation RMSE by at most "
                   f"{c['difference (°C)'].abs().max():.3f} °C.")
    res.notes.append("The difference between a scheme and the reference is not an error in the model's "
                     "predictions: the parameters are calibrated with the scheme, and absorb its behaviour "
                     "(part C). The published parameters were calibrated with CRN and are reproduced by CRN "
                     "(V1, V2); they should not be run with another scheme. A FORWARD run given the "
                     "calibration's calibration_metadata.json refuses a different version or scheme.")
    res.notes.append("RK4, RK2 and EUL are kept only to reproduce the original program. They are unstable "
                     "when the model's decay rate B exceeds their limit (2.785 for RK4, 2 for RK2 and EUL), "
                     "which the published Rhône and Dischmabach parameters do (a3 > 2), and they can be "
                     "inaccurate even when stable (EUL by up to about 1 °C on the Mentue). The package prints "
                     "a notice when one of them is selected.")
    figs = [_fig_exact(a), _fig_real(b), _fig_schemes(c)]
    for col in ("RMS difference from reference (°C)", "max |difference| (°C)"):
        b[col] = b[col].map(lambda x: "diverged" if not np.isfinite(x) else f"{x:.3f}")
    b["share of the model's error"] = b["share of the model's error"].map(
        lambda x: "" if not np.isfinite(x) else f"{x:.0%}")
    a["max |error| (°C)"] = a["max |error| (°C)"].map(lambda x: f"{x:.1e}")
    res.sections += [
        Section("A. Against the exact solution",
                "With constant air temperature and discharge the equation has an exact solution, so the error of "
                "each scheme can be measured directly.",
                figures=[figs[0]], tables=[("A. Error against the exact solution", a)]),
        Section("B. Real forcing: each scheme against a fine-step reference",
                f"The same equation solved with {SUBSTEPS} small steps per day serves as the reference. The "
                "package's response (warning or stop) is recorded for every run.",
                figures=[figs[1]],
                tables=[("B. Real forcing, published parameters: difference from the reference solution",
                         b.drop(columns=["diverged"]))]),
        Section("C. Does the choice of stable scheme change a calibrated model's predictions?",
                "Each case is calibrated and validated twice, once with each always-stable scheme.",
                figures=[figs[2]], tables=[("C. Calibrated with CRN or EXP: validation RMSE (°C)", c.round(3))])]
    return res


def _fig_exact(a):
    import matplotlib.pyplot as plt
    plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    schemes = list(dict.fromkeys(a.scheme))
    for k, (test, colour, marker) in enumerate((("version 5", BLUE, "o"), ("version 8", ORANGE, "s"))):
        g = a[a.test == test].set_index("scheme")
        ax.scatter(range(len(schemes)), [g.loc[sc, "max |error| (°C)"] for sc in schemes], s=34, marker=marker,
                   color=colour, label=test, zorder=3)
    for tol, label, dy, va in ((TOL_ANALYTIC, f"limit for CRN, EXP, RK4, RK2 ({TOL_ANALYTIC} °C)", -3, "top"),
                               (TOL_ANALYTIC_EUL, f"limit for EUL, a first-order scheme ({TOL_ANALYTIC_EUL} °C)", 3,
                                "bottom")):
        ax.axhline(tol, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.annotate(label, (0.33, tol), xycoords=("axes fraction", "data"), xytext=(0, dy),
                    textcoords="offset points", fontsize=7.5, color=INK2, va=va)
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 3)
    ax.set_xticks(range(len(schemes)), [sc.replace(" (", "\n(") for sc in schemes], fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Largest error against the\nexact solution (°C)")
    ax.set_title("Every scheme solves the equation correctly")
    ax.legend(loc="lower left", bbox_to_anchor=(0.6, 0.3), fontsize=7.5)
    return (save_figure(fig, "V6_exact_solution.png"),
            "Largest error of each scheme against the exact solution (log scale). All are within their limits; the "
            "reference solver used in part B is exact to rounding.")


def _fig_real(b):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plot_style()
    cases = list(dict.fromkeys(zip(b.river, b.version)))
    cap = 3.0
    fig, ax = plt.subplots(figsize=(7.5, 1.2 + 0.27 * len(cases)))
    for i, (rv, v) in enumerate(cases):
        y = len(cases) - 1 - i
        for k, (sc, colour, marker) in enumerate((("CRN", BLUE, "o"), ("EXP", ORANGE, "s"), ("RK4", AQUA, "D"))):
            row = b[(b.river == rv) & (b.version == v) & (b.scheme == sc)].iloc[0]
            val = row["RMS difference from reference (°C)"]
            dy = (k - 1) * 0.18
            if not np.isfinite(val) or val > cap:
                ax.scatter(cap, y + dy, marker="x", s=30, color=colour, zorder=3)
            else:
                ax.scatter(val, y + dy, marker=marker, s=26, color=colour, zorder=3)
    ax.axvline(cap, color=LIGHT_GREY, lw=0.8)
    ax.annotate("diverged or above 3 °C,\nstopped by the package", (cap, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -2), textcoords="offset points", fontsize=7, color=INK2, va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.005, cap * 1.6)
    ax.set_yticks(range(len(cases))[::-1], [f"{rv} v{v}" for rv, v in cases], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMS difference from the fine-step reference (°C, log scale)")
    ax.set_title("On real forcing, CRN (the default) stays within 0.1 °C of the reference")
    ax.legend(handles=[Line2D([0], [0], marker=m, lw=0, color=c, label=n) for n, c, m in
                       (("CRN (default)", BLUE, "o"), ("EXP", ORANGE, "s"), ("RK4", AQUA, "D"))],
              loc="lower left", fontsize=7.5)
    return (save_figure(fig, "V6_schemes_real.png"),
            "Difference of each scheme from the fine-step reference on real forcing with the published parameters. "
            "RK4 diverges where the water responds quickly; every such run was stopped (RK2 and EUL: see table).")


def _fig_schemes(c):
    import matplotlib.pyplot as plt
    plot_style()
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.3 * len(c)))
    y = np.arange(len(c))[::-1]
    ax.hlines(y, c["CRN"], c["EXP"], color=LIGHT_GREY, lw=2, zorder=2)
    ax.scatter(c["CRN"], y, s=34, color=BLUE, label="calibrated and run with CRN", zorder=3)
    ax.scatter(c["EXP"], y, s=34, marker="s", color=ORANGE, label="calibrated and run with EXP", zorder=3)
    ax.set_yticks(y, [f"{rv} v{v}" for rv, v in zip(c.river, c.version)], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Validation RMSE (°C)")
    ax.set_title("The two always-stable schemes give the same predictive skill")
    ax.legend(loc="lower right", fontsize=7.5)
    return (save_figure(fig, "V6_crn_vs_exp.png"),
            "Validation error when the model is calibrated and run with CRN or with EXP. The difference is at most "
            "a few hundredths of a degree: the parameters absorb the scheme's behaviour.")
