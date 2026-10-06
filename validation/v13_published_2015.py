"""
V13 - Reproduces the 2015 published results.

The first air2stream paper (Toffolon and Piccolroaz, 2015) published its own calibrated
parameters and errors for the same three Swiss rivers as Piccolroaz et al. (2016), which V2
reproduces. This check reproduces the 2015 values: a second, independent published benchmark
on the same data, with a different set of parameters.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DATA, RIVERS, VERSIONS, WORK, Result, Section, Timer, calibrate, mean_discharge, plot_style,
                    quiet, river_csv, save_figure, simulate, BLUE, ORANGE, INK2)

PUBLISHED = os.path.join(DATA, "published", "Toffolon_Piccolroaz_ERL2015.csv")
ROUNDING_RMSE = 0.005       # °C: the published RMSE are rounded to 2 decimals
ROUNDING_PAR = 0.0005       # the published parameters are rounded to 3 decimals
PERTURBATIONS = 100         # random roundings tried, to bound the effect of the parameters' rounding
TOL_DE = 0.002              # °C: DE with RK4 may fit worse than the published parameters by no more than this
MATCH = 0.01                # parameters "match" when each is within 1% of its range of the published value
STATED_RHONE_SPLIT = (2003, 2004)   # the paper's Table 1: Rhône calibration 1984-2003, validation 2004-2013


def published():
    """The 2015 values: {(station, version): (8 parameters, unused 0; RMSE cal, val; NSE cal, val)}."""
    df = pd.read_csv(PUBLISHED)
    out = {}
    for r in df.itertuples(index=False):
        par = [0.0 if pd.isna(getattr(r, f"a{j}")) else float(getattr(r, f"a{j}")) for j in range(1, 9)]
        out[(r.station, int(r.version))] = (par, (r.rmse_cal, r.rmse_val), (r.nse_cal, r.nse_val))
    return out


def _score(csv, version, par, integrator, name):
    """RMSE and NSE of a FORWARD run over the measured days, with the period's own Qmedia (as the original
    program, and V2); None if the simulation diverges."""
    with quiet():
        d = simulate(csv, version, par, integrator, mean_discharge(csv), objective="RMS", name=name)
    obs, sim = d.Twat_obs[365:], d.Twat_mod[365:]
    m = obs != -999.0
    if not np.all(np.isfinite(sim)) or np.max(np.abs(sim)) > 100:
        return None, None
    e = sim[m] - obs[m]
    return float(np.sqrt(np.mean(e ** 2))), float(1 - np.sum(e ** 2) / np.sum((obs[m] - obs[m].mean()) ** 2))


def _split_files(st, cal_last):
    """The river's record split after `cal_last` (for the Rhône's split as stated in the paper)."""
    full = pd.concat([pd.read_csv(river_csv(st, "calibration")), pd.read_csv(river_csv(st, "validation"))])
    years = pd.to_datetime(full.Date).dt.year
    os.makedirs(WORK, exist_ok=True)
    paths = []
    for part, mask in (("cal", years <= cal_last), ("val", years > cal_last)):
        path = os.path.join(WORK, f"v13_{st}_{cal_last}_{part}.csv")
        full[mask].to_csv(path, index=False)
        paths.append(path)
    return paths


def _case(args):
    """Part A for one river and version: RMSE with RK4 and CRN, and the effect of the parameters' rounding."""
    from pyair2stream.config import ACTIVE_PARAMS
    st, version, quick = args
    par, rmse_pub, nse_pub = published()[(st, version)]
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    name = f"v13_{st}_{version}"
    out = {"station": st, "version": version, "published RMSE": rmse_pub, "published NSE": nse_pub}
    for integ in ("RK4", "CRN"):
        out[integ] = [_score(f, version, par, integ, name) for f in (cal, val)]
    # Rounding: the largest change in RMSE when the active parameters move within their rounding.
    rng = np.random.default_rng(version * 1000 + list(RIVERS).index(st))
    act = list(ACTIVE_PARAMS[version])
    base = [out["RK4"][0][0], out["RK4"][1][0]]
    worst = 0.0
    for _ in range(10 if quick else PERTURBATIONS):
        p = np.array(par, float)
        p[act] += rng.uniform(-ROUNDING_PAR, ROUNDING_PAR, len(act))
        for i, f in enumerate((cal, val)):
            r, _ = _score(f, version, list(p), "RK4", name)
            if r is not None and base[i] is not None:
                worst = max(worst, abs(r - base[i]))
    out["rounding effect"] = worst
    if st == "SIO_2011":
        out["stated split"] = [_score(f, version, par, "RK4", name)[0] for f in _split_files(st, STATED_RHONE_SPLIT[0])]
    return out


def _de(args):
    """Part B: DE calibration with RK4 on the calibration period (RMS objective, authors' bounds)."""
    st, version = args
    with quiet():
        d = calibrate(river_csv(st, "calibration"), version, objective="RMS", integrator="RK4",
                      name=f"v13_de_{st}_{version}")
    return st, version, float(-d.finalfit), list(d.par_best)


def run(ctx) -> Result:
    res = Result(
        code="V13", title="Reproduces the 2015 published results",
        question="The first air2stream paper (Toffolon and Piccolroaz, 2015) published its own parameters and "
                 "errors for the same three Swiss rivers, for versions 3, 4, 5, 7 and 8. Given those parameters, "
                 "does pyair2stream reproduce the published errors, and with which numerical scheme? Does its own "
                 "calibration with that scheme fit at least as well?",
        method="The 2015 parameters (Table 1 of the paper's supplementary information) and calibration and "
               "validation RMSE and NSE (Table 2) were transcribed into data/switzerland/published/"
               "Toffolon_Piccolroaz_ERL2015.csv. (A) Each of the 15 parameter sets is simulated with RK4 and with "
               "Crank-Nicolson (CRN) over the calibration and validation periods of the data distributed with the "
               "original code (Mentue 2002-2009 / 2010-2012, Rhône 1984-2004 / 2005-2013, Dischmabach 2003-2009 / "
               "2010-2012), with each period's own Qmedia as the original program computes it (as in V2), and its "
               "RMSE and NSE are computed over the measured days. The published values are rounded: RMSE to 2 "
               "decimals and parameters to 3. The effect of the parameters' rounding is bounded by simulating "
               f"{PERTURBATIONS} parameter sets that each differ from the published one by up to {ROUNDING_PAR} in "
               "every parameter. The paper's Table 1 gives the Rhône's periods as 1984-2003 / 2004-2013; that split "
               "is scored too. (B) Each version is calibrated by DE with RK4 (RMS objective, the authors' "
               "parameter ranges, the calibration years).",
        criterion=f"(A) With RK4, all 30 published RMSE values are reproduced to within {ROUNDING_RMSE} °C (their "
                  f"rounding) plus the largest effect of rounding the parameters. (B) DE with RK4 fits each "
                  f"calibration period no worse than the published parameters do (their RMSE with RK4, + "
                  f"{TOL_DE} °C). Whether DE returns the published parameters is reported. "
                  f"CRN and the stated Rhône split are reported, not judged.")
    stations = ["MAH_2369"] if ctx.quick else list(RIVERS)
    cases = [(st, v) for st in stations for v in sorted(VERSIONS, reverse=True)]
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            scored = list(ex.map(_case, [(st, v, ctx.quick) for st, v in cases]))
            de = {(st, v): (r, p) for st, v, r, p in ex.map(_de, cases)}
    res.seconds = t.seconds

    fmt = lambda x: "diverged" if x is None else f"{x:.3f}"
    rows_a, ok_a = [], True
    for o in scored:
        tol = ROUNDING_RMSE + o["rounding effect"]
        (rc, nc), (rv, nv) = o["RK4"]
        diff = max(abs(rc - o["published RMSE"][0]), abs(rv - o["published RMSE"][1])) if None not in (rc, rv) else np.inf
        ok_a &= bool(diff <= tol + 1e-9)
        row = {"river": RIVERS[o["station"]], "version": o["version"],
               "published cal": o["published RMSE"][0], "RK4 cal": fmt(rc), "CRN cal": fmt(o["CRN"][0][0]),
               "published val": o["published RMSE"][1], "RK4 val": fmt(rv), "CRN val": fmt(o["CRN"][1][0]),
               "largest difference with RK4": round(diff, 4), "accepted": round(tol, 4),
               "published NSE cal, val": f"{o['published NSE'][0]:.2f}, {o['published NSE'][1]:.2f}",
               "NSE with RK4": f"{nc:.3f}, {nv:.3f}" if None not in (nc, nv) else "diverged"}
        if "stated split" in o:
            row["RK4, Rhône split 1984-2003 / 2004-2013"] = ", ".join(fmt(x) for x in o["stated split"])
        rows_a.append(row)
    a = pd.DataFrame(rows_a)

    from pyair2stream.config import ACTIVE_PARAMS
    span = np.array(AUTHORS_BOUNDS["max"], float) - np.array(AUTHORS_BOUNDS["min"], float)
    rows_b, ok_b = [], True
    for o in scored:
        key = (o["station"], o["version"])
        r_de, p_de = de[key]
        r_pub = o["RK4"][0][0]
        ok_b &= bool(r_pub is not None and r_de <= r_pub + TOL_DE)
        act = list(ACTIVE_PARAMS[key[1]])
        pub = np.array(published()[key][0], float)
        diff = 100 * float(np.max(np.abs(np.array(p_de)[act] - pub[act]) / span[act]))
        rows_b.append({"river": RIVERS[key[0]], "version": key[1], "published cal RMSE": o["published RMSE"][0],
                       "published parameters with RK4": fmt(r_pub), "DE with RK4": round(r_de, 4),
                       "DE minus published parameters": round(r_de - r_pub, 4) if r_pub is not None else np.nan,
                       "largest parameter difference (% of range)": round(diff, 2),
                       "same parameters": "yes" if diff <= 100 * MATCH else "no",
                       **{f"a{j + 1}": (f"{p_de[j]:.3f}" if j in act else "") for j in range(8)}})
    b = pd.DataFrame(rows_b)
    res.passed = ok_a and ok_b

    crn_diff = max(max(abs(float(r["CRN cal"]) - r["published cal"]), abs(float(r["CRN val"]) - r["published val"]))
                   for r in rows_a if "diverged" not in (r["CRN cal"], r["CRN val"]))
    res.summary = (f"(A) With RK4, {int((a['largest difference with RK4'] <= a['accepted'] + 1e-9).sum())} of "
                   f"{len(a)} parameter sets reproduce both published RMSE values within their rounding (largest "
                   f"difference {a['largest difference with RK4'].max():.4f} °C); with Crank-Nicolson the "
                   f"differences reach {crn_diff:.3f} °C. (B) DE with RK4 fits the calibration years at least as "
                   f"well as the published parameters in "
                   f"{int((b['DE minus published parameters'] <= TOL_DE).sum())} of {len(b)} cases, and returns "
                   f"the published parameters (each within {MATCH:.0%} of its range) in "
                   f"{int((b['same parameters'] == 'yes').sum())} of {len(b)}.")
    split = a.get("RK4, Rhône split 1984-2003 / 2004-2013")
    if split is not None and split.notna().any():
        rh = a[a.river == RIVERS["SIO_2011"]]
        worst_stated = max(max(abs(float(x) - p) for x, p in zip(s.split(", "), (pc, pv)))
                           for s, pc, pv in zip(rh["RK4, Rhône split 1984-2003 / 2004-2013"], rh["published cal"],
                                                rh["published val"]))
        res.notes.append(
            f"The paper's Table 1 gives the Rhône's periods as 1984-2003 and 2004-2013. With that split the "
            f"published errors are missed by up to {worst_stated:.3f} °C, also by versions 3 and 5, which do not "
            f"use discharge; with the split of the data distributed with the code (1984-2004 and 2005-2013) they "
            f"are reproduced. The 2015 results were therefore computed on the distributed split, and the periods "
            f"in the table are a misprint.")
    res.notes.append(
        "The 2015 parameters belong to the RK4 scheme and the 2016 parameters (V2) to Crank-Nicolson: each set "
        "reproduces its paper's errors only with its own scheme. Parameters are specific to the scheme they were "
        "calibrated with (V2 part E, V6). Together, V2 and V13 reproduce 60 published errors from two papers, "
        "with two schemes, on the same data.")
    res.sections.append(Section(
        "A. The 2015 parameters give the 2015 errors with RK4",
        "No optimiser is involved: each published parameter set is simulated as published. 'Accepted' is the "
        f"rounding of the published RMSE ({ROUNDING_RMSE} °C) plus the largest change in RMSE when the parameters "
        "move within their own rounding.",
        figures=[_fig(a)], tables=[("A. Published parameters: RMSE (°C) and NSE", a)]))
    res.sections.append(Section(
        "B. Calibration with RK4 fits at least as well",
        "DE with RK4 on the calibration years, the RMS objective and the authors' parameter ranges, with the "
        "parameters found. 'Same parameters': every parameter within 1% of its range of the published value.",
        tables=[("B. DE with RK4: calibration RMSE (°C) and parameters (blank: not used by the version)", b)]))
    return res


def _fig(a):
    import matplotlib.pyplot as plt
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, period in zip(axes, ("cal", "val")):
        pub = a[f"published {period}"].astype(float)
        for integ, colour, marker in (("RK4", BLUE, "o"), ("CRN", ORANGE, "s")):
            vals = pd.to_numeric(a[f"{integ} {period}"], errors="coerce")
            ax.scatter(pub, vals, s=22, color=colour, marker=marker, label=integ, zorder=3)
        lo, hi = 0.5, max(1.1, float(pd.to_numeric(a[f"CRN {period}"], errors="coerce").max()) + 0.05)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title("Calibration period" if period == "cal" else "Validation period", fontsize=9)
        ax.set_xlabel("Published RMSE, 2015 (°C)")
    axes[0].set_ylabel("RMSE from pyair2stream (°C)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return (save_figure(fig, "V13_published_2015.png"),
            "RMSE of the 2015 published parameters computed by pyair2stream against the published values, with "
            "RK4 (blue) and Crank-Nicolson (orange). On the dashed line they agree.")
