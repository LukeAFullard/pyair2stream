"""
V19 - Conformal margins: do they make the prediction intervals hold on every river?

V18 found the prediction intervals too narrow on the British Columbia rivers. The package's optional split
conformal margins (cross_validation.conformal_margins, forward_options.conformal_margins; docs/METHODS.md
§13) widen an interval by how far the held-out years' measurements fell outside it. This check tests them
on all 26 real rivers of the suite: the 23 British Columbia streams, where the intervals were too narrow, and
the 3 Swiss rivers, where they held (V5), so a margin there must not make them too wide. (A) Later years
never used for calibration or for the margins; (B) every held-out year, with a margin from the river's
other years. The pass criterion was fixed in this file before the check was first run.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DE_SETTINGS, LEVELS, RIVERS, WORK, Result, Section, Timer, accepted_real, load,
                    mean_discharge, quiet, river_csv, plot_style, save_figure, BLUE, ORANGE, INK2, LIGHT_GREY)
from v15_british_columbia import QUICK_STATIONS, _load
from v18_bc_intervals import (HEAT_DOME, SUMMER, WINDOWS, _held_out_years, _later_years, _resumable,
                              _ranges)

JUDGED_A = (50, 80, 90, 95)                     # as V18: days in the later years
JUDGED_B = (80, 90, 95)                         # as V18: held-out years
MULTI_DAY_RANGE = (0.85, 0.95)                  # 7- and 30-day means inside the 90% interval, as V18
SWISS = tuple(RIVERS)                           # DAV_2327, MAH_2369, SIO_2011
SWISS_MCMC = {"mcmc_walkers": 32, "mcmc_steps": 20000}     # as V5
PREDICTION_LEVEL = 90.0


def _margins(folds):
    """The package's margins at every level, for days, 7-day and 30-day means."""
    from pyair2stream.cross_validation import conformal_margins
    return conformal_margins(folds, levels=LEVELS, windows=tuple(WINDOWS.values()), seed=1)


# --- Switzerland: the same steps as for British Columbia (V18), through the package's own outputs --------

def _swiss_config(st, tag, quick, **extra):
    cal = river_csv(st, "calibration")
    cfg = {"version": 8, "integrator": "CRN", "objective_function": "NSE", "random_seed": 1,
           "Qmedia": mean_discharge(cal), "parameter_bounds": AUTHORS_BOUNDS,
           "optimization": {"n_run": 40, "n_particles": 8} if quick else dict(DE_SETTINGS),
           "uncertainty_options": {"noise_model": "ar1", "prediction_interval": PREDICTION_LEVEL},
           "paths": {"input_data": cal, "output_dir": os.path.join(WORK, tag, "out")}}
    cfg.update(extra)
    return cfg


def _swiss_cv(args):
    """Leave-one-year-out cross-validation of the calibration years, and cv_conformal_margins.csv as a
    cross-validation run writes it."""
    st, quick = args
    from pyair2stream.cross_validation import cross_validate
    from pyair2stream.main import write_conformal_margins
    tag = f"v19_cv_{st}"
    cfg = _swiss_config(st, tag, quick, run_mode="DE",
                        cross_validation={"enabled": True, "unit": "year", "min_train_years": 0,
                                          "skip_first_year": True})
    data = load(cfg, tag)
    if quick:   # the last four years only
        data.cross_validation.min_train_years = max(0, len(set(data.date[365:, 0])) - 5)
    with quiet():
        _, folds = cross_validate(data, "DE", return_folds=True)
        write_conformal_margins(data, folds, PREDICTION_LEVEL)
    return {"station": st, "fold results": folds, "margins_csv": os.path.join(data.folder, "cv_conformal_margins.csv")}


def _swiss_mcmc(args):
    st, quick = args
    from pyair2stream.optimization import DE_MCMC_mode
    tag = f"v19_mcmc_{st}"
    cfg = _swiss_config(st, tag, quick, run_mode="DE-MCMC")
    cfg["uncertainty_options"]["strict_convergence"] = False
    cfg["optimization"].update({"mcmc_walkers": 16, "mcmc_steps": 600} if quick else SWISS_MCMC)
    data = load(cfg, tag)
    with quiet():
        DE_MCMC_mode(data, seed=1)
    out = os.path.join(WORK, tag, "out")
    meta = json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))
    return {"station": st, "par_best": [float(x) for x in data.par_best], "chain": os.path.join(out, "MCMC_chain_S_c_1d.csv"),
            "converged": bool(meta["converged"]), "steps": int(meta["steps_run"])}


def _swiss_forward(args):
    """FORWARD run of the validation years with forward_options.conformal_margins, as a user runs it. The
    widened daily interval it writes must equal the package's scenario.conformal_range of its ensemble."""
    st, quick, mcmc, margins_csv = args
    from pyair2stream import scenario
    from pyair2stream.optimization import forward_mode
    tag = f"v19_fwd_{st}"
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    fcfg = {"version": 8, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": mean_discharge(cal),
            "parameters_forward": mcmc["par_best"],
            "uncertainty_options": {"noise_model": "ar1", "save_ensemble": True, "prediction_interval": PREDICTION_LEVEL},
            "forward_options": {"enable_prediction_intervals": True, "mcmc_chain_path": mcmc["chain"],
                                "n_samples": 100 if quick else 1000, "random_seed": 1,
                                "conformal_margins": margins_csv},
            "paths": {"input_data": val, "output_dir": os.path.join(WORK, tag, "out")}}
    data = load(fcfg, tag)
    with quiet():
        forward_mode(data)
    out = os.path.join(WORK, tag, "out")
    ens, dates = scenario.load_ensemble(os.path.join(out, "Forward_Prediction_Ensemble_S_c_1d.npz"))
    env = pd.read_csv(os.path.join(out, "Forward_Prediction_Envelopes_S_c_1d.csv"))
    lo, hi = scenario.conformal_range(ens, PREDICTION_LEVEL, scenario.conformal_margin(margins_csv, PREDICTION_LEVEL),
                                      floor=data.Tice_cover)
    if not (np.allclose(lo, env.Twat_mod_lower_conformal) and np.allclose(hi, env.Twat_mod_upper_conformal)):
        raise RuntimeError(f"{st}: the FORWARD run's widened interval differs from scenario.conformal_range")
    obs = pd.read_csv(val).T_water.to_numpy(float)
    return {"station": st, "ranges": _ranges(ens, obs, dates), "ice_floor": float(data.Tice_cover)}


# --- Scoring -----------------------------------------------------------------------------------------

def _later_counts(ranges, margins, floor, mask=None):
    """{(values, level): (inside without margin, inside with margin, n, mean width without, with)}."""
    from pyair2stream.scenario import widen_range
    out = {}
    for what, w in WINDOWS.items():
        r = ranges[what]
        o = r["obs"]
        sel = np.isfinite(o) & (np.ones(len(o), bool) if mask is None or w > 1 else mask)
        for lev in LEVELS:
            lo, hi = r["lower"][lev][sel], r["upper"][lev][sel]
            m = margins.get((w, lev), np.nan)
            clo, chi = widen_range(lo, hi, m, floor=floor)
            out[(what, lev)] = (int(np.sum((o[sel] >= lo) & (o[sel] <= hi))),
                                int(np.sum((o[sel] >= clo) & (o[sel] <= chi))), int(sel.sum()),
                                float(np.mean(hi - lo)) if sel.any() else np.nan,
                                float(np.mean(chi - clo)) if sel.any() else np.nan)
    return out


def _station_rows(region, st, table, later, converged, steps):
    """Rows of the per-station results: part B from the margins table, part A from the later years."""
    rows = []
    margins = {(int(r.window_days), int(r.level)): float(r.margin) for r in table.itertuples()}
    names = {w: what for what, w in WINDOWS.items()}
    for r in table.itertuples():
        rows.append({"region": region, "station": st, "part": "B", "values": names[int(r.window_days)],
                     "level": int(r.level), "margin": float(r.margin), "held-out years": int(r.held_out_years),
                     "n": int(r.values), "inside before": int(round(r.inside_before * r.values)),
                     "inside after": int(round(r.inside_after * r.values)), "converged": True})
    if later is not None:
        counts = _later_counts(later["ranges"], margins, later["floor"])
        for (what, lev), (k0, k1, n, w0, w1) in counts.items():
            rows.append({"region": region, "station": st, "part": "A", "values": what, "level": lev,
                         "margin": margins.get((WINDOWS[what], lev), np.nan), "n": n, "inside before": k0,
                         "inside after": k1, "width before": w0, "width after": w1, "converged": converged,
                         "steps": steps})
        dates = later["ranges"]["dates"]
        for label, mask in (("summer days", dates.month.isin(SUMMER)),
                            ("heat-dome days", (dates >= pd.Timestamp(HEAT_DOME[0])) & (dates <= pd.Timestamp(HEAT_DOME[1])))):
            if region != "British Columbia" and label == "heat-dome days":
                continue
            k0, k1, n, w0, w1 = _later_counts(later["ranges"], margins, later["floor"], mask=np.asarray(mask))[("days", 90)]
            rows.append({"region": region, "station": st, "part": "A", "values": label, "level": 90,
                         "margin": margins.get((1, 90), np.nan), "n": n, "inside before": k0, "inside after": k1,
                         "width before": w0, "width after": w1, "converged": converged, "steps": steps})
    return rows


def _bc_margins(args):
    st, quick = args
    b, _ = _resumable((_held_out_years, (st, quick)))
    return st, _margins(b["fold results"]), b["fold results"][0].ice_floor


def run(ctx) -> Result:
    res = Result(
        code="V19", title="Conformal margins on 26 real rivers",
        question="The package's optional split conformal margins widen a prediction interval by how far the "
                 "measurements of years held out by cross-validation fell outside it. On the 23 British Columbia "
                 "rivers, where V18 found the intervals too narrow, do the widened intervals hold at their stated "
                 "levels, for days, 7-day and 30-day means, in later years and in held-out years? On the 3 Swiss "
                 "rivers, where the intervals already held (V5), do they still hold, rather than become too wide?",
        method="For each river, the package's leave-one-year-out cross-validation of the calibration years (DE, "
               "every year but the first held out once) gives the margins (cross_validation.conformal_margins, as "
               "written to cv_conformal_margins.csv): each held-out value is scored by how far it lies outside the "
               "central range of 1000 simulations of its fold (fold parameters plus AR(1) error with the training "
               "years' sigma and rho), and the margin at level L is the L% quantile of the scores, each held-out "
               "year weighted equally; for days, 7-day and 30-day moving means, at 50, 80, 90, 95 and 99%. "
               "(A) Later years: the central ranges of a FORWARD run of 1000 series from a DE-MCMC chain on the "
               "calibration years (British Columbia: V18's runs, 2021-2022; Switzerland: version 8, NSE, the "
               "authors' parameter ranges, 32 walkers, at most 20,000 steps, as V5, then the validation years, "
               "2010-2012 or 2005-2013, with forward_options.conformal_margins), widened by the margins from all "
               "held-out years, never below Tice_cover (scenario.conformal_range). Nothing of the later years is "
               "used. (B) Held-out years: each held-out year's values against the ranges of its fold, widened by "
               "a margin from the river's other held-out years only (inside_after in cv_conformal_margins.csv). "
               "Shares are pooled over rivers, weighted by the number of values, separately for each region. The "
               "Swiss FORWARD runs check that the widened interval the package writes equals "
               "scenario.conformal_range of its ensemble.",
        criterion="Fixed before the check was first run. With the margins, in each region (British Columbia; "
                  "Switzerland): (A) for the rivers whose DE-MCMC converged, pooled, the share of later-year days "
                  f"inside the interval is within the accepted range at {', '.join(f'{x}%' for x in JUDGED_A)} (a "
                  "miss rate between half and 1.5 times the stated one, as V5 and V18), and the share of 7-day and "
                  f"of 30-day means inside the 90% interval is between {MULTI_DAY_RANGE[0]:.0%} and "
                  f"{MULTI_DAY_RANGE[1]:.0%}; (B) pooled over every held-out year, days, 7-day and 30-day means "
                  f"inside the {', '.join(f'{x}%' for x in JUDGED_B)} intervals are within the accepted range. The "
                  "same criterion as V18, so that the two can be compared. The coverage without margins, 99% "
                  "intervals, single rivers, unconverged rivers, summer and the heat dome are reported, not judged.")
    _, par_table, _ = _load()
    bc = list(QUICK_STATIONS) if ctx.quick else list(par_table.index)
    swiss = ["MAH_2369"] if ctx.quick else list(SWISS)
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            f_bc_a = {st: ex.submit(_resumable, (_later_years, (st, "weekly", ctx.quick))) for st in bc}
            f_cv = {st: ex.submit(_swiss_cv, (st, ctx.quick)) for st in swiss}
            f_mc = {st: ex.submit(_swiss_mcmc, (st, ctx.quick)) for st in swiss}
            f_bc_b = {st: ex.submit(_bc_margins, (st, ctx.quick)) for st in bc}
            cv = {st: f.result() for st, f in f_cv.items()}
            mc = {st: f.result() for st, f in f_mc.items()}
            f_fw = {st: ex.submit(_swiss_forward, (st, ctx.quick, mc[st], cv[st]["margins_csv"])) for st in swiss}
            f_sw_b = {st: ex.submit(_margins, cv[st]["fold results"]) for st in swiss}
            bc_a = {st: f.result()[0] for st, f in f_bc_a.items()}
            bc_b = {st: f.result() for st, f in f_bc_b.items()}
            fw = {st: f.result() for st, f in f_fw.items()}
            sw_b = {st: f.result() for st, f in f_sw_b.items()}
    res.seconds = t.seconds

    rows = []
    for st in bc:
        _, table, floor = bc_b[st]
        a = bc_a[st]
        rows += _station_rows("British Columbia", st, table, {"ranges": a["ranges"], "floor": floor},
                              a["converged"], a["steps"])
    for st in swiss:
        rows += _station_rows("Switzerland", RIVERS[st], sw_b[st],
                              {"ranges": fw[st]["ranges"], "floor": fw[st]["ice_floor"]},
                              mc[st]["converged"], mc[st]["steps"])
    df = pd.DataFrame(rows)
    pool_conv = not ctx.quick           # short quick-run chains never converge: the quick run pools every river

    def pooled(region, part):
        g = df[(df.region == region) & (df.part == part)]
        if part == "A" and pool_conv:
            g = g[g.converged.astype(bool)]
        out = []
        for (what, lev), h in g.groupby(["values", "level"], sort=False):
            n = h.n.sum()
            row = {"region": region, "part": part, "values": what, "level": lev, "rivers": h.station.nunique(),
                   "n": int(n), "inside, no margin": h["inside before"].sum() / n if n else np.nan,
                   "inside, with margin": h["inside after"].sum() / n if n else np.nan,
                   "median margin (°C)": float(np.median(h.margin))}
            if part == "A":
                row["mean width, no margin (°C)"] = float(np.average(h["width before"], weights=h.n)) if n else np.nan
                row["mean width, with margin (°C)"] = float(np.average(h["width after"], weights=h.n)) if n else np.nan
            out.append(row)
        return pd.DataFrame(out)

    regions = ("British Columbia", "Switzerland")
    pools = {(reg, part): pooled(reg, part) for reg in regions for part in ("A", "B")}

    def share(reg, part, what, lev, col="inside, with margin"):
        g = pools[(reg, part)]
        g = g[(g["values"] == what) & (g.level == lev)]
        return float(g[col].iloc[0]) if len(g) else np.nan

    fails = []
    for reg in regions:
        for lev in JUDGED_A:
            x = share(reg, "A", "days", lev)
            lo, hi = accepted_real(lev)
            if not (lo <= x <= hi):
                fails.append(f"{reg} (A) days, {lev}%: {x:.1%} (accepted {lo:.1%}-{hi:.1%})")
        for what in ("7-day means", "30-day means"):
            x = share(reg, "A", what, 90)
            if not (MULTI_DAY_RANGE[0] <= x <= MULTI_DAY_RANGE[1]):
                fails.append(f"{reg} (A) {what}, 90%: {x:.1%}")
        for what in WINDOWS:
            for lev in JUDGED_B:
                x = share(reg, "B", what, lev)
                lo, hi = accepted_real(lev)
                if not (lo <= x <= hi):
                    fails.append(f"{reg} (B) {what}, {lev}%: {x:.1%} (accepted {lo:.1%}-{hi:.1%})")
    res.passed = True if ctx.quick else not fails

    def line(reg, part):
        return "; ".join(f"{what} {share(reg, part, what, 90, 'inside, no margin'):.1%} -> {share(reg, part, what, 90):.1%}"
                         for what in WINDOWS)
    n_conv = {reg: df[(df.region == reg) & (df.part == "A") & df.converged.astype(bool)].station.nunique()
              for reg in regions}
    n_all = {reg: df[(df.region == reg)].station.nunique() for reg in regions}
    res.summary = " ".join(
        f"{reg}, share inside the 90% intervals without -> with the margins: in the later years ({n_conv[reg]} of "
        f"{n_all[reg]} rivers converged) {line(reg, 'A')}; in the held-out years {line(reg, 'B')}." for reg in regions)
    if ctx.quick:
        res.summary += " Quick run: coverage reported, not judged."
    elif fails:
        res.summary += " Criteria not met: " + "; ".join(fails) + "."

    def fmt(t):
        t = t.copy()
        for c in t.columns:
            if c.startswith("inside"):
                t[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
            elif "(°C)" in c:
                t[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}" if c.startswith("median margin") else f"{x:.2f}")
        return t.drop(columns=["region", "part"])

    accepted = pd.DataFrame([{"level": f"{x}%", "accepted share inside": "{:.1%}-{:.1%}".format(*accepted_real(x)),
                              "judged": ("days in A and B" if x in JUDGED_B else "days in A" if x in JUDGED_A
                                         else "no (reported)")} for x in LEVELS])
    by_station = []
    for (reg, st), g in df[df.level == 90].groupby(["region", "station"], sort=False):
        row = {"region": reg, "station": st}
        for part in ("A", "B"):
            for what in WINDOWS:
                h = g[(g.part == part) & (g["values"] == what)]
                if len(h) and h.n.iloc[0]:
                    r = h.iloc[0]
                    row[f"{part}: {what}"] = f"{r['inside before'] / r.n:.0%} -> {r['inside after'] / r.n:.0%}"
        h = g[(g.part == "B") & (g["values"] == "days")]
        row["margin, days (°C)"] = round(float(h.margin.iloc[0]), 2) if len(h) else np.nan
        row["held-out years"] = int(h["held-out years"].iloc[0]) if len(h) else 0
        a = g[g.part == "A"]
        row["DE-MCMC converged (steps)"] = (f"{'yes' if bool(a.converged.iloc[0]) else 'NO'} ({int(a.steps.iloc[0])})"
                                            if len(a) else "")
        by_station.append(row)
    res.sections.append(Section(
        "A. Later years",
        "Share of measured values inside the central range of the FORWARD run, without and with the margins from "
        "the river's held-out years, pooled over the rivers whose DE-MCMC converged; mean width of the range. The "
        "margins are estimated from fold simulations without parameter uncertainty and added to a FORWARD range "
        "that includes it, so they err on the wide side.",
        figures=[_fig_levels(pools)],
        tables=[(f"A. {reg}", fmt(pools[(reg, "A")])) for reg in regions]
               + [("Accepted coverage (miss rate between half and 1.5 times the stated one)", accepted)]))
    res.sections.append(Section(
        "B. Held-out years",
        "Share of held-out values inside the range of their fold, without a margin and with a margin from the "
        "river's other held-out years only.",
        tables=[(f"B. {reg}", fmt(pools[(reg, "B")])) for reg in regions]))
    res.sections.append(Section(
        "C. By river, 90% intervals",
        "Share inside without -> with the margin, for each river and part, the margin of single days from all "
        "held-out years, and the number of held-out years it comes from.",
        tables=[("C. By river", pd.DataFrame(by_station))]))
    margins_all = df[df.part == "B"].pivot_table(index=["region", "station"], columns=["values", "level"],
                                                 values="margin").round(2)
    margins_all.columns = [f"{w}, {lev}%" for w, lev in margins_all.columns]
    res.sections.append(Section("D. The margins (°C)", "Margin added to both ends of each central range, from all "
                                "held-out years of the river (negative: the range is narrowed).",
                                tables=[("D. Margins", margins_all.reset_index())]))
    df.to_csv(os.path.join(WORK, "v19_rows.csv"), index=False)
    return res


def _fig_levels(pools):
    """Stated level against coverage, without and with the margins, both regions and parts."""
    import matplotlib.pyplot as plt
    plot_style()
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.5), sharex=True, sharey=True)
    grid = np.linspace(45, 100, 200)
    keys = [(reg, part) for reg in ("British Columbia", "Switzerland") for part in ("A", "B")]
    names = {"A": "later years", "B": "held-out years"}
    for i, (reg, part) in enumerate(keys):
        df = pools[(reg, part)]
        for j, what in enumerate(WINDOWS):
            ax = axes[i, j]
            ax.fill_between(grid, 100 * (1 - 1.5 * (1 - grid / 100)), 100 * (1 - 0.5 * (1 - grid / 100)),
                            color=LIGHT_GREY, alpha=0.35, lw=0, zorder=0)
            ax.plot([45, 100], [45, 100], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
            g = df[df["values"] == what].sort_values("level")
            for col, colour, label in (("inside, no margin", ORANGE, "without margin"),
                                       ("inside, with margin", BLUE, "with conformal margin")):
                if len(g):
                    ax.plot(g.level, 100 * g[col], marker="o", ms=3.5, lw=1.2, color=colour, label=label)
            ax.set_xlim(45, 100)
            ax.set_ylim(45, 100)
            ax.set_aspect("equal")
            ax.set_title(f"{reg}, {names[part]}: {what}", fontsize=8.5)
            if i == len(keys) - 1:
                ax.set_xlabel("Stated level (%)")
        axes[i, 0].set_ylabel("Measured values inside (%)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return (save_figure(fig, "V19_levels.png"),
            "Stated level against the share of measured values inside the interval, without (orange) and with "
            "(blue) the conformal margins, for days and 7-day and 30-day means. Rows: British Columbia later years "
            "(2021-2022, converged rivers) and held-out years; Switzerland later years and held-out years. Dashed: "
            "stated = achieved; shaded: the accepted range.")
