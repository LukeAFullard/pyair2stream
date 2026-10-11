"""
V20 - The final method, once, on rivers it has never seen (north-eastern United States).

The error model was developed on the three Swiss rivers (V4, V5, V9, V11). V18 then found its intervals
too narrow on the 23 British Columbia rivers, and the conformal margins were added and tested on those same
rivers (V19). Neither region is independent of the method any more. This check runs the package, as it
stands after V19 and with nothing changed since, on up to 30 rivers of the north-eastern United States,
chosen by a fixed rule before any model was run on them (data/usa_northeast/download.py): USGS daily water
temperature and discharge, Daymet air temperature. It is run once; the method, the stations, the settings
and the pass criterion are all fixed in this file and in the data folder before the first run.

(A) Later years: DE-MCMC on 2011-2020, then a FORWARD run of 2021-2023, without and with the conformal
margins from a cross-validation of 2011-2020. (B) Held-out years of 2011-2020, each judged with a margin from
the river's other years only. (C) How the errors of new years compare with the error model (as V18, part C).
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DE_SETTINGS, LEVELS, REPO, WORK, Result, Section, Timer, accepted_real,
                    cluster_interval, load, quiet, plot_style, save_figure, BLUE, ORANGE, INK2, LIGHT_GREY)
from v18_bc_intervals import SUMMER, WINDOWS, _error_diagnostics, _ranges, _resumable
from v19_conformal import _later_counts, _margins

DATA = os.path.join(REPO, "data", "usa_northeast")
QUICK_STATIONS = 2
MCMC_WALKERS, MCMC_STEPS = 32, 100000          # as V15 and V18
PREDICTION_LEVEL = 90.0
JUDGED_A = (50, 80, 90, 95)                     # as V18 and V19: days in the later years
JUDGED_B = (80, 90, 95)                         # as V18 and V19: held-out years
MULTI_DAY_RANGE = (0.85, 0.95)                  # 7- and 30-day means inside the 90% interval, as V18 and V19


def stations(quick=False):
    table = pd.read_csv(os.path.join(DATA, "stations.csv"), dtype={"site_no": str})
    return list(table.site_no[:QUICK_STATIONS] if quick else table.site_no)


def _config(st, tag, quick, **extra):
    """Version 8, NSE, Crank-Nicolson, the authors' parameter ranges, the calibration Qmedia: the
    settings of V19's Swiss runs."""
    cal = os.path.join(DATA, f"{st}_calibration.csv")
    q = pd.read_csv(cal).Discharge
    cfg = {"version": 8, "integrator": "CRN", "objective_function": "NSE", "random_seed": 1,
           "Qmedia": float(q[q > 0].mean()), "parameter_bounds": AUTHORS_BOUNDS,
           "optimization": {"n_run": 40, "n_particles": 8} if quick else dict(DE_SETTINGS),
           "uncertainty_options": {"noise_model": "ar1", "prediction_interval": PREDICTION_LEVEL},
           "paths": {"input_data": cal, "output_dir": os.path.join(WORK, tag, "out")}}
    cfg.update(extra)
    return cfg


def _cross_validation(args):
    """Leave-one-year-out cross-validation of 2011-2020 (every year but the first held out once), the
    conformal margins as a cross-validation run writes them, and how each held-out year's errors compare
    with its fold's error model."""
    st, quick = args
    from pyair2stream.cross_validation import cross_validate
    from pyair2stream.main import write_conformal_margins
    tag = f"v20_cv_{st}"
    cfg = _config(st, tag, quick, run_mode="DE",
                  cross_validation={"enabled": True, "unit": "year", "min_train_years": 0, "skip_first_year": True})
    data = load(cfg, tag)
    if quick:   # the last four years only
        data.cross_validation.min_train_years = max(0, len(set(data.date[365:, 0])) - 5)
    with quiet():
        _, folds = cross_validate(data, "DE", return_folds=True)
        write_conformal_margins(data, folds, PREDICTION_LEVEL)
    diagnostics = []
    for f in folds:
        obs = np.where(f.obs_held_out == -999.0, np.nan, f.obs_held_out)
        sim = np.where(f.sim_held_out == -999.0, np.nan, f.sim_held_out)
        d = _error_diagnostics(sim - obs, f.sigma, f.rho)
        if d is not None:
            diagnostics.append({"fold": f.label, **d})
    if os.path.abspath(os.path.join(data.folder, "cv_conformal_margins.csv")) != _margins_csv(st):
        raise RuntimeError(f"{st}: the margins were written to {data.folder}")
    return {"station": st, "folds": len(folds), "margins": _margins(folds), "diagnostics": diagnostics}


def _margins_csv(st):
    return os.path.abspath(os.path.join(WORK, f"v20_cv_{st}", "out", "cv_conformal_margins.csv"))


def _later_years(args):
    """DE-MCMC on 2011-2020, then a FORWARD run of 2021-2023 from the chain, with the conformal margins
    when the cross-validation wrote them."""
    st, quick = args
    margins_csv = _margins_csv(st) if os.path.exists(_margins_csv(st)) else None
    from pyair2stream import scenario
    from pyair2stream.optimization import DE_MCMC_mode, forward_mode
    tag = f"v20_{st}"
    cfg = _config(st, tag, quick, run_mode="DE-MCMC")
    cfg["uncertainty_options"].update({"likelihood": "least_squares", "strict_convergence": False})
    cfg["optimization"].update({"mcmc_walkers": 16, "mcmc_steps": 600} if quick
                               else {"mcmc_walkers": MCMC_WALKERS, "mcmc_steps": MCMC_STEPS})
    data = load(cfg, tag)
    with quiet():
        DE_MCMC_mode(data, seed=1)
    out = os.path.join(WORK, tag, "out")
    meta = json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))
    later = os.path.join(DATA, f"{st}_later.csv")
    fcfg = {"version": 8, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": cfg["Qmedia"],
            "parameters_forward": [float(x) for x in data.par_best],
            "uncertainty_options": {"noise_model": "ar1", "save_ensemble": True,
                                    "prediction_interval": PREDICTION_LEVEL},
            "forward_options": {"enable_prediction_intervals": True,
                                "mcmc_chain_path": os.path.join(out, "MCMC_chain_S_c_1d.csv"),
                                "n_samples": 100 if quick else 1000, "random_seed": 1},
            "paths": {"input_data": later, "output_dir": os.path.join(WORK, tag, "fwd")}}
    if margins_csv:
        fcfg["forward_options"]["conformal_margins"] = margins_csv
    fdata = load(fcfg, tag + "_fwd")
    with quiet():
        forward_mode(fdata)
    ens, dates = scenario.load_ensemble(os.path.join(WORK, tag, "fwd", "Forward_Prediction_Ensemble_S_c_1d.npz"))
    obs = pd.read_csv(later).T_water.to_numpy(float)
    if ens.shape[1] != len(obs):
        raise RuntimeError(f"{st}: ensemble of {ens.shape[1]} days for {len(obs)} later days")
    ok = np.isfinite(obs)
    best = np.asarray(fdata.Twat_mod[365:], float)
    diag = _error_diagnostics(np.where(ok, best - obs, np.nan), float(meta["sigma"]), float(meta["rho"]))
    return {"station": st, "converged": bool(meta["converged"]), "steps": int(meta["steps_run"]),
            "rho": float(meta["rho"]), "sigma": float(meta["sigma"]), "ranges": _ranges(ens, obs, dates),
            "ice_floor": float(fdata.Tice_cover), "diagnostics": diag}


def _rows(st, cv, later):
    """Per-station counts for both parts, as V19 (_station_rows), without the heat dome."""
    rows = []
    table = cv["margins"]
    margins = {(int(r.window_days), int(r.level)): float(r.margin) for r in table.itertuples()}
    names = {w: what for what, w in WINDOWS.items()}
    for r in table.itertuples():
        rows.append({"station": st, "part": "B", "values": names[int(r.window_days)], "level": int(r.level),
                     "margin": float(r.margin), "held-out years": int(r.held_out_years), "n": int(r.values),
                     "inside before": int(round(r.inside_before * r.values)),
                     "inside after": int(round(r.inside_after * r.values)), "converged": True})
    counts = _later_counts(later["ranges"], margins, later["ice_floor"])
    dates = later["ranges"]["dates"]
    counts.update({("summer days", lev): v for (w, lev), v in _later_counts(
        later["ranges"], margins, later["ice_floor"], mask=np.asarray(dates.month.isin(SUMMER))).items() if w == "days"})
    for (what, lev), (k0, k1, n, w0, w1) in counts.items():
        window = 1 if what == "summer days" else WINDOWS[what]
        rows.append({"station": st, "part": "A", "values": what, "level": lev,
                     "margin": margins.get((window, lev), np.nan), "n": n, "inside before": k0, "inside after": k1,
                     "width before": w0, "width after": w1, "converged": later["converged"], "steps": later["steps"]})
    return rows


def run(ctx) -> Result:
    sts = stations(ctx.quick)
    res = Result(
        code="V20", title="The final method on rivers it has never seen (north-eastern United States)",
        question="Run once, unchanged, on rivers that played no part in developing or correcting it, do the "
                 "package's prediction intervals hold at their stated levels, for days, 7-day and 30-day means, "
                 "in later years and in held-out years: as they are, and widened by the conformal margins?",
        method=f"{len(sts)} USGS stream gauges in New England and the Mid-Atlantic (hydrologic regions 01 and 02), "
               "chosen by the rule in data/usa_northeast/download.py before any model was run on them: discharge "
               "on every day of 2011-2023, at least 300 daily water temperatures in at least 5 of the years "
               "2011-2020 and at least 300 in 2021-2023, at most 30 stations drawn at random. Air temperature "
               "from Daymet, (tmax + tmin) / 2 at the gauge. Version 8, NSE, Crank-Nicolson, the authors' "
               "parameter ranges and the calibration Qmedia, as V19's Swiss runs. (A) Later years: DE-MCMC with "
               f"the default least-squares likelihood and rho ({MCMC_WALKERS} walkers, run until converged, at "
               f"most {MCMC_STEPS} steps, as V18) on 2011-2020, then a FORWARD run of 1000 series from the "
               "chain for 2021-2023 with forward_options.conformal_margins set to the margins of (B). (B) "
               "Held-out years: the package's leave-one-year-out cross-validation of 2011-2020 (DE, every year "
               "but the first held out once), which writes cv_conformal_margins.csv; each held-out year is "
               "judged without a margin and with a margin from the river's other held-out years only "
               "(inside_after). (C) For each predicted period, its mean error against what the error model "
               "allows (z, as V18 part C). Shares are pooled over stations, weighted by the number of values; "
               "95% confidence intervals resample stations (cluster bootstrap).",
        criterion="Fixed in this file before the check was first run, with the method unchanged since V19. With "
                  "the conformal margins: (A) for the stations whose DE-MCMC converged, pooled, the share of "
                  f"later-year days inside the interval is within the accepted range at "
                  f"{', '.join(f'{x}%' for x in JUDGED_A)} (a miss rate between half and 1.5 times the stated one), "
                  f"and the share of 7-day and of 30-day means inside the 90% interval is between "
                  f"{MULTI_DAY_RANGE[0]:.0%} and {MULTI_DAY_RANGE[1]:.0%}; (B) pooled over every held-out year, "
                  f"days, 7-day and 30-day means inside the {', '.join(f'{x}%' for x in JUDGED_B)} intervals are "
                  "within the accepted range. These are V19's criteria. The intervals without margins are judged "
                  "by the same criteria (V18's), and the outcome is reported in the summary but does not decide "
                  "the result: the margins are what the package recommends where cross-validation shows "
                  "intervals that hold too rarely. The 99% intervals, single stations, unconverged stations and "
                  "summer are reported, not judged.")
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            f_cv = {st: ex.submit(_resumable, (_cross_validation, (st, ctx.quick))) for st in sts}
            cv = {st: f.result()[0] for st, f in f_cv.items()}
            f_a = {st: ex.submit(_resumable, (_later_years, (st, ctx.quick))) for st in sts}
            later = {st: f.result()[0] for st, f in f_a.items()}
    res.seconds = t.seconds
    no_margins = [st for st in sts if cv[st]["margins"].empty]
    df = pd.DataFrame([r for st in sts if st not in no_margins for r in _rows(st, cv[st], later[st])])
    pool_conv = not ctx.quick

    def pooled(part):
        g = df[df.part == part]
        if part == "A" and pool_conv:
            g = g[g.converged.astype(bool)]
        out = []
        for (what, lev), h in g.groupby(["values", "level"], sort=False):
            n = h.n.sum()
            row = {"values": what, "level": lev, "stations": h.station.nunique(), "n": int(n)}
            for col, label in (("inside before", "no margin"), ("inside after", "with margin")):
                row[f"inside, {label}"] = h[col].sum() / n if n else np.nan
                lo, hi = cluster_interval(zip(h[col], h.n))
                row[f"{label}: 95% CI"] = "" if not np.isfinite(lo) else f"{lo * 100:.1f}-{hi * 100:.1f}%"
            row["median margin (°C)"] = float(np.median(h.margin))
            if part == "A":
                row["mean width, no margin (°C)"] = float(np.average(h["width before"], weights=h.n)) if n else np.nan
                row["mean width, with margin (°C)"] = float(np.average(h["width after"], weights=h.n)) if n else np.nan
            out.append(row)
        return pd.DataFrame(out)
    pools = {part: pooled(part) for part in ("A", "B")}

    def share(part, what, lev, col="inside, with margin"):
        g = pools[part]
        g = g[(g["values"] == what) & (g.level == lev)]
        return float(g[col].iloc[0]) if len(g) else np.nan

    def ci(part, what, lev, label="with margin"):
        g = pools[part]
        g = g[(g["values"] == what) & (g.level == lev)]
        return g[f"{label}: 95% CI"].iloc[0] if len(g) else ""

    def judge(col):
        fails = []
        for lev in JUDGED_A:
            x = share("A", "days", lev, col)
            lo, hi = accepted_real(lev)
            if not (lo <= x <= hi):
                fails.append(f"(A) days, {lev}%: {x:.1%} (accepted {lo:.1%}-{hi:.1%})")
        for what in ("7-day means", "30-day means"):
            x = share("A", what, 90, col)
            if not (MULTI_DAY_RANGE[0] <= x <= MULTI_DAY_RANGE[1]):
                fails.append(f"(A) {what}, 90%: {x:.1%}")
        for what in WINDOWS:
            for lev in JUDGED_B:
                x = share("B", what, lev, col)
                lo, hi = accepted_real(lev)
                if not (lo <= x <= hi):
                    fails.append(f"(B) {what}, {lev}%: {x:.1%} (accepted {lo:.1%}-{hi:.1%})")
        return fails
    fails_margin, fails_plain = judge("inside, with margin"), judge("inside, no margin")
    res.passed = True if ctx.quick else not fails_margin

    n_conv = df[(df.part == "A") & df.converged.astype(bool)].station.nunique()

    def line(part):
        return "; ".join(f"{what} {share(part, what, 90, 'inside, no margin'):.1%} ({ci(part, what, 90, 'no margin')}) "
                         f"-> {share(part, what, 90):.1%} ({ci(part, what, 90)})" for what in WINDOWS)
    res.summary = (f"{len(sts)} rivers ({n_conv} with a converged DE-MCMC). Share inside the 90% intervals without -> "
                   f"with the margins (95% confidence intervals, stations resampled): in the later years 2021-2023 "
                   f"{line('A')}; in the held-out years {line('B')}. ")
    if ctx.quick:
        res.summary += "Quick run: coverage reported, not judged."
    else:
        res.summary += ("With the margins: " + ("criteria met." if not fails_margin
                                                else "criteria not met: " + "; ".join(fails_margin) + ".")
                        + " Without them (V18's criteria): " + ("met." if not fails_plain
                                                                 else "not met: " + "; ".join(fails_plain) + "."))
    if no_margins:
        res.notes.append(f"No margins (fewer than 3 held-out years with measurements) at {', '.join(no_margins)}; "
                         "those stations are left out.")

    def fmt(t):
        t = t.copy()
        for c in t.columns:
            if c.startswith("inside"):
                t[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
            elif "(°C)" in c:
                t[c] = t[c].map(lambda x: "" if pd.isna(x) else f"{x:+.2f}" if c.startswith("median margin") else f"{x:.2f}")
        return t

    info = pd.read_csv(os.path.join(DATA, "stations.csv"), dtype={"site_no": str}).set_index("site_no")
    by_station = []
    for st in sts:
        g = df[(df.station == st) & (df.level == 90)]
        row = {"station": st, "name": info.station_nm.get(st, ""), "rho": round(later[st]["rho"], 3),
               "sigma (°C)": round(later[st]["sigma"], 3),
               "DE-MCMC converged (steps)": f"{'yes' if later[st]['converged'] else 'NO'} ({later[st]['steps']})"}
        for part in ("A", "B"):
            for what in WINDOWS:
                h = g[(g.part == part) & (g["values"] == what)]
                if len(h) and h.n.iloc[0]:
                    r = h.iloc[0]
                    row[f"{part}: {what}"] = f"{r['inside before'] / r.n:.0%} -> {r['inside after'] / r.n:.0%}"
        h = g[(g.part == "B") & (g["values"] == "days")]
        row["margin, days (°C)"] = round(float(h.margin.iloc[0]), 2) if len(h) else np.nan
        row["held-out years"] = int(h["held-out years"].iloc[0]) if len(h) else 0
        by_station.append(row)
    accepted = pd.DataFrame([{"level": f"{x}%", "accepted share inside": "{:.1%}-{:.1%}".format(*accepted_real(x)),
                              "judged": ("days in A and B" if x in JUDGED_B else "days in A" if x in JUDGED_A
                                         else "no (reported)")} for x in LEVELS])
    res.sections.append(Section(
        "A. Later years (2021-2023)",
        "Share of measured values inside the central range of the FORWARD run, without and with the margins from "
        "the river's held-out years, pooled over the stations whose DE-MCMC converged; mean width of the range.",
        figures=[_fig_levels(pools)],
        tables=[("A. Pooled over converged stations", fmt(pools["A"])),
                ("Accepted coverage (miss rate between half and 1.5 times the stated one)", accepted)]))
    res.sections.append(Section(
        "B. Held-out years (2012-2020)",
        "Share of held-out values inside the range of their fold, without a margin and with a margin from the "
        "river's other held-out years only.",
        tables=[("B. Pooled over every held-out year", fmt(pools["B"]))]))
    res.sections.append(Section("By station, 90% intervals", "Share inside without -> with the margin.",
                                tables=[("By station", pd.DataFrame(by_station))]))

    diag_b = [d for st in sts for d in cv[st]["diagnostics"]]
    diag_a = [later[st]["diagnostics"] for st in sts if later[st]["diagnostics"] and later[st]["converged"]]
    diag_rows = []
    for part, ds in (("A. later years (2021-2023), converged stations", diag_a), ("B. held-out years", diag_b)):
        if ds:
            ratio = float(np.sqrt(sum(d["rmse"] ** 2 * d["n"] for d in ds) / sum(d["sigma"] ** 2 * d["n"] for d in ds)))
            diag_rows.append({"part": part, "periods": len(ds), "RMSE / sigma": round(ratio, 2),
                              "RMS of z": round(float(np.sqrt(np.mean([d["z"] ** 2 for d in ds]))), 2),
                              "periods with |z| > 2": f"{sum(abs(d['z']) > 2 for d in ds)} of {len(ds)} "
                                                      "(error model: about 5%)"})
    if diag_rows:
        res.sections.append(Section(
            "C. The errors of new periods against the error model",
            "As V18 part C: the RMSE of each predicted period over the sigma it was predicted with, and its mean "
            "error over the standard deviation the error model gives the mean of that many days (z).",
            tables=[("C. Pooled", pd.DataFrame(diag_rows))]))
    res.notes.append(
        "These rivers were chosen by a fixed rule (data/usa_northeast/download.py) and the check was written, with "
        "its criterion, before any model was run on them; the package was not changed between V19 and this run. "
        "They differ from both earlier regions: lowland and upland rivers of a humid continental climate, many "
        "small, several urban or regulated.")
    df.to_csv(os.path.join(WORK, "v20_rows.csv"), index=False)
    return res


def _fig_levels(pools):
    import matplotlib.pyplot as plt
    plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), sharex=True, sharey=True)
    grid = np.linspace(45, 100, 200)
    for i, (part, title) in enumerate((("A", "Later years (2021-2023)"), ("B", "Held-out years"))):
        df = pools[part]
        for j, what in enumerate(WINDOWS):
            ax = axes[i, j]
            ax.fill_between(grid, 100 * (1 - 1.5 * (1 - grid / 100)), 100 * (1 - 0.5 * (1 - grid / 100)),
                            color=LIGHT_GREY, alpha=0.35, lw=0, zorder=0)
            ax.plot([45, 100], [45, 100], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
            g = df[df["values"] == what].sort_values("level")
            for col, colour, marker, label in (("inside, no margin", ORANGE, "o", "without margins"),
                                               ("inside, with margin", BLUE, "s", "with conformal margins")):
                if len(g):
                    ax.plot(g.level, 100 * g[col], marker=marker, ms=3.5, lw=1.2, color=colour, label=label)
            ax.set_xlim(45, 100)
            ax.set_ylim(45, 100)
            ax.set_aspect("equal")
            ax.set_title(f"{title}: {what}", fontsize=9)
            if i == 1:
                ax.set_xlabel("Stated level (%)")
        axes[i, 0].set_ylabel("Measured values inside (%)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return (save_figure(fig, "V20_levels.png"),
            "North-eastern United States, rivers never used to develop or correct the method: stated level against "
            "the share of measured values inside the interval, without (orange circles) and with (blue squares) "
            "the conformal margins. Top: later years 2021-2023 (stations whose DE-MCMC converged). Bottom: years "
            "held out by cross-validation. Dashed: stated = achieved; shaded: the accepted range.")
