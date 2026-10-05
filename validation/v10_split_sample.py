"""
V10 - Predicting warmer or lower-flow years than those calibrated on.

Temperature limits are usually at stake in warm, low-flow summers, and scenarios
(lower flows, warmer air) ask the model to predict conditions outside those it
was calibrated on. This is the differential split-sample test (Klemeš, 1986):
calibrate on the coolest (or highest-flow) years and predict the warmest (or
lowest-flow) ones. The same years are also predicted from a calibration on the
middle years, so the cost of extrapolating can be measured.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DE_SETTINGS, RIVERS, WORK, Result, Section, Timer, load, quiet, river_csv,
                    metrics, plot_style, save_figure, BLUE, ORANGE, AQUA, INK2, LIGHT_GREY, MUTED)
from v5_real_rivers import _benchmarks

VERSIONS_TESTED = (5, 8)
SUMMER = (6, 7, 8)
MIN_SUMMER_OBSERVED = 0.8         # a year takes part if at least this share of its summer days is measured
MIN_COVERAGE = 0.80               # 90% intervals must contain at least this share of the test years' days
SPLITS = {
    "warm": ("summer air temperature", "coolest", "warmest", lambda g: g["summer air (°C)"]),
    "low flow": ("summer discharge", "highest-flow", "lowest-flow", lambda g: -g["summer discharge (m3/s)"]),
}


def full_record(st):
    """Calibration and validation files joined into one continuous record."""
    return pd.concat([pd.read_csv(river_csv(st, p), parse_dates=["Date"]) for p in ("calibration", "validation")],
                     ignore_index=True)


def year_table(df):
    s = df[df.Date.dt.month.isin(SUMMER)]
    g = s.groupby(s.Date.dt.year).agg(**{"summer air (°C)": ("T_air", "mean"),
                                         "summer discharge (m3/s)": ("Discharge", "mean"),
                                         "summer observed": ("T_water", lambda x: x.notna().mean())})
    return g[g["summer observed"] >= MIN_SUMMER_OBSERVED]


def split_years(years, split):
    """Test = the most extreme third, control = the next third, differential = the opposite third."""
    key = SPLITS[split][3](years).sort_values(ascending=False)
    k = len(key) // 3
    order = list(key.index)
    return {"test": sorted(order[:k]), "control": sorted(order[k:2 * k]), "differential": sorted(order[-k:])}


def _fit(args):
    """Calibrate (DE-MCMC, defaults) on `cal_years` of the full record and predict `test_years` with 90%
    intervals. The forcing stays continuous; water temperatures outside the calibration years are hidden."""
    st, version, split, role, cal_years, test_years = args
    from pyair2stream.optimization import DE_MCMC_mode, forward_mode
    tag = f"v10_{st}_{version}_{split.replace(' ', '')}_{role}"
    folder = os.path.join(WORK, tag)
    os.makedirs(folder, exist_ok=True)
    df = full_record(st)
    year = df.Date.dt.year
    in_cal = year.isin(cal_years)
    cal_csv = os.path.join(folder, "calibration.csv")
    df.assign(T_water=df.T_water.where(in_cal), Date=df.Date.dt.strftime("%Y-%m-%d")).to_csv(cal_csv, index=False)
    qmedia = float(df.Discharge[in_cal].mean())        # as a user calibrating on these years would get
    out = os.path.join(folder, "out")
    cfg = {"version": version, "integrator": "CRN", "run_mode": "DE-MCMC", "objective_function": "NSE",
           "random_seed": 1, "Qmedia": qmedia, "parameter_bounds": AUTHORS_BOUNDS,
           "optimization": {**DE_SETTINGS, "mcmc_walkers": 32, "mcmc_steps": 20000},
           "paths": {"input_data": cal_csv, "output_dir": out}}
    row = {"river": RIVERS[st], "version": version, "split": split, "calibrated on": role}
    data = load(cfg, tag)
    try:
        with quiet():
            DE_MCMC_mode(data, seed=1)
    except RuntimeError:
        return {**row, "converged": False}
    full_csv = os.path.join(folder, "full.csv")
    df.assign(Date=df.Date.dt.strftime("%Y-%m-%d")).to_csv(full_csv, index=False)
    fcfg = {"version": version, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": qmedia,
            "parameters_forward": [float(x) for x in data.par_best],
            "forward_options": {"enable_prediction_intervals": True,
                                "mcmc_chain_path": os.path.join(out, "MCMC_chain_S_c_1d.csv"),
                                "n_samples": 1000, "random_seed": 1},
            "paths": {"input_data": full_csv, "output_dir": os.path.join(folder, "fwd")}}
    fdata = load(fcfg, tag + "_fwd")
    with quiet():
        forward_mode(fdata)
    env = pd.read_csv(os.path.join(folder, "fwd", "Forward_Prediction_Envelopes_S_c_1d.csv"))
    obs = df.T_water.to_numpy(float)
    sim = fdata.Twat_mod[365:].astype(float)
    test = year.isin(test_years).to_numpy() & np.isfinite(obs)
    summer = test & df.Date.dt.month.isin(SUMMER).to_numpy()
    inside = (obs >= env.Twat_mod_lower.to_numpy()) & (obs <= env.Twat_mod_upper.to_numpy())
    m_all, m_sum = metrics(obs[test], sim[test]), metrics(obs[summer], sim[summer])
    # The simple alternatives, fitted on the same calibration years.
    bench_cal = os.path.join(folder, "bench_cal.csv")
    bench_test = os.path.join(folder, "bench_test.csv")
    df[in_cal].to_csv(bench_cal, index=False)
    df[year.isin(test_years)].to_csv(bench_test, index=False)
    obs_t = df.T_water[year.isin(test_years)].to_numpy(float)
    simple = {name: metrics(obs_t, pred)["RMSE"] for name, pred in _benchmarks(bench_cal, bench_test).items()}
    return {**row, "converged": True, "Qmedia": qmedia,
            "RMSE": m_all["RMSE"], "summer RMSE": m_sum["RMSE"], "summer bias": m_sum["bias"],
            "coverage": float(inside[test].mean()), "summer coverage": float(inside[summer].mean()),
            "best simple alternative": min(simple, key=simple.get),
            "best simple alternative RMSE": min(simple.values()),
            "_series": (df.Date.to_numpy(), obs, sim, env.Twat_mod_lower.to_numpy(), env.Twat_mod_upper.to_numpy())
            if (st, version, split, role) == ("MAH_2369", 8, "warm", "differential") else None}


def run(ctx) -> Result:
    res = Result(
        code="V10", title="Predicting warmer or lower-flow years than those calibrated on",
        question="Calibrated only on the coolest (or highest-flow) years of a record, does the model still "
                 "predict the warmest (or lowest-flow) years, when temperature limits are usually at stake? How "
                 "much does extrapolating cost, and do the 90% intervals still hold?",
        method="Differential split-sample test (Klemeš, 1986). For each Swiss river the calibration and "
               "validation files are joined into one continuous record, and the years with at least 80% of "
               "their summer (June-August) water temperatures measured are ranked twice: by summer air "
               "temperature and by summer discharge. The record is divided into thirds. The most extreme third "
               "(warmest, or lowest-flow) is predicted (test years) from two calibrations of equal length: on "
               "the opposite third (coolest, or highest-flow: the differential calibration) and on the middle "
               "third (the control). A calibration uses the whole continuous forcing but only the water "
               "temperatures of its own years, with Qmedia the mean discharge of those years. Each calibration "
               "is DE-MCMC with the default settings, followed by a FORWARD run with 90% prediction intervals "
               "over the whole record; versions 5 and 8. The two simple alternatives of V5 (day-of-year average "
               "and air-temperature regression) are fitted on the same years.",
        criterion=f"For every river, version and split, calibrated on the opposite third, the model predicts the "
                  f"test years with a lower RMSE than both simple alternatives fitted on the same years, and its "
                  f"90% intervals contain at least {MIN_COVERAGE:.0%} of the test years' measurements. (The "
                  f"criterion for intervals is lower than V5's 85% because these years lie outside the "
                  f"calibration conditions by design; the cost of extrapolating is reported, not judged.)")
    stations = ["MAH_2369"] if ctx.quick else list(RIVERS)
    versions = (8,) if ctx.quick else VERSIONS_TESTED
    splits = ("warm",) if ctx.quick else tuple(SPLITS)
    designs, jobs = [], []
    for st in stations:
        years = year_table(full_record(st))
        for split in splits:
            sets = split_years(years, split)
            designs.append((st, split, years, sets))
            for v in versions:
                for role in ("differential", "control"):
                    jobs.append((st, v, split, role, sets[role], sets["test"]))
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            out = list(ex.map(_fit, jobs))
    res.seconds = t.seconds
    example = next((o.pop("_series") for o in out if o.get("_series") is not None), None)
    for o in out:
        o.pop("_series", None)
    fits = pd.DataFrame(out)

    rows, ok = [], bool(fits.converged.all())
    for (river, v, split), g in fits.groupby(["river", "version", "split"], sort=False):
        d = g[g["calibrated on"] == "differential"].iloc[0]
        c = g[g["calibrated on"] == "control"].iloc[0]
        if not (d.converged and c.converged):
            rows.append({"river": river, "version": v, "split": split, "converged": False, "pass": False})
            ok = False
            continue
        st = {r: s for s, r in RIVERS.items()}[river]
        sets = next(s for s_, sp, _, s in designs if s_ == st and sp == split)
        passed = bool(d["RMSE"] < d["best simple alternative RMSE"] and d["coverage"] >= MIN_COVERAGE)
        ok &= passed
        rows.append({
            "river": river, "version": v, "split": split, "converged": True,
            "test years": ", ".join(map(str, sets["test"])),
            "calibrated on (differential)": ", ".join(map(str, sets["differential"])),
            "control years": ", ".join(map(str, sets["control"])),
            "RMSE, control (°C)": round(c["RMSE"], 3), "RMSE, differential (°C)": round(d["RMSE"], 3),
            "cost of extrapolating (°C)": round(d["RMSE"] - c["RMSE"], 3),
            "summer bias, control (°C)": round(c["summer bias"], 2),
            "summer bias, differential (°C)": round(d["summer bias"], 2),
            "best simple alternative, differential years (RMSE °C)":
                f"{d['best simple alternative']} {d['best simple alternative RMSE']:.3f}",
            "90% interval coverage, differential": f"{d['coverage']:.1%}",
            "summer coverage, differential": f"{d['summer coverage']:.1%}",
            "90% interval coverage, control": f"{c['coverage']:.1%}",
            "pass": passed})
    table = pd.DataFrame(rows)
    res.passed = ok
    good = table[table.converged]
    cost = good["cost of extrapolating (°C)"]
    cov = good["90% interval coverage, differential"].str.rstrip("%").astype(float)
    res.summary = (f"Calibrated on the opposite third of the years, the model predicted the warmest and the "
                   f"lowest-flow years {'better than both simple alternatives in every case' if bool(good['pass'].all()) else 'with mixed results (see table)'}; "
                   f"extrapolating changed the RMSE by {cost.min():+.2f} to {cost.max():+.2f} °C compared with "
                   f"calibrating on the middle years, and the 90% intervals contained {cov.min():.1f}-{cov.max():.1f}% "
                   f"of the test years' measurements.")
    res.sections.append(Section(
        "Which years were used",
        "Each point is a year. The test years (the most extreme third) are predicted from a calibration on the "
        "opposite third (differential) and, for comparison, on the middle third (control).",
        figures=[_fig_design(designs)]))
    res.sections.append(Section(
        "Results",
        "RMSE over all days of the test years; summer bias is the mean of simulated minus measured temperature in "
        "June-August of the test years (negative: the model is too cool). The cost of extrapolating is the RMSE "
        "from the differential calibration minus the RMSE from the control.",
        figures=[f for f in (_fig_results(fits), _fig_example(example)) if f],
        tables=[("Differential split-sample test", table)]))
    worst_bias = good.loc[good["summer bias, differential (°C)"].abs().idxmax()]
    res.notes.append(
        f"The largest summer bias after extrapolating was {worst_bias['summer bias, differential (°C)']:+.2f} °C "
        f"({worst_bias['river']}, version {worst_bias['version']}, {worst_bias['split']} split), against "
        f"{worst_bias['summer bias, control (°C)']:+.2f} °C from the control calibration. A model that is too cool in "
        f"warm years understates the chance that a warm-water limit was exceeded; check the bias in the season "
        f"of interest on your own data (USER_GUIDE §9).")
    res.notes.append("With three to ten years per third, these results describe these rivers and years; they are "
                     "evidence about how the model extrapolates, not a guarantee for other rivers or larger changes.")
    return res


def _fig_design(designs):
    import matplotlib.pyplot as plt
    plot_style()
    splits = list(dict.fromkeys(sp for _, sp, _, _ in designs))
    stations = list(dict.fromkeys(st for st, _, _, _ in designs))
    fig, axes = plt.subplots(len(stations), len(splits), figsize=(4.6 * len(splits), 1.3 * len(stations) + 0.8),
                             squeeze=False)
    colour = {"test": ORANGE, "control": AQUA, "differential": BLUE, "unused": LIGHT_GREY}
    for st, split, years, sets in designs:
        ax = axes[stations.index(st)][splits.index(split)]
        col = "summer air (°C)" if split == "warm" else "summer discharge (m3/s)"
        role = {y: r for r, ys in sets.items() for y in ys}
        for y, x in years[col].items():
            ax.scatter(x, 0, s=34, color=colour[role.get(y, "unused")], zorder=3)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{RIVERS[st]}: {SPLITS[split][0]}", fontsize=8.5, loc="left")
        ax.set_xlabel(col, fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", lw=0, color=colour[r], label=lab) for r, lab in
                        (("differential", "calibration years (opposite third)"), ("control", "control years (middle third)"),
                         ("test", "test years (most extreme third)"))],
               loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return (save_figure(fig, "V10_years.png"),
            "The years of each river by summer air temperature (left) and summer discharge (right), and their role.")


def _fig_results(fits):
    import matplotlib.pyplot as plt
    plot_style()
    ok = fits[fits.converged]
    keys = list(dict.fromkeys(zip(ok.river, ok.version, ok.split)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.36 * len(keys) + 1.4), sharey=True)
    for i, (river, v, split) in enumerate(keys):
        y = len(keys) - 1 - i
        g = ok[(ok.river == river) & (ok.version == v) & (ok.split == split)]
        d = g[g["calibrated on"] == "differential"].iloc[0]
        c = g[g["calibrated on"] == "control"].iloc[0]
        a1.plot([c["RMSE"], d["RMSE"]], [y, y], color=LIGHT_GREY, lw=1.2, zorder=1)
        a1.scatter(d["best simple alternative RMSE"], y, marker="|", s=90, color=MUTED, zorder=2)
        a1.scatter(c["RMSE"], y, s=26, color=AQUA, zorder=3)
        a1.scatter(d["RMSE"], y, s=26, color=BLUE, zorder=4)
        a2.plot([c["summer bias"], d["summer bias"]], [y, y], color=LIGHT_GREY, lw=1.2, zorder=1)
        a2.scatter(c["summer bias"], y, s=26, color=AQUA, zorder=3)
        a2.scatter(d["summer bias"], y, s=26, color=BLUE, zorder=4)
    a1.set_yticks(range(len(keys))[::-1], [f"{r} v{v}, {s}" for r, v, s in keys], fontsize=7.5)
    a1.set_xlabel("RMSE on the test years (°C)")
    a2.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    a2.set_xlabel("Summer bias on the test years (°C)")
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([0], [0], marker="o", lw=0, color=BLUE, label="calibrated on the opposite third"),
                       Line2D([0], [0], marker="o", lw=0, color=AQUA, label="calibrated on the middle third"),
                       Line2D([0], [0], marker="|", lw=0, ms=10, color=MUTED,
                              label="best simple alternative (opposite third)")],
              loc="upper center", bbox_to_anchor=(1.05, -0.1), ncol=3, fontsize=7.5)
    return (save_figure(fig, "V10_results.png"),
            "Prediction error and summer bias on the warmest (or lowest-flow) years, from a calibration on the "
            "opposite third of the years (blue) and on the middle third (aqua). The grey tick is the better of the "
            "two simple alternatives fitted on the opposite third.")


def _fig_example(example):
    if example is None:
        return None
    import matplotlib.pyplot as plt
    plot_style()
    dates, obs, sim, lo, hi = example
    dates = pd.DatetimeIndex(dates)
    win = (dates >= "2003-05-01") & (dates <= "2003-09-30")
    if not win.any():
        return None
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.fill_between(dates[win], lo[win], hi[win], color=BLUE, alpha=0.18, lw=0, label="90% interval")
    ax.plot(dates[win], sim[win], color=BLUE, lw=1.2, label="model (calibrated on the three coolest summers)")
    ax.plot(dates[win], obs[win], color="#0b0b0b", lw=0.9, label="measured")
    ax.set_ylabel("Water temperature (°C)")
    ax.set_title("Mentue, summer 2003 (the warmest in the record), version 8")
    ax.legend(loc="lower center", fontsize=7.5, ncol=3, bbox_to_anchor=(0.5, -0.32))
    return (save_figure(fig, "V10_example_2003.png"),
            "The 2003 heatwave summer on the Mentue, predicted by a model calibrated only on the three coolest "
            "summers of the record (2007, 2008, 2011).")
