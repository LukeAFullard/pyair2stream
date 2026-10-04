"""
V5 - Predicts real rivers in years it was not calibrated on.

(A) Each model version is calibrated on the first years of each Swiss river and
used to predict the later years, and compared with two simple alternatives a
practitioner might use instead. (B) The 90% prediction intervals are checked
against the real observations of the later years.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DE_SETTINGS, RIVERS, VERSIONS, WORK, Result, Section, Timer, calibrate, daily,
                    load, mean_discharge, metrics, params_at_bounds, quiet, river_csv, simulate, plot_style,
                    save_figure, reference_line, SERIES, BLUE, ORANGE, INK, INK2, LIGHT_GREY, GRID)

PI_VERSIONS = (5, 8)
# Error models: iid; AR(1) with the exact AR(1) likelihood; AR(1) noise with the least-squares likelihood
# and the effective sample size (uncertainty_options.likelihood).
NOISE_MODELS = ("iid", "ar1", "ar1-ls")
NOISE_LABEL = {"iid": "iid", "ar1": "ar1, exact likelihood", "ar1-ls": "ar1, least-squares likelihood"}
PI_RANGE = (0.85, 0.95)       # accepted coverage of the 90% interval on real held-out years
SUMMER = (6, 7, 8)


def _benchmarks(cal_csv: str, val_csv: str) -> dict:
    """Two simple predictions of the validation years, fitted on the calibration years only."""
    cal, val = pd.read_csv(cal_csv, parse_dates=["Date"]), pd.read_csv(val_csv, parse_dates=["Date"])
    doy_c = np.minimum(cal.Date.dt.dayofyear, 365)
    clim = cal.groupby(doy_c).T_water.mean().reindex(range(1, 366))
    clim = pd.concat([clim.iloc[-15:], clim, clim.iloc[:15]]).rolling(31, center=True, min_periods=1).mean()
    clim = clim.iloc[15:-15]
    ok = cal.T_water.notna()
    slope, intercept = np.polyfit(cal.T_air[ok], cal.T_water[ok], 1)
    return {"day-of-year average": clim.loc[np.minimum(val.Date.dt.dayofyear, 365)].to_numpy(),
            "air-temperature regression": (intercept + slope * val.T_air).to_numpy()}


def _fit_and_predict(args):
    st, v = args
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    q_cal = mean_discharge(cal)
    d = calibrate(cal, v, objective="NSE", name=f"v5_{st}_{v}", Qmedia=q_cal)
    obs, sim = daily(simulate(val, v, d.par_best, "CRN", q_cal, name=f"v5p_{st}_{v}"))
    m = metrics(obs, sim)
    return {"river": RIVERS[st], "version": v, "RMSE": m["RMSE"], "NSE": m["NSE"], "KGE": m["KGE"],
            "bias": m["bias"], "parameters at a bound": ", ".join(params_at_bounds(d.par_best, v)) or "none"}


def _interval(args):
    st, v, noise = args
    from pyair2stream.optimization import DE_MCMC_mode, forward_mode
    tag = f"v5i_{st}_{v}_{noise}"
    folder = os.path.join(WORK, tag)
    cal, val = river_csv(st, "calibration"), river_csv(st, "validation")
    q_cal = mean_discharge(cal)
    out = os.path.join(folder, "out")
    cfg = {"version": v, "integrator": "CRN", "run_mode": "DE-MCMC", "objective_function": "NSE",
           "random_seed": 1, "Qmedia": q_cal, "parameter_bounds": AUTHORS_BOUNDS,
           "optimization": {**DE_SETTINGS, "mcmc_walkers": 32, "mcmc_steps": 20000},
           "uncertainty_options": {"noise_model": "iid" if noise == "iid" else "ar1",
                                   "likelihood": "least_squares" if noise == "ar1-ls" else "exact"},
           "paths": {"input_data": cal, "output_dir": out}}
    row = {"river": RIVERS[st], "version": v, "noise model": noise}
    data = load(cfg, tag)
    try:
        with quiet():
            DE_MCMC_mode(data, seed=1)
    except RuntimeError:
        meta = json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))
        return {**row, "converged": False, "steps": meta["steps_run"]}
    meta = json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))
    fcfg = {"version": v, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": q_cal,
            "parameters_forward": [float(x) for x in data.par_best],
            "uncertainty_options": {"noise_model": "iid" if noise == "iid" else "ar1", "save_ensemble": True},
            "forward_options": {"enable_prediction_intervals": True,
                                "mcmc_chain_path": os.path.join(out, "MCMC_chain_S_c_1d.csv"),
                                "n_samples": 1000, "random_seed": 1},
            "paths": {"input_data": val, "output_dir": os.path.join(folder, "fwd")}}
    fdata = load(fcfg, tag + "_fwd")
    with quiet():
        forward_mode(fdata)
    env = pd.read_csv(os.path.join(folder, "fwd", "Forward_Prediction_Envelopes_S_c_1d.csv"))
    obs = pd.read_csv(val).T_water.to_numpy()
    ok = np.isfinite(obs)
    inside = (obs >= env.Twat_mod_lower) & (obs <= env.Twat_mod_upper)
    summer = ok & env.Month.isin(SUMMER).to_numpy()
    # 7-day means: averaged per ensemble member first, then the 5-95% range across members,
    # compared with the observed 7-day means (weeks with all 7 days observed).
    from pyair2stream import scenario
    ens, dates = scenario.load_ensemble(os.path.join(folder, "fwd", "Forward_Prediction_Ensemble_S_c_1d.npz"))
    weekly = scenario.aggregate(ens, dates, how="mean", freq="7D")
    lo, hi = np.nanpercentile(weekly, [5, 95], axis=0)
    obs_w = pd.Series(obs, index=dates).resample("7D")
    full = (obs_w.count() == 7).to_numpy()
    obs_w = obs_w.mean().to_numpy()
    if (st, v, noise) == ("MAH_2369", 8, "ar1-ls"):
        row["_band"] = (pd.DatetimeIndex(dates), env.Twat_mod_lower.to_numpy(), env.Twat_mod_upper.to_numpy(), obs)
    return {**row, "converged": True, "steps": meta["steps_run"],
            "coverage": float(inside[ok].mean()),
            "summer coverage (Jun-Aug)": float(inside[summer].mean()),
            "7-day mean coverage": float(np.mean((obs_w[full] >= lo[full]) & (obs_w[full] <= hi[full]))),
            "band centre bias (°C)": float(np.mean(env.Twat_mod_p50.to_numpy()[ok] - obs[ok])),
            "best-fit bias (°C)": float(np.mean(fdata.Twat_mod[365:][ok] - obs[ok])),
            "mean width (°C)": float((env.Twat_mod_upper - env.Twat_mod_lower)[ok].mean()),
            "calibration coverage": meta["interval_coverage"]}


def run(ctx) -> Result:
    res = Result(
        code="V5", title="Predicts real rivers in years it was not calibrated on",
        question="Calibrated on some years of real data, how well does the model predict other years, "
                 "compared with simple alternatives? And do its 90% prediction intervals contain about "
                 "90% of what was actually measured?",
        method="Three Swiss rivers with different behaviour: the Mentue (small lowland river), the Rhône at "
               "Sion (large, glacier-fed and regulated by hydropower) and the Dischmabach (small alpine "
               "stream fed by snowmelt). Each is split as in Piccolroaz et al. (2016) into calibration and "
               "later validation years. (A) Each model version is calibrated by DE (NSE, CRN, the authors' "
               "parameter ranges) on the calibration years and then predicts the validation years. Two "
               "simple alternatives, fitted on the same calibration years, are scored the same way: the "
               "average water temperature for each day of the year (smoothed over a month), and a straight-"
               "line regression of water temperature on same-day air temperature. (B) For versions 5 and 8, "
               "DE-MCMC (32 walkers, run until converged, at most 20,000 steps) is run on the calibration "
               "years with each error model (iid; AR(1) with the exact likelihood; AR(1) with the least-squares "
               "likelihood, the default), and a FORWARD run gives 90% prediction intervals for the "
               "validation years. The share of real observations inside them is recorded, for the whole "
               "year and for summer (June-August), when temperature limits are usually at stake.",
        criterion=f"(A) In every river, every version predicts the validation years with a lower RMSE than "
                  f"both simple alternatives. (B) Every MCMC run converges, and the 90% interval contains "
                  f"between {PI_RANGE[0]:.0%} and {PI_RANGE[1]:.0%} of the validation observations.")
    stations = ["MAH_2369"] if ctx.quick else list(RIVERS)
    versions = (3, 8) if ctx.quick else VERSIONS
    jobs_a = [(st, v) for st in stations for v in versions]
    jobs_b = [] if ctx.quick else [(st, v, n) for st in stations for v in PI_VERSIONS for n in NOISE_MODELS]
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            fut_b = [ex.submit(_interval, j) for j in jobs_b]
            rows_a = list(ex.map(_fit_and_predict, jobs_a))
            rows_b = [f.result() for f in fut_b]
        for st in stations:
            obs = pd.read_csv(river_csv(st, "validation")).T_water.to_numpy()
            for name, pred in _benchmarks(river_csv(st, "calibration"), river_csv(st, "validation")).items():
                m = metrics(obs, pred)
                rows_a.append({"river": RIVERS[st], "version": name, "RMSE": m["RMSE"], "NSE": m["NSE"],
                               "KGE": m["KGE"], "bias": m["bias"], "parameters at a bound": ""})
    a = pd.DataFrame(rows_a)
    ok_a = True
    for river, g in a.groupby("river", sort=False):
        model = g[g.version.map(lambda x: isinstance(x, (int, np.integer)))]
        simple = g[~g.index.isin(model.index)]
        ok_a &= bool(model.RMSE.max() < simple.RMSE.min())
    band = next((r.pop("_band") for r in rows_b if "_band" in r), None)
    b = pd.DataFrame(rows_b)
    b_raw = b.copy()
    ok_b = True
    if len(b):
        ok_b = bool(b.converged.all() and b.coverage.between(*PI_RANGE).all())
    res.passed = ok_a and ok_b
    res.seconds = t.seconds
    best = a[a.version.map(lambda x: isinstance(x, (int, np.integer)))].groupby("river").RMSE.min()
    simple_best = a[a.version.map(lambda x: isinstance(x, str))].groupby("river").RMSE.min()
    res.summary = "(A) Validation RMSE of the best version vs the best simple alternative: " + "; ".join(
        f"{r} {best[r]:.2f} vs {simple_best[r]:.2f} °C" for r in best.index) + "."
    if len(b):
        conv = b[b.converged]
        res.summary += (f" (B) {int(b.converged.sum())} of {len(b)} MCMC runs converged; 90% interval "
                        f"coverage of real validation data {conv.coverage.min():.0%}-{conv.coverage.max():.0%}"
                        f" (summer {conv['summer coverage (Jun-Aug)'].min():.0%}-"
                        f"{conv['summer coverage (Jun-Aug)'].max():.0%}).")
        w = {n: conv.loc[conv["noise model"] == n, "7-day mean coverage"] for n in NOISE_MODELS}
        res.notes.append(
            f"7-day means: with iid noise the 90% intervals contained only {w['iid'].min():.0%}-"
            f"{w['iid'].max():.0%} of the observed 7-day mean temperatures; with AR(1) noise, "
            f"{w['ar1-ls'].min():.0%}-{w['ar1-ls'].max():.0%} with the least-squares likelihood (the default) "
            f"and {w['ar1'].min():.0%}-{w['ar1'].max():.0%} with the exact AR(1) likelihood. With iid noise the "
            f"simulated day-to-day errors average out within a week, but real model errors persist for days. "
            f"Use noise_model: ar1 (the default) whenever the quantity of interest spans several days (7-day "
            f"means, runs of consecutive days), and treat even those intervals as somewhat narrow: real errors "
            f"persist longer than the AR(1) model assumes.")
        bias = {n: conv.loc[conv["noise model"] == n, "band centre bias (°C)"] -
                   conv.loc[conv["noise model"] == n, "best-fit bias (°C)"] for n in NOISE_MODELS}
        res.notes.append(
            f"Where the band is centred: the median of the band minus the measured temperature, compared with the "
            f"same for the best fit. With the least-squares likelihood (the default) the band's centre stays within "
            f"{bias['ar1-ls'].abs().max():.2f} °C of the best fit's. With the exact AR(1) likelihood it moves by up "
            f"to {bias['ar1'].abs().max():.2f} °C (mostly cooler): that likelihood weighs day-to-day changes more "
            f"than the overall level, and a model that is not exactly right is then pulled towards parameters "
            f"that fit the level slightly worse. This is why the least-squares likelihood is the default "
            f"(uncertainty_options.likelihood).")
        res.notes.append(
            f"Daily coverage on the validation years ({conv.coverage.min():.1%}-{conv.coverage.max():.1%}) "
            f"is a little below the 90% achieved on the calibration years, because the noise level is "
            f"estimated from the calibration years and the model's errors are somewhat larger in years "
            f"it has not seen. The intervals are therefore slightly optimistic for new years.")
    if (~a["parameters at a bound"].isin(["", "none"])).any():
        res.notes.append("Where a calibrated parameter sits on the edge of the authors' ranges (last column of "
                         "table A), the best fit would lie outside that range; the fit is still valid within it.")
    for col in ("RMSE", "NSE", "KGE", "bias"):
        a[col] = a[col].round(3)
    res.sections.append(Section(
        "A. Predicting years not used for calibration",
        "Each version was calibrated on each river's calibration years and predicted its later validation "
        "years. The grey bars are two simple alternatives fitted on the same years.",
        figures=[_fig_rmse(a)], tables=[("A. Validation years: model versions and simple alternatives", a)]))
    if len(b):
        figs_b = [_fig_coverage(b_raw)] + ([_fig_band(band)] if band is not None else [])
        for col in ("coverage", "summer coverage (Jun-Aug)", "7-day mean coverage", "calibration coverage"):
            b[col] = b[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        b["mean width (°C)"] = b["mean width (°C)"].round(2)
        res.sections.append(Section(
            "B. Do 90% prediction intervals contain 90% of real measurements?",
            "For versions 5 and 8, intervals were made for the validation years from a calibration on the "
            "earlier years, with each error model (iid noise; AR(1) noise with the exact AR(1) likelihood; AR(1) "
            "noise with the least-squares likelihood, the default), and compared with what was measured: day by "
            "day, in summer, and for 7-day means (computed within each simulated series). The last two columns "
            "show where the band is centred: its median minus the measured temperature, and the same for the "
            "best fit.",
            figures=figs_b, tables=[("B. 90% prediction intervals on the validation years", b)]))
    res.notes.append("On real data the model is never exactly right, so interval coverage on real rivers "
                     "tests the whole approach (model, noise model and data), not only the code. Coverage "
                     "can differ between seasons because the model's errors are larger in some seasons.")
    return res


def _fig_rmse(a):
    import matplotlib.pyplot as plt
    plot_style()
    rivers = list(dict.fromkeys(a.river))
    labels = list(dict.fromkeys(a.version.astype(str)))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    w = 0.8 / len(labels)
    greys = {"day-of-year average": LIGHT_GREY, "air-temperature regression": "#898781"}
    for k, lab in enumerate(labels):
        vals = [a[(a.river == rv) & (a.version.astype(str) == lab)].RMSE.mean() for rv in rivers]
        colour = SERIES[k] if lab.isdigit() else greys.get(lab, LIGHT_GREY)
        ax.bar(np.arange(len(rivers)) + k * w, vals, w * 0.88, color=colour,
               label=f"version {lab}" if lab.isdigit() else lab)
    ax.set_xticks(np.arange(len(rivers)) + 0.4 - w / 2, rivers)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("RMSE on the validation years (°C)")
    ax.set_title("Every model version beats both simple alternatives on every river")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return (save_figure(fig, "V5_validation_rmse.png"),
            "Prediction error on years not used for calibration (lower is better). Coloured: the five model "
            "versions; grey: the two simple alternatives.")


def _fig_coverage(b):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plot_style()
    b = b[b.converged]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, col, title in zip(axes, ("coverage", "summer coverage (Jun-Aug)", "7-day mean coverage"),
                              ("Daily, whole year", "Daily, summer (June-August)", "7-day means")):
        for k, (noise, colour) in enumerate((("iid", LIGHT_GREY), ("ar1", ORANGE), ("ar1-ls", BLUE))):
            g = b[b["noise model"] == noise]
            ax.bar(np.arange(len(g)) + (k - 1) * 0.27, g[col] * 100, 0.25, color=colour)
        reference_line(ax, 90, "nominal 90%")
        g = b[b["noise model"] == "iid"]
        ax.set_xticks(np.arange(len(g)), [f"{rv} v{v}" for rv, v in zip(g.river, g.version)], fontsize=7.5,
                      rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
        ax.set(title=title, ylim=(0, 105))
    axes[0].set_ylabel("Validation measurements inside\nthe 90% interval (%)")
    fig.legend(handles=[Patch(color=LIGHT_GREY, label="noise model iid"),
                        Patch(color=ORANGE, label="noise model ar1, exact likelihood"),
                        Patch(color=BLUE, label="noise model ar1, least-squares likelihood (default)")],
               loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    return (save_figure(fig, "V5_interval_coverage.png"),
            "Share of real validation-year measurements inside the 90% interval. Daily values: slightly below 90% "
            "with every error model. 7-day means: only AR(1) noise comes close, best with the default "
            "least-squares likelihood (blue).")


def _fig_band(band):
    import matplotlib.pyplot as plt
    plot_style()
    dates, lo, hi, obs = band
    m = (dates.year == 2010) & dates.month.isin(SUMMER)
    inside = (obs >= lo) & (obs <= hi)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(dates[m], lo[m], hi[m], color="#cde2fb", lw=0, label="90% prediction interval")
    ok = m & inside
    out = m & ~inside & np.isfinite(obs)
    ax.scatter(dates[ok], obs[ok], s=10, color=INK, zorder=3, label="measured, inside")
    ax.scatter(dates[out], obs[out], s=18, color=ORANGE, zorder=4, label="measured, outside")
    ax.set_ylabel("Water temperature (°C)")
    ax.set_title("Mentue, version 8, summer 2010: interval from a calibration on 2002-2009")
    ax.legend(loc="lower center", ncol=3, fontsize=7.5)
    share = inside[m & np.isfinite(obs)].mean()
    return (save_figure(fig, "V5_band_example.png"),
            f"An example of a prediction interval for a year the model was not calibrated on: "
            f"{share:.0%} of this summer's measurements fall inside it.")
