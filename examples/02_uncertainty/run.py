"""
Run example 02 and refresh the figure its README shows.

    python examples/02_uncertainty/run.py

Runs the four steps the README describes with `pyair2stream.run` (each the same
as a `pyair2stream --config` command), prints the checks to make before
trusting the results, draws the prediction interval for the summer of 2010, and
compares the interval with and without the conformal margins of step 3.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyair2stream
from pyair2stream import scenario

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.chdir(REPO)      # the paths in the settings files are relative to the repository's top folder

results = {step: pyair2stream.run(f"examples/02_uncertainty/{step}.yaml")
           for step in ("calibrate", "predict", "check", "predict_conformal")}

# Check 1: did the MCMC converge? Check 2: does the interval contain about 90% of observations?
cal = json.load(open(os.path.join(results["calibrate"].output_dir, "MCMC_chain_Mentue_c_1d_meta.json")))
pred = json.load(open(os.path.join(results["predict"].output_dir, "Forward_Prediction_Ensemble_Mentue_c_1d_meta.json")))
print(f"\nConverged: {cal['converged']} after {cal['steps_run']} steps "
      f"(autocorrelation time {cal['max_autocorr_time']:.0f} steps, split-Rhat {cal['max_split_rhat']:.3f})")
print(f"Share of observed days inside the 90% interval: calibration years {cal['interval_coverage']:.1%}, "
      f"validation years {pred['interval_coverage']:.1%}")

# Figure: one summer of the validation years.
env = pd.read_csv(os.path.join(results["predict"].output_dir, "Forward_Prediction_Envelopes_Mentue_c_1d.csv"))
env["Date"] = pd.to_datetime(env[["Year", "Month", "Day"]])
obs = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv"), parse_dates=["Date"])
d = env.merge(obs, on="Date")
d = d[(d.Date >= "2010-06-01") & (d.Date <= "2010-09-30")]
inside = (d.T_water >= d.Twat_mod_lower) & (d.T_water <= d.Twat_mod_upper)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.fill_between(d.Date, d.Twat_mod_lower, d.Twat_mod_upper, color="tab:blue", alpha=0.25, lw=0,
                label="90% prediction interval")
ax.plot(d.Date, d.Twat_mod_p50, color="tab:blue", lw=1.2, label="median prediction")
ax.scatter(d.Date[inside], d.T_water[inside], s=10, color="black", label="observed, inside", zorder=3)
ax.scatter(d.Date[~inside], d.T_water[~inside], s=18, color="tab:red", label="observed, outside", zorder=3)
ax.set(ylabel="Water temperature (°C)", title="Mentue, summer 2010 (not used for calibration)")
ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower center")
fig.autofmt_xdate()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "interval_summer_2010.png"), dpi=130, bbox_inches="tight")


# Steps 3 and 4: the conformal margins, and their effect on the 2010-2012 intervals.
margins = scenario.read_conformal_margins(os.path.join(results["check"].output_dir, "cv_conformal_margins.csv"))
print("\nConformal margins from the 8 held-out years (step 3), 90% interval:")
names = {1: "days", 7: "7-day means", 30: "30-day means"}
for r in margins[margins.level == 90].itertuples():
    print(f"  {names[r.window_days]:13s} margin {r.margin:+.2f} °C; held-out values inside {r.inside_before:.1%} "
          f"without, {r.inside_after:.1%} with a margin from the other years")

# The same 1,000 series as step 2 (same chain, seed and settings), saved by step 4.
conformal_dir = results["predict_conformal"].output_dir
ens, dates = scenario.load_ensemble(os.path.join(conformal_dir, "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
measured = obs.set_index("Date").T_water.reindex(dates).to_numpy(float)
rows = []
for window, what in names.items():
    e = pd.DataFrame(ens.T).rolling(window).mean().to_numpy().T if window > 1 else ens      # moving means, per series
    o = pd.Series(measured).rolling(window).mean().to_numpy() if window > 1 else measured
    ok = np.isfinite(o) & np.all(np.isfinite(e), axis=0)
    for level in (50, 80, 90, 95):
        lo, hi = scenario.central_range(e[:, ok], level, axis=0)
        m = scenario.conformal_margin(margins, level, window_days=window)
        clo, chi = scenario.conformal_range(e[:, ok], level, m, floor=0.0)      # floor: Tice_cover (0 °C)
        rows.append({"values": what, "level": level, "margin (°C)": round(m, 2),
                     "inside, without": np.mean((o[ok] >= lo) & (o[ok] <= hi)),
                     "inside, with": np.mean((o[ok] >= clo) & (o[ok] <= chi)),
                     "mean width without (°C)": np.mean(hi - lo), "mean width with (°C)": np.mean(chi - clo)})
effect = pd.DataFrame(rows)
print("\n2010-2012, share of measured values inside the interval, without and with the margins:")
print(effect.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Figure: spring 2011, the months of 2010-2012 in which the margin changed most, with and without it.
env_c = pd.read_csv(os.path.join(conformal_dir, "Forward_Prediction_Envelopes_Mentue_c_1d.csv"))
env_c["Date"] = pd.to_datetime(env_c[["Year", "Month", "Day"]])
d = env_c.merge(obs, on="Date")
d = d[(d.Date >= "2011-03-01") & (d.Date <= "2011-06-30")]
inside = (d.T_water >= d.Twat_mod_lower) & (d.T_water <= d.Twat_mod_upper)
inside_c = (d.T_water >= d.Twat_mod_lower_conformal) & (d.T_water <= d.Twat_mod_upper_conformal)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [2.1, 1]})
ax = axes[0]
ax.fill_between(d.Date, d.Twat_mod_lower, d.Twat_mod_upper, color="tab:blue", alpha=0.25, lw=0,
                label="90% interval")
ax.plot(d.Date, d.Twat_mod_lower_conformal, color="tab:blue", lw=0.9, ls="--", label="with conformal margin")
ax.plot(d.Date, d.Twat_mod_upper_conformal, color="tab:blue", lw=0.9, ls="--")
ax.scatter(d.Date[inside], d.T_water[inside], s=10, color="black", label="observed, inside", zorder=3)
ax.scatter(d.Date[~inside & inside_c], d.T_water[~inside & inside_c], s=22, color="tab:orange",
           label="inside only with the margin", zorder=3)
ax.scatter(d.Date[~inside_c], d.T_water[~inside_c], s=18, color="tab:red", label="outside both", zorder=3)
ax.set(ylabel="Water temperature (°C)", title="Mentue, spring 2011: 90%% interval, margin %+.2f °C" % margins[
    (margins.window_days == 1) & (margins.level == 90)].margin.iloc[0])
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha("right")
ax = axes[1]
for what, marker in (("days", "o"), ("30-day means", "s")):
    g = effect[effect["values"] == what]
    ax.plot(g.level, 100 * g["inside, without"], marker=marker, ms=5, lw=1, color="tab:gray",
            label=f"{what}, without")
    ax.plot(g.level, 100 * g["inside, with"], marker=marker, ms=5, lw=1, color="tab:blue", label=f"{what}, with")
ax.plot([40, 100], [40, 100], color="black", lw=0.8, ls=":")
ax.set(xlim=(40, 100), ylim=(40, 100), xlabel="Stated level (%)", ylabel="Measured values inside (%)",
       title="2010-2012, every level")
ax.set_aspect("equal")
ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "interval_conformal.png"), dpi=130, bbox_inches="tight")
