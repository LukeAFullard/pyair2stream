"""
Run example 02 and refresh the figure its README shows.

    python examples/02_uncertainty/run.py

Runs the four steps the README describes with `pyair2stream.run` (each the same
as a `pyair2stream --config` command), prints the checks to make before
trusting the results, and compares the interval with and without the conformal
margins of step 3. Draws the README's figures with pyair2stream.plots.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyair2stream
from pyair2stream import plots, scenario

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

# The measured temperatures of 2010-2012, used here only to check the intervals.
obs = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv"),
                  parse_dates=["Date"], index_col="Date").T_water

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
measured = obs.reindex(dates).to_numpy(float)
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

# Figures, with the plotting helpers in pyair2stream.plots. Step 4 drew the same 1,000 series as
# step 2, so its saved series give step 2's interval too.
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
summer = (dates >= "2010-06-01") & (dates <= "2010-09-30")
ax = plots.prediction_range(ens[:, summer], dates[summer], level=90, measured=obs, outside=True)
ax.set_title("Mentue, summer 2010 (not used for calibration)", fontsize=10)
ax.figure.savefig(os.path.join(FIG, "interval_summer_2010.png"), dpi=130, bbox_inches="tight")
plt.close(ax.figure)

# Spring 2011, the months of 2010-2012 in which the margin changed most, and every level.
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})
spring = (dates >= "2011-03-01") & (dates <= "2011-06-30")
day_margin = scenario.conformal_margin(margins, 90, window_days=1)
plots.prediction_range(ens[:, spring], dates[spring], level=90, measured=obs, margin=day_margin, floor=0.0,
                       outside=True, ax=left)
left.set_title(f"Mentue, spring 2011: 90% interval, margin {day_margin:+.2f} °C", fontsize=10)
windows = {what: days for days, what in names.items()}
plots.coverage(effect.assign(window_days=effect["values"].map(windows)).rename(
    columns={"inside, without": "inside_before", "inside, with": "inside_after"}), ax=right)
right.set_ylabel("2010-2012 measured values inside it (%)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "interval_conformal.png"), dpi=130, bbox_inches="tight")
