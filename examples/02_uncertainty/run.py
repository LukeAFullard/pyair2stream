"""
Run example 02 and refresh the figure its README shows.

    python examples/02_uncertainty/run.py

Runs the two steps the README describes (the same as the two pyair2stream
commands), prints the checks to make before trusting the results, and draws
the prediction interval for the summer of 2010.
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")

for step in ("calibrate", "predict"):
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", f"examples/02_uncertainty/{step}.yaml"],
                   cwd=REPO, check=True)

# Check 1: did the MCMC converge? Check 2: does the interval contain about 90% of observations?
cal = json.load(open(os.path.join(OUT, "calibration", "MCMC_chain_Mentue_c_1d_meta.json")))
pred = json.load(open(os.path.join(OUT, "prediction", "Forward_Prediction_Ensemble_Mentue_c_1d_meta.json")))
print(f"\nConverged: {cal['converged']} after {cal['steps_run']} steps "
      f"(autocorrelation time {cal['max_autocorr_time']:.0f} steps, split-Rhat {cal['max_split_rhat']:.3f})")
print(f"Share of observed days inside the 90% interval: calibration years {cal['interval_coverage']:.1%}, "
      f"validation years {pred['interval_coverage']:.1%}")

# Figure: one summer of the validation years.
env = pd.read_csv(os.path.join(OUT, "prediction", "Forward_Prediction_Envelopes_Mentue_c_1d.csv"))
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
