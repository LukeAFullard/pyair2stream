"""
Run example 03: the probability that a temperature limit was exceeded.

    python examples/03_compliance/run.py

Steps 1 and 2 are the two pyair2stream commands in the README. Step 3, the
analysis, is below: it uses every simulated series, not the daily interval.
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyair2stream import scenario

LIMIT_7DAY = 20.0      # °C, illustrative limit on the 7-day mean water temperature
WARM_DAY = 18.0        # °C, illustrative threshold for counting warm days
YEARS = (2010, 2011, 2012)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")

# Steps 1 and 2: calibrate with uncertainty, then simulate 2010-2012 1000 times.
for step in ("calibrate", "predict"):
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", f"examples/03_compliance/{step}.yaml"],
                   cwd=REPO, check=True)

# Step 3. `ens` holds 1000 simulated series (rows) of daily water temperature, each
# with its own parameters and its own model error; `dates` labels the columns.
ens, dates = scenario.load_ensemble(os.path.join(OUT, "prediction", "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
sims = pd.DataFrame(ens.T, index=dates)            # one column per simulation
week = sims.rolling(7).mean()                      # 7-day means, within each simulation

# The measured temperatures, used here only to check the answer.
obs = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv"),
                  parse_dates=["Date"], index_col="Date").T_water
obs_week = obs.rolling(7).mean()

rows, peaks = [], {}
for year in YEARS:
    y = str(year)
    peak = week.loc[y].max().to_numpy()            # each simulation's highest 7-day mean
    warm = scenario.exceedance(sims.loc[y].T.to_numpy(), WARM_DAY)   # warm days, per simulation
    peaks[year] = peak
    rows.append({
        "year": year,
        f"P(7-day mean > {LIMIT_7DAY:g} °C)": round(float(np.mean(peak > LIMIT_7DAY)), 2),
        "highest 7-day mean (°C), 90% range": f"{np.percentile(peak, 5):.1f} to {np.percentile(peak, 95):.1f}",
        "measured highest 7-day mean (°C)": round(float(obs_week.loc[y].max()), 1),
        f"days above {WARM_DAY:g} °C, median (90% range)":
            f"{np.median(warm):.0f} ({np.percentile(warm, 5):.0f} to {np.percentile(warm, 95):.0f})",
        f"measured days above {WARM_DAY:g} °C": int((obs.loc[y] > WARM_DAY).sum()),
    })
table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "compliance_summary.csv"), index=False)
with pd.option_context("display.width", 200, "display.max_columns", 10):
    print("\n" + table.to_string(index=False))

fig, axes = plt.subplots(1, len(YEARS), figsize=(9, 3), sharey=True)
for ax, year in zip(axes, YEARS):
    ax.hist(peaks[year], bins=np.arange(17.5, 23.01, 0.25), color="tab:blue", alpha=0.6)
    ax.axvline(LIMIT_7DAY, color="black", ls="--", lw=1)
    ax.axvline(obs_week.loc[str(year)].max(), color="tab:red", lw=2)
    ax.set(title=f"{year}: P(exceeded) = {np.mean(peaks[year] > LIMIT_7DAY):.2f}",
           xlabel="Highest 7-day mean (°C)")
axes[0].set_ylabel("Simulations")
fig.suptitle(f"Blue: 1000 simulations. Dashed: the {LIMIT_7DAY:g} °C limit. Red: measured.", fontsize=9, y=0.02)
fig.tight_layout(rect=(0, 0.06, 1, 1))
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "peak_7day_mean.png"), dpi=130, bbox_inches="tight")
