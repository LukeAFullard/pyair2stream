"""
Run example 04: the effect of abstracting 30% of the flow, with its uncertainty.

    python examples/04_scenario/run.py

Makes the scenario's input file, runs the three pyair2stream steps in the
README, then computes the paired difference between the two scenarios.
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyair2stream import scenario

FLOW_KEPT = 0.7        # the abstraction scenario keeps 70% of the measured discharge
SUMMER = (6, 7, 8)
WINTER = (12, 1, 2)
WARM_DAY = 18.0        # °C, illustrative threshold for counting warm days

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")


def pyair2stream(config):
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", f"examples/04_scenario/{config}"],
                   cwd=REPO, check=True)


# The scenario's input: the measured file with less discharge. Water temperature is
# removed: it was not measured under this scenario.
os.makedirs(OUT, exist_ok=True)
measured = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv"))
measured.assign(Discharge=measured.Discharge * FLOW_KEPT, T_water=np.nan).to_csv(
    os.path.join(OUT, "abstraction_input.csv"), index=False)

for config in ("calibrate.yaml", "baseline.yaml", "abstraction.yaml"):
    pyair2stream(config)

ensemble = "Forward_Prediction_Ensemble_Mentue_c_1d.npz"
base_file, abst_file = os.path.join(OUT, "baseline", ensemble), os.path.join(OUT, "abstraction", ensemble)
# Abstraction minus baseline, simulation by simulation (checks both used the same parameter sets).
diff = scenario.paired_difference_from_files(abst_file, base_file)
base, dates = scenario.load_ensemble(base_file)
abst, _ = scenario.load_ensemble(abst_file)
summer = np.isin(dates.month, SUMMER)
winter = np.isin(dates.month, WINTER)

# Each simulation's average summer warming, then its range across simulations.
effect = np.nanmean(diff[:, summer], axis=1)
# For comparison: the same statistic if the two runs were NOT paired (simulations matched at random).
shuffled = abst[np.random.default_rng(0).permutation(len(abst))]
unpaired = np.nanmean(shuffled[:, summer] - base[:, summer], axis=1)

extra_warm = (scenario.exceedance(abst, WARM_DAY) - scenario.exceedance(base, WARM_DAY)) / 3   # per year
rows = [
    ("Average summer (Jun-Aug) change, °C", effect, 2),
    ("Average winter (Dec-Feb) change, °C", np.nanmean(diff[:, winter], axis=1), 2),
    ("Largest daily warming, °C", np.nanmax(diff, axis=1), 2),
    (f"Extra days per year above {WARM_DAY:g} °C", extra_warm, 1),
    ("Summer change if the runs were NOT paired, °C", unpaired, 2),
]
table = pd.DataFrame([{"quantity": name, "median": round(float(np.median(v)), dp),
                       "90% range": f"{np.percentile(v, 5):.{dp}f} to {np.percentile(v, 95):.{dp}f}"}
                      for name, v, dp in rows])
table.to_csv(os.path.join(OUT, "scenario_summary.csv"), index=False)
print("\n" + table.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
lo, med, hi = np.nanpercentile(diff, [5, 50, 95], axis=0)
ax1.fill_between(dates, lo, hi, color="tab:red", alpha=0.25, lw=0, label="90% range")
ax1.plot(dates, med, color="tab:red", lw=1, label="median")
ax1.axhline(0, color="black", lw=0.8)
ax1.set(ylabel="Change from abstraction (°C)", title="Daily effect, 2010-2012")
ax1.legend(frameon=False, fontsize=8)
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
bins = np.linspace(min(unpaired.min(), effect.min()), max(unpaired.max(), effect.max()), 40)
ax2.hist(unpaired, bins=bins, color="0.7", label="not paired")
ax2.hist(effect, bins=bins, color="tab:red", alpha=0.8, label="paired")
ax2.set(xlabel="Average summer change (°C)", ylabel="Simulations", title="Summer average")
ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "abstraction_effect.png"), dpi=130, bbox_inches="tight")
