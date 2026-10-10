"""
Run example 04: the effect of abstracting 30% of the flow, with its uncertainty.

    python examples/04_scenario/run.py

Makes the scenario's input, runs the three steps in the README with
pyair2stream.Model, then computes the paired difference between the two scenarios.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyair2stream
from pyair2stream import plots, scenario

FLOW_KEPT = 0.7        # the abstraction scenario keeps 70% of the measured discharge
SUMMER = (6, 7, 8)
WINTER = (12, 1, 2)
WARM_DAY = 18.0        # °C, illustrative threshold for counting warm days

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")
os.chdir(REPO)      # the paths in the settings file are relative to the repository's top folder

# The scenario's input: the measured file with less discharge. Water temperature is
# removed: it was not measured under this scenario.
validation = "data/switzerland/MAH_2369_validation.csv"
measured = pd.read_csv(validation)
abstraction = measured.assign(Discharge=measured.Discharge * FLOW_KEPT, T_water=np.nan)

m = pyair2stream.Model("examples/04_scenario/settings.yaml")
m.calibrate()                                                          # with uncertainty, as in example 02
m.predict(validation, name="baseline")                                 # 1,000 simulations, measured discharge
m.predict(abstraction, name="abstraction", paired_with="baseline")    # the same parameter sets, 70% of the flow

# Abstraction minus baseline, simulation by simulation (checks both used the same parameter sets).
diff = m.difference("abstraction", "baseline")
base, dates = m.ensemble("baseline")
abst, _ = m.ensemble("abstraction")
summer = np.isin(dates.month, SUMMER)
winter = np.isin(dates.month, WINTER)

# Each simulation's average summer warming, then its range across simulations.
effect = np.nanmean(diff[:, summer], axis=1)
# For comparison: the same statistic if the two runs were NOT paired (simulations matched at random).
shuffled = abst[np.random.default_rng(0).permutation(len(abst))]
unpaired = np.nanmean(shuffled[:, summer] - base[:, summer], axis=1)

# What pairing means, on the first three simulations (README, "Paired and not paired").
base_summer = np.nanmean(base[:, summer], axis=1)
abst_summer = np.nanmean(abst[:, summer], axis=1)
print("\nAverage summer water temperature in the 1000 baseline simulations: 90% range "
      "{:.1f} to {:.1f} °C".format(*np.percentile(base_summer, [5, 95])))
for i in range(3):
    print(f"Simulation {i + 1}: baseline {base_summer[i]:.2f} °C, abstraction {abst_summer[i]:.2f} °C, "
          f"difference {abst_summer[i] - base_summer[i]:+.2f} °C")
print(f"Not paired: abstraction of simulation 3 minus baseline of simulation 2: "
      f"{abst_summer[2] - base_summer[1]:+.2f} °C; abstraction of simulation 2 minus baseline of simulation 3: "
      f"{abst_summer[1] - base_summer[2]:+.2f} °C")

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
# Left: the daily change with its 90% range (pyair2stream.plots). Right: why pairing matters.
plots.change(diff, dates, by="day", ax=ax1, colors=[plots.PALETTE[1]])
ax1.set(ylabel="Change from abstraction (°C)", title="Daily effect, 2010-2012")
bins = np.linspace(min(unpaired.min(), effect.min()), max(unpaired.max(), effect.max()), 40)
ax2.hist(unpaired, bins=bins, color=plots.MUTED, alpha=0.6, label="not paired")
ax2.hist(effect, bins=bins, color=plots.PALETTE[1], alpha=0.8, label="paired")
ax2.set(xlabel="Average summer change (°C)", ylabel="Simulations", title="Summer average")
ax2.legend(frameon=False, fontsize=8)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "abstraction_effect.png"), dpi=130, bbox_inches="tight")
