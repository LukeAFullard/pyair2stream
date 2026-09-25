"""
Run example 06: leave-one-year-out cross-validation of model versions 5 and 8.

    python examples/06_cross_validation/run.py
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VERSIONS = (5, 8)

results = {}
for v in VERSIONS:
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config",
                    f"examples/06_cross_validation/version{v}.yaml"], cwd=REPO, check=True)
    results[v] = pd.read_csv(os.path.join(HERE, "output", f"version{v}", "cv_results.csv"))

# Error on each held-out year, and over all held-out days together ("pooled").
table = pd.DataFrame({"held-out year": results[5].fold})
for v in VERSIONS:
    table[f"RMSE version {v} (°C)"] = results[v].RMSE.round(2)
table = table[table["held-out year"].isin([str(y) for y in range(1900, 2100)] + ["pooled"])]
print("\n" + table.to_string(index=False))

# How much each parameter changes when a different year is held out.
for v in VERSIONS:
    years = results[v][results[v].fold.str.isdigit()]
    active = [c for c in years.columns if c.startswith("p") and years[c].abs().max() > 0]
    spread = (years[active].std() / years[active].mean().abs()).round(2)
    print(f"Version {v}: parameter spread between folds (SD / |mean|): " +
          ", ".join(f"a{c[1:]} {s:.2f}" for c, s in spread.items()))

years = results[5][results[5].fold.str.isdigit()].fold
fig, ax = plt.subplots(figsize=(6, 3.2))
for k, v in enumerate(VERSIONS):
    r = results[v][results[v].fold.str.isdigit()]
    ax.bar(np.arange(len(r)) + (k - 0.5) * 0.38, r.RMSE, 0.38, label=f"version {v}")
ax.set_xticks(np.arange(len(years)), years)
ax.set(xlabel="Year held out", ylabel="RMSE on that year (°C)", title="Mentue, leave-one-year-out")
ax.legend(frameon=False)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "rmse_by_year.png"), dpi=130, bbox_inches="tight")
