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

from pyair2stream.cross_validation import jackknife_rows

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

# The parameters fitted without each year, and the jackknife 90% intervals (cv_results.csv).
for v in VERSIONS:
    r = results[v].set_index("fold")
    active = [f"p{i}" for i in range(1, 9) if r[f"p{i}"].abs().max() > 0]
    rows = [f for f in r.index if f.isdigit()] + ["mean", "jackknife_90_lower", "jackknife_90_upper"]
    params = r.loc[rows, active].rename(columns=lambda c: f"a{c[1:]}").round(3)
    print(f"\nVersion {v}: parameters fitted without each year, and 90% jackknife intervals\n"
          + params.to_string())

# Parameters that trade off against each other move together from fold to fold. Their
# combinations can be much better determined than the parameters themselves: a2/a3 is how
# much the balance water temperature rises per degree of air temperature, a5/a8 the
# temperature the discharge-weighted terms pull the water towards.
r8 = results[8].set_index("fold")
folds = r8.loc[[f for f in r8.index if f.isdigit()], [f"p{i}" for i in range(1, 9)]].to_numpy()
n_years = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_calibration.csv"),
                      parse_dates=["Date"]).Date.dt.year.nunique()     # blocks in the whole record
quantities = {"a1  constant": folds[:, 0], "a2  air temperature": folds[:, 1],
              "a3  relaxation": folds[:, 2], "a5  discharge term: constant": folds[:, 4],
              "a6  discharge term: seasonal amplitude": folds[:, 5],
              "a7  discharge term: seasonal timing": folds[:, 6],
              "a8  discharge term: relaxation": folds[:, 7],
              "a2/a3  water warming per °C of air": folds[:, 1] / folds[:, 2],
              "a5/a8  temperature the discharge terms pull to": folds[:, 4] / folds[:, 7]}
print("\nVersion 8: change between folds and 90% jackknife interval, as % of the mean of the folds")
rows = []
for name, x in quantities.items():
    lo, hi = (jackknife_rows(x[:, None], n_years)[k]["p1"] for k in (1, 2))
    rows.append((name, 100 * (x - x.mean()) / abs(x.mean()), 100 * (lo - x.mean()) / abs(x.mean()),
                 100 * (hi - x.mean()) / abs(x.mean())))
    print(f"  {name:48s} folds {x.min():7.3f} to {x.max():7.3f}   interval {lo:7.3f} to {hi:7.3f}"
          f"   (±{rows[-1][3]:.0f}%)")
a4 = folds[:, 3]
lo, hi = (jackknife_rows(a4[:, None], n_years)[k]["p1"] for k in (1, 2))
print(f"  a4 (close to zero; not shown as %): folds {a4.min():.3f} to {a4.max():.3f}, "
      f"interval {lo:.3f} to {hi:.3f}")

fig, ax = plt.subplots(figsize=(8, 4.4))
blue = "#2a78d6"
for k, (name, dev, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - k + (0 if "/" in name else 0.6)      # a gap between parameters and ratios
    ax.plot([lo, hi], [y, y], color=blue, alpha=0.3, lw=7, solid_capstyle="round",
            label="90% jackknife interval" if k == 0 else None)
    ax.scatter(dev, np.full(len(dev), y), s=34, color=blue, edgecolor="white", linewidth=1.2, zorder=3,
               label="fitted without one year (6 folds)" if k == 0 else None)
    ax.text(-62, y, name, ha="right", va="center", fontsize=8.5, color="#0b0b0b")
ax.axvline(0, color="#8a8984", lw=1)
ax.set(xlim=(-60, 60), yticks=[], xlabel="Difference from the mean of the folds (% of its value)",
       title="Version 8 on the Mentue: how firmly the data fix each parameter")
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "parameters_by_fold.png"), dpi=130, bbox_inches="tight")

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
