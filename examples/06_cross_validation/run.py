"""
Run example 06: leave-one-year-out cross-validation of model versions 5 and 8.

    python examples/06_cross_validation/run.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import pyair2stream
from pyair2stream import plots
from pyair2stream.cross_validation import jackknife_rows

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VERSIONS = (5, 8)
os.chdir(REPO)      # the paths in the settings files are relative to the repository's top folder

results = {}
for v in VERSIONS:
    result = pyair2stream.run(f"examples/06_cross_validation/version{v}.yaml")
    results[v] = pd.read_csv(os.path.join(result.output_dir, "cv_results.csv"))

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

# The figures, with the plotting helpers in pyair2stream.plots (a cross-validation run also draws
# them for itself: cv_parameters_by_fold.png and cv_error_by_fold.png in its output folder).
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
names = {"a1": "a1  constant", "a2": "a2  air temperature", "a3": "a3  relaxation",
         "a5": "a5  discharge term: constant", "a6": "a6  discharge term: seasonal amplitude",
         "a7": "a7  discharge term: seasonal timing", "a8": "a8  discharge term: relaxation"}
ratios = {"a2/a3  water warming per °C of air": lambda p: p["a2"] / p["a3"],
          "a5/a8  temperature the discharge terms pull to": lambda p: p["a5"] / p["a8"]}
ax = plots.cv_parameters(results[8], combinations=ratios, n_blocks=n_years, labels=names)
ax.figure.suptitle("Version 8 on the Mentue: how firmly the data fix each parameter", x=0.02, ha="left", y=1.02)
ax.figure.savefig(os.path.join(FIG, "parameters_by_fold.png"), dpi=130, bbox_inches="tight")
plt.close(ax.figure)

ax = plots.cv_by_fold({f"version {v}": results[v] for v in VERSIONS})
ax.figure.savefig(os.path.join(FIG, "rmse_by_year.png"), dpi=130, bbox_inches="tight")
plt.close(ax.figure)

# Which parameters change the simulated temperature most? A single calibration of version 8,
# then each parameter moved up and down by 1% of its bound range, one at a time.
result = pyair2stream.run("examples/06_cross_validation/sensitivity.yaml")
sens = pd.read_csv(os.path.join(result.output_dir, "sensitivity_DE_NSE_Mentue.csv"))
sens = sens[sens.Status.isin(["Active", "Bounded"])].assign(parameter=lambda d: "a" + d.Parameter.str[4:])
# The index is in degC per full bound range; per 1% of the range it is easier to read.
sens = sens.assign(per_1pct=sens.Sensitivity_Index / 100).sort_values("per_1pct", ascending=False)
print("\nMean change in simulated water temperature (°C) when the parameter moves by 1% of its bound range")
print(sens[["parameter", "per_1pct", "Status"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.barh(sens.parameter[::-1], sens.per_1pct[::-1], color=plots.PALETTE[0])
ax.set(xlabel="Mean change in water temperature (°C)\nwhen the parameter moves by 1% of its bound range",
       title="Version 8 on the Mentue: local sensitivity")
for side in ("right", "top"):
    ax.spines[side].set_visible(False)
fig.savefig(os.path.join(HERE, "figures", "sensitivity.png"), dpi=130, bbox_inches="tight")
