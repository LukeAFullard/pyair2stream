"""
Run example 08: a warmer climate, with and without less summer flow.

    python examples/08_climate/run.py

Calibrates with uncertainty on 2002-2009, simulates 2010-2012 1,000 times as measured
(baseline) and with two changes (the air 2 degC warmer; and the air 2 degC warmer with 20%
less discharge in June-September), using the same parameter sets in every run, and
compares them simulation by simulation. A cross-validation of 2002-2009 checks and corrects
the yearly peaks (as in example 03). Writes the tables to output/ and the README's figures
to figures/.
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyair2stream import scenario

WARMING = 2.0          # degC added to the air temperature on every day
FLOW_KEPT = 0.8        # share of the discharge kept in June-September in the second scenario
DRY_MONTHS = (6, 7, 8, 9)
LIMIT_7DAY = 20.0      # degC, illustrative limit on the 7-day mean water temperature
WARM_DAY = 18.0        # degC, illustrative threshold for counting warm days
LEVEL = 90.0
YEARS = (2010, 2011, 2012)
SCENARIOS = ("baseline", "warmer", "warmer_drier")
NAMES = {"baseline": "as measured", "warmer": "air +2 °C", "warmer_drier": "air +2 °C, 20% less summer flow"}
SHORT = {"baseline": "as measured", "warmer": "air +2 °C", "warmer_drier": "+ less flow"}
COLOURS = {"baseline": "tab:gray", "warmer": "tab:orange", "warmer_drier": "tab:red"}

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def run(config):
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", f"examples/08_climate/{config}"],
                   cwd=REPO, check=True)


# --- The scenario inputs: the measured 2010-2012 file, changed. Water temperature is removed:
# it was not measured under these conditions.
measured = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv"), parse_dates=["Date"])
warmer = measured.assign(T_air=measured.T_air + WARMING, T_water=np.nan)
warmer.to_csv(os.path.join(OUT, "warmer_input.csv"), index=False, date_format="%Y-%m-%d")
dry = measured.Date.dt.month.isin(DRY_MONTHS)
warmer.assign(Discharge=measured.Discharge.where(~dry, measured.Discharge * FLOW_KEPT)).to_csv(
    os.path.join(OUT, "warmer_drier_input.csv"), index=False, date_format="%Y-%m-%d")

for config in ("calibrate.yaml", "baseline.yaml", "warmer.yaml", "warmer_drier.yaml", "check.yaml"):
    run(config)

ensemble = "Forward_Prediction_Ensemble_Mentue_c_1d.npz"
ens, dates = {}, None
for s in SCENARIOS:
    ens[s], dates = scenario.load_ensemble(os.path.join(OUT, s, ensemble))

# --- How far outside the calibrated conditions are the scenarios? --------------------------------
cal = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_calibration.csv"), parse_dates=["Date"])
above = float((warmer.T_air > cal.T_air.max()).mean())
summer = measured.Date.dt.month.isin((6, 7, 8))
cal_summer = cal[cal.Date.dt.month.isin((6, 7, 8))]
warmest_summer = cal_summer.groupby(cal_summer.Date.dt.year).T_air.mean().max()
print(f"\nHighest daily air temperature of 2002-2009: {cal.T_air.max():.1f} °C; days of the warmer scenario above it: "
      f"{100 * above:.1f}%.")
print(f"Mean summer air temperature: 2010-2012 {measured.T_air[summer].mean():.1f} °C as measured, "
      f"{warmer.T_air[summer].mean():.1f} °C in the scenario; warmest summer of 2002-2009 {warmest_summer:.1f} °C.")

# The model's own sensitivity to air temperature: at the balance, a 1 degC warmer air raises the
# water temperature by a2 / (a3 + a8 * theta) (version 8; theta = discharge / mean discharge).
meta = json.load(open(os.path.join(OUT, "calibration", "calibration_metadata.json")))
a2, a3, a8 = meta["par_best"][1], meta["par_best"][2], meta["par_best"][7]
theta_summer = float((measured.Discharge[summer] / meta["qmedia"]).median())
print(f"Balance sensitivity a2/(a3 + a8*theta): {a2 / a3:.2f} at very low flow, {a2 / (a3 + a8):.2f} at mean flow, "
      f"{a2 / (a3 + a8 * theta_summer):.2f} at the median summer flow (theta {theta_summer:.2f}).")

# --- The change, simulation by simulation ------------------------------------------------------
rows = []
months = dates.month
for s in ("warmer", "warmer_drier"):
    diff = scenario.paired_difference_from_files(os.path.join(OUT, s, ensemble), os.path.join(OUT, "baseline", ensemble))
    for label, mask in (("summer (Jun-Aug) mean", np.isin(months, (6, 7, 8))),
                        ("winter (Dec-Feb) mean", np.isin(months, (12, 1, 2))),
                        ("whole-year mean", np.ones(len(months), bool))):
        change = np.nanmean(diff[:, mask], axis=1)
        lo, hi = scenario.central_range(change, LEVEL)
        rows.append({"scenario": NAMES[s], "change in": label, "median (°C)": round(float(np.median(change)), 2),
                     f"{LEVEL:g}% range (°C)": f"{lo:+.2f} to {hi:+.2f}"})
change_table = pd.DataFrame(rows)
change_table.to_csv(os.path.join(OUT, "changes.csv"), index=False)
print("\n" + change_table.to_string(index=False))

# Monthly change (mean over 2010-2012), for the figure.
monthly = {}
for s in ("warmer", "warmer_drier"):
    diff = ens[s] - ens["baseline"]
    monthly[s] = np.array([np.nanmean(diff[:, months == m], axis=1) for m in range(1, 13)])   # (12, draws)

# --- Yearly peaks and warm days, checked and corrected by cross-validation (example 03) --------
check = pd.read_csv(os.path.join(OUT, "check", "cv_yearly_statistics.csv"))
dev = {name: check.loc[check.statistic == name, "deviation"] for name in scenario.YEARLY_STATISTICS}
stats = {s: scenario.year_statistics(ens[s], dates, threshold=WARM_DAY) for s in SCENARIOS}
rows, peaks = [], {}
for year in YEARS:
    row = {"year": year}
    for s in SCENARIOS:
        # The same seed in every scenario: the correction's uncertainty is drawn alike for all three.
        peak = scenario.correct_statistic(stats[s][year]["highest 7-day mean"], dev["highest 7-day mean"], seed=year)
        warm = scenario.correct_statistic(stats[s][year]["days above threshold"], dev["days above threshold"], seed=year)
        peaks[(year, s)] = peak
        row[f"P(7-day mean > {LIMIT_7DAY:g} °C), {NAMES[s]}"] = round(float(np.mean(peak > LIMIT_7DAY)), 2)
        row[f"days above {WARM_DAY:g} °C, {NAMES[s]}"] = "{:.0f} ({:.0f} to {:.0f})".format(
            np.median(warm), *scenario.central_range(warm, LEVEL))
    rows.append(row)
yearly = pd.DataFrame(rows)
yearly.to_csv(os.path.join(OUT, "yearly.csv"), index=False)
with pd.option_context("display.width", 250, "display.max_columns", 12):
    print("\n" + yearly.to_string(index=False))
peak_change = np.concatenate([peaks[(y, s)] - peaks[(y, "baseline")] for y in YEARS for s in ("warmer",)])
print(f"\nChange in the yearly highest 7-day mean, air +2 °C: median {np.median(peak_change):+.2f} °C, "
      f"{LEVEL:g}% range {scenario.central_range(peak_change, LEVEL)[0]:+.2f} to "
      f"{scenario.central_range(peak_change, LEVEL)[1]:+.2f} °C")

# --- Figures ---------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(1, 13)
for s, dx in (("warmer", -0.12), ("warmer_drier", 0.12)):
    lo, hi = np.percentile(monthly[s], [5, 95], axis=1)
    ax.vlines(x + dx, lo, hi, color=COLOURS[s], lw=5, alpha=0.5)
    ax.plot(x + dx, np.median(monthly[s], axis=1), "o", color=COLOURS[s], label=NAMES[s])
ax.axhline(WARMING, color="black", lw=0.8, ls="--")
ax.text(0.6, WARMING + 0.04, "the change in air temperature", fontsize=8, va="bottom")
ax.set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_xlim(0.5, 12.5)
ax.set_ylim(0, WARMING + 0.35)
ax.set_ylabel("Change in water temperature (°C)")
ax.set_title("Monthly mean change in water temperature, 2010-2012 (median and 90% range)", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "monthly_change.png"), dpi=130)
plt.close(fig)

fig, axes = plt.subplots(1, len(YEARS), figsize=(10, 3.6), sharey=True)
bins = np.arange(17.0, 25.01, 0.25)
tallest = 0
for ax, year in zip(axes, YEARS):
    lines = [f"P(above {LIMIT_7DAY:g} °C)"]
    for s in SCENARIOS:
        p = peaks[(year, s)]
        style = dict(alpha=0.45) if s == "baseline" else dict(histtype="step", lw=1.8)
        counts, _, _ = ax.hist(p, bins=bins, color=COLOURS[s], label=NAMES[s], **style)
        tallest = max(tallest, counts.max())
        lines.append(f"{SHORT[s]}: {np.mean(p > LIMIT_7DAY):.2f}")
    ax.axvline(LIMIT_7DAY, color="black", ls="--", lw=1)
    ax.text(0.97, 0.96, "\n".join(lines), transform=ax.transAxes, fontsize=7.5, va="top", ha="right")
    ax.set_title(str(year), fontsize=10)
    ax.set_xlabel("Highest 7-day mean (°C)")
axes[0].set_ylim(0, tallest * 1.6)
axes[0].set_ylabel("Simulations")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=8, frameon=False)
fig.suptitle(f"The year's highest 7-day mean in 1,000 simulations (corrected); dashed: the {LIMIT_7DAY:g} °C limit",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(FIG, "yearly_peaks.png"), dpi=130)
plt.close(fig)
