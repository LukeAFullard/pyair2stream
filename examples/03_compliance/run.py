"""
Run example 03: the probability that a temperature limit was exceeded.

    python examples/03_compliance/run.py

Steps 1-3 are the three pyair2stream commands in the README. Step 4, the
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

from pyair2stream import plots, scenario

LIMIT_7DAY = 20.0      # °C, illustrative limit on the 7-day mean water temperature
WARM_DAY = 18.0        # °C, illustrative threshold for counting warm days (also in check.yaml)
LEVEL = 90.0           # % width of the reported ranges, e.g. 95 (a level other than 50, 80, 90 or 95 also
                       # needs uncertainty_options.prediction_interval in check.yaml)
YEARS = (2010, 2011, 2012)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")

# Steps 1-3: calibrate with uncertainty, simulate 2010-2012 1000 times, and check the
# yearly statistics by cross-validation of the calibration years.
for step in ("calibrate", "predict", "check"):
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", f"examples/03_compliance/{step}.yaml"],
                   cwd=REPO, check=True)

# Step 4. `ens` holds 1000 simulated series (rows) of daily water temperature, each
# with its own parameters and its own model error; `dates` labels the columns.
ens, dates = scenario.load_ensemble(os.path.join(OUT, "prediction", "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
stats = scenario.year_statistics(ens, dates, threshold=WARM_DAY)    # each year's statistics, per simulation

# The cross-validation check: how far the measured statistic was from the predicted median
# in each held-out calibration year (measured minus median).
check = pd.read_csv(os.path.join(OUT, "check", "cv_yearly_statistics.csv"))
deviations = {name: check.loc[check.statistic == name, "deviation"] for name in scenario.YEARLY_STATISTICS}
print("\n" + pd.read_csv(os.path.join(OUT, "check", "cv_yearly_statistics_summary.csv"))[
    ["statistic", "n_years", f"share_inside_{LEVEL:g}", "mean_deviation", "mean_deviation_ci95_lower",
     "mean_deviation_ci95_upper"]].round(2).to_string(index=False))

# The measured temperatures, used here only to check the answer.
obs = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv"),
                  parse_dates=["Date"], index_col="Date").T_water
measured = scenario.year_statistics(obs.to_numpy(), obs.index, threshold=WARM_DAY)

rows, peaks = [], {}
for year in YEARS:
    peak = stats[year]["highest 7-day mean"]                                       # one value per simulation
    peak_c = scenario.correct_statistic(peak, deviations["highest 7-day mean"], seed=year)
    warm = stats[year]["days above threshold"]
    warm_c = scenario.correct_statistic(warm, deviations["days above threshold"], seed=year)
    peaks[year] = (peak, peak_c)
    rows.append({
        "year": year,
        f"P(7-day mean > {LIMIT_7DAY:g} °C)": round(float(np.mean(peak > LIMIT_7DAY)), 2),
        "P, corrected": round(float(np.mean(peak_c > LIMIT_7DAY)), 2),
        f"highest 7-day mean, {LEVEL:g}% range (°C)": "{:.1f} to {:.1f}".format(*scenario.central_range(peak, LEVEL)),
        f"{LEVEL:g}% range, corrected": "{:.1f} to {:.1f}".format(*scenario.central_range(peak_c, LEVEL)),
        "measured (°C)": round(float(measured[year]["highest 7-day mean"][0]), 1),
        f"days above {WARM_DAY:g} °C, median ({LEVEL:g}% range)":
            "{:.0f} ({:.0f} to {:.0f})".format(np.median(warm), *scenario.central_range(warm, LEVEL)),
        "days, corrected": "{:.0f} ({:.0f} to {:.0f})".format(np.median(warm_c), *scenario.central_range(warm_c, LEVEL)),
        "days, measured": int(measured[year]["days above threshold"][0]),
    })
table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "compliance_summary.csv"), index=False)
with pd.option_context("display.width", 250, "display.max_columns", 12):
    print("\n" + table.to_string(index=False))

# Figures, with the plotting helpers in pyair2stream.plots.
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# The 7-day mean in the 1000 simulations (uncorrected), the measurements and the limit.
ax = plots.prediction_range(ens, dates, window=7, level=LEVEL, measured=obs, limit=LIMIT_7DAY)
ax.set_title("7-day mean water temperature, 2010-2012: 1000 simulations", fontsize=10)
ax.figure.savefig(os.path.join(FIG, "prediction_7day_mean.png"), dpi=130, bbox_inches="tight")
plt.close(ax.figure)

# Each year's highest 7-day mean, corrected and not, against the limit; the numbers are P(exceeded).
ax = plots.yearly_statistic({"uncorrected": {y: peaks[y][0] for y in YEARS},
                             "corrected": {y: peaks[y][1] for y in YEARS}},
                            limit=LIMIT_7DAY, measured=measured, level=LEVEL,
                            colors=[plots.MUTED, plots.PALETTE[0]])
ax.figure.savefig(os.path.join(FIG, "peak_7day_mean.png"), dpi=130, bbox_inches="tight")
plt.close(ax.figure)
