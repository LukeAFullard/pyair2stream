"""
Example 04, other flow changes: adding 30% of the flow, and adding or taking 30% on the
days of lowest or highest flow only, with versions 4, 7 and 8.

    python examples/04_scenario/flow_regimes.py

Uses the calibrations and baselines of compare_versions.py (run it first). Each scenario is
paired with its version's baseline. Writes output/regimes/<version>/ and
output/flow_regimes_summary.csv, the table in the README's "Other flow changes".
About 10 minutes.
"""
import os

import numpy as np
import pandas as pd
import yaml

import pyair2stream
from pyair2stream import scenario

REGIME = 0.10          # the lowest and the highest 10% of the 2010-2012 days, by measured flow
WARM_DAY = 18.0        # °C
HOTTEST = 0.10         # the hottest 10% of days, by the baseline's median simulation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VERSIONS = os.path.join(HERE, "output", "versions")
os.chdir(REPO)      # the paths in the settings file are relative to the repository's top folder

with open("examples/04_scenario/settings.yaml") as f:
    settings = yaml.safe_load(f)
measured = pd.read_csv("data/switzerland/MAH_2369_validation.csv")
flow = measured.Discharge
low = flow <= flow.quantile(REGIME)
high = flow >= flow.quantile(1 - REGIME)
everyday = pd.Series(True, index=flow.index)
SCENARIOS = {          # name: (days changed, factor)
    "30% taken, every day": (everyday, 0.7),
    "30% added, every day": (everyday, 1.3),
    "30% taken, lowest flows": (low, 0.7),
    "30% added, lowest flows": (low, 1.3),
    "30% taken, highest flows": (high, 0.7),
    "30% added, highest flows": (high, 1.3),
}
print(f"Lowest 10% of flows: below {flow.quantile(REGIME):.3f} m3/s, months "
      f"{sorted(set(pd.to_datetime(measured.Date[low]).dt.month))}; highest 10%: above "
      f"{flow.quantile(1 - REGIME):.2f} m3/s, months {sorted(set(pd.to_datetime(measured.Date[high]).dt.month))}")


def median_range(values, dp):
    return f"{np.median(values):+.{dp}f} ({np.percentile(values, 5):+.{dp}f} to {np.percentile(values, 95):+.{dp}f})"


def mean_of_years(stats, name):
    """The yearly statistic `name`, averaged over the years: one value per simulation."""
    return np.mean([stats[year][name] for year in stats], axis=0)


rows = []
for version in (4, 7, 8):
    folder = os.path.join(VERSIONS, f"v{version}")
    if not os.path.isdir(os.path.join(folder, "baseline")):
        raise SystemExit(f"{folder}/baseline not found: run examples/04_scenario/compare_versions.py first")
    m = pyair2stream.Model({**settings, "version": version},
                           output_dir=os.path.join(HERE, "output", "regimes", f"v{version}"))
    m.use_calibration(os.path.join(folder, "calibration"))
    m.use_prediction("baseline", os.path.join(folder, "baseline"))
    base, dates = m.ensemble("baseline")
    hottest = np.median(base, axis=0) >= np.quantile(np.median(base, axis=0), 1 - HOTTEST)
    base_stats = scenario.year_statistics(base, dates, WARM_DAY)
    for k, (name, (days, factor)) in enumerate(SCENARIOS.items()):
        data = measured.assign(Discharge=np.where(days, flow * factor, flow), T_water=np.nan)
        m.predict(data, name=f"scenario{k + 1}", paired_with="baseline")
        diff = m.difference(f"scenario{k + 1}", "baseline")
        ens, _ = m.ensemble(f"scenario{k + 1}")
        stats = scenario.year_statistics(ens, dates, WARM_DAY)
        rows.append({
            "version": version, "scenario": name,
            "days changed": median_range(np.mean(diff[:, days.to_numpy()], axis=1), 2),
            "hottest 10% of days": median_range(np.mean(diff[:, hottest], axis=1), 2),
            "highest 7-day mean": median_range(mean_of_years(stats, "highest 7-day mean")
                                               - mean_of_years(base_stats, "highest 7-day mean"), 2),
            "highest daily mean": median_range(mean_of_years(stats, "highest daily mean")
                                               - mean_of_years(base_stats, "highest daily mean"), 2),
            f"days above {WARM_DAY:g} °C per year": median_range(
                mean_of_years(stats, "days above threshold") - mean_of_years(base_stats, "days above threshold"), 1),
            "largest daily warming": median_range(np.max(diff, axis=1), 2),
            "largest daily cooling": median_range(np.min(diff, axis=1), 2),
        })

table = pd.DataFrame(rows)
table.to_csv(os.path.join(HERE, "output", "flow_regimes_summary.csv"), index=False)
print(f"\nShare of the hottest 10% of days that are also among the lowest-flow days: "
      f"{np.mean(low.to_numpy()[hottest]):.0%}")
print("\nChange from the baseline, °C (days: per year); median and 90% range over the paired simulations:")
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(table.to_string(index=False))
