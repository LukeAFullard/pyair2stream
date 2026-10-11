"""
Example 04's abstraction for each model version that uses discharge (4, 7 and 8).

    python examples/04_scenario/compare_versions.py

The same settings, data and seed as run.py, with only `version` changed. Each version
is calibrated with uncertainty, then predicts 2010-2012 with the measured flow and with
70% of it, paired. Writes output/versions/<version>/ and output/versions_summary.csv,
the tables in the README's "Does the model version matter?". About 10 minutes.
"""
import json
import os

import numpy as np
import pandas as pd
import yaml

import pyair2stream
from pyair2stream import scenario

FLOW_KEPT = 0.7
SUMMER = (6, 7, 8)
WINTER = (12, 1, 2)
WARM_DAY = 18.0

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.chdir(REPO)      # the paths in the settings file are relative to the repository's top folder

settings = yaml.safe_load(open("examples/04_scenario/settings.yaml"))
validation = "data/switzerland/MAH_2369_validation.csv"
measured = pd.read_csv(validation)
abstraction = measured.assign(Discharge=measured.Discharge * FLOW_KEPT, T_water=np.nan)


def median_range(values, dp):
    return f"{np.median(values):+.{dp}f} ({np.percentile(values, 5):+.{dp}f} to {np.percentile(values, 95):+.{dp}f})"


rows = []
for version in (4, 7, 8):
    m = pyair2stream.Model({**settings, "version": version},
                           output_dir=os.path.join(HERE, "output", "versions", f"v{version}"))
    calibration = m.calibrate()
    baseline = m.predict(validation, name="baseline")
    m.predict(abstraction, name="abstraction", paired_with="baseline")
    diff = m.difference("abstraction", "baseline")
    base, dates = m.ensemble("baseline")
    abst, _ = m.ensemble("abstraction")
    inside = scenario.central_range(base, 90, axis=0)
    obs = measured.set_index(pd.to_datetime(measured.Date)).T_water.reindex(dates).to_numpy(float)
    ok = np.isfinite(obs)
    p = {f"a{i}": 0.0 for i in range(1, 9)} | calibration.parameters      # unused parameters are 0
    rows.append({
        "version": version,
        "RMSE 2010-2012 (°C)": round(baseline.scores["forward"]["RMSE"], 2),
        "inside the 90% interval": f"{np.mean((obs[ok] >= inside[0][ok]) & (obs[ok] <= inside[1][ok])):.1%}",
        "a4": round(p["a4"], 3), "a8": round(p["a8"], 3),
        "summer change (°C)": median_range(np.nanmean(diff[:, np.isin(dates.month, SUMMER)], axis=1), 2),
        "winter change (°C)": median_range(np.nanmean(diff[:, np.isin(dates.month, WINTER)], axis=1), 2),
        "largest daily warming (°C)": median_range(np.nanmax(diff, axis=1), 2),
        f"extra days per year above {WARM_DAY:g} °C": median_range(
            (scenario.exceedance(abst, WARM_DAY) - scenario.exceedance(base, WARM_DAY)) / 3, 1),
    })

table = pd.DataFrame(rows)
table.to_csv(os.path.join(HERE, "output", "versions_summary.csv"), index=False)
print("\n" + table.to_string(index=False))

# Why versions 7 and 8 warm the summer and cool the winter. The water is pulled towards the
# equilibrium A/B (USER_GUIDE §9.1; the 1/theta^a4 factor cancels in it, so a4 changes only how
# fast the water gets there). With version 8's A and B, the equilibrium is a weighted average:
#   (1 - w) * (a1 + a2*Ta) / a3   +   w * (a5 + a6*cos(2*pi*(t - a7))) / a8,
# with w = a8*theta / (a3 + a8*theta), the share of the term weighted by discharge.
meta = json.load(open(os.path.join(HERE, "output", "versions", "v8", "calibration", "calibration_metadata.json")))
a1, a2, a3, a4, a5, a6, a7, a8 = meta["par_best"]
d = measured.assign(Date=pd.to_datetime(measured.Date))
t = d.Date.dt.dayofyear / np.where(d.Date.dt.is_leap_year, 366, 365)      # time of year (METHODS §5)
theta = d.Discharge / meta["qmedia"]
air = (a1 + a2 * d.T_air) / a3
flow = (a5 + a6 * np.cos(2 * np.pi * (t - a7))) / a8


def share(th):
    return a8 * th / (a3 + a8 * th)


def equilibrium(th):
    return (1 - share(th)) * air + share(th) * flow


parts = pd.DataFrame({
    "month": d.Date.dt.month,
    "air-driven (°C)": air,
    "flow-linked (°C)": flow,
    "share w, measured flow": share(theta),
    f"share w, {FLOW_KEPT:.0%} of the flow": share(FLOW_KEPT * theta),
    "change in equilibrium (°C)": equilibrium(FLOW_KEPT * theta) - equilibrium(theta),
}).groupby("month").mean().round(2)
parts.to_csv(os.path.join(HERE, "output", "version8_equilibrium_by_month.csv"))
print("\nVersion 8, 2010-2012, monthly means:\n" + parts.to_string())
