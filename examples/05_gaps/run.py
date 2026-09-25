"""
Run example 05: missing air temperature, handled two ways.

    python examples/05_gaps/run.py

Removes three weeks and ten scattered single days of air temperature from the
Mentue's 2002-2009 record, then calibrates (1) after filling the gaps by
interpolation and (2) in gap-tolerant mode, and compares both on 2010-2012.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")

# A record with gaps: a three-week logger failure and ten single missing days.
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_calibration.csv"), parse_dates=["Date"])
gap = (df.Date >= "2005-07-04") & (df.Date <= "2005-07-24")
single = np.random.default_rng(1).choice(np.arange(400, len(df) - 400), size=10, replace=False)
gap.iloc[single] = True
gappy = df.assign(T_air=df.T_air.mask(gap))
gappy.to_csv(os.path.join(OUT, "gappy_calibration.csv"), index=False, date_format="%Y-%m-%d")
print(f"Removed {int(gap.sum())} days of air temperature.")

# Option 1: fill the gaps by straight-line interpolation (in your own work, prefer a nearby station).
filled = gappy.assign(T_air=gappy.T_air.interpolate())
filled.to_csv(os.path.join(OUT, "filled_calibration.csv"), index=False, date_format="%Y-%m-%d")

# In the standard mode, a gap in air temperature stops the run with a clear message.
cfg = os.path.join(OUT, "standard.yaml")
with open(os.path.join(HERE, "filled.yaml")) as f:
    open(cfg, "w").write(f.read().replace("filled_calibration.csv", "gappy_calibration.csv")
                         .replace("output/filled", "output/standard"))
run = subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", cfg], cwd=REPO,
                     capture_output=True, text=True)
print("Standard mode on the gappy record:", (run.stderr.strip().splitlines() or ["?"])[-1])

# Options 1 and 2.
rows = []
for mode in ("filled", "gap_tolerant"):
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", f"examples/05_gaps/{mode}.yaml"],
                   cwd=REPO, check=True)
    out = os.path.join(OUT, mode)
    cal = pd.read_csv(os.path.join(out, "2_DE_NSE_Mentue_cc_1d.csv"))
    fit = pd.read_csv(os.path.join(out, "goodness_of_fit_validation_DE_NSE_Mentue.csv"), index_col="Metric").Value
    rows.append({"approach": mode, "days scored in calibration": int((cal.Twat_obs_agg != -999).sum()),
                 "validation NSE": round(fit["NSE"], 3), "validation RMSE (°C)": round(fit["RMSE"], 3)})
table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "gaps_summary.csv"), index=False)
print("\n" + table.to_string(index=False))
