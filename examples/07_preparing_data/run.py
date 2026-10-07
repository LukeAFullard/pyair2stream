"""
Run example 07: from raw logger files to checked daily input files.

    python examples/07_preparing_data/run.py

1. Makes three raw files, as a user might receive them: hourly air temperature, water
   temperature every 15 minutes, and daily discharge. They are built from the Mentue's daily
   data (data/switzerland/) with a daily cycle added, and with typical faults: a logger outage,
   unreadable timestamps, an "ERR" code, a day with daytime readings only, and a sensor fault
   coded -999.
2. Merges them into one daily file with pyair2stream.merge_timeseries.
3. Checks it with pyair2stream.analyze_timeseries, fills the short air temperature gaps, and
   checks it again.
4. Splits it into a calibration and a validation file and runs the model on them.
Writes the README's figures to figures/.
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

import pyair2stream

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")
RAW = os.path.join(OUT, "raw")
FIG = os.path.join(HERE, "figures")
for d in (OUT, RAW, FIG):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(7)

# --- 1. The raw files -------------------------------------------------------------------------
daily = pd.concat([pd.read_csv(os.path.join(REPO, "data", "switzerland", f"MAH_2369_{p}.csv"), parse_dates=["Date"])
                   for p in ("calibration", "validation")], ignore_index=True)

# Air temperature, hourly, from a weather station. A daily cycle (4 degC, warmest at 15:00)
# around the daily mean; over 24 hourly readings it averages to the daily mean exactly.
hours = pd.date_range(daily.Date.iloc[0], daily.Date.iloc[-1] + pd.Timedelta(hours=23), freq="h")
day_mean = daily.set_index("Date").T_air.reindex(hours.normalize()).to_numpy()
air = pd.DataFrame({"timestamp": hours.strftime("%d.%m.%Y %H:%M"),
                    "air_temp": np.round(day_mean + 4.0 * np.sin(2 * np.pi * (hours.hour - 9) / 24), 2)})
air["air_temp"] = air["air_temp"].astype(object)
outage = (hours >= "2004-03-10") & (hours < "2004-03-13")               # logger outage: no rows
daytime_only = (hours.normalize() == "2005-06-15") & ((hours.hour < 9) | (hours.hour > 17))
air.loc[rng.choice(np.flatnonzero(~outage), 6, replace=False), "timestamp"] = "##.##.#### ##:##"
air.loc[rng.choice(np.flatnonzero(~outage & ~daytime_only), 4, replace=False), "air_temp"] = "ERR"
air[~outage & ~daytime_only].to_csv(os.path.join(RAW, "air_station.csv"), index=False)

# Water temperature, every 15 minutes, from a logger in the river (1 degC daily cycle).
quarters = pd.date_range(daily.Date.iloc[0], daily.Date.iloc[-1] + pd.Timedelta(minutes=23 * 60 + 45), freq="15min")
w_mean = daily.set_index("Date").T_water.reindex(quarters.normalize()).to_numpy()
hour = quarters.hour + quarters.minute / 60
water = pd.DataFrame({"Time": quarters.strftime("%Y-%m-%dT%H:%M"),
                      "Tw": np.round(w_mean + 1.0 * np.sin(2 * np.pi * (hour - 11) / 24), 3)})
water = water[np.isfinite(w_mean)]                                      # days not measured: no rows
water.loc[water.Time.str.startswith(("2007-02-01", "2007-02-02", "2007-02-03")), "Tw"] = -999.0   # sensor fault
water.to_csv(os.path.join(RAW, "river_logger.csv"), index=False)

# Discharge, daily means from the gauging agency.
daily[["Date", "Discharge"]].rename(columns={"Date": "date", "Discharge": "Q"}).to_csv(
    os.path.join(RAW, "flow_gauge.csv"), index=False, date_format="%Y-%m-%d")

# --- 2. Merge them into one daily file ----------------------------------------------------------
files = [
    {"file_path": os.path.join(RAW, "air_station.csv"), "date_col": "timestamp", "value_col": "air_temp",
     "standard_col_name": "T_air", "date_format": "%d.%m.%Y %H:%M"},
    {"file_path": os.path.join(RAW, "river_logger.csv"), "date_col": "Time", "value_col": "Tw",
     "standard_col_name": "T_water"},
    {"file_path": os.path.join(RAW, "flow_gauge.csv"), "date_col": "date", "value_col": "Q",
     "standard_col_name": "Discharge"},
]
print("\n--- First attempt ---")
try:
    pyair2stream.merge_timeseries(files)
except ValueError as e:
    print("ValueError:", e)

print("\n--- With the station's error code named, and a minimum number of readings per day ---")
files[0].update(na_values=["ERR"], min_readings_per_day=20)            # hourly: at least 20 of 24
files[1].update(min_readings_per_day=80)                                # 15-minute: at least 80 of 96
merged = pyair2stream.merge_timeseries(files, output_file=os.path.join(OUT, "merged.csv"))

# --- 3. Check it as a run would, fill short gaps, check again -----------------------------------
summary, report = pyair2stream.analyze_timeseries(merged, version=8, source="merged.csv")
print("\n" + report.split("--- Missing Data")[0])

filled = merged.copy()
filled["T_air"] = filled["T_air"].interpolate(limit=3, limit_area="inside")    # gaps of up to 3 days only
summary2, report2 = pyair2stream.analyze_timeseries(filled, version=8, source="filled.csv")
print(report2.split("--- Missing Data")[0])

# How close the prepared daily means are to the original daily data.
prepared = filled.set_index(pd.to_datetime(filled.Date))
orig = daily.set_index("Date")
for col in ("T_air", "T_water", "Discharge"):
    d = (prepared[col] - orig[col]).abs()
    print(f"{col}: largest difference from the original daily data {d.max():.3f} (median {d.median():.3f})")

# --- 4. Split into calibration and validation files, and run the model ---------------------------
dates = pd.to_datetime(filled.Date)
filled[dates.dt.year <= 2009].to_csv(os.path.join(OUT, "calibration.csv"), index=False)
filled[dates.dt.year >= 2010].to_csv(os.path.join(OUT, "validation.csv"), index=False)
subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", "examples/07_preparing_data/config.yaml"],
               cwd=REPO, check=True)
fit = pd.read_csv(os.path.join(OUT, "model", "goodness_of_fit_validation_DE_NSE_Mentue.csv"), index_col="Metric").Value
print(f"\nValidation 2010-2012 on the prepared files: NSE {fit['NSE']:.3f}, RMSE {fit['RMSE']:.2f} °C")

# --- Figures ---------------------------------------------------------------------------------------
raw_air = pd.read_csv(os.path.join(RAW, "air_station.csv"))
raw_air["t"] = pd.to_datetime(raw_air.timestamp, format="%d.%m.%Y %H:%M", errors="coerce")
raw_air["v"] = pd.to_numeric(raw_air.air_temp, errors="coerce")
span = raw_air[(raw_air.t >= "2005-06-14") & (raw_air.t < "2005-06-17")]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(span.t, span.v, "o", color="tab:gray", ms=3, label="hourly readings")
for day in pd.date_range("2005-06-14", periods=3):
    r = span[span.t.dt.normalize() == day]
    ax.hlines(r.v.mean(), day, day + pd.Timedelta(hours=23), color="tab:red", lw=2,
              label="mean of the readings" if day.day == 14 else None)
    ax.hlines(orig.T_air[day], day, day + pd.Timedelta(hours=23), color="black", lw=1.2, ls="--",
              label="true daily mean" if day.day == 14 else None)
    ax.text(day + pd.Timedelta(hours=1), r.v.max() + 0.6, f"{len(r)} readings", fontsize=8)
ax.set_ylabel("Air temperature (°C)")
bias = span[span.t.dt.normalize() == "2005-06-15"].v.mean() - orig.T_air["2005-06-15"]
ax.set_title(f"A day with daytime readings only: its mean is {bias:.1f} °C too warm", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(span.v.min() - 1, span.v.max() + 2)
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=12))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "daytime_only_day.png"), dpi=130)
plt.close(fig)

m = pd.to_datetime(merged.Date)
win = (m >= "2004-02-25") & (m <= "2004-03-27")
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(orig.T_air["2004-02-25":"2004-03-27"], color="black", lw=1, label="true daily mean")
ax.plot(m[win], filled.T_air[win], color="tab:blue", lw=1.6, ls="--", label="prepared file (gap filled)")
ax.plot(m[win], merged.T_air[win], "o", color="tab:blue", ms=3.5, label="merged from the readings")
ax.axvspan(pd.Timestamp("2004-03-10"), pd.Timestamp("2004-03-12 23:00"), color="tab:gray", alpha=0.15, lw=0)
ax.text(pd.Timestamp("2004-03-10 03:00"), ax.get_ylim()[1] - 1, "logger\noutage", fontsize=8, va="top")
ax.set_ylabel("Air temperature (°C)")
ax.set_title("A three-day logger outage, filled by interpolation", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "outage_filled.png"), dpi=130)
plt.close(fig)
