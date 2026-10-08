"""
Run example 05: what to do with missing data.

    python examples/05_gaps/run.py

1. Checks a record with gaps in air temperature, as a run would.
2. Missing water temperature: hides July-August 2006 and shows that the model fills it.
3. Missing air temperature: removes three weeks and ten scattered single days from the
   Mentue's 2002-2009 record, then calibrates (a) after filling the gaps by interpolation
   and (b) in gap-tolerant mode, and compares both on 2010-2012.
Writes the tables to output/ and the README's figures to figures/. gap_study.py tests longer
gaps, scattered gaps and shorter warm-ups.
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
import yaml

import pyair2stream

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output")
FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def pyair2stream_run(config):
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", f"examples/05_gaps/{config}"],
                   cwd=REPO, check=True)


def daily_output(folder, name):
    d = pd.read_csv(os.path.join(OUT, folder, name)).replace(-999.0, np.nan)
    d.index = pd.to_datetime(d[["Year", "Month", "Day"]])
    return d


cal = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_calibration.csv"), parse_dates=["Date"])

# --- 1. A record with gaps in air temperature, and the check a run would make ----------------
gap = (cal.Date >= "2005-07-04") & (cal.Date <= "2005-07-24")          # a three-week logger failure
single = np.random.default_rng(1).choice(np.arange(400, len(cal) - 400), size=10, replace=False)
gap.iloc[single] = True                                                   # and ten single missing days
gappy = cal.assign(T_air=cal.T_air.mask(gap))
gappy.to_csv(os.path.join(OUT, "gappy_calibration.csv"), index=False, date_format="%Y-%m-%d")
print(f"Removed {int(gap.sum())} days of air temperature.")

summary, report = pyair2stream.analyze_timeseries(gappy, version=8, source="gappy_calibration.csv")
open(os.path.join(OUT, "data_check.txt"), "w").write(report)
print("\n" + report.split("--- Missing Data")[0])

# The same check, in a real run in the standard mode: it stops before calibrating.
cfg = yaml.safe_load(open(os.path.join(HERE, "filled.yaml")))
cfg["paths"].update(input_data="examples/05_gaps/output/gappy_calibration.csv",
                    output_dir="examples/05_gaps/output/standard")
yaml.safe_dump(cfg, open(os.path.join(OUT, "standard.yaml"), "w"))
run = subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", "examples/05_gaps/output/standard.yaml"],
                     cwd=REPO, capture_output=True, text=True)
print("Standard mode on the gappy record:", (run.stderr.strip().splitlines() or ["?"])[-1])

# --- 2. Missing water temperature: nothing to do, and the model fills the gap ----------------
hidden = (cal.Date >= "2006-07-01") & (cal.Date <= "2006-08-31")
cal.assign(T_water=cal.T_water.mask(hidden)).to_csv(os.path.join(OUT, "water_gaps_calibration.csv"), index=False,
                                                     date_format="%Y-%m-%d")
pyair2stream_run("water_gaps.yaml")
sim = daily_output("water_gaps", "2_DE_NSE_Mentue_cc_1d.csv").Twat_mod
truth = cal.set_index("Date").T_water
err = (sim - truth)[hidden.to_numpy()]
fill = {"hidden days": int(hidden.sum()), "RMSE on hidden days (°C)": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "mean error (°C)": round(float(err.mean()), 2), "largest error (°C)": round(float(err.abs().max()), 2)}
print("\nFilling July-August 2006:", fill)

window = slice("2006-05-15", "2006-10-15")
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axvspan(pd.Timestamp("2006-07-01"), pd.Timestamp("2006-08-31"), color="tab:gray", alpha=0.12, lw=0)
shown = truth[window]
mask = (shown.index >= "2006-07-01") & (shown.index <= "2006-08-31")
ax.plot(shown.index[~mask], shown[~mask], ".", color="black", ms=3, label="measured, used for calibration")
ax.plot(shown.index[mask], shown[mask], "o", mfc="none", color="tab:gray", ms=3.5,
        label="measured, hidden from calibration")
ax.plot(sim[window].index, sim[window], color="tab:orange", lw=1.4, label="model")
ax.set_ylim(shown.min() - 2.2, shown.max() + 1.5)
ax.text(pd.Timestamp("2006-07-03"), shown.max() + 0.7, "July-August: water temperature hidden", fontsize=8,
        color="dimgray")
ax.set_ylabel("Water temperature (°C)")
ax.set_title(f"The model fills two months without measurements: RMSE {fill['RMSE on hidden days (°C)']:.2f} °C "
             f"against the hidden values", fontsize=10)
ax.legend(fontsize=8, loc="lower center", ncol=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "water_gap_filled.png"), dpi=130)
plt.close(fig)

# --- 3. Missing air temperature: fill the gaps, or split the record ---------------------------
filled = gappy.assign(T_air=gappy.T_air.interpolate())     # in your own work, prefer a nearby station
filled.to_csv(os.path.join(OUT, "filled_calibration.csv"), index=False, date_format="%Y-%m-%d")

window = (cal.Date >= "2005-06-20") & (cal.Date <= "2005-08-08")
three_weeks = (cal.Date >= "2005-07-04") & (cal.Date <= "2005-07-24")
interp_err = (filled.T_air - cal.T_air)[three_weeks]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.axvspan(pd.Timestamp("2005-07-04"), pd.Timestamp("2005-07-24"), color="tab:gray", alpha=0.12, lw=0)
ax.plot(cal.Date[window], cal.T_air[window], color="black", lw=1, label="measured (removed in the gap)")
ax.plot(filled.Date[window], filled.T_air[window], color="tab:blue", lw=1.6, ls="--",
        label="filled by straight-line interpolation")
ax.set_ylabel("Air temperature (°C)")
ax.set_title(f"Interpolating across three weeks invents weather: off by {interp_err.abs().mean():.1f} °C on "
             f"average, up to {interp_err.abs().max():.1f} °C", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "air_gap_interpolated.png"), dpi=130)
plt.close(fig)

rows = []
for mode in ("filled", "gap_tolerant"):
    pyair2stream_run(f"{mode}.yaml")
    c = pd.read_csv(os.path.join(OUT, mode, "2_DE_NSE_Mentue_cc_1d.csv"))
    fit = pd.read_csv(os.path.join(OUT, mode, "goodness_of_fit_validation_DE_NSE_Mentue.csv"), index_col="Metric").Value
    rows.append({"approach": mode, "days scored in calibration": int((c.Twat_obs_agg != -999).sum()),
                 "validation NSE": round(fit["NSE"], 3), "validation RMSE (°C)": round(fit["RMSE"], 3)})
table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "gaps_summary.csv"), index=False)
print("\n" + table.to_string(index=False))

# Where gap-tolerant mode scored the model, and where it did not.
g = daily_output("gap_tolerant", "2_DE_NSE_Mentue_cc_1d.csv")
in_segment = g.segment_id.fillna(-999) >= 0
measured = g.Twat_obs.notna()
scored = g.Twat_obs_agg.notna()
n_segments = int(g.segment_id[in_segment].nunique())
lost = int((measured & ~scored).sum())
fig, ax = plt.subplots(figsize=(9, 2.4))
for y, mask, colour, label in ((2, in_segment & scored, "tab:blue", "scored"),
                               (1, in_segment & measured & ~scored, "tab:orange",
                                "measured, not scored: the first 15 days of a stretch"),
                               (0, ~in_segment, "tab:red", "no air temperature (a gap) or a stretch under 30 days")):
    ax.scatter(g.index[mask], np.full(int(mask.sum()), y), marker="|", s=120, color=colour, label=label)
ax.set_yticks([0, 1, 2], ["gap", "warm-up", "scored"])
ax.set_ylim(-0.7, 2.7)
ax.set_title(f"Gap-tolerant mode: {n_segments} stretches; {lost} measured days not scored", fontsize=10)
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "gap_tolerant_stretches.png"), dpi=130)
plt.close(fig)

# The two approaches predict the validation years almost identically.
# (On the first days of 2010 they differ more: gap-tolerant mode starts from the measured
# temperature instead of a warm-up year, and does not score those days.)
a = daily_output("filled", "3_DE_NSE_Mentue_cv_1d.csv")
b = daily_output("gap_tolerant", "3_DE_NSE_Mentue_cv_1d.csv")
diff = (b.Twat_mod - a.Twat_mod)[b.Twat_obs_agg.notna()]
print(f"\nValidation years, scored days, gap-tolerant minus filled: mean {diff.mean():+.3f} °C, "
      f"largest {diff.abs().max():.3f} °C")
