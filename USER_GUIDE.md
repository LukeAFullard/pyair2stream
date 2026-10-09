# pyair2stream User Guide

This guide takes you from installing pyair2stream to reading its results. You
do not need to be a modelling expert. For exactly what the software computes,
see [docs/METHODS.md](docs/METHODS.md).

## Contents

1. [What you're running](#1-what-youre-running)
2. [Install](#2-install)
3. [Your first run: the bundled example](#3-your-first-run-the-bundled-example)
4. [Choosing a model version and integrator](#4-choosing-a-model-version-and-integrator)
5. [Preparing your own data](#5-preparing-your-own-data)
6. [Configuration reference](#6-configuration-reference)
7. [Running the model](#7-running-the-model)
8. [Understanding the output files](#8-understanding-the-output-files)
9. [Troubleshooting](#9-troubleshooting)
10. [Gap-tolerant mode](#10-gap-tolerant-mode)
11. [Uncertainty (DE-MCMC) and sensitivity analysis](#11-uncertainty-de-mcmc-and-sensitivity-analysis)
12. [Scenario runs and prediction intervals](#12-scenario-runs-and-prediction-intervals)
13. [Cross-validation](#13-cross-validation)
14. [Checklist for results that support a decision](#14-checklist-for-results-that-support-a-decision)

## 1. What you're running

pyair2stream predicts a river's daily mean water temperature. It uses daily
mean air temperature and, for some model versions, daily mean discharge (flow).
The model is one small equation with 3 to 8 numbers in it, the **parameters**.
They are not measured. Instead, pyair2stream finds the values that best match
your measured water temperature.

A normal run does two things:

1. **Calibrates**: it finds the parameter values that best reproduce the water
   temperature in your **calibration** file.
2. **Validates**: it runs the model with those parameters on a separate
   **validation** file, if you give one. The model was not fitted to those
   years, so the validation score shows how well it predicts.

You control a run with one settings file (YAML) and your data files (CSV).

**The usual workflow:**

1. Prepare a calibration file and a validation file (§5).
2. Write a settings file (§6).
3. Run the model (§7).
4. Read the results and check the fit (§8).
5. If you need them: uncertainty ranges (§11), scenarios (§12) and
   cross-validation (§13).
6. Before you use the results for a decision, work through the checklist
   (§14).

**Words used in this guide:**

| Word | Meaning |
|---|---|
| calibration | fitting the parameters to measured water temperature |
| validation | testing the fitted model on years it was not fitted to |
| parameters | the 3–8 numbers in the equation (`a1` to `a8`, §6) |
| run mode | what a run does: calibrate, add uncertainty, or run a scenario (§6) |
| residual (error) | simulated minus measured water temperature on a day |
| `Qmedia` | the mean discharge of the calibration file; the model uses flow relative to it (§6) |

## 2. Install

You need Python 3.9 or newer. It is tested on Python 3.9, 3.12 and 3.14.

```bash
git clone https://github.com/LukeAFullard/pyair2stream.git
cd pyair2stream
pip install .
pyair2stream --help        # check that it installed
```

## 3. Your first run: the bundled example

This example uses the Mentue, a small Swiss river (`data/switzerland/`). It
calibrates on 2002–2009 and validates on 2010–2012. **Run it from the
repository's top folder**, because the paths in its settings file start there
([§7.1](#71-run-from-the-right-directory)):

```bash
pyair2stream --config examples/01_quickstart/config.yaml
```

It takes under a minute. After a banner, you should see:

```
mean, TSS and standard deviation (calibration)
9.73069 96513.81949 5.76298
Pop. Size (particles) = 50, Max Generations (runs) = 100
DE Finished. Best internal negated objective: -0.987657
L-BFGS-B Finished. Best internal negated objective: -0.987927
Efficiency Index in calibration 0.9879265649786034
Consistency check passed.
mean, TSS and standard deviation (validation)
9.67611 37744.04425 5.87375
```

The settings file sets `random_seed: 42`, so your numbers should be the same.
What they mean:

- The calibration score (NSE) is 0.988. A perfect fit would be 1.
- In the validation years, NSE is 0.982 and the typical daily error (RMSE) is
  0.78 °C. These are in `goodness_of_fit_validation_DE_NSE_Mentue.csv`.
- `validation_DE_NSE_Mentue.png` shows the measured and simulated
  temperatures.

[§8](#8-understanding-the-output-files) explains every output file.
[examples/](examples/README.md) has seven more worked examples: uncertainty,
temperature limits, flow scenarios, gaps, cross-validation and sensitivity,
preparing data from raw files, and a warmer climate.

## 4. Choosing a model version and integrator

### Model version (`version`)

| Version | Parameters | Needs discharge? | Seasonal term? | Use it when... |
|:-:|:-:|:-:|:-:|---|
| 3 | 3 | no | no | you want a quick baseline from air temperature only |
| 4 | 4 | yes | no | discharge matters, but there is no extra seasonal effect |
| 5 | 5 | no | yes | you have no reliable discharge, but there is a seasonal effect that air temperature does not explain (for example groundwater or snowmelt) |
| 7 | 7 | yes | yes | you want the full model without the discharge exponent `a4` |
| 8 | 8 | yes | yes | you have good discharge data: the usual starting point |

Start with version 8, or version 5 if you have no discharge data. Then compare
it with simpler versions on the **validation** years. A version that fits the
calibration years better but predicts the validation years worse is
over-fitted. Choose the simplest version that predicts well. Cross-validation
(§13, [example 06](examples/06_cross_validation/README.md)) is the most
thorough way to compare versions.

On the three Swiss rivers tested, versions 7 and 8 predicted unseen years best
([validation V5](validation/REPORT.md#v5)). The Rhône's summer temperature
depends strongly on its flow. There, the versions without discharge (3–5) did
hardly better than simple alternatives
([V9](validation/REPORT.md#v9), [V10](validation/REPORT.md#v10)).

### Integrator (`integrator`)

The integrator is the numerical method that steps the equation forward one day
at a time. **Keep the default, `CRN`** (Crank–Nicolson). It is stable for any
flow and parameters, and the model's authors recommend it. `EXP` is also
always stable.

`RK4`, `RK2` and `EUL` are there to reproduce the original Fortran exactly.
They can become unstable when the flow differs from the calibration flows.
Then they give wrong numbers. Never use them for scenario runs
([§9.1](#91-numerical-stability-and-the-choice-of-integrator)).

## 5. Preparing your own data

Make one CSV file for each period: one for calibration and, ideally, one for
validation. Each needs these columns:

| Column | Needed? | Notes |
|---|---|---|
| `Date` | yes | for example `2020-01-31`; any date format pandas can read |
| `T_air` | yes | daily mean air temperature (°C) |
| `T_water` | yes | daily mean measured water temperature (°C); gaps are allowed |
| `Discharge` | for versions 4, 7 and 8 | daily mean flow, in any unit; can be left out for versions 3 and 5 |

```csv
Date,T_air,T_water,Discharge
2020-01-01,5.2,4.1,12.5
2020-01-02,4.8,,11.8
2020-01-03,6.1,4.0,10.2
```

**The rules.** A run checks every file it reads (the calibration file, a
scenario file and the validation file) before it calibrates anything. If a rule
is broken, it stops with a message that names the file, the column and the
first line with the problem.

- **Give a row for every calendar day.** Never skip a date. Show a missing
  value as an empty cell or `-999`.
- **Use only those two codes for missing values.** Other codes, such as `-99`
  or `9999`, are read as real temperatures. The program warns if a temperature
  looks implausible, but replace such codes with empty cells first.
- **`T_air` and `Discharge` must have no gaps.** If yours have gaps, fill them
  or use [gap-tolerant mode](#10-gap-tolerant-mode).
- **Discharge must be above zero** for versions 4, 7 and 8
  ([§9.2](#92-zero-or-negative-discharge)).
- **Calibration and validation files must start on 1 January and be at least
  365 days long.** If your water temperature record starts later in the year,
  start the file on 1 January anyway. Fill in air temperature and discharge
  from 1 January, and leave `T_water` empty until your measurements begin.
- **Calibration and validation files need a `T_water` column with some
  measurements.** A scenario file does not.
- **Scenario (`FORWARD`) files may start on any day, but must also be at least
  365 days long**, because the model's warm-up repeats the first year. A
  validation file shorter than a year is skipped, with a warning.
- **Keep the validation years separate from the calibration years.** If the
  same measured days are in both files, the run warns: the validation score
  then does not test the model on new data.

**Choosing the calibration and validation years.** Use most of your years for
calibration: several years are better than one. Keep at least one full year
for validation, preferably two or three. The validation years should include
the conditions you care about, such as hot summers or low flows. If you cannot
spare any years, use cross-validation (§13) instead.

**Climate-model data** often uses a 365-day calendar with no leap days, or a
360-day calendar. Declare it with `calendar: "noleap"` or
`calendar: "360_day"`. Do not pad such data with invented dates, or the
seasonal term will drift out of step.

**Building the file from raw data.** pyair2stream has two helpers:

```python
import pyair2stream

# Daily means from separate raw files, on a complete daily calendar
df = pyair2stream.merge_timeseries([
    {"file_path": "raw/air.csv",   "date_col": "time", "value_col": "temp", "standard_col_name": "T_air"},
    {"file_path": "raw/water.csv", "date_col": "time", "value_col": "temp", "standard_col_name": "T_water"},
    {"file_path": "raw/flow.csv",  "date_col": "time", "value_col": "flow", "standard_col_name": "Discharge"},
], output_file="data/all_days.csv")

# The checks a run would make, and a report of missing data and usable stretches
summary, report = pyair2stream.analyze_timeseries(df, version=8, source="data/all_days.csv")
print(report)
```

`analyze_timeseries` makes exactly the checks a run makes, with the same
settings (`version`, `gap_tolerant`, `calendar`, and `period`: `"calibration"`,
`"validation"` or `"scenario"`). Its report starts with "A run would accept this
data" or "A run would STOP on this data", followed by the reasons. It also
counts dates with no row as missing days.

`merge_timeseries` reports what it cannot use:

- rows whose time it cannot read are left out, with a warning;
- values of `-999` are treated as missing;
- text that is not a number (such as `ERR`) stops it. If the text is a code for
  "missing", list it, for example `"na_values": ["ERR"]` in that file's
  settings;
- days with fewer than half the usual number of readings get a warning. Their
  daily mean may be wrong: a day with only daytime readings gives a mean that is
  too high. To leave such days blank, set `min_readings_per_day`, for example
  `merge_timeseries([...], min_readings_per_day=20)` for hourly data.

[Example 07](examples/07_preparing_data/README.md) builds a daily file from raw
logger files and checks it.

## 6. Configuration reference

Every key is optional, except where marked. The defaults are shown.

```yaml
# --- Names (used in the output file names) ---
project_name: "pyair2stream_project"
station_name: "AirStation"
water_station: "AirStation"       # default: the same as station_name
series: "series"                  # any short label

# --- Model ---
version: 8                  # 3, 4, 5, 7 or 8 (§4)
integrator: "CRN"           # CRN, EXP, RK4, RK2 or EUL (§4)
Tice_cover: 0.0             # water temperature is never simulated below this (°C)
calendar: "standard"        # standard, noleap or 360_day (§5)
min_theta_floor: null       # e.g. 1.0e-6 to allow days with zero flow (§9.2)
Qmedia: null                # mean discharge used to scale the flow; needed for FORWARD (below)

# --- Calibration ---
run_mode: "DE"              # DE, PSO, LATHYP, DE-MCMC or FORWARD (below)
objective_function: "NSE"   # the score to optimise: NSE, KGE or RMS (docs/METHODS.md §7)
time_resolution: "1d"       # score daily values ("1d"), N-week means (e.g. "2w") or monthly means ("1m")
prc: 1.0                    # weekly/monthly scoring: the share of days in a week or month that must be measured (above 0, at most 1)
random_seed: null           # an integer makes results exactly repeatable

parameter_bounds:           # needed for calibration: 8 values each, for a1..a8
  min: [-5, -5, -5, -1, 0,  0,  0, -1]
  max: [15, 1.5, 5,  1, 20, 10, 1,  5]

# parameters_forward: [8 values]   # FORWARD only; default: from paths.calibration_metadata

optimization:
  n_run: 100                # DE: the most generations; PSO: iterations; LATHYP: samples
  n_particles: 50           # DE: population = n_particles x 8; PSO: number of particles
  tol: 0.001                # DE only: stop early once the population's scores agree this closely
  c1: 2.0                   # PSO only: pull towards each particle's own best
  c2: 2.0                   # PSO only: pull towards the swarm's best
  wmax: 0.9                 # PSO only: starting inertia
  wmin: 0.4                 # PSO only: final inertia
  mcmc_walkers: 32          # DE-MCMC only: at least 2 x the number of calibrated parameters
  mcmc_steps: 20000         # DE-MCMC only: the most steps; it stops once converged (§11)

paths:
  input_data: "data/calibration.csv"          # required
  validation_data: "data/validation.csv"      # optional
  output_dir: "output"                        # default: "<project_name>/output_<version>"
  calibration_metadata: null                  # FORWARD: a calibration's calibration_metadata.json (§12)

# --- Safety checks (docs/METHODS.md §15) ---
max_plausible_twat: 60.0            # stop if a simulated temperature is above this (°C)
stability_error_fraction: 0.10      # stop if more than this share of days is unstable (§9.1)

# --- Optional features ---
gap_tolerant: false                 # §10
warmup_drop_days: 15                # §10
min_segment_days: 30                # §10
sensitivity_analysis: false         # §11
sensitivity_perturbations: [1.0]    # % changes to test, e.g. [1.0, 5.0]
sensitivity_perturbation_mode: "value"   # "value" or "range" (§11)
# uncertainty_options:              # §11 and §12
# forward_options:                  # §12
# cross_validation:                 # §13
```

`mineff_index`, from Fortran settings files, is accepted but does nothing.
`0_*.csv` always records every parameter set tried.

### Parameters `a1`–`a8`

| Parameter | What it does in the equation (docs/METHODS.md §5) |
|---|---|
| `a1` | a constant heat input |
| `a2` | the effect of air temperature |
| `a3` | how fast water temperature returns to balance |
| `a4` | how discharge changes the river's thermal inertia (an exponent) |
| `a5` | a constant term, scaled by discharge |
| `a6` | the size of an extra yearly cycle |
| `a7` | the timing of that cycle, as a fraction of the year |
| `a8` | a return-to-balance term, scaled by discharge |

You always give 8 bounds. Parameters your version does not use are set to
zero for you. The bounds shown above are the original authors' ranges and a
good start. If a fitted value ends up exactly on a bound, widen that bound and
calibrate again.

**With weekly or monthly scoring** (`time_resolution` other than `"1d"`), set
the minimum of `a2` and `a3` to 0. Otherwise the fit can end on parameters
whose daily simulation zigzags between 0 °C and high values. The zigzag
averages out over a week or a month, so it can score well. In a test with known
parameters, this happened in 9 of 30 weekly-scored calibrations
([V4](validation/REPORT.md#v4)). pyair2stream warns when it happens.

### Run modes (`run_mode`)

| Mode | What it does |
|---|---|
| `DE` | Calibrates with Differential Evolution, then a local search. **Recommended.** |
| `PSO` | Calibrates with Particle Swarm Optimisation, as the original Fortran does. Less reliable: check that it converged. |
| `LATHYP` | Tries many parameter sets spread over the bounds (Latin hypercube). This explores; it does not optimise. |
| `DE-MCMC` | Calibrates with `DE`, then measures the uncertainty of the parameters and predictions (§11). |
| `FORWARD` | Does not calibrate. Runs given parameters, for example on a scenario (§12). |

### Qmedia: keep it fixed when discharge changes

The model never uses discharge directly. It uses `Discharge / Qmedia`, where
`Qmedia` is the mean discharge of the calibration file. The fitted parameters
only make sense with that `Qmedia`.

So when you run the model on a changed flow, keep the calibration's `Qmedia`.
If you recomputed it from the new flows, the change would cancel out. For
example, if all flows rose by half and `Qmedia` was recomputed, the model would
show no change in temperature at all.

A `FORWARD` run therefore needs the calibration's `Qmedia`. The best way is to
point `paths.calibration_metadata` at the `calibration_metadata.json` the
calibration wrote. This also:

- supplies the fitted parameters;
- checks that `version` and `integrator` match the calibration;
- warns if the scenario's flows go outside the calibrated range.

You can also set `Qmedia:` yourself.

## 7. Running the model

```bash
pyair2stream --config path/to/config.yaml
```

### 7.1 Run from the right directory

The paths under `paths:` are relative to the folder you run the command
**from**, not to the settings file. So either run from the folder they are
relative to, or use full paths.

### 7.2 From Python

The command line is the simplest way to run pyair2stream. To run it from a
Python script, call the same command:

```python
import subprocess
subprocess.run(["pyair2stream", "--config", "config.yaml"], check=True)
```

For more control, you can call the calibration steps directly:

```python
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis
from pyair2stream.optimization import DE_mode

data = read_calibration(config_file="config.yaml")
read_Tseries(data, "c")        # load the calibration data
aggregation(data)
statis(data)
DE_mode(data, seed=42)         # sets data.par_best and data.finalfit
print(data.par_best, data.finalfit)
```

This calibrates only. It does not validate or write the output files.

## 8. Understanding the output files

All files go to `output_dir`. Their names include the run mode, the score
(objective), the station, the series label and the time resolution. Every plot
is saved as a PNG and as a PDF.

### What to check first

After a calibration, look at these, in this order:

1. **The validation score.** Open `goodness_of_fit_validation_*.csv`. NSE close
   to 1 and a small RMSE are good. Compare them with the calibration scores in
   `goodness_of_fit_calibration_*.csv`. Validation is usually a little worse.
   Much worse means the model does not predict well.
2. **The time-series plot.** Open `validation_*.png`. The lower panel shows the
   residuals (the errors). Look for long stretches where they stay on one side
   of zero. Those are periods the model gets wrong, for example after the river
   froze, or in an unusual summer.
3. **The error by month.** Open `bias_by_month_validation_*.png`. It shows the
   average error in each month and season, with a 95% interval. If the interval
   does not include zero, the model is too warm or too cool in that month.
   Check the months your question is about.
4. **The parameters.** In `dottyplots_*.png`, check that no parameter is
   squeezed against the edge of its bounds. If one is, widen that bound.
5. **The search.** In `convergence_*.png`, the best score should flatten out.
   If it is still climbing at the end, increase `n_run`.

### Every output file

| File | Contents |
|---|---|
| `1_*.out` | line 1: the 8 fitted parameters; line 2: the calibration score; line 3: the validation score (if run) |
| `2_*.csv` / `3_*.csv` | one row per day of the calibration / validation file: `Year, Month, Day, Tair, Twat_obs, Twat_mod, Twat_obs_agg, Twat_mod_agg, Q`. `_agg` are the values actually scored; `-999` means none. Gap-tolerant runs add `Tair_gap, Q_gap, segment_id`. |
| `goodness_of_fit_<period>_*.csv` | N, NSE, R² (squared correlation), RMSE, MAE, AIC and BIC, for `calibration` and `validation`. The `full_simulation` file repeats the calibration scores, because only measured days are scored. |
| `calibration_*.png`, `validation_*.png` | measured and simulated water temperature on the measured days, with the residuals below |
| `full_simulation_*.png` | the calibration period again, on every day, including days without a measurement |
| `predicted_vs_measured_*.png` | simulated against measured temperature |
| `residual_diagnostics_*.png` | the residuals' histogram, normal Q-Q plot and autocorrelation. Use it to check the assumptions behind the uncertainty ranges. |
| `bias_by_month_<period>_*.csv` / `.png` | the mean error (simulated minus measured) for each month, each season and the whole year, with a 95% interval ([docs/METHODS.md §7](docs/METHODS.md#7-measuring-the-fit)) |
| `0_*.csv`, `convergence_*.png`, `dottyplots_*.png` | every parameter set tried; the best score so far against the number of tries (it should flatten out); the score against each parameter |
| `calibration_metadata.json` | `Qmedia`, the calibrated flow range, version, integrator, parameters and seed. Later runs reuse it. Not written by `FORWARD` runs. |
| `parameters.txt` | the bounds actually used |
| `gaps_summary.txt` | gap-tolerant runs: the stretches of data used |
| `sensitivity_*` | §11 |
| `MCMC_*`, `parameter_significance_*`, `parameter_correlation_*` | §11 |
| `Forward_Prediction_*`, `forward_projection*.png` | §12 |
| `cv_results.csv`, `cv_bias_by_month.*`, `cv_yearly_statistics*.csv`, `cv_interval_coverage.csv` | §13 |

### Reading the scores

- **NSE** (Nash–Sutcliffe efficiency): 1 is perfect. 0 means the model is no
  better than always guessing the average temperature. Above 0.9 is common for
  daily water temperature with this model.
- **RMSE** (root-mean-square error) and **MAE** (mean absolute error) are in °C.
  They are the typical size of the daily error.
- **R²** only measures correlation. It can be high even when the model is
  always too warm. Always check NSE or RMSE too.
- **AIC and BIC** (lower is better) compare model versions on the same data.
  They assume that the daily errors are independent. Real errors last for
  days, which makes AIC and BIC favour the more complex versions. Use them as a
  rough guide, next to the validation scores.

## 9. Troubleshooting

| Message | What to do |
|---|---|
| `Configuration file not found` | Check the path and the folder you run from ([§7.1](#71-run-from-the-right-directory)). |
| `Invalid version` / `run_mode` / `integrator` / `objective_function` / `time_resolution` | Use one of the values the message lists. |
| `parameter_bounds.min must be a list of exactly 8 numbers` (or `parameters_forward`) | Give 8 values, one for each parameter, a1 to a8. |
| `parameter_bounds: min > max` | Swap or correct the bounds of the parameter it names. |
| `No parameter is free to calibrate` | Add `parameter_bounds`. |
| `Missing calibration data file` / `Missing validation data file` | Check `paths.input_data` or `paths.validation_data`. |
| `must start on January 1st` | Start the file on 1 January ([§5](#5-preparing-your-own-data)), or use gap-tolerant mode. |
| `must be continuous at a daily time scale` | The message says which problem it is: a date with no row (add a row; its values may be empty), a date that appears twice (remove one of the rows), or dates out of order (sort the rows by date). |
| `Date is blank on line ...` / `cannot be read as a date` | Write every date as `YYYY-MM-DD`, in the same format on every row. |
| `value(s) that are not numbers` | Replace text such as `n.a.` with an empty cell, or write `-999`. Use a point, not a comma, as the decimal separator. |
| `The series of observed air temperature / discharge ... must be complete` | Fill the gaps, or use [gap-tolerant mode](#10-gap-tolerant-mode). The message names the first missing day. |
| `Missing 'T_air' column` (or another column) `(found 'T_air ' ...)` | Column names must match exactly: check for spaces and capital letters. |
| `Missing 'Discharge' column` | Versions 4, 7 and 8 need discharge. |
| `Missing 'T_water' column` / `has no water temperature measurements` | Calibration and validation files need measured water temperature. If a validation file has none, remove `paths.validation_data`. |
| `has only ... day(s); at least 365 are required` | Calibration and scenario files need at least a year of data. |
| `Warning: ... validation will be skipped` | The validation file is shorter than a year. Give it at least 365 days, or remove it. |
| `Warning: ... are also measured days of the calibration file` | The validation file repeats calibration days. Use separate years ([§5](#5-preparing-your-own-data)). |
| `Non-positive discharge (Q <= 0)` | See [§9.2](#92-zero-or-negative-discharge). |
| `FORWARD mode requires an explicit Qmedia` | Set `paths.calibration_metadata` or `Qmedia:` ([§6](#qmedia-keep-it-fixed-when-discharge-changes)). |
| `n_dat is 0 after aggregation` | No usable `T_water` values. Check the column, or lower `prc`. |
| `prc must be a fraction above 0 and at most 1` | Set `prc` to, for example, `0.6` (60% of the days in each week or month). |
| `Warning: ... value(s) of T_air` (or `T_water`) `... are outside` | Usually a missing-value code other than `-999`. Replace it with an empty cell. Otherwise check that the units are °C. |
| `No valid segments found` | Gap-tolerant mode: no stretch without gaps is at least `min_segment_days` long. |
| `Qmedia is zero or negative` | Gap-tolerant mode: too little valid discharge. Set `Qmedia:`. |
| `NumericalDivergenceError` / `exceed the ... stability limit` | Use `CRN` or `EXP` ([§9.1](#91-numerical-stability-and-the-choice-of-integrator)). |
| `Warning: the relaxation rate B is negative` / `zigzags from one day to the next` | The fitted parameters are physically impossible. This usually follows weekly or monthly scoring with bounds that allow a negative `a2` or `a3`. Set their minimum to 0 and calibrate again ([§6](#parameters-a1a8)). Do not use the results. |
| `Efficiency mismatch in forward run` | An internal check failed. Please report it, with your settings file. |
| `mcmc_walkers ... must be at least 2x` | Increase `mcmc_walkers`. |
| `MCMC did not converge within ... steps` | Try a simpler model version, or increase `mcmc_steps` ([§11](#11-uncertainty-de-mcmc-and-sensitivity-analysis)). |
| `FORWARD mode needs parameters` | Set `paths.calibration_metadata` (or `parameters_forward`) ([§12](#12-scenario-runs-and-prediction-intervals)). |
| `Note: integrator RK4/RK2/EUL is kept to reproduce the original Fortran` | Use `CRN`, unless you need results identical to the Fortran ([§9.1](#91-numerical-stability-and-the-choice-of-integrator)). |
| `enable_prediction_intervals is True but residual_sigma is 0.0/unavailable` | Point `mcmc_chain_path` at a chain that has its `_meta.json`, or set `residual_sigma`. |
| `draws ... were excluded as numerically divergent` | Use `CRN` or `EXP`, or check the chain and the bounds ([§12](#12-scenario-runs-and-prediction-intervals)). |
| `paired_difference_from_files: ... differs` | The two scenario runs did not use the same parameter sets ([§12](#12-scenario-runs-and-prediction-intervals)). |
| `The MCMC chain ... was fitted with ...` | The FORWARD run's model version, integrator or `Qmedia` differs from the chain's calibration. Use `paths.calibration_metadata` from that calibration ([§12](#12-scenario-runs-and-prediction-intervals)). |
| `Warning: warmup_drop_days=... is shorter than` | Gap-tolerant mode: increase `warmup_drop_days` as the message suggests ([§10](#10-gap-tolerant-mode)). |
| `Note: the calibrated model forgets its restart within about ... days` | Gap-tolerant mode: a shorter warm-up would score more measured days. Set the values it gives and calibrate again ([§10](#10-gap-tolerant-mode)). |
| (no message) Good overall scores, but `bias_by_month_*.png` shows the model too warm or too cool in some months | A score over the whole year can hide an error in one season. Compare model versions ([§4](#4-choosing-a-model-version-and-integrator)). Where discharge drives the summer temperature, use version 7 or 8. If an error remains in the season of your limit, report it. A model that is too warm overstates the chance that a warm-water limit was exceeded; one that is too cool understates it. |

### 9.1 Numerical stability and the choice of integrator

Each day, the model pulls the water temperature back towards a balance with
the air. The speed of that pull, `B`, is a rate per day. For version 8,
`B = (a3 + a8·θ) / θ^a4`, with `θ = Discharge/Qmedia`.

With a one-day step, the explicit methods are only stable while B stays below
a limit: 2.0 for `EUL` and `RK2`, and 2.785 for `RK4`. `CRN` and `EXP` are
stable for any positive B. B depends on discharge. So parameters that are
stable for the calibration flows can be unstable for other flows. **An
unstable run can give plausible-looking but wrong numbers, with no error.** A
negative B is physically impossible (the water would move away from balance),
and pyair2stream warns about it.

To protect you, pyair2stream:

- uses `CRN` by default;
- warns before simulating if B goes above the chosen method's limit on some
  days, and stops if this happens on more than `stability_error_fraction` of
  the days;
- stops if a simulated temperature is not a number, or is above
  `max_plausible_twat`;
- prints a note when you choose `RK4`, `RK2` or `EUL`.

Use `RK4`, `RK2` or `EUL` only to reproduce Fortran results, never for
scenarios. Even when stable, they can be inaccurate with a one-day step: `EUL`
was off by up to about 1 °C on the Mentue ([validation V6](validation/REPORT.md#v6)).
Calibrating with `EXP` instead of `CRN` changed predictions by less than
0.03 °C on the Swiss rivers.

**Parameters belong to the method they were calibrated with.** Run them with
the same one. This matters for parameters taken from a paper. The parameters
of Piccolroaz et al. (2016) were calibrated with Crank–Nicolson (`CRN`): with
`RK4`, 8 of the 15 sets are unstable ([V2](validation/REPORT.md#v2)). Those of
Toffolon and Piccolroaz (2015) were calibrated with `RK4`: with `CRN`, their
errors differ from the published ones by up to 0.32 °C
([V13](validation/REPORT.md#v13)). A
`FORWARD` run given `paths.calibration_metadata` refuses a different method or
model version.

### 9.2 Zero or negative discharge

Versions 4, 7 and 8 divide by `θ^a4`. That is undefined when the flow is zero.
So a zero or negative discharge stops the run when the data are loaded. The
message names the first bad date. You can:

1. correct the data;
2. use `gap_tolerant: true`, which treats those days as gaps; or
3. set `min_theta_floor` (for example `1.0e-6`), which keeps θ at least that
   large. Use this only for days that genuinely had no flow. Expect large
   simulated responses on those days, because θ^a4 becomes very large or very
   small as the flow approaches zero.

## 10. Gap-tolerant mode

With `gap_tolerant: true`, `T_air` and `Discharge` may have gaps. This is how
it works:

- The record is split at each gap into **segments**: stretches with no gaps.
- Segments shorter than `min_segment_days` (default 30) are dropped.
- Each segment is simulated on its own. It starts from the measured water
  temperature on its first day, or, if there is none, from the average for
  that day of the year.
- The first `warmup_drop_days` (default 15) of each segment are not scored.
  This gives the model time to forget its approximate starting value. After
  the calibration, the program works out how long the fitted model needs:
  about three relaxation times, 3/B days (B is defined in §9.1). It warns if
  the warm-up is shorter. If it is much longer and that costs many measured
  days, a note gives a shorter warm-up and how many more days it would score.
- The record does not need to start on 1 January.
- Water temperature measurements inside a gap are not used. The model gives no
  water temperatures inside a gap.

When to use it. [Example 05](examples/05_gaps/README.md) tests this on the
Mentue:

- **Long gaps (weeks to a year): use gap-tolerant mode.** It changed the
  calibrated model least, for every gap from a month to a year. Even a whole
  missing year changed the predictions for other years by at most 0.07 °C.
  Filling a gap of a quarter to a year with a straight line changed them by
  0.10 to 1.14 °C on average. Filling it with the seasonal average changed them
  by 0.06 to 0.12 °C.
- **Many short gaps: shorten the warm-up.** With the default 15 days,
  scattered gaps throw away much more data than their share. With 5% of days
  missing at random, only about a third of the measured days were scored. The
  Mentue forgets its restart within 4 days. With a 4-day warm-up, three
  quarters were scored, and the predictions changed by 0.03 °C. So run once
  with the defaults, and if the run prints a note, use its values. On 26 rivers,
  the suggested warm-up was 2–11 days.
- **Do not set the warm-up to 0.** The first days after each restart start
  from the measured water temperature, so the fit looks better than it is.
- **Gaps on one day in five or more:** even a short warm-up scores only about
  a quarter of the record. Fill single missing days instead, from a nearby
  station or by interpolating over a day or two.

Be aware:

- **Check how much data was used** in `gaps_summary.txt` and in the console
  warnings.
- **Scores are not directly comparable with scores on a complete record.** Gaps
  often remove unusual periods, such as floods or freezes. That can make the
  fit look better than it would be on the full record.
- **Set `Qmedia` yourself** if periods of high flow are missing. Otherwise the
  computed mean flow is too low.

## 11. Uncertainty (DE-MCMC) and sensitivity analysis

### Parameter and prediction uncertainty

A single best fit has no error bars. `run_mode: "DE-MCMC"` adds them. It first
calibrates with DE. Then it collects many parameter sets that fit the data
almost as well, in proportion to how well they fit. This is Markov chain Monte
Carlo (MCMC), done with the `emcee` package. From these, it gives a
**prediction interval**: a band that should contain the stated share of the
measured daily temperatures, for example 90%.

**New to uncertainty?** [docs/UNCERTAINTY.md](docs/UNCERTAINTY.md) explains
every range, probability and check in plain language, with figures. The method
is in [docs/METHODS.md §12](docs/METHODS.md#12-parameter-and-prediction-uncertainty-de-mcmc).

```yaml
run_mode: "DE-MCMC"
optimization:
  mcmc_walkers: 32
  mcmc_steps: 20000             # the most steps; it stops once converged
uncertainty_options:
  noise_model: "ar1"            # the default; or "iid" (see below)
  likelihood: "least_squares"   # the default; or "exact" (see below)
  rho_timescale: "weekly"       # the default; or "daily" (see below)
  prediction_interval: 90       # the width of the band, in %; any level, e.g. 95
  parameter_interval: 90        # the width of the parameter ranges, in %
  save_ensemble: false          # true: also save every simulated series (.npz)
  strict_convergence: true      # the default: stop with an error if not converged
  burnin_fraction: null         # override the automatic burn-in (0 to 1)
  on_divergent_draw: "drop"     # or "raise"
  max_divergent_fraction: 0.10
  # ar1_rho: 0.8                # set the error persistence yourself (FORWARD runs); normally from the chain
```

[Example 02](examples/02_uncertainty/README.md) walks through a run. Check
these before you use the results:

- **Convergence.** The sampler runs in blocks of 1,000 steps until its results
  are stable. It then prints `MCMC converged after ... steps`. (Technically:
  the run is at least 50 times its longest autocorrelation time, and split-R̂
  is below 1.01.) If it has not converged by `mcmc_steps`, the run stops with
  an error and gives no interval. This usually means the data cannot pin down
  all the parameters. Try a simpler model version, or more steps. With
  `strict_convergence: false` the run continues, but marks every result as not
  converged. Do not use such results for decisions.
- **Coverage.** The console and `MCMC_chain_*_meta.json` report
  `interval_coverage`: the share of measured days inside the band. It should be
  close to the level you asked for. If it is much lower, the band is too
  narrow.
- **The level.** `prediction_interval` can be any level, for example 95. On the
  Swiss rivers, daily bands from 50% to 95% held in years not used for
  calibration, but 99% bands held on only 98.0–98.2% of days
  ([V11](validation/REPORT.md#v11)). Before you report a 95% or 99% band, check
  it at your site: cross-validation (§13) writes `cv_interval_coverage.csv`
  for each level.

**The three error settings.** Keep the defaults. In short:

- **`noise_model: "ar1"`** treats the model's errors as lasting from day to
  day, as real errors do. For single days, `"iid"` (independent errors) gives
  about the same band. For anything over several days, such as a 7-day mean,
  `"iid"` gives bands that are far too narrow. On the Swiss rivers, its 90%
  bands for 7-day means held only 39–62% of the time
  ([V5](validation/REPORT.md#v5)).
- **`rho_timescale: "weekly"`** sets how long the errors last (ρ). Real errors
  have a fast part, which fades in days, and a slow part, which lasts weeks.
  `"weekly"` includes the slow part; `"daily"` does not. Daily bands are about
  the same either way. Bands for weekly means, yearly peaks and probabilities
  of exceeding a limit are wider with `"weekly"`, and held better on real
  rivers. Use `"daily"` only to reproduce results from version 0.4.1 or
  earlier. Why: [docs/UNCERTAINTY.md §5](docs/UNCERTAINTY.md#5-the-models-errors-their-size-and-how-long-they-last).
- **`likelihood: "least_squares"`** keeps the band centred on the best fit.
  The alternative, `"exact"`, moved the band's centre by up to 0.11 °C on the
  Swiss rivers, and made predictions slightly worse
  ([V5](validation/REPORT.md#v5)).

**About the parameters.** For versions with many parameters, especially 8,
several combinations fit almost equally well. So individual values are
uncertain, and parameters that trade off move together. In a test with known
parameters, the 90% parameter ranges contained the true values about 90% of
the time or more ([V4](validation/REPORT.md#v4)). Rely on the predictions
rather than on single parameter values. Do not combine the ends of several
ranges: such combinations do not fit the data.

**Outputs:**

- `MCMC_chain_*.csv`: the parameter sets collected;
- `MCMC_chain_*_meta.json`: the settings, the diagnostics, the error size (σ)
  and persistence (ρ), and the coverage;
- `MCMC_envelopes_*.csv`: the band for each day of the calibration period
  (`Twat_mod_lower`, `Twat_mod_p50`, `Twat_mod_upper`);
- `parameter_significance_*.csv`: each parameter's mean, standard deviation
  and range at `parameter_interval`, and whether it differs from zero (at the
  5% level);
- `parameter_correlation_*.png`: how the parameters move together.

The band in `MCMC_envelopes_*.csv` covers the calibration years. For other
years, run `FORWARD` with intervals (§12).

### Sensitivity analysis

`sensitivity_analysis: true` changes each parameter, one at a time, up and
down by each percentage in `sensitivity_perturbations`. It reports the mean
change in simulated water temperature (`sensitivity_*.csv` and `.png`). The
result is in °C per 100% change of the parameter's value (`"value"` mode), or
per its full bound range (`"range"` mode, which is easier to compare across
parameters). It describes the model's behaviour near the fitted values only.

## 12. Scenario runs and prediction intervals

`run_mode: "FORWARD"` runs known parameters on any input file. Use it for
naturalised flows, an abstraction scenario, climate projections, or simply
years without water temperature measurements.

```yaml
run_mode: "FORWARD"
version: 8
integrator: "CRN"
paths:
  input_data: "data/scenario.csv"
  output_dir: "output/scenario"
  # From the calibration: its parameters, Qmedia (§6), version and integrator.
  calibration_metadata: "output/calibration_metadata.json"
# parameters_forward: [8 values]  # only to run parameters other than the calibrated ones
uncertainty_options:
  noise_model: "ar1"          # the same as in the calibration
forward_options:
  enable_prediction_intervals: true
  mcmc_chain_path: "output/MCMC_chain_Station_A_series_1d.csv"   # from a DE-MCMC run
  n_samples: 1000
  random_seed: 42
  residual_sigma: null        # default: taken from the chain's _meta.json
```

With prediction intervals, the run:

1. draws parameter sets from the chain;
2. runs the model with each;
3. adds random error with the calibration's typical size and persistence (σ
   and ρ, from the chain's `_meta.json`).

It stops if the chain was fitted with another model version, integrator or
`Qmedia`. If the input file has water temperature measurements, the run
reports the fit and the share of measured days inside the band. The lower
edge of the band can fall below `Tice_cover`, because the error is added after
the simulation. Method: [docs/METHODS.md §13](docs/METHODS.md#13-forward-runs-and-scenario-comparisons).

**Weekly means, yearly peaks and counts of days** must be computed in each
simulated series, not from the daily band. The upper edge of the daily band is
not the upper edge of a weekly mean. Set
`uncertainty_options.save_ensemble: true` to keep every series, then use the
`pyair2stream.scenario` module:

- `load_ensemble` reads the saved series;
- `year_statistics` gives each year's highest daily mean, highest 7-day mean
  and number of days above a threshold, in every series;
- `aggregate` gives means, sums or maxima over fixed periods;
- `exceedance` counts days above a threshold.

[Example 03](examples/03_compliance/README.md) computes the probability that a
7-day mean limit was exceeded.

**Check and correct yearly statistics.** The model can be biased on the hottest
days. On the Mentue, its yearly peaks came out 0.6–0.8 °C too high in years it
was not calibrated on. On the Swiss rivers, uncorrected 90% ranges for yearly
statistics held in only 73–92% of years ([V11](validation/REPORT.md#v11)). So,
for a yearly statistic:

1. Run a cross-validation of the calibration years (§13), with your limit's
   threshold and season (`cross_validation.threshold`, `season_months`).
2. Correct the simulated statistic with the errors that cross-validation found
   (`cv_yearly_statistics.csv`):

```python
import pandas as pd
from pyair2stream import scenario

ens, dates = scenario.load_ensemble("output/prediction/Forward_Prediction_Ensemble_<...>.npz")
peak = scenario.year_statistics(ens, dates, threshold=18)[2010]["highest 7-day mean"]  # one value per series

check = pd.read_csv("output/check/cv_yearly_statistics.csv")
dev = check[check.statistic == "highest 7-day mean"].deviation   # measured minus predicted, per held-out year
p_exceeded = (scenario.correct_statistic(peak, dev) > 20).mean() # share of series above a 20 °C limit
```

With the correction, the 90% ranges held in 85–94% of years (V11). Report the
corrected probability together with the check's summary
(`cv_yearly_statistics_summary.csv`). With fewer than about 5 held-out years,
the check says little and the correction is wide.

**Comparing two scenarios.** To get an uncertainty band for the *difference*
(for example abstraction minus natural flow), both runs must use the same
parameter sets:

1. Run scenario A with `save_ensemble: true`.
2. Run scenario B with the same `mcmc_chain_path`, `save_ensemble: true` and
   `forward_options.reuse_sample_indices_from:
   "output/scenario_a/Forward_Prediction_Ensemble_<...>_meta.json"`.
3. Compute the difference:

```python
from pyair2stream import scenario
diff = scenario.paired_difference_from_files(
    "output/scenario_b/Forward_Prediction_Ensemble_<...>.npz",
    "output/scenario_a/Forward_Prediction_Ensemble_<...>.npz",
)   # B minus A: one row per parameter set, one column per day
```

This first checks that the two runs used the same parameter sets. Each pair
also gets the same random error in both runs, so the error cancels. What is
left is the uncertainty of the effect itself, from the parameters. This
assumes the model's error on a given day would be the same in both scenarios.
[Example 04](examples/04_scenario/README.md) works through a flow abstraction.

## 13. Cross-validation

Cross-validation calibrates the model again and again. Each time it hides one
year, and then scores the model on that hidden year. It shows:

- how well the model predicts years it has not seen;
- whether the parameters stay the same from year to year;
- whether its uncertainty ranges hold.

Only the hidden year's water temperatures are hidden. The year is predicted
from its own air temperature and discharge, like any prediction. This is the
standard design. Hiding the inputs too changed no year's RMSE by more than
0.01 °C ([V12](validation/REPORT.md#v12);
[docs/METHODS.md §11](docs/METHODS.md#11-cross-validation) explains why).

Each year is predicted from a calibration that includes later years as well
as earlier ones. For predictions of future years, also calibrate on the
earlier years and validate on the later ones (`paths.validation_data`). That
test is slightly harder.

```yaml
run_mode: "DE"              # DE, PSO or LATHYP
cross_validation:
  enabled: true
  unit: "year"              # or "n_years" with n_years_per_fold
  n_years_per_fold: 1
  water_year_start_month: 1 # e.g. 10 for October-September years
  min_train_years: 1        # extra first years that are never hidden
  skip_first_year: true     # the first year is never hidden
  min_valid_obs: 10         # skip years with fewer measurements
  optimizer_overrides:      # optional cheaper settings for each fold
    n_run: 50
```

With these defaults, the first two years are always used for calibration
only. For versions 4, 7 and 8, also set `Qmedia:` (the mean discharge of the
file). Then every fold scales discharge the same way. Otherwise the parameters
also move with each fold's own mean discharge.

A cross-validation run writes these files instead of the usual outputs:

- **`cv_results.csv`**: one row per hidden year, with NSE, KGE and RMSE on
  daily values and the fitted parameters. Then rows for the `mean`, the `std`
  (spread), the scores over all hidden days together (`pooled`), and the
  parameters' confidence intervals (`jackknife_90_lower`, `jackknife_90_upper`;
  see below). Large differences in the parameters between years mean the data
  do not pin them down well.
- **`cv_bias_by_month.csv`** and **`.png`**: the mean error by month and season
  over the hidden years. A seasonal bias shows up here, in years the model did
  not see.
- **`cv_yearly_statistics.csv`**: for each hidden year, 1,000 simulations of its
  highest daily mean, highest 7-day mean and number of days above
  `threshold`, and where the measured value fell among them. The `deviation`
  column (measured minus the predicted median) is what
  `scenario.correct_statistic` uses (§12).
- **`cv_yearly_statistics_summary.csv`**: for each statistic, how often the 50%
  and 90% ranges held (with the range expected by chance), and the mean
  deviation with a 95% interval. If that interval does not include zero, the
  model is biased in that statistic.
- **`cv_interval_coverage.csv`**: for the 50%, 80%, 90% and 95% bands and your
  `prediction_interval`, the share of hidden days and 7-day means that fell
  inside. Check the level you intend to report.

Settings for the yearly statistics:

```yaml
cross_validation:
  threshold: 18             # °C, for "days above threshold" (default: the 90th percentile of the measurements)
  season_months: [6, 7, 8, 9]   # a year counts if 80% of these months was measured (default: the 4 warmest)
  min_train_years: 0        # hide every year but the first: the check needs as many years as possible
```

The check uses the error settings of `uncertainty_options` (`noise_model`,
`rho_timescale`), as a FORWARD run does. It reports ranges at
`prediction_interval`. With fewer than about 5 hidden years, it says little.

**Parameter confidence intervals.** The `jackknife_90_lower` and
`jackknife_90_upper` rows give approximate 90% intervals for each parameter.
(The level is `uncertainty_options.parameter_interval`, and the row names
follow it.) They come from how much the parameters move between folds (the
jackknife, [docs/METHODS.md §11](docs/METHODS.md#11-cross-validation)). In a
test with known parameters, they contained the true values 83–95% of the time
([V4](validation/REPORT.md#v4)). Keep in mind:

- Do not use the `std` row as an uncertainty. The folds share most of their
  data, so it is far too small. It contained the true values only 35–52% of
  the time.
- Each interval is for one parameter on its own. Parameters that trade off
  move together, so combining the ends of several intervals gives parameter
  sets that do not fit ([example 06](examples/06_cross_validation/README.md)
  shows this).
- The intervals also depend a little on the optimiser's random start. Where
  parameters trade off, one fold can end on a distant set with almost the same
  fit. Another `random_seed` alone changed the intervals' typical width by a
  factor of 0.65 to 1.06 ([V12](validation/REPORT.md#v12)). For poorly
  determined parameters, repeat the run with a second seed.

Cross-validation runs only with `run_mode` `DE`, `PSO` or `LATHYP`. In other
run modes it is ignored, with a warning.

## 14. Checklist for results that support a decision

Before you use the results for a compliance assessment, a permit or any other
decision, check:

1. **Validation.** The model scores well on years it was not calibrated on
   (a validation file, or cross-validation).
2. **Plausible parameters.** No parameter sits exactly on a bound (the dotty
   plots and `1_*.out`).
3. **Residuals.** There is no strong pattern over time or with temperature
   (`residual_diagnostics_*.png`). There is no clear bias in the months your
   limit applies to (`bias_by_month_validation_*.png`, or `cv_bias_by_month.png`
   from cross-validation). A 95% interval that excludes zero means the model is
   too warm or too cool in that month.
4. **Uncertainty.** If you report a band, its coverage is close to the level
   you asked for: on validation years, and in the season your limit applies
   to. Bands for new years are usually slightly narrow: 90% bands held on
   85–89% of days on the Swiss rivers ([V5](validation/REPORT.md#v5)).
   - For 7-day means, runs of warm days and other quantities over several days,
     keep `noise_model: "ar1"` and `rho_timescale: "weekly"` (the defaults).
     Compute them from the saved simulations (§12).
   - For a yearly statistic (a peak, a count of days), check it by
     cross-validation and correct it (§12, §13).
   - Report probabilities with their ranges and the check, not as a yes or no.
5. **Scope.** The model gives **daily means**. A limit on daily maxima, or on
   values within a day, needs a separate, justified step. Scenario inputs
   outside the calibrated range of air temperature or flow are extrapolation.
6. **Reproducibility.** Keep the settings file (with `random_seed`), the input
   files, the pyair2stream commit (`git rev-parse HEAD`), the output of
   `pip freeze`, and the output folder.
7. **The software.** [validation/REPORT.md](validation/REPORT.md) records the
   checks that pyair2stream reproduces the original model and the published
   results, and how well its ranges hold. Cite it with the commit.

[docs/METHODS.md §16](docs/METHODS.md#16-limitations-and-good-practice) lists
every assumption and limitation.
[docs/UNCERTAINTY.md §15](docs/UNCERTAINTY.md#15-is-it-defensible) explains
how to present a result so that you can defend it.
