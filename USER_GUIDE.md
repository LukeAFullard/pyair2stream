# pyair2stream User Guide

This guide walks a first-time user through installing `pyair2stream`, running the bundled example, preparing your own data, writing a configuration file, and reading the results. If you just need a quick reference to config options, jump to [6. Configuration reference](#6-configuration-reference). If something goes wrong, jump to [9. Troubleshooting](#9-troubleshooting).

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
11. [Sensitivity analysis and uncertainty (DE-MCMC)](#11-sensitivity-analysis-and-uncertainty-de-mcmc)
12. [Forward prediction intervals](#12-forward-prediction-intervals)
13. [Cross-validation](#13-cross-validation)
14. [Where to go next](#14-where-to-go-next)

## 1. What you're running

`pyair2stream` fits a small ordinary differential equation (ODE) to your river's water temperature, using daily air temperature (and optionally discharge) as inputs. It does this in two steps every time you run it:

1. **Calibrate**: search for the 3–8 model parameters that best reproduce your *observed* water temperature.
2. **Simulate**: run the calibrated model forward and (if you provided one) evaluate it against a separate validation period.

You'll interact with it through a single YAML configuration file and CSV data files. No Fortran or compiling is required.

## 2. Install

Requires Python 3.9+.

```bash
git clone https://github.com/LukeAFullard/pyair2stream.git
cd pyair2stream
pip install .
```

Check it installed correctly:

```bash
pyair2stream --help
```

If you plan to run the test suite later, also install `pytest` and (optionally) `gfortran` (see the main [README](README.md#testing)).

## 3. Your first run: the bundled example

Before touching your own data, run the bundled `quickstart` example so you know what a successful run looks like. It uses four years of synthetic daily data for a fictional "River Alpha" (`examples/quickstart/data/`) (three years for calibration, one held back for validation) using a fast `PSO` configuration so the whole thing finishes in well under a minute.

**Run this from the root of the cloned repository** because the example's config uses paths relative to that directory (see the [working-directory note](#71-run-from-the-right-directory) below).

```bash
pyair2stream --config examples/quickstart/config.yaml
```

You should see something like:

```
pyair2stream Version 1.0.0 (Python Port)

mean, TSS and standard deviation (calibration)
10.00641 28570.62781 5.10802
N. particles = 20, N. run = 30
Progress: 40.0 %
Progress: 50.0 %
...
Progress: 100.0 %
Efficiency Index in calibration 0.8371758152864042
Consistency check passed.
mean, TSS and standard deviation (validation)
10.03896 9041.37700 4.98387
Computation time was 4.76 seconds.
Starting post-processing visualizations...
Post-processing completed.
```

("Progress: X%" / "Consistency check passed." are progress and self-check messages which are informational, not errors. Because PSO involves randomised search, your efficiency index and exact timings may differ slightly from run to run.)

Everything is written to `examples/quickstart/output/`. Open `calibration_PSO_NSE_River_Alpha.png` to see simulated vs. observed water temperature for the calibration period, and `validation_PSO_NSE_River_Alpha.png` for the held-out year. An efficiency index (here, NSE ≈ 0.84 on calibration) close to 1.0 indicates a good fit. See [§8](#8-understanding-the-output-files) for an explanation of all output files. Once this runs cleanly for you, move on to your own data.

## 4. Choosing a model version and integrator

### Model version

Set via `version` in the config. Pick the simplest version your data supports:

| Version | Parameters | Needs discharge? | Has a seasonal term? | Use when... |
|:-------:|:----------:|:-----------------:|:----------------------:|--------------|
| 3 | 3 | No | No | You only have air temperature and want a quick baseline |
| 4 | 4 | Yes | No | Discharge matters but seasonality is already explained by air temperature |
| 5 | 5 | No | Yes | No reliable discharge data, but there's a seasonal signal air temperature doesn't fully capture (e.g. groundwater influence) |
| 7 | 7 | Yes | Yes | Full model, without the discharge-attenuation exponent |
| 8 | 8 | Yes | Yes | Full model. This is the usual starting point if you have good discharge data. |

If you're unsure, start with **version 8**, then compare against version 7 (equivalent but without the discharge exponent `a4`) or version 5 (if discharge data is unreliable) using the same objective function and check which one calibrates better *and* validates better, as a version that only fits the calibration period well may be overfit.

### Integrator

Set via `integrator`. **`RK4`** (4th-order Runge-Kutta) is the default and recommended choice for accuracy. `CRN` (Crank-Nicolson) is more numerically stable for stiff parameter combinations; `RK2` and `EUL` are simpler/faster but less accurate. These are mainly useful for quick tests.

## 5. Preparing your own data

Your calibration and (optional) validation CSVs need these columns:

| Column | Always required? | Notes |
|---|---|---|
| `Date` | Yes | Any format `pandas.to_datetime` understands, e.g. `YYYY-MM-DD` |
| `T_air` | Yes | Air temperature (°C) |
| `T_water` | Yes | Observed water temperature (°C). May contain gaps. |
| `Discharge` | **Yes, even for versions 3 and 5** | The column must exist in the file for the file to load, even though versions 3 and 5 don't use it in the model equation. If you don't have discharge, fill the column with a placeholder (e.g. `1.0`) for those versions. |

A few hard requirements the reader will enforce, taken from the original model's data conventions:

- **One row per calendar day, with no missing dates.** This applies even in [gap-tolerant mode](#10-gap-tolerant-mode). Gaps must appear as `NaN`/`-999.0` *values* in an existing row, not as a skipped date.
- **The series must start on 1 January** (unless `gap_tolerant: true`). If your real record starts later in the year, back-fill `T_air` (reconstructed if necessary) and mark `T_water` as missing for the lead-in days.
- **`T_air` and `Discharge` must be gap-free** (unless `gap_tolerant: true`). Only `T_water` is allowed to have missing values in the default mode.
- Missing values can be left blank (pandas reads them as `NaN`) or written explicitly as the legacy sentinel `-999.0`.

Example:

```csv
Date,T_air,T_water,Discharge
2020-01-01,5.2,4.1,12.5
2020-01-02,4.8,4.0,11.8
2020-01-03,6.1,,10.2
2020-01-04,5.9,3.9,9.7
```

You need at least a full year of data, and ideally several years, since the model always discards its first simulated year as a warm-up (see [§8](#8-understanding-the-output-files)), so a one-year file leaves nothing left to evaluate.

## 6. Configuration reference

Create a `config.yaml` (any filename is fine; pass it with `--config`). Every field below is optional unless marked **required**, and defaults are as shown.

```yaml
# --- Identity & labelling (used to name output files/folders) ---
project_name: "my_river_project"     # default: "pyair2stream_project"
station_name: "Station_A"            # default: "AirStation"
water_station: "Station_B"           # default: same as station_name
series: "c"                          # free-text label only (see note below)

# --- Model setup ---
version: 8                # required in practice: 3, 4, 5, 7, or 8 (see §4)
integrator: "RK4"          # RK4 (default), EUL, RK2, CRN
Tice_cover: 0.0            # water temperature floor (°C); simulated Tw is clamped at this value
time_resolution: "1d"      # 1d = daily; "Nw" = N weeks (e.g. "2w"); "Nm" = N months (e.g. "1m")
prc: 1.0                   # for time_resolution other than 1d: minimum fraction (0-1) of days
                            # that must have valid T_water within a period for that period's
                            # aggregate to be used in calibration

# --- Calibration ---
objective_function: "NSE"  # NSE, KGE, or RMS (all reported as "higher is better" internally)
run_mode: "DE"              # DE (recommended default), PSO, LATHYP, FORWARD, DE-MCMC (see below)
mineff_index: 0.0           # only parameter sets scoring >= this are kept in the "0_*.csv" history.
                             # Must be a TOP-LEVEL key (see callout below the table) (not nested)
                             # under `optimization:`, despite what the bundled example configs show.

paths:
  input_data: "data/calibration_data.csv"        # required
  validation_data: "data/validation_data.csv"    # optional (validation is skipped if absent)
  output_dir: "output"                            # default: "{project_name}/output_{version}"

parameter_bounds:           # required for DE/PSO/LATHYP/DE-MCMC; 8 values each (unused
  min: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # parameters for your chosen version are ignored)
  max: [10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0]

parameters_forward: [1.2, 0.3, 0.2, 0.5, 0.1, 1.5, 0.4, 0.1]  # only used when run_mode: FORWARD

optimization:
  n_run: 100          # DE: max generations. PSO/LATHYP: max iterations.
  n_particles: 50      # DE: population size. PSO: swarm size.
  # PSO-only:
  c1: 2.0               # cognitive (personal-best) weight
  c2: 2.0               # social (global-best) weight
  wmax: 0.9             # initial inertia weight
  wmin: 0.4             # final inertia weight
  # DE-MCMC-only:
  mcmc_walkers: 32       # number of emcee walkers
  mcmc_steps: 1000       # steps per walker (30% discarded as burn-in automatically)

# --- Gap-tolerant mode (see §10) ---
gap_tolerant: false
Qmedia: 15.3               # optional: supply a known long-term mean discharge instead of
                            # computing it from (possibly gappy) data
warmup_drop_days: 15
min_segment_days: 30

# --- Optional analyses ---
sensitivity_analysis: false
sensitivity_perturbations: [1.0, 2.0, 5.0]   # % perturbations for one-at-a-time sensitivity
```

**Note on `series`**: in the original Fortran model this switched between a continuous daily series (`c`) and a repeated "mean year" climatology (`m`). This Python port only implements the continuous-series behaviour; `series` here is used solely as a text label in output filenames, whatever value you set.

> **Heads-up on `mineff_index`**: the code reads this as a top-level config key (`config.get('mineff_index', ...)`), but every bundled example config in `examples/` nests it under `optimization:` instead. Nested that way, it's silently ignored and the default of `0.0` is used regardless of what you set. This matters if you were relying on a very low threshold (e.g. `-999.0`, as several examples set it) to capture every parameter set tried, including poor ones, in the `0_*.csv` history. Until this is resolved upstream, put `mineff_index` at the top level of your config as shown above, and double check your `0_*.csv` history file actually contains the range of scores you expect.

### Parameter meaning (`a1`–`a8`)

The 8 parameters correspond to physical terms in the governing equation (Toffolon & Piccolroaz, 2015):

| Parameter | Role |
|---|---|
| `a1` | Constant offset |
| `a2` | Sensitivity of water temperature to air temperature |
| `a3` | Linear heat-loss coefficient (damps water temperature back toward equilibrium) |
| `a4` | Exponent linking discharge to the river's effective thermal capacity (rating-curve exponent) |
| `a5` | Constant offset, scaled by relative discharge |
| `a6` | Amplitude of an additional seasonal (annual) cycle not explained by air temperature |
| `a7` | Phase of that seasonal cycle (as a fraction of the year, 0–1) |
| `a8` | Discharge-scaled heat-loss coefficient |

Only the subset relevant to your chosen `version` is actually calibrated (see the table in [§4](#4-choosing-a-model-version-and-integrator)); the rest are fixed at zero automatically. As a starting point for `parameter_bounds`, the original model's own example datasets used:

```yaml
parameter_bounds:
  min: [-5, -5, -5, -1, 0,  0,  0, -1]
  max: [15, 1.5, 5,  1, 20, 10, 1,  5]
```

Treat these as a wide starting range, not universal defaults. Narrow them once you have looked at your dotty plots (see [§8](#8-understanding-the-output-files)) and confirmed the calibrated values sit well inside the bounds, not pinned against an edge.

### Calibration modes (`run_mode`)

| Mode | What it does | When to use it |
|---|---|---|
| `DE` | Differential Evolution (global search) followed by L-BFGS-B polishing | Recommended default. This is robust and fast for most cases. |
| `PSO` | Particle Swarm Optimization | Legacy-compatible alternative to DE; tune `c1`/`c2`/`wmax`/`wmin` if used |
| `LATHYP` | Latin Hypercube sampling (no optimization, just space-filling exploration) | Exploring the response surface / a cheap uncertainty screen |
| `DE-MCMC` | Runs `DE` to find the best fit, then samples the posterior with MCMC (`emcee`) | You want calibration *and* parameter/prediction uncertainty envelopes |
| `FORWARD` | Runs the model once with a fixed parameter set (`parameters_forward`), no calibration | You already know the parameters (e.g. from a previous calibration) and just want to simulate |

## 7. Running the model

```bash
pyair2stream --config path/to/config.yaml
```

### 7.1 Run from the right directory

Paths under `paths:` in the config (`input_data`, `validation_data`, `output_dir`) are resolved **relative to the directory you run the command from**, not relative to the config file's location. If you run `pyair2stream` from somewhere other than where your data/config live, either:

- use absolute paths in `paths:`, or
- `cd` into the directory containing your config and data before running.

### 7.2 From Python

You can also drive individual stages directly, e.g. to script batch runs or inspect intermediate results:

```python
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis
from pyair2stream.optimization import DE_mode

data = read_calibration(config_file="config.yaml")
read_Tseries(data, "c")
aggregation(data)
statis(data)

DE_mode(data)                      # populates data.par_best and data.finalfit
print(data.par_best, data.finalfit)
```

## 8. Understanding the output files

Everything is written to `output_dir` (or `{project_name}/output_{version}` by default):

| File | Contents |
|---|---|
| `parameters.txt` | The parameter bounds actually used (after fixing version-inactive parameters to 0) |
| `0_*.csv` | Every parameter set tried during optimization, with its objective score. This serves as the raw material for dotty plots. |
| `1_*.out` | Best-fit parameters, plus the final efficiency index (calibration, then validation if run) |
| `2_*.csv` | Full simulated vs. observed time series for the **calibration** period |
| `3_*.csv` | Same, for the **validation** period (only created if validation data was supplied) |
| `goodness_of_fit_calibration_*.csv`, `goodness_of_fit_validation_*.csv`, `goodness_of_fit_full_simulation_*.csv` | R², RMSE, MAE, AIC, BIC (one file per period, each named after the plot it accompanies) |
| `calibration_*.png` / `.pdf` | Time-series plot: observed vs. simulated water temperature, restricted to periods with observations, calibration period only |
| `validation_*.png` / `.pdf` | Same, for the validation period |
| `full_simulation_*.png` / `.pdf` | Same, but plotted over the entire simulated record (including where there's no observation to compare against) |
| `convergence_*.png` / `.pdf` | Objective-function value vs. optimizer iteration. Check that this flattens out rather than still trending upward at the end. |
| `dottyplots_*.png` / `.pdf` | Parameter value vs. objective score, one panel per parameter. Use this to check your bounds (see [§6](#parameter-meaning-a1a8)) |
| `predicted_vs_measured_*.png` / `.pdf` | Simulated vs. observed scatter (one per period: calibration/validation/full_simulation) |
| `residual_diagnostics_*.png` / `.pdf` | Residual plots: histogram, Q-Q plot, and autocorrelation (one per period) |
| `sensitivity_*.csv` / `.png` / `.pdf` | Only if `sensitivity_analysis: true` (see [§11](#11-sensitivity-analysis-and-uncertainty-de-mcmc)) |
| `MCMC_chain_*.csv`, `MCMC_envelopes_*.csv` | Only for `run_mode: DE-MCMC` |
| `Forward_Prediction_Envelopes_*.csv` | Only for a `FORWARD` run with `forward_options.enable_prediction_intervals: true` (see [§12](#12-forward-prediction-intervals)) |
| `cv_results.csv` | Only if `cross_validation.enabled: true` (see [§13](#13-cross-validation)). Replaces the usual forward-run/plotting outputs above for that run. |
| `gaps_summary.txt` | Only for `gap_tolerant: true` (segment/gap diagnostics) |

### The first 365 rows of `2_*.csv` / `3_*.csv` are not real data

Every run internally prepends a duplicate of the first simulated year as a numerical warm-up, to reduce sensitivity to the initial condition. In the output CSVs, these warm-up rows are easy to spot: **`Year`, `Month`, and `Day` are all `-999`**. Ignore/filter out any row where `Year == -999` before analysing results. The real, evaluated time series starts at the first row with a genuine date.

## 9. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `Configuration file not found: ...` | Wrong `--config` path, or wrong working directory | Check the path; see [§7.1](#71-run-from-the-right-directory) |
| `Missing required calibration data file: ...` | `paths.input_data` doesn't resolve to an existing file | Check the path is correct relative to your working directory |
| `The time series in ... must start on January 1st.` | Non-gap-tolerant mode requires the record to start Jan 1 | Back-fill `T_air`/`Discharge` and mark `T_water` as missing for the lead-in period, or set `gap_tolerant: true` |
| `The time series in ... must be continuous at a daily time scale with no missing dates.` | A calendar day is missing from the CSV | Insert a row for every date in range, even if values are blank/`-999` |
| `The series of observed air temperature in ... must be complete.` / same for discharge | `T_air` or `Discharge` has gaps outside gap-tolerant mode | Fill the gaps, or set `gap_tolerant: true` (see [§10](#10-gap-tolerant-mode)) |
| `Missing 'Discharge' column in ...` | The `Discharge` column is absent, even for a version that doesn't use it | Add the column (dummy values are fine for versions 3/5) |
| `No valid segments found after gap detection and filtering.` | Gap-tolerant mode couldn't find any block of valid forcing data long enough (`min_segment_days`) | Loosen `min_segment_days`, or check your gap pattern |
| `Qmedia is zero or negative. Please supply Qmedia...` | Discharge is effectively all missing/invalid in gap-tolerant mode | Supply `Qmedia:` explicitly in the config |
| `n_dat is 0 after aggregation. No T_water observations survived.` | No usable `T_water` observations after filtering (e.g. `prc` too strict, or all observations fall in gaps) | Lower `prc`, check for excessive missing `T_water`, or check for a `gap_tolerant` misconfiguration |
| Results look implausible (e.g. temperatures diverging or oscillating wildly) | Parameter bounds too wide for `EUL`/`RK2`, or version/integrator mismatch | Switch to `RK4` or `CRN`; tighten `parameter_bounds` |

## 10. Gap-tolerant mode

By default `pyair2stream` requires `T_air` (and `Discharge`, if your version uses it) to be complete for the whole record. Setting `gap_tolerant: true` lets the model split the record into contiguous valid segments and calibrate across all of them together.

Before relying on it:

- **Gaps bias performance metrics upward.** Missing data often coincides with floods or freeze events. Removing those extremes from the fitted record tends to inflate NSE/KGE relative to what you'd get on a continuous record. Consequently, gap-tolerant metrics are not directly comparable to continuous-record ones. NSE and RMS degrade more gracefully than KGE here.
- **Supply `Qmedia` explicitly if high-flow periods are missing.** The internally computed mean discharge will be biased low otherwise.
- **`T_water` observations inside a forcing gap are dropped**, not interpolated.
- Each segment discards its first `warmup_drop_days` (default 15) from evaluation, since a segment restarts from a day-of-year climatology rather than a true initial condition.
- A segment shorter than `min_segment_days` (default 30) is discarded entirely. Check the console warnings and `gaps_summary.txt` for what was dropped.

## 11. Sensitivity analysis and uncertainty (DE-MCMC)

- Set `sensitivity_analysis: true` to get a one-at-a-time local sensitivity analysis around your best-fit parameters: each parameter is perturbed by ±`sensitivity_perturbations`% of its own value (bounded by `parameter_bounds`), and the mean absolute change in simulated water temperature is reported per parameter. This tells you which parameters the fit is most sensitive to, which is useful for deciding which bounds are worth tightening.
- Set `run_mode: DE-MCMC` to get full parameter and predictive uncertainty: it runs `DE` to find the best fit, then samples the posterior around it with `emcee`, producing an MCMC chain (`MCMC_chain_*.csv`) and 5th/50th/95th percentile prediction envelopes (`MCMC_envelopes_*.csv`). This is more expensive than `DE` alone. Expect it to take noticeably longer, scaling with `mcmc_walkers × mcmc_steps`.

## 12. Forward prediction intervals

When running the model in `FORWARD` mode using a previously generated MCMC parameter chain (e.g., from a `DE-MCMC` calibration run), you can opt to generate probabilistic prediction envelopes around the forward simulation.

### Prediction Interval Options

To enable forward prediction intervals, configure the `forward_options` block in your YAML file:

```yaml
forward_options:
  enable_prediction_intervals: true
  mcmc_chain_path: "output_v4/MCMC_chain_test_station_c_1d.csv"
  residual_sigma: 1.0  # Observation error variance (optional)
  n_samples: 1000      # Number of parameter sets to sample
  random_seed: 42      # Seed for reproducibility
```

### Noise Models

By default, the noise applied to generate the prediction envelopes assumes independent and identically distributed (i.i.d.) residuals. However, river water temperature residuals typically exhibit significant serial correlation. To generate more realistic uncertainty bounds, you can opt into an Autoregressive AR(1) noise model via the `uncertainty_options` block:

```yaml
uncertainty_options:
  noise_model: "ar1"  # "iid" (default) or "ar1"
  ar1_rho: null       # optional override, between -1 and 1
```

If `noise_model: "ar1"` is selected, `pyair2stream` will resolve the lag-1 autocorrelation coefficient (`rho`) using the following priority order:
1. **Explicit Override:** If `ar1_rho` is explicitly provided in the config, it is used directly.
2. **Own Observations:** If the forward run dataset includes actual water temperature observations, `rho` is estimated from the forward run's own daily residuals.
3. **Sidecar Carry-forward:** If a `_meta.json` sidecar file exists next to your `mcmc_chain_path` (generated automatically during `DE-MCMC` calibration), `rho` is read from that file.
4. **Fallback:** If none of the above are available, it falls back to `rho = 0.0` (equivalent to `iid`) and logs a warning.

**Important Caveats:**
- Uncertainty (`rho` and `sigma`) is quantified on daily residuals, even if the calibration objective function used aggregated (e.g., weekly) data.
- Both the `iid` and `ar1` methods add noise *after* the physical integration. This means lower prediction bounds might dip below the physical ice-cover floor (`Tice_cover`).
- The `rho` value used for the AR(1) interval is fixed and is not jointly calibrated with the physical parameters.

## 13. Cross-validation

`pyair2stream` supports date-based leave-one-year-out (or leave-N-years-out) cross-validation. This repeatedly holds out an entire seasonal block of `T_water` observations, recalibrates on the remaining years, and scores the held-out block.

Unlike traditional point-wise leave-one-out cross validation (LOOCV), which is poorly suited for stateful ODE integrators and suffers from autocorrelation leakage, the grouped block scheme robustly tests parameter generalization across different types of years.

**Note on Integrator State during Cross Validation:**
* If `gap_tolerant: true` is configured, cross-validation safely handles the held-out target window by invoking the model's segmented restart and climatology mechanisms.
* If `gap_tolerant: false`, the ODE free-integrates right through the held-out window using actual forcing data. There is no equivalent restart protection against state drift during the held-out window. This asymmetry is acceptable for most applications because the model is structurally mean-reverting, but users should be aware of the difference.

To enable cross-validation, add the `cross_validation` block to your YAML file:

```yaml
cross_validation:
  enabled: true
  unit: year                  # Either 'year' or 'n_years'
  n_years_per_fold: 1         # The number of years to hold out per fold
  water_year_start_month: 1   # Start of the seasonal block (1 = calendar year)
  min_train_years: 1          # Skip the first N eligible years for spin up
  skip_first_year: true       # First calendar/water year is skipped (warm-up only)
  optimizer_overrides:
    n_run: 20                 # Optional: Reduced iterations for CV folds to save time
```


### When to use `water_year_start_month != 1`

The seasonal cyclic term (phase-locked) is best grouped such that a fold boundary lands on a seasonal trough (a "water year") rather than splitting a seasonal cycle mid-winter/mid-summer. Adjust this month based on where your river's seasonal cycle is quietest.

### Performance considerations

Since cross-validation runs a full calibration process for every fold, N folds take N× a single production calibration's time. You can use `optimizer_overrides` to provide a cheaper configuration for cross-validation runs compared to your production run (e.g. smaller `n_run`, `n_particles`).

### Output

When enabled, the normal `forward()` validation and single post-processing steps are skipped. Instead, a `cv_results.csv` file will be generated in your output directory containing one row per fold with metrics (NSE, KGE, RMSE) and the calibrated parameters `p1`..`pN`.

## 14. Where to go next

- Browse `examples/` for runnable configs covering gap-tolerant mode, sensitivity analysis, forward prediction intervals, cross-validation, optimizer comparisons, and real river case studies. See the [Examples table in the README](README.md#examples) for what each one demonstrates.
- If your data has missing days of `T_air` or `Discharge`, read [§10 Gap-tolerant mode](#10-gap-tolerant-mode).
- If you want uncertainty bounds around your calibration or a forward projection, read [§11](#11-sensitivity-analysis-and-uncertainty-de-mcmc) and [§12](#12-forward-prediction-intervals).
- If you want to test how well your calibrated parameters generalize across years, read [§13 Cross-validation](#13-cross-validation).
- Compare different optimization algorithms (`PSO` vs `DE` vs `DE-MCMC`) on your dataset (see [Calibration modes](#calibration-modes-run_mode) in §6).
- See the main [README.md](README.md) for the model-version equations, output-file reference, testing instructions, and how to cite the original model.
- The original model is described in Toffolon, M. and Piccolroaz, S. (2015), *A hybrid model for river water temperature as a function of air temperature and discharge*, Environmental Research Letters, 10(11), 114011, https://doi.org/10.1088/1748-9326/10/11/114011. This is worth reading before calibrating a real river, particularly for guidance on choosing parameter bounds and interpreting dotty plots.
