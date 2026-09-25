# pyair2stream User Guide

This guide takes you from installation to reading results. You do not need to be
a modelling expert. For what the software does internally, step by step, see
[docs/METHODS.md](docs/METHODS.md).

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

`pyair2stream` fits a small equation to your river's daily mean water
temperature, driven by daily mean air temperature (and, for some versions,
discharge). Each run:

1. **Calibrates**: finds the 3–8 parameter values that best reproduce your
   observed water temperature.
2. **Simulates**: runs the model with those parameters on the calibration period
   and, if you give one, on a separate **validation** period it was not fitted
   to. The validation score is the best guide to how well the model predicts.

You control everything with one YAML configuration file and CSV data files.

## 2. Install

Requires Python 3.9 or newer.

```bash
git clone https://github.com/LukeAFullard/pyair2stream.git
cd pyair2stream
pip install .
pyair2stream --help        # check it installed
```

## 3. Your first run: the bundled example

The quick-start example calibrates the model on the Mentue, a small Swiss river
(`data/switzerland/`): 2002–2009 to calibrate, 2010–2012 to validate. **Run it
from the repository's top folder** (paths in the config are relative to where
you run the command, see [§7.1](#71-run-from-the-right-directory)):

```bash
pyair2stream --config examples/01_quickstart/config.yaml
```

After a banner, you should see:

```
mean, TSS and standard deviation (calibration)
9.73069 96513.81949 5.76298
Pop. Size (particles) = 50, Max Generations (runs) = 100
DE Finished. Best internal negated objective: -0.985672
L-BFGS-B Finished. Best internal negated objective: -0.987927
Efficiency Index in calibration 0.9879266378568211
Consistency check passed.
mean, TSS and standard deviation (validation)
9.67611 37744.04425 5.87375
```

The config sets `random_seed: 42`, so your numbers should be identical. The
calibration NSE is 0.988 (1.0 would be perfect). On the validation years, NSE is
0.982 and RMSE 0.78 °C (`goodness_of_fit_validation_DE_NSE_Mentue.csv`). Open
`validation_DE_NSE_Mentue.png` to see observed and simulated temperatures.
[§8](#8-understanding-the-output-files) explains every file, and
[examples/](examples/README.md) has five more worked examples: uncertainty,
compliance with a temperature limit, scenarios, gaps and cross-validation.

## 4. Choosing a model version and integrator

### Model version (`version`)

| Version | Parameters | Needs discharge? | Seasonal term? | Use when... |
|:-:|:-:|:-:|:-:|---|
| 3 | 3 | no | no | a quick baseline from air temperature only |
| 4 | 4 | yes | no | discharge matters, no extra seasonal effect |
| 5 | 5 | no | yes | no reliable discharge, but a seasonal effect air temperature does not explain (e.g. groundwater, snowmelt) |
| 7 | 7 | yes | yes | full model without the discharge exponent `a4` |
| 8 | 8 | yes | yes | full model; the usual starting point with good discharge data |

Start with 8 (or 5 without discharge) and compare with simpler versions on the
**validation** period. A version that fits calibration better but validates worse
is over-fitted; prefer the simplest version that validates well.
Cross-validation (example [06](examples/06_cross_validation/README.md)) is the
most thorough comparison. On the three Swiss rivers, versions 7 and 8 predicted
unseen years best ([validation V5](validation/REPORT.md#v5)).

### Integrator (`integrator`)

Keep the default **`CRN`**. It is stable for any discharge and parameters (and
is the choice recommended by the model's authors). `EXP` is also always stable.
`RK4`, `RK2` and `EUL` exist to reproduce the original Fortran exactly; they can
become unstable, giving wrong numbers, when discharge differs from calibration,
so never use them for scenario runs ([§9.1](#91-numerical-stability-and-the-choice-of-integrator)).

## 5. Preparing your own data

One CSV per period (calibration, and optionally validation) with these columns:

| Column | Required? | Notes |
|---|---|---|
| `Date` | yes | e.g. `2020-01-31`; any format pandas understands |
| `T_air` | yes | daily mean air temperature (°C) |
| `T_water` | yes | daily mean observed water temperature (°C); gaps allowed |
| `Discharge` | versions 4, 7, 8 | daily mean flow (any unit); may be omitted for versions 3 and 5 |

Rules the program enforces:

- **One row for every calendar day.** Show a missing value as an empty cell or
  `-999` in an existing row; never skip the date.
- **`T_air` and `Discharge` must have no gaps** unless you use
  [gap-tolerant mode](#10-gap-tolerant-mode).
- **Discharge must be above zero** for versions 4, 7 and 8 ([§9.2](#92-zero-or-negative-discharge)).
- **Calibration and validation files must start on 1 January and be at least
  365 days long.** If your record starts later, start the file on 1 January with
  real air temperature and discharge and leave `T_water` empty until your
  observations begin. Several years of data is better than one. `FORWARD`
  (scenario) files may start on any day.

```csv
Date,T_air,T_water,Discharge
2020-01-01,5.2,4.1,12.5
2020-01-02,4.8,,11.8
2020-01-03,6.1,4.0,10.2
```

**Climate-model output** often uses a 365-day (`calendar: "noleap"`) or 360-day
(`calendar: "360_day"`) calendar. Declare it; do not pad such data with fake
dates and leave `calendar: "standard"`, or the seasonal term will drift out of
phase.

**Checking and preparing data:**
`pyair2stream.analyze_timeseries(df)` reports missing data and usable segments
before you calibrate, and `pyair2stream.merge_timeseries(...)` builds a daily
file (daily means of all readings on each day) from separate raw files.

## 6. Configuration reference

All keys are optional except where marked; defaults are shown.

```yaml
# --- Names (used in output file names) ---
project_name: "pyair2stream_project"
station_name: "AirStation"
water_station: "AirStation"       # default: same as station_name
series: "series"                  # free-text label

# --- Model ---
version: 8                  # 3, 4, 5, 7 or 8 (§4)
integrator: "CRN"           # CRN, EXP, RK4, RK2 or EUL (§4)
Tice_cover: 0.0             # water temperature is never simulated below this (°C)
calendar: "standard"        # standard, noleap or 360_day (§5)
min_theta_floor: null       # e.g. 1.0e-6 to allow zero-flow days (§9.2)
Qmedia: null                # mean discharge used to scale flow; required for FORWARD (below)

# --- Calibration ---
run_mode: "DE"              # DE, PSO, LATHYP, DE-MCMC or FORWARD (below)
objective_function: "NSE"   # NSE, KGE or RMS (docs/METHODS.md §7)
time_resolution: "1d"       # "1d" daily, "Nw" N-week means (e.g. "2w"), "1m" monthly means
prc: 1.0                    # weekly/monthly: minimum fraction of days with an observation
random_seed: null           # set an integer to make results exactly repeatable

parameter_bounds:           # required for calibration: 8 values each, for a1..a8
  min: [-5, -5, -5, -1, 0,  0,  0, -1]
  max: [15, 1.5, 5,  1, 20, 10, 1,  5]

# parameters_forward: [8 values]   # FORWARD only; default: from paths.calibration_metadata

optimization:
  n_run: 100                # DE: max generations; PSO: iterations; LATHYP: samples
  n_particles: 50           # DE: population = n_particles x 8; PSO: number of particles
  c1: 2.0                   # PSO only: pull towards each particle's own best
  c2: 2.0                   # PSO only: pull towards the swarm's best
  wmax: 0.9                 # PSO only: starting inertia
  wmin: 0.4                 # PSO only: final inertia
  mcmc_walkers: 32          # DE-MCMC only (at least 2 x number of calibrated parameters)
  mcmc_steps: 20000         # DE-MCMC only: the most steps; it stops once converged (§11)

paths:
  input_data: "data/calibration.csv"          # required
  validation_data: "data/validation.csv"      # optional
  output_dir: "output"                        # default: "<project_name>/output_<version>"
  calibration_metadata: null                  # FORWARD: a calibration's calibration_metadata.json (§12)

# --- Safety checks (docs/METHODS.md §15) ---
max_plausible_twat: 60.0            # stop if a simulated temperature exceeds this (°C)
stability_error_fraction: 0.10      # stop if more than this share of days is unstable (§9.1)

# --- Optional features ---
gap_tolerant: false                 # §10
warmup_drop_days: 15                # §10
min_segment_days: 30                # §10
sensitivity_analysis: false         # §11
sensitivity_perturbations: [1.0]    # % changes to test, e.g. [1.0, 5.0]
sensitivity_perturbation_mode: "value"   # "value" or "range" (§11)
# uncertainty_options:              # see §11-12
# forward_options:                  # see §12
# cross_validation:                 # see §13
```

`mineff_index` (from Fortran configs) is accepted but has no effect: `0_*.csv`
always records every parameter set tried.

### Parameters `a1`–`a8`

| Parameter | Role in the equation (docs/METHODS.md §5) |
|---|---|
| `a1` | constant heat input |
| `a2` | effect of air temperature |
| `a3` | how fast water temperature relaxes back to balance |
| `a4` | how discharge changes the river's thermal inertia (exponent) |
| `a5` | constant term scaled by discharge |
| `a6` | amplitude of an extra annual cycle |
| `a7` | timing (phase) of that cycle, as a fraction of the year |
| `a8` | relaxation term scaled by discharge |

Parameters your version does not use are fixed at zero automatically, but you
still give 8 bounds. The ranges above are the original authors' and are a good
start; if a calibrated value ends up exactly on a bound, widen that bound.

### Run modes (`run_mode`)

| Mode | What it does |
|---|---|
| `DE` | Differential Evolution then a local polish. **Recommended.** |
| `PSO` | Particle Swarm Optimisation, as in the original Fortran. Less reliable: check it converged. |
| `LATHYP` | Latin Hypercube sampling of the bounds (exploration, not optimisation). |
| `DE-MCMC` | `DE`, then parameter and prediction uncertainty (§11). |
| `FORWARD` | No calibration: runs given parameters, e.g. on a scenario (§12). |

### Qmedia: keep it fixed when discharge changes

The model sees discharge only as `Discharge / Qmedia`, where `Qmedia` is the
mean discharge of the calibration record. Fitted parameters only make sense with
the `Qmedia` they were fitted with. If you ran the model on a changed-flow
scenario and recomputed `Qmedia` from the scenario's discharge, the change would
be cancelled out: for a uniform 1.5× flow increase, the simulated temperature
change would be exactly zero instead of the real response.

So a `FORWARD` run must be given the calibration `Qmedia`, either as `Qmedia:`
or, better, with `paths.calibration_metadata` pointing to the
`calibration_metadata.json` written by the calibration. The latter also
supplies the calibrated parameters, checks that `version` and `integrator`
match, and warns if the scenario's flows go outside the calibrated range.

## 7. Running the model

```bash
pyair2stream --config path/to/config.yaml
```

### 7.1 Run from the right directory

Paths in `paths:` are relative to the folder you run the command **from**, not
to the config file. Either run from the folder they are relative to, or use
absolute paths.

### 7.2 From Python

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

## 8. Understanding the output files

All files go to `output_dir`. File names include the run mode, objective,
station, series label and time resolution.

| File | Contents |
|---|---|
| `1_*.out` | line 1: the 8 best parameters; line 2: calibration score; line 3: validation score (if run) |
| `2_*.csv` / `3_*.csv` | one row per day of the calibration / validation period: `Year, Month, Day, Tair, Twat_obs, Twat_mod, Twat_obs_agg, Twat_mod_agg, Q` (`_agg` = the values actually scored; `-999` = none). Gap-tolerant runs add `Tair_gap, Q_gap, segment_id`. |
| `goodness_of_fit_<period>_*.csv` | N, NSE, R² (squared correlation), RMSE, MAE, AIC, BIC for `calibration`, `validation`, `full_simulation` |
| `calibration_*.png`, `validation_*.png` | observed vs. simulated water temperature over the observed period, with residuals below |
| `full_simulation_*.png` | the same over the whole record |
| `predicted_vs_measured_*.png` | scatter of simulated vs. observed |
| `residual_diagnostics_*.png` | residual histogram, normal Q-Q plot and autocorrelation: check the assumptions behind uncertainty bands |
| `0_*.csv`, `convergence_*.png`, `dottyplots_*.png` | every parameter set tried; best score so far vs. evaluations (should flatten out); score vs. each parameter |
| `calibration_metadata.json` | `Qmedia`, calibrated flow range, version, integrator, parameters, seed (not written by `FORWARD` runs) |
| `parameters.txt` | the bounds actually used |
| `gaps_summary.txt` | gap-tolerant runs: segments found |
| `sensitivity_*` | §11 |
| `MCMC_*`, `parameter_significance_*`, `parameter_correlation_*` | §11 |
| `Forward_Prediction_*`, `forward_projection*.png` | §12 |
| `cv_results.csv` | §13 |

**Reading the scores.** NSE: 1 is perfect, 0 is no better than the long-term mean;
NSE above 0.9 is common for daily water temperature with this model.
RMSE and MAE are in °C. R² measures only correlation, so it can be high even if
the simulation is biased: always check NSE or RMSE too.

**AIC/BIC** (lower is better) are for comparing versions on the same data. They
assume independent daily errors; real errors are usually correlated, which
favours more complex versions. Use them as a rough guide, alongside validation
scores.

## 9. Troubleshooting

| Message | Fix |
|---|---|
| `Configuration file not found` | Check the path and your working folder ([§7.1](#71-run-from-the-right-directory)). |
| `Invalid version` / `run_mode` / `integrator` / `objective_function` / `time_resolution` | Use one of the values listed in the message. |
| `parameter_bounds.min must be a list of exactly 8 numbers` (or `parameters_forward`) | Give 8 values, one per parameter a1..a8. |
| `parameter_bounds: min > max` | Swap or correct the listed parameter's bounds. |
| `No parameter is free to calibrate` | Add `parameter_bounds`. |
| `Missing required calibration data file` | Check `paths.input_data`. |
| `must start on January 1st` | Start the file on 1 January (see [§5](#5-preparing-your-own-data)) or use gap-tolerant mode. |
| `must be continuous at a daily time scale` | Add a row for every missing date (values may be empty). |
| `The series of observed air temperature / discharge ... must be complete` | Fill the gaps or use [gap-tolerant mode](#10-gap-tolerant-mode). |
| `Missing 'Discharge' column` | Versions 4/7/8 need discharge. |
| `Non-positive discharge (Q <= 0)` | See [§9.2](#92-zero-or-negative-discharge). |
| `FORWARD mode requires an explicit Qmedia` | Set `Qmedia:` or `paths.calibration_metadata` ([§6](#qmedia-keep-it-fixed-when-discharge-changes)). |
| `n_dat is 0 after aggregation` | No usable `T_water` values: check the column, or lower `prc`. |
| `No valid segments found` | Gap-tolerant: no gap-free stretch is at least `min_segment_days` long. |
| `Qmedia is zero or negative` | Gap-tolerant: too little valid discharge; set `Qmedia:`. |
| `NumericalDivergenceError` / `exceed the ... stability limit` | Use `CRN` or `EXP` ([§9.1](#91-numerical-stability-and-the-choice-of-integrator)). |
| `Efficiency mismatch in forward run` | Internal consistency check failed; please report it with your config. |
| `mcmc_walkers ... must be at least 2x` | Increase `mcmc_walkers`. |
| `MCMC did not converge within ... steps` | Try a simpler model version, or increase `mcmc_steps` ([§11](#11-uncertainty-de-mcmc-and-sensitivity-analysis)). |
| `FORWARD mode needs parameters` | Set `paths.calibration_metadata` (or `parameters_forward`) ([§12](#12-scenario-runs-and-prediction-intervals)). |
| `Note: integrator RK4/RK2/EUL is kept to reproduce the original Fortran` | Use `CRN` unless you need Fortran-identical results ([§9.1](#91-numerical-stability-and-the-choice-of-integrator)). |
| `enable_prediction_intervals is True but residual_sigma is 0.0/unavailable` | Point `mcmc_chain_path` at a chain with its `_meta.json`, or set `residual_sigma`. |
| `draws ... were excluded as numerically divergent` | Use `CRN`/`EXP`, or check the chain and bounds ([§12](#12-scenario-runs-and-prediction-intervals)). |
| `paired_difference_from_files: ... differs` | The two scenario runs did not use the same parameter draws ([§12](#12-scenario-runs-and-prediction-intervals)). |
| `Warning: warmup_drop_days=... is shorter than` | Gap-tolerant: increase `warmup_drop_days` as suggested ([§10](#10-gap-tolerant-mode)). |

### 9.1 Numerical stability and the choice of integrator

The model relaxes water temperature towards a balance at a rate `B` per day; for
version 8, `B = (a3 + a8·θ) / θ^a4` with `θ = Discharge/Qmedia`. With a one-day
step, the explicit integrators are only stable while B stays below a limit:
`EUL` and `RK2` 2.0, `RK4` 2.785. `CRN` and `EXP` are always stable. Because B
depends on discharge, parameters that are stable for the calibration flows can
be unstable for other flows. An unstable run can produce plausible-looking but
wrong numbers without any error.

The program therefore defaults to `CRN`; warns before simulating if B exceeds
the chosen integrator's limit on some days (and stops if more than
`stability_error_fraction` of days do); and stops if any simulated temperature
is not a number or exceeds `max_plausible_twat`. Use `RK4`/`RK2`/`EUL` only to
reproduce Fortran results, never for scenarios: even when stable they can be
inaccurate with a one-day step (by up to about 1 °C for `EUL` on the Mentue,
[validation V6](validation/REPORT.md#v6)), and the program prints a note when
one is selected. Calibrating with `EXP` instead of `CRN` changed predictions by
less than 0.03 °C on the Swiss rivers.

Parameters belong to the integrator they were calibrated with: run them with the
same one. A `FORWARD` run given `paths.calibration_metadata` refuses a
different integrator or model version.

### 9.2 Zero or negative discharge

For versions 4, 7 and 8 the equation divides by `θ^a4`, which is undefined at
zero flow. A zero or negative discharge therefore stops the run when the data
are loaded, naming the first bad date. Options:

1. correct the data;
2. use `gap_tolerant: true`, which treats those days as gaps; or
3. set `min_theta_floor: 1.0e-6` (for example) to keep θ above that value. Use
   this only for genuine zero-flow days, and expect large simulated responses
   there, since θ^a4 grows very large as flow approaches zero.

## 10. Gap-tolerant mode

With `gap_tolerant: true`, `T_air` and `Discharge` may have gaps. The record is
split into gap-free **segments**; each is simulated separately, starting from the
observed water temperature on its first day (or the average for that day of the
year). Segments shorter than `min_segment_days` (30) are dropped, and the first
`warmup_drop_days` (15) of each segment are not scored, so the approximate start
value does not affect the score. The program warns if 15 days is too short for
your calibrated model. The record need not start on 1 January.

Be aware:

- **Scattered gaps discard far more data than their share suggests**: with 5% of
  days missing at random, only about a third of the observations could be
  scored ([validation V7](validation/REPORT.md#v7)), because few gap-free
  stretches reach `min_segment_days`. Fill short gaps instead (from a nearby
  station, or by interpolation over a day or two) and keep gap-tolerant mode for
  long gaps. Example [05](examples/05_gaps/README.md) compares the two. Check
  `gaps_summary.txt` for how many observations were used.
- **Scores are not directly comparable with complete-record scores.** Gaps often
  remove unusual periods such as floods or freezes, which can make the fit look
  better than it would be on the full record.
- **Set `Qmedia` yourself** if high-flow periods are missing, as the computed
  mean flow will then be too low.
- `T_water` observations inside a gap are not used.
- Check `gaps_summary.txt` and the console warnings for dropped segments.

## 11. Uncertainty (DE-MCMC) and sensitivity analysis

### Parameter and prediction uncertainty

`run_mode: "DE-MCMC"` calibrates with DE, then explores all parameter sets
consistent with the data (Markov chain Monte Carlo, `emcee`) and gives a
**prediction interval**: a band that should contain the stated share of observed
daily temperatures. Method: [docs/METHODS.md §12](docs/METHODS.md#12-parameter-and-prediction-uncertainty-de-mcmc).

```yaml
run_mode: "DE-MCMC"
optimization:
  mcmc_walkers: 32
  mcmc_steps: 20000             # the most steps it may take; it stops once converged
uncertainty_options:
  noise_model: "ar1"            # the default; "iid" is also available (see below)
  prediction_interval: 90       # % width of the band
  save_ensemble: false          # true: also save every simulated series (.npz)
  strict_convergence: true      # default: stop with an error if not converged
  burnin_fraction: null         # override the automatic burn-in (0-1)
  on_divergent_draw: "drop"     # or "raise"
  max_divergent_fraction: 0.10
```

Check before using the results (example [02](examples/02_uncertainty/README.md)
walks through this):

- **Convergence.** The sampler runs in blocks of 1,000 steps until its results
  are stable: the chain is at least 50 times its autocorrelation time and
  split-R̂ is below 1.01. It then prints `MCMC converged after ... steps`. If
  that has not happened by `mcmc_steps`, the run stops with an error and no
  interval is produced. This usually means the data cannot pin down all the
  parameters: try a simpler model version. With `strict_convergence: false` it
  continues instead, and marks every result as not converged; do not use such
  results for decisions.
- **Coverage.** The console and `MCMC_chain_*_meta.json` report
  `interval_coverage`: the share of observed days inside the band. It should be
  close to `prediction_interval`. Much lower means the band is too narrow.
- **`noise_model`.** Real model errors persist from day to day. For a single
  day, `"iid"` and `"ar1"` give bands of about the same width. For anything
  spanning several days they do not: on the Swiss rivers, 90% bands for 7-day
  means contained 39–62% of observed values with `"iid"` and 76–88% with
  `"ar1"` ([validation V5](validation/REPORT.md#v5)). Keep the default `"ar1"`.
- **Parameters.** For versions with many parameters (especially 8), several
  combinations fit almost equally well. Their intervals are then too narrow
  ([V4](validation/REPORT.md#v4)), and with `"ar1"` they can be centred away
  from the DE best fit, which assumes independent errors. The predictions are
  hardly affected. Rely on predictions, not on individual parameter values.

Outputs: `MCMC_chain_*.csv` (parameter samples), `MCMC_chain_*_meta.json`
(settings, diagnostics, residual σ and ρ, coverage), `MCMC_envelopes_*.csv`
(`Twat_mod_lower`, `Twat_mod_p50`, `Twat_mod_upper` per day),
`parameter_significance_*.csv` (mean, SD and 95% interval of each parameter)
and `parameter_correlation_*.png`.


### Sensitivity analysis

`sensitivity_analysis: true` moves each parameter up and down by each percentage
in `sensitivity_perturbations`, one at a time, and reports the mean change in
simulated water temperature (`sensitivity_*.csv`, `.png`). The index is in °C
per 100% change of the parameter's value (`"value"` mode) or per its full bound
range (`"range"`, which is easier to compare across parameters). It describes
behaviour near the calibrated values only.

## 12. Scenario runs and prediction intervals

`run_mode: "FORWARD"` runs known parameters on any input file: for example
naturalised flows, an abstraction scenario or climate projections.

```yaml
run_mode: "FORWARD"
version: 8
integrator: "CRN"
paths:
  input_data: "data/scenario.csv"
  output_dir: "output/scenario"
  # From the calibration: its parameters, Qmedia (§6), version and integrator.
  calibration_metadata: "output/calibration_metadata.json"
# parameters_forward: [8 values]  # only to run other parameters than the calibrated ones
uncertainty_options:
  noise_model: "ar1"          # the same as in the calibration
forward_options:
  enable_prediction_intervals: true
  mcmc_chain_path: "output/MCMC_chain_Station_A_series_1d.csv"   # from DE-MCMC
  n_samples: 1000
  random_seed: 42
  residual_sigma: null        # default: taken from the chain's _meta.json
```

With prediction intervals, parameter sets are drawn from the chain, each is run,
and random error of the calibration's typical size and persistence (σ and ρ,
from the chain's `_meta.json`) is added
([docs/METHODS.md §13](docs/METHODS.md#13-forward-runs-and-scenario-comparisons)).
If the scenario file has water temperature observations, the fit and the
interval coverage are reported. Lower bounds can fall below `Tice_cover` because
the error is added after the simulation.

**Multi-day quantities** (weekly means, days in a row above a limit, degree-days)
must be computed from the individual simulations, not from the daily band:
set `uncertainty_options.save_ensemble: true` and use `pyair2stream.scenario`
(`load_ensemble`, `aggregate`, `exceedance`). This is where `noise_model: "ar1"`
(the default) matters: it keeps each simulated error series realistically
persistent.
Example [03](examples/03_compliance/README.md) computes the probability that a
7-day mean limit was exceeded.

**Comparing two scenarios.** To get an uncertainty band for the *difference*
(for example abstraction minus natural flow), both runs must use the same
parameter draws:

1. Run scenario A with `save_ensemble: true`.
2. Run scenario B with the same `mcmc_chain_path`, `save_ensemble: true` and
   `forward_options.reuse_sample_indices_from:
   "output/scenario_a/Forward_Prediction_Ensemble_<...>_meta.json"`.
3. Compute the difference:

```python
from pyair2stream import scenario
diff = scenario.paired_difference_from_files(
    "output/scenario_a/Forward_Prediction_Ensemble_<...>.npz",
    "output/scenario_b/Forward_Prediction_Ensemble_<...>.npz",
)   # one row per parameter draw, one column per day
```

This checks that the two runs really used the same draws before subtracting.
Each draw also gets the same random error in both runs, so the error cancels
and the spread of the difference is the parameter uncertainty of the effect.
This assumes the model's error on a given day would be the same under both
scenarios. Example [04](examples/04_scenario/README.md) works through a flow
abstraction.

## 13. Cross-validation

Cross-validation calibrates the model repeatedly, each time hiding one year, and
scores it on the hidden year. It shows how well the model predicts years it has
not seen, and whether the parameters are stable from year to year.

```yaml
run_mode: "DE"              # DE, PSO or LATHYP
cross_validation:
  enabled: true
  unit: "year"              # or "n_years" with n_years_per_fold
  n_years_per_fold: 1
  water_year_start_month: 1 # e.g. 10 for October-September years
  min_train_years: 1        # extra leading years never held out
  skip_first_year: true     # the first year is never held out
  min_valid_obs: 10         # skip years with fewer observations
  optimizer_overrides:      # optional cheaper settings for each fold
    n_run: 50
```

With the defaults, the first two years are always used for training only. The run
writes `cv_results.csv` (one row per held-out year with NSE, KGE, RMSE on daily
values and the fitted parameters, plus `mean`, `std` and `pooled` rows) instead
of the usual outputs. Large differences in parameters between years mean the
data do not pin them down well. The spread of the parameters between folds is
not a confidence interval: each fold shares most of its data with the others, so
the spread is much smaller than the real uncertainty (in a test with known
parameters it contained the true values only about half the time,
[validation V4](validation/REPORT.md#v4)). For parameter uncertainty use `DE-MCMC`
(§11). Cross-validation is ignored (with a warning) in other run modes.

## 14. Checklist for results that support a decision

Before using results for a compliance assessment, a permit or any other
decision, check:

1. **Validation**: the model scores well on a period it was not calibrated on
   (validation file or cross-validation).
2. **Plausible parameters**: none sits exactly on a bound (dotty plots,
   `1_*.out`).
3. **Residuals**: no strong pattern over time or with temperature
   (`residual_diagnostics_*.png`).
4. **Uncertainty**: if you report a band, the reported coverage is close to the
   nominal level, ideally on validation data. Bands for new years are usually
   slightly narrow (85–89% for 90% bands on the Swiss rivers,
   [validation V5](validation/REPORT.md#v5)). For 7-day means, runs of days or
   other multi-day quantities, keep `noise_model: "ar1"` (the default) and compute them from
   the saved simulations (§12). Report probabilities with their ranges, not as
   a yes or no.
5. **Scope**: the model gives **daily means**. A limit on daily maxima or on
   sub-daily values needs a separate, justified step. Scenario inputs outside
   the calibrated range of air temperature or flow are extrapolation.
6. **Reproducibility**: keep the config (with `random_seed`), the input files,
   the pyair2stream commit (`git rev-parse HEAD`), `pip freeze` output, and the
   output folder.
7. **The software**: [validation/REPORT.md](validation/REPORT.md) records the
   checks that pyair2stream reproduces the original model and the published
   results, and that its intervals are calibrated. Cite it with the commit.

See [docs/METHODS.md §16](docs/METHODS.md#16-limitations-and-good-practice) for
the full list of assumptions and limitations.
