# How pyair2stream works

This document describes every step `pyair2stream` takes to turn your data into
results. It is written for someone who has to understand, check, or explain a
result (for example in a report or a hearing), not only for modelling
specialists. Where a step differs from the original Fortran `air2stream`, this is
stated. Section numbers are cited by the program's own error messages.

For how to *run* the software, see the [User Guide](../USER_GUIDE.md).

**Contents**

1. [What the model is](#1-what-the-model-is)
2. [Input data](#2-input-data)
3. [The warm-up year](#3-the-warm-up-year)
4. [Discharge scaling (Qmedia)](#4-discharge-scaling-qmedia)
5. [The equation and model versions](#5-the-equation-and-model-versions)
6. [Solving the equation](#6-solving-the-equation)
7. [Measuring the fit](#7-measuring-the-fit)
8. [Calibration](#8-calibration)
9. [Validation](#9-validation)
10. [Gap-tolerant mode](#10-gap-tolerant-mode)
11. [Cross-validation](#11-cross-validation)
12. [Parameter and prediction uncertainty (DE-MCMC)](#12-parameter-and-prediction-uncertainty-de-mcmc)
13. [Forward runs and scenario comparisons](#13-forward-runs-and-scenario-comparisons)
14. [Sensitivity analysis](#14-sensitivity-analysis)
15. [Automatic checks](#15-automatic-checks)
16. [Limitations and good practice](#16-limitations-and-good-practice)
17. [Differences from the Fortran original](#17-differences-from-the-fortran-original)
18. [References](#18-references)

---

## 1. What the model is

`air2stream` (Toffolon and Piccolroaz, 2015) predicts the **daily mean water
temperature** of a river from **daily mean air temperature** and, optionally,
**daily mean discharge** (flow). It is a *hybrid* model: a single equation shaped
by the physics of a river's heat budget, whose 3 to 8 coefficients (parameters
`a1`–`a8`) are not measured but *fitted* to observed water temperature at your
site.

A run has two stages:

1. **Calibrate**: search for the parameter values that make simulated water
   temperature match your observations best (§8).
2. **Simulate**: run the model with those parameters, on the calibration period
   and, if supplied, on a separate validation period (§9), or on new inputs such
   as a scenario (§13).

## 2. Input data

Each input file is a CSV with one row per calendar day and the columns `Date`,
`T_air` (°C), `T_water` (°C) and `Discharge`. When a file is loaded:

- **Every calendar day must have a row.** A missing day is an error. A missing
  *value* is fine: leave the cell blank or write `-999`; both mean "missing".
- **`T_water` may have gaps.** Days without an observation are simulated but not
  scored.
- **`T_air` and `Discharge` must be complete** in the default mode. If they have
  gaps, use gap-tolerant mode (§10). Versions 3 and 5 do not use discharge, so the
  column may be absent or incomplete for them.
- **Discharge must be positive** for versions 4, 7 and 8, because the equation
  divides by a power of discharge. A zero or negative value is an error unless you
  set `min_theta_floor` (§5) or use gap-tolerant mode.
- **Calibration and validation files must start on 1 January** (as in the
  Fortran; not required in gap-tolerant mode) and be at least 365 days long.
  FORWARD runs (§13) may start on any day.
- Dates must be real (Gregorian) dates, unless you declare `calendar: "noleap"`
  (365-day years) or `"360_day"` (twelve 30-day months) for climate-model output.

## 3. The warm-up year

The simulated temperature on any day depends on the day before, so the simulation
needs a starting value. To make that starting value irrelevant, the first 365
rows of the record are copied and run **once before** the real record (a
"warm-up" or "spin-up" year). The simulation starts from the observed water
temperature on the first day (or 4 °C if that day has no observation), runs
through the copied year, and then continues into the real record.

The warm-up year is never scored and never written to output files. (Gap-tolerant
mode handles start values differently; see §10.)

## 4. Discharge scaling (Qmedia)

The model never sees discharge directly. It only uses the ratio

    θ = Discharge / Qmedia

where `Qmedia` is the mean of all valid (non-missing, positive) discharge values
in the **calibration** record, unless you set `Qmedia:` yourself. The fitted
parameters are only meaningful together with the `Qmedia` they were fitted with,
so:

- the validation period is simulated with the calibration `Qmedia`;
- every calibration writes it to `calibration_metadata.json`;
- a FORWARD run must be given it explicitly (`Qmedia:` or
  `paths.calibration_metadata`). Recomputing it from new discharge data would
  rescale θ and cancel part or all of the discharge change being studied.

## 5. The equation and model versions

The full (8-parameter) model is

    dTw/dt = [ a1 + a2·Ta − a3·Tw + θ·( a5 + a6·cos(2π(t − a7)) − a8·Tw ) ] / θ^a4

with `Tw` water temperature, `Ta` air temperature, `t` the time of year as a
fraction (day-of-year ÷ days in that year) and θ the scaled discharge (§4). In
words: water temperature is pulled towards a balance with air temperature (`a1`,
`a2`, `a3`). Further terms are weighted by discharge θ: a constant (`a5`), an
annual cycle with amplitude `a6` and timing `a7` (for effects air temperature does
not capture, such as groundwater or snowmelt), and extra relaxation (`a8`). The
whole rate is divided by θ^a4, which represents how discharge changes the
river's thermal inertia.

The simpler versions fix some parameters at zero:

| Version | Parameters used | Uses discharge | Seasonal term | Equation |
|:-:|---|:-:|:-:|---|
| 3 | a1, a2, a3 | no | no | `a1 + a2·Ta − a3·Tw` |
| 4 | a1–a4 | yes | no | `(a1 + a2·Ta − a3·Tw) / θ^a4` |
| 5 | a1, a2, a3, a6, a7 | no | yes | `a1 + a2·Ta − a3·Tw + a6·cos(2π(t − a7))` |
| 7 | all except a4 | yes | yes | full equation with θ^a4 = 1 |
| 8 | a1–a8 | yes | yes | full equation |

Parameters a version does not use are forced to zero everywhere, including
values typed into `parameters_forward`.

If `min_theta_floor` is set, θ is raised to at least that value before θ^a4 is
evaluated, so a zero-flow day does not make the equation undefined.

## 6. Solving the equation

The equation is stepped forward one day at a time (time step = 1 day) with the
daily inputs. Five methods (`integrator`) are available:

| Integrator | Method | Stability |
|---|---|---|
| `CRN` (default) | Crank–Nicolson (semi-implicit, 2nd order) | always stable |
| `EXP` | exponential / integrating factor | always stable |
| `RK4` | Runge–Kutta 4th order | only while B < 2.785 |
| `RK2` | Heun (Runge–Kutta 2nd order) | only while B < 2.0 |
| `EUL` | explicit Euler, Fortran variant (inputs of the next day) | only while B < 2.0 |

`B` is how fast water temperature relaxes (per day); for version 8,
`B = (a3 + a8·θ) / θ^a4`. Because B depends on discharge, the explicit methods
(RK4, RK2, EUL) can become unstable on flows different from calibration and then
give wrong numbers without any error. `CRN` is therefore the default and is the
method recommended by the original authors. `RK4`, `RK2` and `EUL` reproduce the
Fortran exactly and are mainly useful for that purpose.

After every step, water temperature is not allowed below `Tice_cover` (default
0 °C), as in the Fortran.

## 7. Measuring the fit

**Scored days.** A day is scored if it has an observed water temperature and is
not in the warm-up year (and, in gap-tolerant mode, not in the unscored start of
a segment, §10).

**Time resolution.** With `time_resolution: "1d"`, each scored day is compared
directly. With `"Nw"` (N weeks) the record is cut into consecutive blocks of N×7
days starting on the first day; with `"1m"` into calendar months. A block is used
only if the fraction of its days with a scored observation is at least `prc`
(default 1.0, i.e. every day). The block's observed value is the mean of those
observations, and the simulated value is the mean of the simulation **on the same
days**.

**Objective function** (`objective_function`), computed over the n scored values:

- **NSE** (Nash–Sutcliffe efficiency) = 1 − Σ(sim − obs)² / Σ(obs − mean obs)².
  1 is perfect; 0 means no better than always predicting the observed mean.
- **KGE** (Kling–Gupta efficiency) = 1 − √[(r − 1)² + (α − 1)² + (β − 1)²], with
  r the correlation, α the ratio of standard deviations (sim/obs) and β the ratio
  of means (sim/obs). 1 is perfect.
- **RMS** = root-mean-square error in °C (smaller is better; internally the
  negative is maximised).

**Other reported measures** (in `goodness_of_fit_*.csv` and plot titles): N,
NSE, R² (squared correlation between simulated and observed), RMSE, MAE, and
AIC = n·ln(SSE/n) + 2(k+1) and BIC = n·ln(SSE/n) + (k+1)·ln(n), where SSE is the
sum of squared errors and k the number of fitted parameters (the +1 is the error
variance). AIC and BIC assume independent errors (§16).

## 8. Calibration

Parameters are searched only within `parameter_bounds` (8 minimum and 8 maximum
values). The search maximises the objective function (§7).

- **`DE` (recommended)** — Differential Evolution (SciPy defaults: `best1bin`
  strategy, mutation 0.5–1.0, crossover 0.7, Latin-hypercube start) with a
  population of `n_particles` × 8 candidates for up to `n_run` generations,
  stopping earlier once the population has converged. The best candidate is then
  refined by a local L-BFGS-B search within the same bounds; the refined result
  is kept only if it is better.
- **`PSO`** — Particle Swarm Optimisation as in the Fortran: `n_particles`
  particles, `n_run` iterations, inertia decreasing linearly from `wmax` to
  `wmin`, attraction weights `c1`, `c2`. A particle that reaches a bound stops
  there. The search stops early once 90% of particles have converged on the best
  position.
- **`LATHYP`** — `n_run` Latin-hypercube samples of the bounds; the best is kept.
  This explores the parameter space rather than optimising it.

Every parameter set tried is written to `0_*.csv` with its objective, NSE, R² and
MAE. After the search, the model is re-run with the best parameters and the
objective is recomputed; if it does not match, the run stops with an error.

**Reproducibility.** With `random_seed:` set, DE, PSO, LATHYP and DE-MCMC give
identical results on every run. Without it, results can differ between runs.

## 9. Validation

If `paths.validation_data` is given, the calibrated parameters are run on that
record, with its own warm-up year (§3) and the calibration `Qmedia` (§4), and
scored exactly as in §7. Validation needs at least 365 days; otherwise it is
skipped with a message. Validation shows how well the model predicts data it was
not fitted to, which is the better guide to its reliability.

## 10. Gap-tolerant mode

With `gap_tolerant: true`, `T_air` and `Discharge` may have gaps:

1. The record is split into **segments**: runs of consecutive days with valid
   `T_air` (and, for versions 4/7/8, positive discharge). Segments shorter than
   `min_segment_days` (default 30) are dropped.
2. Each segment is simulated **separately**. It starts from the observed water
   temperature on its first day if there is one, otherwise from the average
   observed water temperature for that day of the year in the calibration record
   (missing days of the year are interpolated).
3. The first `warmup_drop_days` (default 15) of every segment are simulated but
   **not scored**, so the approximate start value can be forgotten. The program
   warns if this is shorter than about three relaxation times (3/B days) of the
   calibrated model.
4. There is no separate 365-day warm-up (§3); the record need not start on 1 January.
5. Water-temperature observations inside a gap are not used.

`gaps_summary.txt` lists the segments and how many observations were scored.
Scattered gaps can exclude much more data than their share: on the Dischmabach
record, removing 5% of days at random left 781 of 2,197 observations scorable.
Scores from gap-tolerant runs are not directly comparable to runs on complete
records, because the removed days (often floods or freezes) are rarely typical
(§16).

## 11. Cross-validation

Cross-validation tests how well calibrated parameters predict years they were
not fitted to. With `cross_validation.enabled: true` and `run_mode` DE, PSO or
LATHYP:

1. Years are labelled by calendar year, or by a water year starting in
   `water_year_start_month`.
2. The first year (and the next `min_train_years`, default 1) are never held out,
   because the model needs earlier data to start from. Later years become folds
   (one year each, or blocks of `n_years_per_fold`). Folds with fewer than
   `min_valid_obs` observations are skipped.
3. For each fold: its water-temperature observations are hidden; `Qmedia` (and,
   in gap-tolerant mode, the day-of-year climatology) is recomputed without the
   fold; the model is calibrated on the rest; the full record is simulated; and
   NSE, KGE and RMSE are computed on the hidden days only (daily values).
   In gap-tolerant mode the fold's air temperature and discharge are also hidden
   during calibration, so the fold becomes a gap.
4. `cv_results.csv` lists each fold's scores and parameters, plus the mean and
   standard deviation across folds and "pooled" scores over all held-out days.

Large variation of the parameters between folds means they are poorly determined
by the data (equifinality). A cross-validation run does not also produce a
single final calibration.

## 12. Parameter and prediction uncertainty (DE-MCMC)

`run_mode: DE-MCMC` first calibrates with DE (§8), then estimates how uncertain
the parameters and predictions are, using Markov chain Monte Carlo (MCMC):

1. **Prior.** Every parameter value inside the bounds is considered equally
   plausible beforehand; values outside are impossible. Results therefore depend
   on the bounds.
2. **Likelihood** (how well a parameter set explains the data), computed on the
   same scored values as the objective (§7), assuming normally distributed errors
   of constant size (the size is estimated, not supplied):
   - `noise_model: "iid"` (default) treats every day's error as independent:
     log L = −(n/2)·ln(SSE/n).
   - `noise_model: "ar1"` allows each day's error to carry over part of the
     previous day's (lag-1 autocorrelation ρ, estimated once from the DE fit's
     daily residuals, limited to 0–0.99). Errors are converted to independent
     "innovations" (e₀·√(1−ρ²); eₜ − ρ·eₜ₋₁) within each unbroken run of
     scored days, and log L = −(n/2)·ln(SSE_innovations/n) + (runs/2)·ln(1−ρ²).
   River temperature errors are usually strongly autocorrelated; `iid` then
   understates parameter uncertainty. `ar1` is recommended for daily data. (With
   weekly or monthly scoring there are no consecutive days, so `ar1` behaves like
   `iid`.)
3. **Sampling.** The `emcee` affine-invariant ensemble sampler runs
   `mcmc_walkers` (default 32) chains for `mcmc_steps` (default 1000) steps,
   starting close to the DE optimum (spread 0.1% of each parameter's bound range,
   reflected back inside the bounds). `DE-CV-MCMC` instead starts with the
   parameter spread found by cross-validation (§11); this only affects how fast
   the sampler settles, not what it converges to.
4. **Burn-in and convergence.** The first max(30% of steps, 5× the
   autocorrelation time) steps are discarded (or `burnin_fraction`). The program
   reports the autocorrelation time (warns if the run is shorter than 50× it),
   split-R̂ (warns above 1.01) and the acceptance fraction. **Treat results as
   unreliable while these warnings appear**; increase `mcmc_steps`.
5. **Prediction interval.** Up to 1000 parameter sets are drawn from the chain.
   For each, the model is run and random error is added to every day: normally
   distributed with standard deviation equal to that parameter set's daily
   root-mean-square residual (`iid`), or an AR(1) series with the same standard
   deviation and ρ (`ar1`). The `prediction_interval` (default 90%) is the band
   between the matching lower and upper percentiles of these simulations on each
   day. The program then reports the **coverage**: the share of observed days
   inside the band (it should be close to the nominal percentage).
6. Any drawn parameter set whose simulation diverges (§15) is excluded and
   reported; if more than `max_divergent_fraction` (default 10%) diverge, the run
   stops.

On any single day, iid and AR(1) errors have the same spread, so the daily band
is about the same width under both. They differ for multi-day quantities (weekly
means, days in a row above a threshold): use the raw ensemble for those
(`save_ensemble: true`, and `pyair2stream.scenario`), never averages of the
daily percentiles.

**Outputs:** `MCMC_chain_*.csv` (post-burn-in samples), `MCMC_chain_*_meta.json`
(σ, ρ, diagnostics, coverage, excluded draws), `MCMC_envelopes_*.csv`, and the
parameter summary `parameter_significance_*.csv` (posterior mean, standard
deviation, 95% credible interval, and whether that interval excludes zero).

## 13. Forward runs and scenario comparisons

`run_mode: FORWARD` runs the model with known parameters (`parameters_forward`,
8 values) on any input file, without calibrating. It requires the calibration
`Qmedia` (§4). When given `paths.calibration_metadata`, it also warns if more
than 1% of days have θ outside the range seen in calibration (extrapolation). If the file contains water-temperature
observations, the fit is reported as in §7.

**Prediction intervals** (`forward_options.enable_prediction_intervals: true`)
reuse a DE-MCMC chain (`mcmc_chain_path`): `n_samples` (default 1000) parameter
sets are drawn, each is run, and error is added with standard deviation σ =
`forward_options.residual_sigma`, or else the calibration's daily residual
standard deviation stored in the chain's `_meta.json`. For `ar1`, ρ is taken from
`ar1_rho`, else from this run's own residuals (if it has observations), else from
the calibration `_meta.json`. Coverage is reported if observations exist.

**Comparing two scenarios** (for example observed versus naturalised flow): run
FORWARD once per scenario from the same chain with `save_ensemble: true`, and
for the second run set `forward_options.reuse_sample_indices_from` to the first
run's `Forward_Prediction_Ensemble_*_meta.json`, so both use exactly the same
parameter sets. `scenario.paired_difference_from_files()` then checks this before
computing the difference draw by draw, which gives an uncertainty band for the
*difference* itself.

## 14. Sensitivity analysis

With `sensitivity_analysis: true`, after calibration each used parameter is
moved up and down by p% (each value in `sensitivity_perturbations`) of its
calibrated value (`sensitivity_perturbation_mode: "value"`) or of its bound range
(`"range"`), limited by the bounds, one parameter at a time. The index is the
mean absolute change in simulated water temperature over scored days, divided by
the relative size of the change: °C per 100% change of the parameter (or per one
bound range). It is local (valid near the calibrated values only) and ignores
interactions between parameters. Rows are marked `Bounded` when a bound made the
change one-sided.

## 15. Automatic checks

| Check | When | Effect |
|---|---|---|
| Missing dates, incomplete `T_air`/`Discharge`, non-positive discharge, short record | loading data | error |
| Invalid version, run mode, integrator, objective, time resolution, bounds | loading config | error |
| Stability of the chosen integrator (B vs. limit, §6) | before each user-facing simulation | warning; error if >10% of days exceed it |
| Simulated temperature not finite or above `max_plausible_twat` (60 °C) | after each user-facing simulation | error |
| Recomputed objective matches the calibration result | after calibration | error |
| Discharge outside the calibrated range | FORWARD runs | warning |
| Segment warm-up too short | gap-tolerant runs | warning |
| MCMC convergence and diverging draws | DE-MCMC, FORWARD intervals | warning / error |

## 16. Limitations and good practice

- **Daily means only.** Inputs and outputs are daily means. The model cannot
  predict daily maxima, minima or sub-daily peaks; a criterion defined on those
  needs a separate, justified relationship.
- **Fitted, site-specific model.** Parameters describe one site and period.
  Predictions for conditions outside the calibration range (air temperature,
  discharge) are extrapolations; check the θ-range warning.
- **Different parameter sets can fit equally well** (equifinality), especially
  for versions 7 and 8. Inspect the dotty plots; a parameter at a bound suggests
  the bounds are too narrow. Prefer the simplest version that validates well.
- **Prediction intervals rest on assumptions**: normally distributed errors of
  constant size, independent or AR(1). Check the residual plots (histogram,
  Q-Q, autocorrelation) and the reported coverage, ideally on validation data.
- **Converged sampling.** MCMC results are only valid once the convergence
  warnings (§12) are gone.
- **Metric caveats.** KGE's ratio of means is unstable when mean water
  temperature is near 0 °C. AIC/BIC assume independent errors and favour more
  complex versions when errors are autocorrelated. Gap-tolerant scores are not
  comparable to complete-record scores.
- **Record what you ran**: the config file, `random_seed`, the pyair2stream
  commit (`git rev-parse HEAD`), the Python package versions (`pip freeze`), and
  the output files `calibration_metadata.json` and any `_meta.json`.

**Evidence that the implementation is correct:** the test suite compiles the
original Fortran from source and checks that every version (3, 4, 5, 7, 8) with
each Fortran integrator gives the same daily water temperatures to within
6×10⁻⁶ °C (the precision of the Fortran's printed output); it also checks each
safeguard above. Running the published parameter sets for three Swiss rivers
reproduces the published calibration NSE to within 0.006 (see the README). On
the bundled synthetic example, the 90% DE-MCMC interval contained 89% of
calibration days and 92% of the held-out validation year.

## 17. Differences from the Fortran original

These are deliberate; each is covered by tests.

- **Default integrator is `CRN`** (the choice suggested in the Fortran's example input);
  RK4/RK2/EUL are unchanged and match the Fortran exactly.
- **Default optimiser is DE + L-BFGS-B**; PSO and LATHYP are kept.
- **PSO**: the best-so-far starts at −10³⁰ rather than 0 and ignores failed (NaN)
  evaluations, and the early-stop test uses a tolerance of 1e-4 (the Fortran's
  test can never be met, so it always runs to the end).
- **`Qmedia`** also excludes discharge ≤ 0, and is fixed at the calibration value
  for validation and FORWARD runs instead of being recomputed.
- **Seasonal phase** is computed from each row's real date (equivalent for
  records starting on 1 January); FORWARD runs may start on any date, with the
  warm-up year taking the phase of the rows it copies.
- **Checks added**: missing values, non-positive discharge, unused parameters,
  integrator stability and divergence (§15); the Fortran would run on silently.
- **Time resolution**: an out-of-range index in the Fortran's weekly aggregation
  of the last, partial block is avoided; monthly aggregation accepts only `1m`
  (the Fortran ignores the number of months).
- **`0_*.csv`** records every evaluated parameter set (the Fortran's
  `mineff_index` filter is not applied).
- **Added features** not in the Fortran: gap-tolerant mode, cross-validation,
  DE-MCMC uncertainty, forward prediction intervals and scenario pairing,
  sensitivity analysis, non-standard calendars, and diagnostic plots.

## 18. References

- Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water
  temperature as a function of air temperature and discharge. *Environmental
  Research Letters*, 10, 114011. https://doi.org/10.1088/1748-9326/10/11/114011
- Piccolroaz, S., Calamita, E., Majone, B., Gallice, A., Siviglia, A. and
  Toffolon, M. (2016). Prediction of river water temperature: a comparison
  between a new family of hybrid models and statistical approaches.
  *Hydrological Processes*, 30, 3901–3917.
- Storn, R. and Price, K. (1997). Differential evolution. *Journal of Global
  Optimization*, 11, 341–359.
- Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm
  for bound constrained optimization. *SIAM Journal on Scientific Computing*, 16,
  1190–1208.
- Kennedy, J. and Eberhart, R. (1995). Particle swarm optimization. *Proceedings
  of ICNN'95*, 1942–1948.
- Nash, J. E. and Sutcliffe, J. V. (1970). River flow forecasting through
  conceptual models part I. *Journal of Hydrology*, 10, 282–290.
- Gupta, H. V., Kling, H., Yilmaz, K. K. and Martinez, G. F. (2009).
  Decomposition of the mean squared error and NSE performance criteria.
  *Journal of Hydrology*, 377, 80–91.
- Goodman, J. and Weare, J. (2010). Ensemble samplers with affine invariance.
  *Communications in Applied Mathematics and Computational Science*, 5, 65–80.
- Foreman-Mackey, D., Hogg, D. W., Lang, D. and Goodman, J. (2013). emcee: the
  MCMC hammer. *Publications of the Astronomical Society of the Pacific*, 125,
  306–312.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and
  Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd edn. CRC Press (split-R̂).
