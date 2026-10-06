# How pyair2stream works

This document describes every step `pyair2stream` takes to turn your data into
results. It is written for someone who has to understand, check, or explain a
result (for example in a report or a hearing), not only for modelling
specialists. Where a step differs from the original Fortran `air2stream`, this is
stated. Section numbers are cited by the program's own error messages.

For how to *run* the software, see the [User Guide](../USER_GUIDE.md). For a
plain-language explanation of the uncertainty statistics and tests (§11–§13),
with figures, see [UNCERTAINTY.md](UNCERTAINTY.md).

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

A run has two stages, and optionally a third:

1. **Calibrate**: search for the parameter values that make simulated water
   temperature match your observations best (§8).
2. **Simulate**: run the model with those parameters, on the calibration period
   and, if supplied, on a separate validation period (§9), or on new inputs such
   as a scenario (§13).
3. **Quantify uncertainty** (optional): find every parameter set consistent
   with the data and the typical size and persistence of the model's errors
   (§12), and use them to give ranges, probabilities that a limit was exceeded,
   and differences between scenarios (§13). Cross-validation checks whether such
   ranges hold in years the model was not calibrated on (§11).

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
- **Implausible values are reported**, not changed: a warning lists `T_air`
  outside −60 to 60 °C and `T_water` outside −2 to 50 °C. Such values usually
  mean a missing-value code other than blank or `-999` (for example `-99`), which
  would otherwise be used as a real temperature.

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
| `CRN` (default) | Crank–Nicolson (semi-implicit, 2nd order) | stable for any B ≥ 0 |
| `EXP` | exponential / integrating factor | stable for any B ≥ 0 |
| `RK4` | Runge–Kutta 4th order | only while B < 2.785 |
| `RK2` | Heun (Runge–Kutta 2nd order) | only while B < 2.0 |
| `EUL` | explicit Euler, Fortran variant (inputs of the next day) | only while B < 2.0 |

`B` is how fast water temperature relaxes (per day); for version 8,
`B = (a3 + a8·θ) / θ^a4`. A negative B is physically impossible (water
temperature would move away from equilibrium); see §7 for how it can arise and
the warning about it. Because B depends on discharge, the explicit methods
(RK4, RK2, EUL) can become unstable on flows different from calibration and then
give wrong numbers without any error. `CRN` is therefore the default and is the
method recommended by the original authors. `RK4`, `RK2` and `EUL` reproduce the
Fortran exactly and are mainly useful for that purpose.

After every step, water temperature is not allowed below `Tice_cover` (default
0 °C), as in the Fortran.

With a one-day step, no scheme follows the equation exactly when water
temperature responds within a day (large B). Compared with a fine-step
solution of the same equation, CRN differs by 0.04–0.10 °C RMS on the Swiss
rivers, and EUL by up to about 1 °C even where stable (validation V6). This is
not an error in the predictions, because the parameters are calibrated with
the scheme and absorb its behaviour. It does mean that **parameters belong to
the scheme they were calibrated with**: the published parameters are CRN
parameters. Calibrating with EXP instead of CRN changed validation RMSE by less
than 0.03 °C.

## 7. Measuring the fit

**Scored days.** A day is scored if it has an observed water temperature and is
not in the warm-up year (and, in gap-tolerant mode, not in the unscored start of
a segment, §10).

**Time resolution.** With `time_resolution: "1d"`, each scored day is compared
directly. With `"Nw"` (N weeks) the record is cut into consecutive blocks of N×7
days starting on the first day; with `"1m"` into calendar months. A block is used
only if the fraction of its days with a scored observation is at least `prc`
(above 0 and at most 1; default 1.0, i.e. every day). As in the Fortran, a last,
incomplete month counts only its days in the record, while a last, incomplete
block of weeks is compared with the full N×7 days. The block's observed value is
the mean of those observations, and the simulated value is the mean of the
simulation **on the same days**.

Weekly or monthly means cannot see what happens from one day to the next. A
parameter set whose daily simulation zigzags (in air2stream, a negative
relaxation rate a3 makes the simulation swing between the 0 °C floor and high
values) can then score as well as the true one. With the authors' bounds,
which allow a negative a2 and a3, 9 of 30 weekly-scored calibrations of version
5 on synthetic data ended on such a set: daily errors of about 11 °C, weekly
means that fit (validation V4, case J). Physically, water warms with the air
and relaxes towards equilibrium, so a2 and a3 should be at least 0; with those
bounds no calibration ended there (case K). After every calibration and
forward run pyair2stream warns if the relaxation rate B (§6) is negative on
any day, or if successive daily changes of the simulation are correlated below
−0.5 (sensible fits give about +0.5).

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

**Mean error by month and season** (`bias_by_month_*.csv` and `.png`, for the
calibration and validation periods; `cv_bias_by_month.*` for the held-out years
of a cross-validation). A model can score well over the year and still be too
warm in summer and too cool in spring. For each calendar month, each season
(December counted with January and February of the same year) and the whole
year, the daily errors (simulated − measured) are first averaged within each
year; a month counts in a year if it has at least 10 days with both values (a
season 30, a year 120). The bias is the mean of these yearly values, with the
95% interval mean ± t₀.₉₇₅,ₙ₋₁ · sd/√n over the n years (none with fewer than
two). Days are not used as independent values because errors persist from day
to day and can last a whole season, so an interval from daily values would be
far too narrow. An interval that excludes zero means the model is consistently
biased in that month or season.

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
3. For each fold: its water-temperature observations are hidden; `Qmedia`
   (unless set with `Qmedia:`) and, in gap-tolerant mode, the day-of-year
   climatology are recomputed without the fold; the model is calibrated on the rest; the full record is simulated; and
   NSE, KGE and RMSE are computed on the hidden days only (daily values; in
   gap-tolerant mode not on the unscored start of a segment, §10).
   In gap-tolerant mode the fold's air temperature and discharge are also hidden
   during calibration, so the fold becomes a gap. Why the inputs are otherwise
   kept is explained below the list.
4. `cv_results.csv` lists each fold's scores and parameters, plus the mean and
   standard deviation across folds and "pooled" scores over all held-out days;
   `cv_bias_by_month.*` gives the mean error by month and season over the
   held-out days (§7).
5. **Check of yearly statistics** (`cv_yearly_statistics.csv` and
   `cv_yearly_statistics_summary.csv`). For each held-out year, 1,000 series are
   made from the fold's simulation plus random error from the fold's own error
   model (σ and ρ estimated on its training years, as in §12). The highest daily
   mean, the highest 7-day moving mean and the number of days above `threshold`
   (default: the 90th percentile of the measured temperatures) are computed in
   each series and in the measurements, over the days that were measured. A
   year counts if at least 80% of `season_months` (default: the four warmest
   months) was measured. For each year and statistic the output gives the
   predicted percentiles, the share of series below the measured value (the
   probability integral transform, PIT, which is uniform between 0 and 1 if the
   predictions are right; Gneiting et al., 2007) and the **deviation**: measured
   minus predicted median. The summary gives the share of years inside the 50%
   and 90% ranges, the shares expected by chance (95% binomial range), and the
   mean deviation with its 95% confidence interval. Parameter uncertainty is not
   included (each fold has one parameter set), so these ranges are slightly
   narrower than a FORWARD run's (§13). Set `threshold` and `season_months` to
   the limit and season in question: the defaults are chosen from all measured
   temperatures, held-out years included. That changes no prediction, but a
   question fixed in advance is easier to defend.
6. **Coverage at each level** (`cv_interval_coverage.csv`). From the same
   simulations, the share of measured held-out days, and of 7-day moving means,
   inside the central 50%, 80%, 90% and 95% ranges and at `prediction_interval`.
   It shows whether intervals at the level you report held at your site.

Every range in items 5 and 6 is reported at `uncertainty_options.prediction_interval`
(default 90%), and coverage at 50%, 80%, 90% and 95% as well.

**What is hidden, and why.** A fold hides the held-out year's measured water
temperatures, the quantity predicted, and keeps its air temperature and
discharge, from which the year is then predicted. This is the standard design.
Cross-validation hides the responses of the held-out cases and predicts them
from their own predictors (Stone, 1974; Hastie et al., 2009, §7.10), and a
hydrological split-sample test simulates the test period from its measured
inputs (Klemeš, 1986; Coron et al., 2012). Using the inputs is not leakage,
which is information about the target that would not be available when
predicting (Kaufman et al., 2012): a prediction always has its inputs. The
model never uses measured water temperature, so hiding it removes all of the
year's information about the answer, and every step of the prediction that
uses it is repeated inside each fold (calibration, σ and ρ, and in gap-tolerant
mode the day-of-year climatology). During calibration the model runs through the held-out year with
its inputs, which shape only the first days to weeks of the next year.

Validation V12 tested the alternatives on 96 held-out river-years. Hiding the
inputs during calibration too (in gap-tolerant mode, where the year then becomes
a gap) changed no year's RMSE by more than 0.034 °C against the same mode with
the inputs kept (another optimizer seed alone: 0.017 °C), and a 60-day
buffer of unused days on each side of the year (h-block cross-validation;
Burman et al., 1994) by no more than 0.041 °C; neither changed the coverage of
the 90% interval or the mean error of the yearly statistics measurably.

**What it does not test.**

- *Errors in the inputs.* Cross-validation uses measured air temperature and
  discharge, so it measures the model's skill given correct inputs, as the
  "perfect predictor" experiments of climate downscaling do (Gutiérrez et al.,
  2019). Errors in projected, transferred or scenario inputs must be assessed
  separately (§16).
- *Forecasting a changing future.* Each fold is calibrated on later as well as
  earlier years, as is usual in hydrology (Coron et al., 2012), but where a
  record changes over time only a test that keeps time order is fully honest
  (Arlot and Celisse, 2010, §8.3). In V12, calibrating on earlier years only
  predicted as accurately (RMSE 0.787 against 0.786 °C on the same 78 years),
  but its 90% intervals held on 87.1% of days against 89.6%, because the error
  size estimated from earlier years only was smaller. V5 (the later years of
  each record) and V10 (warmer years) are the forward tests: quote them, as
  well as the cross-validation, for predictions of future years.

Large variation of the parameters between folds means they are poorly determined
by the data (equifinality). The spread between folds (`std`) is not a confidence
interval: the folds share most of their data, so it understates the uncertainty
(in validation V4 it contained the true values only 35–56% of the time).

`cv_results.csv` therefore also gives **jackknife intervals** for the parameters.
With θᵢ the parameters fitted without block i (m folds), θ̄ their mean, and n the
number of blocks in the whole record (years, or groups of `n_years_per_fold`):

  SE² = (n − 1)/m · Σᵢ (θᵢ − θ̄)²,  interval = θ̄ ± t_(1+L)/2,m−1 · SE,

with L the level `uncertainty_options.parameter_interval` (default 0.90; the
rows are named `jackknife_90_lower` and so on).

When every block is held out (m = n) this is the standard delete-one-block
jackknife; the first years are never held out, so the sum over n blocks is
estimated as n/m times the sum over the m folds. In validation V4 these 90%
intervals contained the true parameters 83–94% of the time for every version.
On the same data, the DE-MCMC parameter intervals (§12, default likelihood)
contained them 97% of the time for version 5 and 91% for version 8. For
versions 4, 7 and 8 set `Qmedia`
explicitly, so that every fold uses the same discharge scaling.

The jackknife intervals inherit the optimizer's randomness. Where parameters
trade off (equifinality), a fold can end on a distant parameter set with almost
the same fit, and that one fold widens the interval. In V12, another optimizer
seed alone changed version 8's jackknife standard errors by a factor of 0.35 to
1.55 (median over its parameters, by river). Read the jackknife intervals of
poorly determined parameters as indicative, and check them with a second
`random_seed`; predictions are not affected.

A cross-validation run does not also produce a single final calibration.

## 12. Parameter and prediction uncertainty (DE-MCMC)

`run_mode: DE-MCMC` first calibrates with DE (§8), then estimates how uncertain
the parameters and predictions are, using Markov chain Monte Carlo (MCMC):

1. **Prior.** Every parameter value inside the bounds is considered equally
   plausible beforehand; values outside are impossible. Results therefore depend
   on the bounds.
2. **Likelihood** (how well a parameter set explains the data), computed on the
   same scored values as the objective (§7), assuming normally distributed errors
   of constant size (the size is estimated, not supplied). A model's daily errors
   persist (AR(1) correlation ρ, estimated once from the DE fit's daily
   residuals, limited to 0–0.99; see "How ρ is estimated" below), so n days of
   errors carry the information of fewer independent ones.
   - `noise_model: "ar1"` with `likelihood: "least_squares"` (the default):
     log L = −(n_eff/2)·ln(SSE/n), with n_eff = n·(1−ρ)/(1+ρ), the usual
     effective number of independent observations for AR(1) errors. Its best
     value is the least-squares fit, the criterion the original authors
     calibrated with, and its spread is widened to allow for the autocorrelation.
   - `noise_model: "ar1"` with `likelihood: "exact"`: the exact AR(1) likelihood.
     Errors are converted to independent "innovations" (e₀·√(1−ρ²); eₜ − ρ·eₜ₋₁)
     within each unbroken run of scored days, and
     log L = −(n/2)·ln(SSE_innovations/n) + (runs/2)·ln(1−ρ²). This weighs
     day-to-day changes in the error far more than its overall level. On real
     rivers, where the model is never exactly right, it moved the parameters away
     from the best fit, to slightly worse and cooler predictions (see below), so
     it is not the default.
   - `noise_model: "iid"` treats every day's error as independent:
     log L = −(n/2)·ln(SSE/n). With autocorrelated errors this understates
     parameter uncertainty, and makes intervals for multi-day quantities far too
     narrow.
   With weekly or monthly scoring (§7) each scored value is the mean of a block
   of m days (m = 7N for `"Nw"`, 30 for `"1m"`), and block means are much less
   correlated from one block to the next than days are. The least-squares
   likelihood then uses n_eff = n / [1 + 2·r_b/(1 − ρ^m)], with r_b the
   correlation between the means of adjacent blocks of AR(1) errors (the
   formula under "How ρ is estimated", with m days in place of 7); for m = 1
   this is n·(1−ρ)/(1+ρ). The
   daily n_eff applied to block means would make parameter intervals 1.6–4.4
   times too wide for ρ = 0.5–0.95. The exact AR(1) likelihood needs consecutive
   scored days; with weekly or monthly scoring it treats the block errors as
   independent and warns, since its intervals are then too narrow if errors
   persist from block to block. Each likelihood replaces the error size by
   its best estimate; this gives the same result as treating the error size as
   unknown with the standard non-informative prior (∝ 1/σ) and averaging over it.
3. **Sampling.** `mcmc_walkers` (default 32) chains ("walkers") are started
   close to the DE optimum (spread 0.1% of each parameter's bound range,
   reflected back inside the bounds) and advanced together by `emcee`'s
   ensemble sampler with the differential-evolution move (ter Braak, 2006): each
   proposal moves a walker along the difference between two others, which suits
   the strongly correlated parameters of air2stream.
4. **Run length and convergence.** The sampler runs in blocks of 1,000 steps
   (at least 2,000). The first max(30% of steps, 5× the longest autocorrelation
   time) steps are discarded as burn-in (or `burnin_fraction` of them). It
   stops when the number of steps is at least 50 times the longest
   autocorrelation time (estimated after burn-in) and split-R̂ is below 1.01 for
   every parameter, or when `mcmc_steps` (default 20,000) is reached. After
   burn-in, every k-th step is kept, with k half the shortest autocorrelation
   time. If the
   chain has not converged, the run stops with an error and writes its
   diagnostics (`strict_convergence: true`, the default); with `false` it
   continues and every output is marked as not converged.
5. **Prediction interval.** Up to 1000 parameter sets are drawn from the chain.
   For each, the model is run and random error is added to every day: normally
   distributed with standard deviation equal to that parameter set's daily
   root-mean-square residual (`iid`), or an AR(1) series with the same standard
   deviation and ρ (`ar1`). The `prediction_interval` (default 90%; any level
   above 0 and below 100) is the band
   between the matching lower and upper percentiles of these simulations on each
   day. The program then reports the **coverage**: the share of observed days
   inside the band (it should be close to the nominal percentage). This band and
   its coverage are for the calibration period; for any other period, including
   validation, use a FORWARD run from the chain (§13).
6. Any drawn parameter set whose simulation diverges (§15) is excluded and
   reported; if more than `max_divergent_fraction` (default 10%) diverge, the run
   stops.

On any single day, iid and AR(1) errors have the same spread, so the daily band
is about the same width under both. They differ for multi-day quantities (weekly
means, days in a row above a threshold): use the raw ensemble for those
(`save_ensemble: true`, and `pyair2stream.scenario`), never averages of the
daily percentiles.

**How ρ is estimated** (`uncertainty_options.rho_timescale`).

- `"weekly"` (the default). Every 7-day window of scored, observed days is paired
  with the complete 7-day window that starts 7 days later, and the correlation r
  of their mean errors is measured. ρ is the AR(1) value whose 7-day means have
  that correlation: r = Σ_{d=1..13} (7 − |d − 7|)·ρ^d / (7 + 2·Σ_{k=1..6} (7 − k)·ρ^k).
  The larger of this and the lag-1 correlation of consecutive days is used. It
  needs at least 140 pairs (about 21 complete weeks); otherwise the lag-1 value
  is used, with a warning.
- `"daily"`: the lag-1 correlation of consecutive scored days.

Both need daily observations. A window counts only if all 7 days are scored,
so with many missing days (for example a measurement every other day) the
lag-1 value is used. With fewer than 30 pairs of consecutive scored days, ρ is
0, with a warning, and bands for multi-day quantities are then too narrow.

ρ sets the noise added to predictions and n_eff in the least-squares likelihood.
The exact likelihood always uses the lag-1 correlation, because removing the
correlation of consecutive days (eₜ − ρ·eₜ₋₁) is a day-scale operation. The
chain's `_meta.json` records `rho`, `rho_timescale`, `rho_likelihood`,
`scoring_block_days` and `likelihood_variance_factor` (n/n_eff). FORWARD runs
use the chain's ρ, and say so when it was estimated at another time scale than
their own `rho_timescale` (chains from version 0.4.1 or earlier used
consecutive days). If ρ reaches its limit of 0.99, a warning says so: errors
that persist for months usually mean a systematic error, such as a bias in one
season (§7, mean error by month and season).

**Why the weekly scale (theory).** Real model errors have two memories at once:
on the Swiss rivers a fast part that fades in about two days (57–76% of the
error variance) and a slow part lasting three to five weeks (24–43%). An AR(1)
process has only one. The choice is which of them it should reproduce.

1. *Predictions.* Compliance quantities are built from many days (7-day means,
   runs of warm days, summer peaks). The uncertainty of an m-day mean depends on
   every error correlation up to lag m: Var = σ²/m · [1 + 2·Σ_{k<m} (1 − k/m)·r_k]
   (Bayley and Hammersley, 1946). Matched to consecutive days, an AR(1) ignores
   the slow part. On the Swiss rivers it understated the variance of 30–90-day
   mean errors by a factor of two to three. Matched to week-to-week persistence,
   the variance it implies was 1.2–1.4 times the measured one for 7–14-day means
   and about right (0.83–1.3) at 60–90 days. A noise model that understates
   low-frequency variability gives intervals that are too narrow (Poppick et al.,
   2017). In hydrology, ignoring error persistence underestimates the
   uncertainty of aggregated quantities (Evin et al., 2014), and reliable
   intervals at several time scales need errors at several time scales
   (McInerney et al., 2020).
2. *Parameters.* With correlated errors, the sampling covariance of least-squares
   estimates is σ²(JᵀJ)⁻¹ JᵀRJ (JᵀJ)⁻¹ (the "sandwich"; R is the correlation
   matrix of the errors, J the sensitivity of the simulated temperature to each
   parameter on each day). The n_eff likelihood widens every parameter's
   variance by the same factor, (1 + ρ)/(1 − ρ). Each parameter needs a factor
   of about 1 + 2·Σ_k r_J(k)·r_e(k), with r_J the autocorrelation of its
   sensitivity and r_e that of the errors.
   That is close to the full allowance for slow errors when the parameter's
   effect varies slowly (the constant a1, the seasonal amplitude a6 and timing
   a7). It is much smaller when the effect varies from day to day (the air
   temperature and relaxation coefficients a2 and a3). With one factor for all
   parameters, the factor must be large enough for the slowest parameter.
   Validation V4 computes this formula and compares it with the measured spread
   of the estimates; they agree closely. With fast + slow errors, the daily ρ
   made a7's interval about 1.5 times too narrow (measured 1.54; the formula
   predicts 1.59), and the weekly ρ gave it the right
   width. The cost is that intervals of the fast-varying parameters are wider
   than necessary, by a factor of about two to three in V4. That happens with
   either ρ: it is a property of a single effective sample size. The two errors
   are not equal: an interval that is too wide errs on the side of caution, one
   that is too narrow claims more than the data show.
3. *The larger of the two estimates.* The weekly option is never less
   persistent than consecutive days show. The week-to-week estimate alone is
   imprecise when errors are only weakly correlated. Taking the larger then errs
   towards wider intervals, and changes nothing for AR(1) errors (V4, V9).

This is a documented, validated approximation, not a published method by name.
The principled refinements are an error model with a fast and a slow part, and
parameter-specific (sandwich) widths for the posterior (Ribatet et al., 2012).
Neither is implemented. Use `rho_timescale: "daily"` to reproduce results made
with the earlier default, or to check how much a conclusion depends on the
choice.

**What the validation shows** ([validation/REPORT.md](../validation/REPORT.md)).
On synthetic data from a known truth, 90% prediction intervals contained about
90% of new observations for versions 5 and 8 with every likelihood, and with the
default least-squares likelihood the 90% parameter intervals contained the true
values at least 90% of the time for both versions (V4). The exact AR(1)
likelihood's intervals for version 8 contained the truth only about 75% of the
time: its parameters trade off against each other and that posterior is far from
normal. The sampler was cross-checked against emcee's stretch move. On three real
rivers, with the default settings, 90% intervals contained 85–89.6% of daily
values in years not used for calibration (84.5–89.6% across all settings
tested), and for 7-day means 89–94%, against 83–88% with `rho_timescale:
"daily"` and 39–62% with `iid` (V5). On the same rivers, the parameters published by Piccolroaz et al.
(2016) lay inside these intervals for every converged run of versions 3–5. For
versions 7 and 8 on the Mentue and version 8 on the Rhône several lay outside:
there many parameter combinations fit almost equally well, and the published
set is not where a least-squares calibration on these data lands (V2).

**At other levels.** On synthetic data the intervals held at every level tested
(50%, 80%, 90%, 95% and 99%: mean coverage within 1.4 points of the level, V4).
On the Swiss rivers, pooled over 48 years held out by cross-validation per
version, daily intervals held from 50% to 95% (95% intervals: 94.4–94.5% of
days), but 99% intervals held only 98.0–98.2%: the model's real errors have
heavier tails than the normal distribution assumed (V11). In the later
validation years of V5, whose errors were larger than in calibration, daily
intervals were narrower than stated, more so at high levels (90%: 85–90%; 95%:
91–95%; 99%: 95–99.6%). 7-day means held at every level (V11: 96–97% at 95%,
98.6–99.2% at 99%).

**Where the chain is centred.** With the default least-squares likelihood the
chain is centred on the DE best fit: in V5 the centre of the band stayed within
0.01 °C of the best fit's. With `likelihood: "exact"` it can be centred
elsewhere, also for versions whose parameters do not trade off: in V5 it moved
by up to 0.11 °C (cooler, Dischmabach version 5). For a limit on warm water, a
cooler band would understate the chance of exceedance.

**Outputs:** `MCMC_chain_*.csv` (post-burn-in samples), `MCMC_chain_*_meta.json`
(σ, ρ, diagnostics, coverage, excluded draws), `MCMC_envelopes_*.csv`, and the
parameter summary `parameter_significance_*.csv` (posterior mean, standard
deviation, central credible interval at `parameter_interval`, default 90%, and
whether zero lies outside the central 95%: a test at the usual 5% level,
whatever the interval's level).
Excluding zero only means something for parameters where zero means "no
effect" (`a2`, `a4`, `a5`, `a6`, `a8`). It says nothing about `a1`, `a3` or the
seasonal timing `a7`.

## 13. Forward runs and scenario comparisons

`run_mode: FORWARD` runs the model with known parameters on any input file,
without calibrating. It requires the calibration `Qmedia` (§4). Given
`paths.calibration_metadata` (the calibration's `calibration_metadata.json`), it
takes `Qmedia` and the calibrated parameters from it (unless
`parameters_forward` is given), refuses a different model version or integrator,
and warns if more than 1% of days have θ outside the range seen in calibration
(extrapolation). If the file contains water-temperature observations, the fit is
reported as in §7.

**Prediction intervals** (`forward_options.enable_prediction_intervals: true`)
reuse a DE-MCMC chain (`mcmc_chain_path`): `n_samples` (default 1000) parameter
sets are drawn, each is run, and error is added with standard deviation σ =
`forward_options.residual_sigma`, or else the calibration's daily residual
standard deviation stored in the chain's `_meta.json`. For `ar1`, ρ is taken from
`ar1_rho`, else from the chain's `_meta.json`, else from this run's own residuals.
With the `_meta.json` that DE-MCMC writes next to the chain, the interval
therefore does not depend on the observations it is checked against. The run
refuses a chain whose `_meta.json` records another model version, integrator or
`Qmedia` (by more than 0.1%) than its own. Coverage is reported if observations exist. The noise
model, σ and ρ used are recorded in the run's
`Forward_Prediction_Ensemble_*_meta.json`. σ and ρ are used at their estimated
values; their own uncertainty is not added.

**Probability that a limit was exceeded.** With `save_ensemble: true` every
simulated series is kept (`.npz`): one per parameter draw, each with its own
error series. A probability is computed in three steps:

1. In each series, compute the quantity the limit is defined on.
   `scenario.year_statistics` gives each year's highest daily mean, highest
   7-day moving mean (the day and the six before it) and number of days above a
   threshold, defined exactly as in the cross-validation check (§11).
   `scenario.aggregate` gives means (or sums, maxima) over consecutive fixed
   periods, and `scenario.exceedance` counts days above a threshold, optionally
   only in runs of at least k consecutive days (days not simulated count as not
   above).
2. The probability of exceedance is the share of series in which that quantity
   exceeds the limit. With 1,000 series it carries a sampling error of at most
   ±0.03 (95%).
3. A range for the quantity (for example 90%) is given by the matching
   percentiles across the series.

Never compute such a quantity from the daily band: the upper edge of the daily
band is not the upper edge of a weekly mean or a yearly peak.

**Checking and correcting yearly statistics.** The model's error on the hottest
days is not always its typical error. In years not used for calibration, the
simulated yearly peak was on average 0.6–0.8 °C too high on the Mentue with
version 8, and with version 5 up to 0.9 °C too high on the Rhône and 0.5 °C too
low on the Dischmabach (V11). Random error of the typical size cannot remove
such a bias, so the ranges of yearly statistics then miss the measured value
more often than they state. Ranges for daily values and 7-day means are much
less affected (V5, V10). For a yearly statistic:

1. Run a cross-validation of the calibration years (§11) with the `threshold`
   and season of the question. `cv_yearly_statistics_summary.csv` shows how
   often the ranges held, and the mean deviation (measured minus predicted
   median) with its 95% interval; an interval that excludes zero is a bias.
2. Correct the statistic of every simulated series with
   `scenario.correct_statistic(values, deviations)`, using that statistic's
   deviations from `cv_yearly_statistics.csv`. Each value is shifted by the
   mean deviation plus a random draw of its uncertainty (its standard error
   times a Student t variable with n − 1 degrees of freedom, for n held-out
   years; at least 3). This is a bias correction estimated out of sample, as in
   model output statistics (Glahn and Lowry, 1972). With few years the shift is
   uncertain, and the corrected range is wider.
3. Compute the probability and range from the corrected values, and report the
   check with them.

The correction assumes that the model's average bias in the statistic is the
same in the years predicted as in the years held out. It is applied always, not
only when the check finds a bias, so the procedure does not depend on the
result.

**What the validation shows** (V9, V11). On synthetic data, where the model and
its error model are exactly right, the stated probabilities for yearly
statistics came true as often as stated, and the 90% ranges contained the
measured value 86–94% of the time (V9 A). On the three Swiss rivers, over 48
held-out years per version (V11), the uncorrected 90% ranges held in 79–92% of
years for version 8 and 73–81% for version 5. After the correction they held
in 85–94% for both versions, within the range expected by chance, and the
measured value sat on average at the middle of the simulations (mean PIT
0.48–0.50, against 0.31–0.52 before). At other levels the corrected ranges held
in 75–85% of years at 80%, 92–100% at 95% and 100% at 99%, all within the range
expected by chance. The highest daily mean remained at the
low end (85%): single-day peaks are the hardest statistic to predict. Applied
as recommended to genuinely later years (cross-validation of the calibration
years, then correction of the FORWARD simulations of the validation years; V9
C, 15 river-years), the correction made the probabilities closer to what
happened for 5 of 6 version and statistic pairs (version 8's Brier skill score
against the share of past years rose from 0.08–0.42 to 0.33–0.50), but version
8's 90% ranges for the highest daily mean still held in only 11 of 15
river-years. Version 5 predicted almost the same peak every year on the Rhône,
where discharge drives summer temperature; the correction removes its bias, but
not its inability to follow the years.

**Comparing two scenarios** (for example observed versus naturalised flow): run
FORWARD once per scenario from the same chain with `save_ensemble: true`, and
for the second run set `forward_options.reuse_sample_indices_from` to the first
run's `Forward_Prediction_Ensemble_*_meta.json`, so both use exactly the same
parameter sets. `scenario.paired_difference_from_files()` then checks this before
computing the difference draw by draw, which gives an uncertainty band for the
*difference* itself. The random error added to a draw is generated from a seed
fixed by the chain's content and the draw's row in it, so both runs add the same
error to the same draw on the same day and it cancels in the difference: the
band is the parameter uncertainty of the effect. This assumes the model's error
on a given day would be the same under both scenarios. The validation checks
that the paired difference equals the exact effect where it is known (V8).

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
| `T_air` or `T_water` outside a plausible range (§2) | loading data | warning |
| Invalid version, run mode, integrator, objective, time resolution, `prc`, bounds | loading config | error |
| Stability of the chosen integrator (B vs. limit, §6) | before each user-facing simulation | warning; error if >10% of days exceed it |
| Simulated temperature not finite or above `max_plausible_twat` (60 °C) | after each user-facing simulation | error |
| Negative relaxation rate B, or a daily simulation that zigzags (§7) | after calibration, before DE-MCMC sampling, FORWARD runs | warning |
| ρ at its limit of 0.99; exact likelihood with weekly or monthly scoring (§12) | DE-MCMC | warning |
| Recomputed objective matches the calibration result | after calibration | error |
| Discharge outside the calibrated range | FORWARD runs | warning |
| Segment warm-up too short | gap-tolerant runs | warning |
| MCMC convergence and diverging draws | DE-MCMC, FORWARD intervals | warning / error |
| Chain fitted with another model version, integrator or `Qmedia` (§13) | FORWARD intervals | error |

## 16. Limitations and good practice

- **Daily means only.** Inputs and outputs are daily means. The model cannot
  predict daily maxima, minima or sub-daily peaks; a criterion defined on those
  needs a separate, justified relationship.
- **Fitted, site-specific model.** Parameters describe one site and period, and
  assume the river behaves the same way in the period predicted (no new dam,
  effluent, abstraction pattern or loss of shading in between). Predictions for
  conditions outside the calibration range (air temperature, discharge) are
  extrapolations; check the θ-range warning. In validation V10, calibrating on
  the coolest (or highest-flow) third of each Swiss river's years and predicting
  the warmest (or lowest-flow) third cost at most 0.07 °C of RMSE compared with
  calibrating on the middle third, but a single extreme period can still be
  missed: in the Mentue's 2003 heatwave, the model calibrated on the three
  coolest summers put August's highest daily temperature 1.9 °C above the
  measured one.
- **Inputs are taken as exact.** The uncertainty covers the parameters and the
  model's own error, not errors in the air temperature or discharge supplied.
  Calibration, validation and cross-validation use measured inputs; a
  prediction from projected, transferred (another station) or scenario inputs
  carries their errors too. Assess those separately, for example by running
  several input series.
- **Choose a version that suits the river.** Where discharge drives the summer
  temperature (the Rhône here), versions without a discharge term (3–5) did
  hardly better than simple alternatives, and version 5's probabilities for
  yearly peaks were no better than going by past years (V5, V9, V10); on the
  hottest days its 90% intervals held on only 83–84% of days, against 91% for
  version 8 (V14). Compare
  versions on validation years or by cross-validation (§11).
- **Yearly statistics need the cross-validation check.** Their computation is
  right (V9, synthetic data), but the model can be biased on the hottest days,
  and uncorrected 90% ranges for yearly statistics held in only 73–92% of
  held-out years on the Swiss rivers. Check them by cross-validation and correct
  them (§13); corrected, they held in 85–94% (V11). Ranges for the highest
  daily mean remained the least reliable (85% in V11, 11 of 15 later years in
  V9 C). The check needs years: with fewer than about 5 held-out years it can
  say little.
- **Different parameter sets can fit equally well** (equifinality), especially
  for versions 7 and 8. Inspect the dotty plots; a parameter at a bound suggests
  the bounds are too narrow. Prefer the simplest version that validates well.
- **Prediction intervals rest on assumptions**: normally distributed errors of
  constant size, independent or AR(1). Check the residual plots (histogram,
  Q-Q, autocorrelation) and the reported coverage, ideally on validation data.
  Errors are often larger in some seasons than others, so also check coverage
  in the season a limit applies to (on the Swiss rivers, summer coverage of 90%
  intervals was 84–97%, validation V5).
- **Converged sampling.** MCMC results are only valid once converged; by
  default the run stops otherwise (§12).
- **Multi-day quantities** (7-day means, runs of days above a limit) need
  `noise_model: "ar1"` and must be computed from the saved simulations. Even
  then their intervals were somewhat narrow on real rivers (§12).
- **Intervals for new years are slightly optimistic**: σ is estimated on the
  calibration years, and errors are usually somewhat larger in other years.
- **Choose the level knowing its record.** Any level can be set
  (`prediction_interval`, `parameter_interval`). On the Swiss rivers daily
  intervals held from 50% to 95%, but 99% daily intervals missed about twice
  as many days as stated (98.0–98.2% held, V11), and in years unlike the
  calibration years 95% intervals held 91–95% and 99% intervals 95–99.6% (V5).
  Before reporting a 95% or 99% daily interval, check it at your site with
  `cv_interval_coverage.csv` (§11) and on validation years.
- **Scenario differences** assume the model's error on a given day would be the
  same under both scenarios, so their band shows parameter uncertainty only
  (§13).
- **Metric caveats.** KGE's ratio of means is unstable when mean water
  temperature is near 0 °C. AIC/BIC assume independent errors and favour more
  complex versions when errors are autocorrelated. Gap-tolerant scores are not
  comparable to complete-record scores.
- **Record what you ran**: the config file, `random_seed`, the pyair2stream
  commit (`git rev-parse HEAD`), the Python package versions (`pip freeze`), and
  the output files `calibration_metadata.json` and any `_meta.json`.

**Evidence that the implementation is correct** is in
[validation/REPORT.md](../validation/REPORT.md), produced by
`validation/run_all.py`: identical results to the original Fortran on real
inputs for every version and Fortran integrator (to 5×10⁻⁶ °C, the precision of
its printed output), and identical calibration scores and weekly and monthly
averages, with and without gaps; all 30 published RMSE values of Piccolroaz et al. (2016)
reproduced to within 0.001 °C, and their parameters recovered by recalibration
except where the parameters trade off (versions 7 and 8 on two rivers, where
recalibration fits slightly better with different parameters and the same
predictions, and where the original program itself returns different
parameters on every run); recovery of a known truth; calibrated intervals
on synthetic data; out-of-sample performance on three real rivers; numerical
accuracy; gaps; exact answers from the workflow and scenario tools;
probabilities of exceeding a limit, on synthetic data and real rivers (V9);
predictions for warmer and lower-flow years than those calibrated on (V10); and
the cross-validated check and correction of yearly statistics (V11); and that
the cross-validation's design does not flatter the model (V12); the 30
published errors of Toffolon and Piccolroaz (2015), computed with RK4, also
reproduced, and their parameters returned by calibration with RK4 (V13); and
prediction intervals on the hottest days (V14); and the simulations published
by an independent group for 23 rivers in British Columbia, reproduced day by day
(V15). V5, V9, V10 and V14 do not pass all their criteria; the report says where
and why. The errors found in the published studies are documented in
[PUBLISHED_RESULTS.md](PUBLISHED_RESULTS.md). The test
suite (`pytest tests/`) also compares against the Fortran and checks each
safeguard above.

## 17. Differences from the Fortran original

These are deliberate; each is covered by tests.

- **Default integrator is `CRN`** (the choice suggested in the Fortran's example input);
  RK4/RK2/EUL are unchanged and match the Fortran exactly.
- **Default optimiser is DE + L-BFGS-B**; PSO and LATHYP are kept.
- **PSO**: the first global best is the best initial particle (the Fortran
  looks it up in a score array that is still all zero, so it always takes the
  first particle); failed (NaN) evaluations are ignored; and the early-stop
  test uses a tolerance of 1e-4 (the Fortran's test can never be met, so it
  always runs to the end).
- **`Qmedia`** also excludes discharge ≤ 0, and is fixed at the calibration value
  for validation and FORWARD runs instead of being recomputed.
- **Seasonal phase** is computed from each row's real date (equivalent for
  records starting on 1 January); FORWARD runs may start on any date, with the
  warm-up year taking the phase of the rows it copies. The Fortran takes it
  from the row number and does not check that a record starts on 1 January: a
  published calibration whose record started on 1 November was run with its
  seasonal cycle ten months out of phase (V15;
  [PUBLISHED_RESULTS.md](PUBLISHED_RESULTS.md)).
- **Checks added**: missing values, implausible values, non-positive discharge,
  unused parameters, integrator stability and divergence (§15); the Fortran
  would run on silently.
- **Time resolution**: an out-of-range index in the Fortran's weekly aggregation
  of the last, partial block is avoided; monthly aggregation accepts only `1m`
  (the Fortran ignores the number of months); a week or month without
  observations is never scored (with `prc` 0 the Fortran divides by zero).
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
- ter Braak, C. J. F. (2006). A Markov chain Monte Carlo version of the genetic
  algorithm Differential Evolution: easy Bayesian computing for real parameter
  spaces. *Statistics and Computing*, 16, 239–249.
- Goodman, J. and Weare, J. (2010). Ensemble samplers with affine invariance.
  *Communications in Applied Mathematics and Computational Science*, 5, 65–80.
- Foreman-Mackey, D., Hogg, D. W., Lang, D. and Goodman, J. (2013). emcee: the
  MCMC hammer. *Publications of the Astronomical Society of the Pacific*, 125,
  306–312.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and
  Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd edn. CRC Press (split-R̂).
- Glahn, H. R. and Lowry, D. A. (1972). The use of model output statistics
  (MOS) in objective weather forecasting. *Journal of Applied Meteorology*, 11,
  1203–1211.
- Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007). Probabilistic
  forecasts, calibration and sharpness. *Journal of the Royal Statistical
  Society B*, 69, 243–268.
- Bayley, G. V. and Hammersley, J. M. (1946). The "effective" number of
  independent observations in an autocorrelated time series. *Supplement to the
  Journal of the Royal Statistical Society*, 8, 184–197.
- Poppick, A., Moyer, E. J. and Stein, M. L. (2017). Estimating trends in the
  global mean temperature record. *Advances in Statistical Climatology,
  Meteorology and Oceanography*, 3, 33–53.
- Evin, G., Thyer, M., Kavetski, D., McInerney, D. and Kuczera, G. (2014).
  Comparison of joint versus postprocessor approaches for hydrological
  uncertainty estimation accounting for error autocorrelation and
  heteroscedasticity. *Water Resources Research*, 50, 2350–2375.
- McInerney, D., Thyer, M., Kavetski, D., Laugesen, R., Tuteja, N. and
  Kuczera, G. (2020). Multi-temporal hydrological residual error modeling for
  seamless subseasonal streamflow forecasting. *Water Resources Research*, 56,
  e2019WR026979.
- Ribatet, M., Cooley, D. and Davison, A. C. (2012). Bayesian inference from
  composite likelihoods, with an application to spatial extremes. *Statistica
  Sinica*, 22, 813–845.
- Stone, M. (1974). Cross-validatory choice and assessment of statistical
  predictions. *Journal of the Royal Statistical Society B*, 36, 111–147.
- Klemeš, V. (1986). Operational testing of hydrological simulation models.
  *Hydrological Sciences Journal*, 31, 13–24.
- Burman, P., Chow, E. and Nolan, D. (1994). A cross-validatory method for
  dependent data. *Biometrika*, 81, 351–358.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of
  Statistical Learning*, 2nd edn. Springer.
- Arlot, S. and Celisse, A. (2010). A survey of cross-validation procedures for
  model selection. *Statistics Surveys*, 4, 40–79.
- Coron, L., Andréassian, V., Perrin, C., Lerat, J., Vaze, J., Bourqui, M. and
  Hendrickx, F. (2012). Crash testing hydrological models in contrasted climate
  conditions: an experiment on 216 Australian catchments. *Water Resources
  Research*, 48, W05552.
- Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). Leakage in data
  mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge
  Discovery from Data*, 6(4), 15.
- Gutiérrez, J. M., Maraun, D., Widmann, M. and others (2019). An
  intercomparison of a large ensemble of statistical downscaling methods over
  Europe: results from the VALUE perfect predictor cross-validation experiment.
  *International Journal of Climatology*, 39, 3750–3785.
- Callahan, L. and Moore, R. D. (2025). Evaluation of the hybrid air2stream
  model for simulating daily stream temperature during extreme summer heat wave
  and autumn drought conditions. *Hydrological Processes*, 39(1), e70033.
  Data: Moore, R. D. and Callahan, L. (2024), Zenodo,
  https://doi.org/10.5281/zenodo.14502248.
