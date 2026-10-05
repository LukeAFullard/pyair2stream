# pyair2stream

[![License: CC BY-SA 3.0](https://img.shields.io/badge/License-CC_BY--SA_3.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

`pyair2stream` predicts the **daily mean water temperature of a river** from daily
air temperature and (optionally) river discharge. It is a Python version of
**air2stream** (Toffolon and Piccolroaz, 2015), a widely used model that fits a
small physically based equation to your site's own measurements.

Typical uses: filling gaps in a water temperature record, checking what a river's
temperature would have been under different flows (for example with and without
water abstraction), projecting temperatures under future climate, and asking
whether a site is likely to meet a temperature limit, with an uncertainty band.

> This is a community port, not the official release by the model's authors. It
> reproduces the original Fortran results (see [Is it correct?](#is-it-correct))
> and adds features listed below.

## What it does

- **Calibrates** the model's 3–8 parameters to your observed water temperature
  (Differential Evolution by default; also Particle Swarm and Latin Hypercube).
- **Validates** it on a separate period you supply.
- **Quantifies uncertainty** in the parameters and gives prediction intervals
  (`DE-MCMC`), including for new scenarios.
- **Runs scenarios** with fixed parameters (`FORWARD`), and compares two
  scenarios with an uncertainty band on the difference.
- Handles **gaps** in air temperature or discharge (gap-tolerant mode),
  **cross-validation** by year (with confidence intervals for the parameters,
  and a check that predicted ranges of yearly statistics hold), and
  **sensitivity analysis**.
- Uses a **YAML config file and CSV files**, and writes CSV results and plots.

## Install

Requires Python 3.9 or newer.

```bash
git clone https://github.com/LukeAFullard/pyair2stream.git
cd pyair2stream
pip install .
```

## Quick start

Calibrate the model on a real Swiss river and test it on later years (under a
minute), from the repository's top folder:

```bash
pyair2stream --config examples/01_quickstart/config.yaml
```

Results and plots appear in `examples/01_quickstart/output/`; the
[example's README](examples/01_quickstart/README.md) explains them.

To use your own data:

1. Make a CSV with one row per day: `Date`, `T_air`, `T_water`, `Discharge`
   (see [Input data](#input-data)).
2. Write a `config.yaml` like the one below.
3. Run `pyair2stream --config config.yaml`.

```yaml
project_name: "my_river"
station_name: "Station_A"
version: 8                 # model version: 3, 4, 5, 7 or 8 (see below)
run_mode: "DE"             # calibrate with Differential Evolution
random_seed: 42            # makes results exactly repeatable

paths:
  input_data: "data/calibration.csv"
  validation_data: "data/validation.csv"   # optional
  output_dir: "output"

optimization:
  n_run: 100               # maximum generations
  n_particles: 10          # population = 10 x 8 parameters

parameter_bounds:          # search ranges for a1..a8 (original authors' ranges)
  min: [-5, -5, -5, -1, 0,  0,  0, -1]
  max: [15, 1.5, 5,  1, 20, 10, 1,  5]
```

Everything not set uses a sensible default (for example the stable `CRN`
integrator and the NSE objective). The [User Guide](USER_GUIDE.md#6-configuration-reference)
lists every option.

## Did a site meet a temperature limit?

The usual route, each step a worked example:

1. **Check the model fits your river**: calibrate on years with measured water
   temperature and check it predicts other years well ([example 01](examples/01_quickstart/README.md)).
2. **Measure its uncertainty** with `run_mode: "DE-MCMC"`, and check that its
   ranges hold on years it was not calibrated on ([example 02](examples/02_uncertainty/README.md)).
3. **Simulate the period in question** 1,000 times with `FORWARD` and
   `save_ensemble: true`. Work out the quantity your limit is defined on (for
   example the highest 7-day mean) in each simulation; the share above the limit
   is the probability that it was exceeded ([example 03](examples/03_compliance/README.md)).
4. **Check that quantity by cross-validation** of the calibration years, and
   correct it for the model's bias on the hottest days
   (`cv_yearly_statistics.csv`, `scenario.correct_statistic`; example 03).
5. **Report a probability with its range and the check**, not a yes or no,
   together with the checks in [User Guide §14](USER_GUIDE.md#14-checklist-for-results-that-support-a-decision).

The model gives daily **means**: a limit on daily maxima needs a separate,
justified step.

## Input data

| Column | Required | Notes |
|---|---|---|
| `Date` | yes | one row for every calendar day, e.g. `2020-01-31` |
| `T_air` | yes | daily mean air temperature (°C); no gaps unless `gap_tolerant: true` |
| `T_water` | yes | daily mean water temperature (°C); gaps allowed |
| `Discharge` | versions 4, 7, 8 | daily mean flow (any unit); no gaps, > 0 |

Leave missing values blank or write `-999`. Calibration and validation files must
start on 1 January and cover at least a year.

## Model versions

| Version | Parameters | Uses discharge | Seasonal term |
|:-:|:-:|:-:|:-:|
| 3 | 3 | no | no |
| 4 | 4 | yes | no |
| 5 | 5 | no | yes |
| 7 | 7 | yes | yes |
| 8 | 8 | yes | yes |

Start with version 8 if you have good discharge data (or 5 if not), and compare
with simpler versions: prefer the simplest one that performs well on the
validation period. The equation is given in [docs/METHODS.md](docs/METHODS.md#5-the-equation-and-model-versions).

## Outputs

| File | Contents |
|---|---|
| `1_*.out` | best parameters (line 1), calibration score (line 2), validation score (line 3) |
| `2_*.csv`, `3_*.csv` | daily observed and simulated water temperature, calibration and validation periods |
| `goodness_of_fit_*.csv` | N, NSE, R², RMSE, MAE, AIC, BIC for each period |
| `calibration_*.png`, `validation_*.png`, `full_simulation_*.png` | time-series plots with residuals |
| `predicted_vs_measured_*.png`, `residual_diagnostics_*.png` | scatter plot; residual histogram, Q-Q and autocorrelation |
| `bias_by_month_*.csv` / `.png` | mean error by month and season, with 95% intervals: is the model off in one season? |
| `convergence_*.png`, `dottyplots_*.png`, `0_*.csv` | every parameter set tried during calibration |
| `calibration_metadata.json`, `parameters.txt` | `Qmedia`, bounds and settings used, needed for later scenario runs |
| `MCMC_*`, `Forward_Prediction_*`, `parameter_significance_*` | uncertainty results (`DE-MCMC` and `FORWARD` with intervals) |
| `cv_results.csv`, `cv_yearly_statistics*.csv` | cross-validation: scores and parameters by held-out year; whether the ranges of yearly statistics held, and the model's bias in them |
| `sensitivity_*`, `gaps_summary.txt` | optional analyses |

Details: [User Guide §8](USER_GUIDE.md#8-understanding-the-output-files).

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** — how to prepare data, configure, run, read
  the results, and fix common errors.
- **[docs/METHODS.md](docs/METHODS.md)** — exactly what the software does, step
  by step, its assumptions and limitations, and how it differs from the Fortran.
  Read §16 there before using results to support a decision.
- **[validation/REPORT.md](validation/REPORT.md)** — the evidence that it works.
- [CHANGELOG.md](CHANGELOG.md) — changes between versions.

## Is it correct?

The [validation suite](validation/README.md) checks this, and its results are in
[validation/REPORT.md](validation/REPORT.md). In short:

- **Same results as the original Fortran**, on real river data, for all five
  model versions and every solution scheme the Fortran has (to 5×10⁻⁶ °C, the
  precision of the Fortran's output). Both programs also compute the same
  calibration scores (RMS, NSE, KGE) from the same daily, weekly and monthly
  averages, with and without gaps in the record.
- **Reproduces the published results.** For three Swiss rivers (Piccolroaz et
  al., 2016), the published parameters give the published calibration and
  validation errors, all 30 of them to within 0.001 °C. Recalibrating with `DE`
  returns the published parameters for versions 3, 4 and 5 (to within 1% of
  their ranges, apart from one flat trade-off on the Rhône) and for versions 7
  and 8 on the Dischmabach. For versions 7 and 8 on the other two rivers it
  finds a slightly better fit than the published one, with different
  parameters but the same predictions (to 0.002 °C). The original program,
  given both sets, computes the same errors as pyair2stream and agrees that
  the new ones fit better, so the difference is not a bug. Those published
  values cannot be reproduced exactly by anyone: the original program itself,
  run with its distributed settings, returns different parameters on every run
  there, because its optimiser stops at a different point each time. The published
  parameters belong to the Crank–Nicolson scheme the paper used: with RK4, 8 of
  the 15 sets are unstable and the rest give different errors. Calibrating with
  RK4 comes close to the published parameters (within 1% of their ranges) only
  where the water temperature responds slowly: the Mentue, versions 3–5.
- **Finds a known truth.** On data made by the model from known parameters,
  calibration predicts other years to within 0.04 °C of the truth (0.06 °C
  with typical gaps in the data).
- **Honest intervals, with known limits.** On such data, 90% prediction
  intervals contain 89–90% of new observations, and 90% parameter intervals
  contain the true values 89–97% of the time, also when calibrating on weekly
  means with `a2` and `a3` bounded at 0 (without that bound, 9 of 30 such
  calibrations ended on a meaningless set that zigzags from day to day, which
  pyair2stream now warns about). On the real rivers, with the
  default settings, prediction intervals contain 85–89.6% of daily values in
  years not used for calibration, so they are slightly optimistic. With three
  of the other settings tested, the Rhône with version 5 falls just below the
  report's 85% threshold (84.5–85.0%), so that check is marked as failed. For
  7-day means the default intervals contain 89–94% of observed values, against
  83–88% with `rho_timescale: "daily"` and 39–62% with `noise_model: "iid"`:
  model errors also have a part that lasts for weeks
  ([docs/METHODS.md §12](docs/METHODS.md#12-parameter-and-prediction-uncertainty-de-mcmc)).
- **The published parameters and the intervals.** The published parameters lie
  inside pyair2stream's 90% parameter intervals for versions 3–5 on all three
  rivers, for version 7 on the Rhône and for every version on the Dischmabach.
  For versions 7 and 8 on the Mentue and version 8 on the Rhône several lie
  outside. These are the cases above where recalibration finds a slightly
  better fit with different parameters: many combinations fit almost equally
  well, and the published set is one of them.
- **Probabilities that a limit was exceeded: right in principle, checked and
  corrected in practice.** On synthetic data the stated chances come true as
  often as they say. On the real rivers the model can be biased on the hottest
  days (on the Mentue, version 8's yearly peaks came out 0.6–0.8 °C too high),
  so over 48 held-out years per version the uncorrected 90% ranges for yearly
  statistics (highest daily mean, highest 7-day mean, days above a threshold)
  held in only 73–92% of years. A cross-validation of the calibration years
  measures that bias, and correcting for it brought the 90% ranges to 85–94%
  ([V11](validation/REPORT.md#v11)). Version 8's probabilities were closer to
  what happened than going by how often the limit was exceeded in past years.
  Version 5, which has no discharge term, could not follow the year-to-year
  changes of the Rhône's summer peaks.
- **Warmer and lower-flow years.** Calibrated only on the coolest (or
  highest-flow) third of the years, the model predicted the warmest (or
  lowest-flow) third almost as well as when calibrated on the middle third (at
  most 0.07 °C worse) and better than the simple alternatives, with 90%
  intervals containing 84–92% of the measurements. The exception is version 5
  on the Rhône, which does no better than the simple alternatives whichever
  years it is calibrated on. On rivers like the Rhône, use a version with
  discharge (7 or 8).
- **Scenario tools give exact answers** where the answer is known.

To run the tests and the validation suite (needs `gfortran`):

```bash
git submodule update --init --recursive
pip install -e . pytest
pytest tests/
python validation/run_all.py --quick     # or without --quick: the full suite, about 105 minutes
```

## Examples

Worked examples on a real river, each with a README
([examples/README.md](examples/README.md)):

| Example | Question |
|---|---|
| [01 Quickstart](examples/01_quickstart/README.md) | Can the model reproduce this river, including years it was not calibrated on? |
| [02 Uncertainty](examples/02_uncertainty/README.md) | What range of temperatures should we expect, and does the range hold? |
| [03 Compliance](examples/03_compliance/README.md) | How likely is it that a temperature limit was exceeded? |
| [04 Scenario](examples/04_scenario/README.md) | What difference would abstracting 30% of the flow make? |
| [05 Gaps](examples/05_gaps/README.md) | What to do with missing data |
| [06 Cross-validation](examples/06_cross_validation/README.md) | Does the model predict every year well? How firmly do the data fix the parameters? Which version to use? |

## Differences from the original Fortran

Results match the Fortran for the same settings. The main differences are safer
defaults (the stable `CRN` integrator and the DE optimizer), checks that stop
with a clear error instead of producing silently wrong numbers, `Qmedia` kept
fixed at its calibration value for validation and scenario runs, and the added
features above. Full list: [docs/METHODS.md §17](docs/METHODS.md#17-differences-from-the-fortran-original).

## Citing

Please cite the original model:

> Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water
> temperature as a function of air temperature and discharge. *Environmental
> Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011

and record the pyair2stream version you used. `pyair2stream` has no tagged
releases yet, so give the git commit (`git rev-parse HEAD`), e.g.:

> Water temperatures were simulated with pyair2stream
> (https://github.com/LukeAFullard/pyair2stream, commit `<sha>`).

## License

[CC BY-SA 3.0](LICENSE), the license of the original air2stream code.
