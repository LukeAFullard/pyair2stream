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
  **cross-validation** by year, and **sensitivity analysis**.
- Uses a **YAML config file and CSV files**, and writes CSV results and plots.

## Install

Requires Python 3.9 or newer.

```bash
git clone https://github.com/LukeAFullard/pyair2stream.git
cd pyair2stream
pip install .
```

## Quick start

Run the bundled example (synthetic data, about 5 seconds):

```bash
pyair2stream --config examples/quickstart/config.yaml
```

Results and plots appear in `examples/quickstart/output/`. The
[User Guide](USER_GUIDE.md#3-your-first-run-the-bundled-example) explains them.

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
| `convergence_*.png`, `dottyplots_*.png`, `0_*.csv` | every parameter set tried during calibration |
| `calibration_metadata.json`, `parameters.txt` | `Qmedia`, bounds and settings used, needed for later scenario runs |
| `MCMC_*`, `Forward_Prediction_*`, `parameter_significance_*` | uncertainty results (`DE-MCMC` and `FORWARD` with intervals) |
| `sensitivity_*`, `cv_results.csv`, `gaps_summary.txt` | optional analyses |

Details: [User Guide §8](USER_GUIDE.md#8-understanding-the-output-files).

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** — how to prepare data, configure, run, read
  the results, and fix common errors.
- **[docs/METHODS.md](docs/METHODS.md)** — exactly what the software does, step
  by step, its assumptions and limitations, and how it differs from the Fortran.
  Read §16 there before using results to support a decision.
- [CHANGELOG.md](CHANGELOG.md) — changes between versions.

## Is it correct?

- **Same results as the original.** The test suite compiles the original Fortran
  (a git submodule, see [`fortran/patches/NOTICE.md`](fortran/patches/NOTICE.md))
  and checks that all five model versions give the same daily temperatures to
  within 6×10⁻⁶ °C.
- **Reproduces published results.** Running the published parameters for three
  Swiss rivers (Piccolroaz et al., 2016) gives the published calibration NSE:

  | River (station) | Flow regime | Published NSE | pyair2stream NSE |
  |---|---|---|---|
  | Mentue (MAH-2369) | natural | 0.989 | 0.9886 |
  | Rhône (SIO-2011) | regulated | 0.923 | 0.9242 |
  | Dischmabach (DAV-2327) | snow-fed | 0.950 | 0.9558 |

  Recalibrating with `DE` recovers the published parameters (for the Mentue, all
  eight within 0.004). Details:
  [`examples/validation/Switzerland/`](examples/validation/Switzerland/README.md).
- **Honest uncertainty.** On the bundled example, the 90% prediction interval
  contained 89% of calibration days and 92% of a held-out year; every run with
  intervals reports this check for your own data.

To run the tests (needs `gfortran`):

```bash
git submodule update --init --recursive
pip install -e . pytest
pytest tests/
```

## Examples

Each folder has a README.

| Folder | Shows |
|---|---|
| `quickstart/` | the smallest complete run |
| `validation/Switzerland/` | reproduction of published results; optimizer and integrator comparisons |
| `forward_prediction_intervals/` | uncertainty bands for a future scenario (iid vs. AR(1) errors) |
| `mcmc_comparison/` | `DE-MCMC` vs. `DE-CV-MCMC` |
| `cross_validation/` | leave-one-year-out cross-validation |
| `gap_experiment/` | effect of gaps on gap-tolerant calibration |
| `optimizer_comparison/`, `optimizer_convergence/` | DE vs. PSO |
| `Hopelands/` | a real river from raw data to results |

Some examples need data that is not included in this repository.

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
