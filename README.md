# pyair2stream

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

A modern Python port of **air2stream**, a hybrid physics-based/empirical model for simulating daily river water temperature from air temperature and (optionally) discharge.

Original model: Toffolon, M. and Piccolroaz, S. (2015). *A hybrid model for river water temperature as a function of air temperature and discharge*, Environmental Research Letters, 10(11), 114011. [doi:10.1088/1748-9326/10/11/114011](https://doi.org/10.1088/1748-9326/10/11/114011)

`pyair2stream` reimplements the original Fortran model in Python/[Numba](https://numba.pydata.org/) and adds:

- **YAML-based configuration** instead of fixed-width text files
- **CSV in, CSV out** — no custom binary formats
- **Gap-tolerant mode** for calibrating on time series with missing air temperature or discharge data
- **Modern calibration algorithms**: Differential Evolution + L-BFGS-B (default), PSO, Latin Hypercube, and DE + MCMC (via [`emcee`](https://emcee.readthedocs.io/)) for uncertainty quantification
- **Automatic post-processing**: calibration/validation plots and parameter dotty-plots
- **One-at-a-time sensitivity analysis** and forward prediction intervals
- **Autoregressive AR(1) Prediction Intervals**: An opt-in noise model for MCMC prediction intervals to account for residual serial correlation, yielding more realistic uncertainty bounds.
- A test suite that validates the Python/Numba integration against the original compiled Fortran source (see [Testing](#testing))

> **Status**: this is a community Python port, not the official model release. See [Relationship to the original Fortran code](#relationship-to-the-original-fortran-code) below.

## Contents

- [Model versions](#model-versions)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Input data format](#input-data-format)
- [Gap-tolerant mode](#gap-tolerant-mode)
- [Outputs](#outputs)
- [Using pyair2stream from Python](#using-pyair2stream-from-python)
- [Testing](#testing)
- [Examples](#examples)
- [Relationship to the original Fortran code](#relationship-to-the-original-fortran-code)
- [Validation against published literature](#validation-against-published-literature)
- [Citing](#citing)
- [License](#license)

## Model versions

`air2stream` supports five model formulations, selected via `version` in the config file. Each trades off complexity/data requirements against the physical processes it represents:

| Version | Parameters | Uses discharge? | Seasonal signal? | Notes |
|:-------:|:----------:|:----------------:|:-----------------:|-------|
| 3 | 3  | No  | No  | Simplest linear air–water relationship |
| 4 | 4  | Yes | No  | Adds discharge-dependent thermal inertia |
| 5 | 5  | No  | Yes | Adds an explicit seasonal (cosine) term |
| 7 | 7  | Yes | Yes | Full model without the discharge-attenuation exponent |
| 8 | 8  | Yes | Yes | Full model (recommended starting point for most rivers) |

Four numerical integrators are available (`integrator` in the config): `EUL` (explicit Euler), `RK2`, `RK4` (default, recommended), and `CRN` (semi-implicit Crank–Nicolson).

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/LukeAFullard/pyair2stream.git
cd pyair2stream
pip install .
```

This installs `pyair2stream` and its dependencies (`numpy`, `pandas`, `matplotlib`, `scipy`, `pyyaml`, `numba`, `emcee`, `openpyxl`), plus the `pyair2stream` command-line entry point.

For development (running tests, editing the source):

```bash
pip install -e .
pip install pytest
```

> The original Fortran source is pulled in as a git submodule under `fortran/upstream/` purely as a reference implementation used by the test suite (see [Testing](#testing)); it is not required to run `pyair2stream`. Clone with `git clone --recurse-submodules`, or run `git submodule update --init --recursive` after a plain clone.

## Quick start

The fastest way to see `pyair2stream` work end-to-end is the bundled quick-start example, which uses a small synthetic dataset so it runs in a few seconds:

```bash
git clone https://github.com/LukeAFullard/pyair2stream.git
cd pyair2stream
pip install .
pyair2stream --config examples/quickstart/config.yaml
```

This calibrates the model, validates it against a held-out year, writes results to `examples/quickstart/output/`, and generates diagnostic plots automatically. See the [User Guide's walkthrough](USER_GUIDE.md#3-your-first-run-the-bundled-example) for what the output should look like and how to interpret it.

Once that runs cleanly, point `pyair2stream` at your own data:

1. Prepare a CSV of daily data with `Date`, `T_air`, `T_water`, and (optionally) `Discharge` columns — see [Input data format](#input-data-format).
2. Create a `config.yaml` (a minimal example is below; see [Configuration](#configuration) for the full reference).
3. Run:

```bash
pyair2stream --config config.yaml
```

### Minimal `config.yaml`

```yaml
project_name: "my_river_project"
station_name: "Station_A"
series: "c"                # c = continuous daily series
time_resolution: "1d"      # 1d, nw (n weeks), or nm (n months)
version: 8                 # 3, 4, 5, 7, or 8
objective_function: "NSE"  # NSE, KGE, or RMS
integrator: "RK4"          # RK4, EUL, RK2, or CRN
run_mode: "DE"             # DE (recommended), PSO, LATHYP, FORWARD, or DE-MCMC

paths:
  input_data: "data/calibration_data.csv"
  validation_data: "data/validation_data.csv"   # optional

optimization:
  n_run: 100
  n_particles: 50

parameter_bounds:
  min: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  max: [10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0]
```

## Configuration

The config file is YAML and supports options for calibration mode, optimizer settings, gap-tolerant mode, sensitivity analysis, and forward prediction intervals. The **full annotated reference** — including every option, its default, and when it applies — is in **[USER_GUIDE.md](USER_GUIDE.md)**.

## Input data format

Input/validation CSVs need a `Date` column parseable by pandas (e.g. `YYYY-MM-DD`) plus:

| Column | Required | Description |
|---|---|---|
| `T_air` | Yes | Air temperature (°C) |
| `T_water` | Yes | Observed water temperature (°C). Missing values allowed. |
| `Discharge` | Only for versions 4, 7, 8 | River discharge |

Missing values can be left blank (read as `NaN`) or given the legacy `-999.0` sentinel.

```csv
Date,T_air,T_water,Discharge
2020-01-01,5.2,4.1,12.5
2020-01-02,4.8,4.0,11.8
2020-01-03,6.1,,10.2
```

## Gap-tolerant mode

By default the model requires `T_air` (and `Discharge`, if used) to be gap-free over the whole record. Setting `gap_tolerant: true` allows the model to split the series into contiguous valid segments and integrate each independently.

**Before relying on gap-tolerant mode, be aware:**

- Gaps often coincide with floods or freeze events. Calibrating on the remaining data excludes these extremes, which can **artificially inflate** NSE/KGE relative to a continuous record — the two are not directly comparable.
- If large high-flow periods are missing, the automatically computed `Qmedia` (mean discharge, used for normalization) will be biased low. Supply a known historical `Qmedia` in the config if this is a concern.
- `T_water` observations that fall inside a forcing gap are excluded from both calibration and evaluation.
- Each segment discards a short warm-up buffer (`warmup_drop_days`, default 15) after restarting, since the restart condition relies on a day-of-year climatology.

See [USER_GUIDE.md](USER_GUIDE.md#10-gap-tolerant-mode) for full details.

## Outputs

Running `pyair2stream` writes to the configured output directory:

| File pattern | Contents |
|---|---|
| `parameters.txt` | Parameter bounds actually used, after fixing version-inactive parameters to 0 |
| `0_*.csv` | Optimization history (every parameter set tried and its objective score) |
| `1_*.out` | Best-fit parameters and final efficiency score (calibration, then validation if run) |
| `2_*.csv` | Simulated vs. observed time series (calibration period) |
| `3_*.csv` | Simulated vs. observed time series (validation period, if validation data was supplied) |
| `calibration_*.png` / `.pdf` | Time-series plot restricted to the calibration period's observations |
| `validation_*.png` / `.pdf` | Same, for the validation period |
| `full_simulation_*.png` / `.pdf` | Same, over the whole simulated record, including where there's no observation |
| `convergence_*.png` / `.pdf` | Objective-function value vs. optimizer iteration |
| `dottyplots_*.png` / `.pdf` | Parameter/objective-function dotty plots |
| `predicted_vs_measured_*.png` / `.pdf` | Simulated vs. observed scatter, one per period (calibration/validation/full_simulation) |
| `residual_diagnostics_*.png` / `.pdf` | Residual histogram, Q-Q plot, and autocorrelation, one per period |
| `goodness_of_fit_*.csv` | R², RMSE, MAE, AIC, BIC, one file per period |
| `sensitivity_*.csv` / `.png` / `.pdf` | One-at-a-time sensitivity analysis (only if `sensitivity_analysis: true`) |
| `MCMC_chain_*.csv`, `MCMC_envelopes_*.csv` | MCMC parameter samples and prediction intervals (`DE-MCMC` mode only) |
| `Forward_Prediction_Envelopes_*.csv` | Prediction envelopes for a `FORWARD` run using a prior MCMC chain (see [forward prediction intervals](USER_GUIDE.md#12-forward-prediction-intervals)) |
| `cv_results.csv` | One row per cross-validation fold with metrics and calibrated parameters (only if `cross_validation.enabled: true`) |
| `gaps_summary.txt` | Gap/segment diagnostics (gap-tolerant mode only) |

See [§8 of the User Guide](USER_GUIDE.md#8-understanding-the-output-files) for a worked explanation of these files, including an important note about warm-up rows in `2_*.csv`/`3_*.csv`.

## Using pyair2stream from Python

```python
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis
from pyair2stream.optimization import DE_mode

data = read_calibration(config_file="config.yaml")
read_Tseries(data, "c")
aggregation(data)
statis(data)

DE_mode(data)  # populates data.par_best and data.finalfit
print(data.par_best, data.finalfit)
```

## Testing

The test suite includes regression tests that compile the original Fortran source with `gfortran` and numerically compare its output against the Python/Numba implementation, in addition to unit tests for I/O, optimization, sensitivity analysis, and post-processing.

The Fortran source itself is not vendored in this repo — it's a git submodule pinned to a specific commit of the upstream [air2stream](https://github.com/spiccolroaz/air2stream) reference implementation, with a small, documented patch applied at build time to make it compile under `gfortran` (the original targets Intel Fortran on Windows). See [`fortran/patches/NOTICE.md`](fortran/patches/NOTICE.md) for exactly what's patched, why, and licensing attribution.

```bash
git submodule update --init --recursive   # fetch the pinned upstream Fortran source
gfortran --version   # gfortran is required for the golden Fortran-comparison tests
pip install -e . pytest
pytest tests/
```

If the submodule isn't initialized, the golden tests fail immediately with a clear message telling you to run the `git submodule update` command above, rather than a cryptic file-not-found error.

## Examples

The `examples/` directory contains runnable end-to-end examples, each with its own README explaining the setup and results in detail:

| Directory | Demonstrates |
|---|---|
| `quickstart/` | Minimal synthetic dataset and config — the fastest way to see a full run (see [Quick start](#quick-start)) |
| `gap_tolerance/` | `gap_tolerant` calibration under 1/2/3 simulated data gaps, and the performance trade-offs involved |
| `gap_experiment/` | How parameter stability and goodness-of-fit degrade as gaps are introduced into `T_air` |
| `forward_prediction_intervals/` | Generating probabilistic prediction envelopes from a prior `DE-MCMC` calibration, including the AR(1) noise model |
| `cross_validation/` | Leave-one-year-out cross-validation on a real river dataset |
| `optimizer_comparison/` | Calibrating the same dataset with `PSO`, `DE`, and `DE-MCMC` and comparing results |
| `optimizer_convergence/` | How `PSO` and `DE` convergence behaviour changes with iteration count |
| `Hopelands/`, `Pukeokahu/` | Full real river-station case studies, including raw-data preprocessing, gap-tolerant calibration, and sensitivity analysis |
| `validation/Switzerland/` | Re-derivation of the Python PSO bugfix (see [Known deviations](#known-deviations-from-the-fortran-reference)) and the literature validation study below |
| `validation/Callahan_Moore_2025/` | Validation against an independently published literature parameter set for a real station |

## Relationship to the original Fortran code

`pyair2stream` reproduces the governing equations, numerical integrators, and objective functions of the original Fortran `air2stream` model. It is not officially maintained by the original authors — if you need the reference implementation, see the original air2stream repository.

### Known deviations from the Fortran reference

These are the known, intentional behavioral differences from the original
Fortran, found during porting and fixed with test coverage:

- **Version 8 parameter zeroing (fixed in `io.py`)**: the original Fortran had
  a duplicated `IF (version == 4)` block where the second occurrence appears
  to have been intended for `version == 8`, causing parameters 5–8 to be
  incorrectly zeroed in Version 8 mode. `pyair2stream` does not reproduce this
  bug — Version 8 uses all 8 parameters. See commit `d78fe17`.
- **PSO initialization/NaN handling (fixed in `optimization.py`)**: the initial
  Python port initialized `fitbest` to zero and did not guard against NaN
  objective values from solver overflow, causing PSO to silently return
  all-zero parameters on some datasets. Fixed by initializing to `-1e30` and
  using NaN-safe comparisons/argmax. See PR #21 and
  `examples/validation/Switzerland/README.md`.
- **`mineff_index` config location (fixed in `io.py`)**: corrected to read
  from the config root rather than nested under `optimization:`, matching
  `USER_GUIDE.md`.
- **Dotty-plot tolerance defaults (fixed in `post_processing.py`)**: the
  default acceptability threshold for highlighting "good" parameter sets in
  diagnostic plots is now objective-function-aware (0.5 for NSE/KGE, 2.0 for
  RMS) instead of a single hardcoded value, which previously produced an
  empty acceptable-parameter region for NSE/KGE calibrations.

None of these affect the core forward-simulation physics (governing equations,
integrators) validated by the golden Fortran tests — they affect calibration
robustness and diagnostic plotting.

## Validation against published literature

Beyond the Fortran golden tests, `pyair2stream` has been validated against the
three Swiss river datasets and literature parameter sets published in the
supplementary material of Piccolroaz et al. (2016), using the `FORWARD` run
mode with literature-derived parameters, and independently re-calibrated with
Differential Evolution:

| River (station) | Flow regime | Literature NSE | pyair2stream NSE (DE) |
|---|---|---|---|
| Mentue (MAH-2369) | Natural | 0.989 | 0.9886 |
| Rhône (SIO-2011) | Regulated | 0.923 | 0.9242 |
| Dischmabach (DAV-2327) | Snow-fed | 0.950 | 0.9558 |

Full methodology, parameter tables, and plots: [`examples/validation/Switzerland/README.md`](examples/validation/Switzerland/README.md).

> **Note on optimizer choice**: this study found that PSO can converge to
> different parameter sets than DE due to equifinality (multiple parameter
> combinations giving similarly good NSE), particularly for versions with 8
> free parameters. DE and DE-MCMC matched literature parameters far more
> closely than PSO in these tests. **For scientific/publication use, prefer
> `DE` or `DE-MCMC` over `PSO`** unless you've independently confirmed PSO
> convergence for your dataset.

## Citing

If you use this software in published work, please cite the original model paper:

> Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *Environmental Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011

Because `pyair2stream` is not yet distributed via PyPI and does not currently
tag releases, please also record the exact git commit hash used for your
study (`git rev-parse HEAD`) so your results are reproducible, e.g.:

> Water temperature simulations were produced using pyair2stream
> (https://github.com/LukeAFullard/pyair2stream, commit `<sha>`).

## License

[MIT](LICENSE)
