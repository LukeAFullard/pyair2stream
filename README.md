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

This installs `pyair2stream` and its dependencies (`numpy`, `pandas`, `matplotlib`, `scipy`, `pyyaml`, `numba`, `emcee`), plus the `pyair2stream` command-line entry point.

For development (running tests, editing the source):

```bash
pip install -e .
pip install pytest
```

> The original Fortran source is included under `fortran/` purely as a reference implementation used by the test suite (see [Testing](#testing)); it is not required to run `pyair2stream`.

## Quick start

1. Prepare a CSV of daily data with `Date`, `T_air`, `T_water`, and (optionally) `Discharge` columns — see [Input data format](#input-data-format).
2. Create a `config.yaml` (a minimal example is below; see [Configuration](#configuration) for the full reference).
3. Run:

```bash
pyair2stream --config config.yaml
```

This calibrates the model, validates it against a validation set (if provided), writes results to the output directory, and generates plots automatically.

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
  n_runs: 100
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

See [USER_GUIDE.md](USER_GUIDE.md#gap-tolerant-mode) for full details.

## Outputs

Running `pyair2stream` writes to the configured output directory:

| File pattern | Contents |
|---|---|
| `0_*.csv` | Optimization history (parameter sets tried and their fit) |
| `1_*.out` | Best-fit parameters and final efficiency score |
| `2_*.csv` | Simulated vs. observed time series (calibration period) |
| `3_*.csv` | Simulated vs. observed time series (validation period) |
| `calibration_*.png` / `.pdf` | Time-series comparison plots |
| `dottyplots_*.png` / `.pdf` | Parameter/objective-function dotty plots |
| `MCMC_chain_*.csv`, `MCMC_envelopes_*.csv` | MCMC parameter samples and prediction intervals (`DE-MCMC` mode only) |
| `gaps_summary.txt` | Gap/segment diagnostics (gap-tolerant mode only) |

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

The test suite includes regression tests that compile the original Fortran source (`fortran/src/`) with `gfortran` and numerically compare its output against the Python/Numba implementation, in addition to unit tests for I/O, optimization, sensitivity analysis, and post-processing.

```bash
gfortran --version   # gfortran is required for the golden Fortran-comparison tests
pip install -e . pytest
pytest tests/
```

## Examples

The `examples/` directory contains runnable end-to-end examples, including:

- `synthetic_example/` — a minimal synthetic dataset and config to get started
- `gap_tolerance/` — demonstrates gap-tolerant calibration
- `sensitivity_example/` — one-at-a-time sensitivity analysis
- `forward_prediction_intervals/` — MCMC-based prediction intervals
- `optimizer_comparison/`, `optimizer_convergence/` — comparing PSO/DE/LATHYP behaviour
- `Hopelands/`, `Pukeokahu/` — real river-station case studies

## Relationship to the original Fortran code

`pyair2stream` reproduces the governing equations, numerical integrators, and objective functions of the original Fortran `air2stream` model, with a small number of intentional corrections to bugs identified in the legacy code (documented inline in the source, e.g. in `optimization.py` and `io.py`). It is not officially maintained by the original authors — if you need the reference implementation, see the original air2stream repository.

## Citing

If you use this software in published work, please cite the original model paper:

> Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *Environmental Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011

## License

[MIT](LICENSE)
