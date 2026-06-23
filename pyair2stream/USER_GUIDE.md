# pyair2stream User Guide

Welcome to the `pyair2stream` User Guide. `pyair2stream` is a modern Python port of the `air2stream` hybrid model for river water temperature as a function of air temperature and discharge.

This guide will explain how to install, configure, run, and visualize the results using the modern Python infrastructure.

## 1. Overview

The original Fortran codebase utilized raw, space-separated text files, which were rigid and prone to formatting errors. `pyair2stream` embraces the modern scientific Python ecosystem. You will configure the model using clear, human-readable **YAML** (or **JSON**) files and provide your time-series inputs using standard **CSV** formats.

The underlying optimization algorithms (PSO, Latin Hypercube) and numerical integrators remain mathematically identical to the original implementation to ensure absolute consistency and correctness.

## 2. Installation

To install `pyair2stream`, you need Python 3.8+ and standard scientific libraries.

```bash
# Clone the repository
git clone <repository_url>
cd pyair2stream

# It is highly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies (such as numpy, pandas, matplotlib, pyyaml)
pip install -r requirements.txt
```

## 3. Configuration & Data Formats

### 3.1. Main Configuration (`config.yaml`)

Instead of `input.txt` and `parameters.txt`, `pyair2stream` expects a `config.yaml` file. This format allows comments and clear structuring.

```yaml
# Example config.yaml
run_mode: "PSO"            # Options: FORWARD, PSO, LATHYP
objective_function: "NSE"  # Options: NSE, KGE, RMS
station_name: "Station_A"
time_resolution: "daily"   # daily, weekly, etc.

integrator: "RK4"          # Options: EUL, RK2, RK4, CRN

optimization:
  n_particles: 50
  n_runs: 1000

paths:
  input_data: "data/input_timeseries.csv"
  output_dir: "results/"
```

### 3.2. Time-Series Input Data (`CSV`)

You provide time-series data using a CSV file with clear headers. The required columns are typically Date, Air Temperature, Water Temperature (if calibrating/validating), and Discharge.

```csv
Date,T_air,T_water,Discharge
2010-01-01,5.2,4.1,120.5
2010-01-02,5.5,4.3,118.2
...
```

*Note: `pyair2stream` uses `pandas` under the hood. Missing data should be represented as empty fields or `NaN`, replacing the legacy `-999` convention where appropriate, though the model will transparently handle backwards compatibility if needed.*

## 4. Running the Model

You execute the model via the main Python script, passing your configuration file:

```bash
python main.py --config config.yaml
```

### Run Modes

- **FORWARD**: Runs a single simulation using the provided parameters. Useful for evaluating a known parameter set or running predictions.
- **PSO (Particle Swarm Optimization)**: Runs the calibration routine to find the best parameter set using a distributed swarm algorithm. This mode automatically leverages multiprocessing to dramatically speed up evaluations across multiple CPU cores.
- **LATHYP (Latin Hypercube)**: Explores the parameter space using Latin Hypercube sampling.

## 5. Output Visualization

The legacy `post_processing.m` MATLAB script has been entirely replaced by Python's `matplotlib` and `pandas`.

After a successful run, the outputs (best parameters, simulation time-series, parameter ensembles) will be saved in the `results/` folder as `CSV` or `JSON` files.

You can automatically generate the standard dotty plots and time-series plots:

```bash
python post_processing.py --result-dir results/ --format pdf
```

This will generate high-quality, color-blind friendly plots for immediate inclusion in reports or papers.
