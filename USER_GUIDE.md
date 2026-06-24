# pyair2stream User Guide

Welcome to **pyair2stream**, the modernized Python port of the `air2stream` model for predicting river water temperature. This guide will help you install dependencies, configure your model, and run calibrations or forward simulations.

## 1. Project Setup

**Prerequisites:**
You will need Python 3.8+ installed on your system.

**Install Dependencies:**
The project dependencies are managed via `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

This will install the necessary scientific packages, including `numpy`, `pandas`, `matplotlib`, `scipy`, and `pyyaml`.

## 2. Configuration (`config.yaml`)

pyair2stream uses a structured YAML configuration file, replacing the legacy `input.txt` and `parameters.txt` space-separated text files. This modern format is less error-prone and easier to read.

By default, pyair2stream looks for `config.yaml` in the current directory. You can specify a different path via the `--config` command-line argument.

### Example `config.yaml`

```yaml
project_name: "my_river_project"
station_name: "Station_A"
water_station: "Station_B" # Optional, defaults to station_name if not provided
series: "c"                # c = continuous, m = mean year
time_resolution: "1d"      # 1d = daily, nw = n weeks, 1m = monthly
version: 8                 # 3, 4, 5, 7, or 8 parameters
Tice_cover: 0.0            # Threshold temperature for ice formation
objective_function: "NSE"  # KGE, NSE, RMS
integrator: "RK4"          # RK4, EUL, RK2, CRN
run_mode: "PSO"            # PSO, LATHYP, or FORWARD
prc: 1.0                   # Minimum percentage of data in input: 0...1

paths:
  input_data: "data/calibration_data.csv"
  validation_data: "data/validation_data.csv" # Optional, skipped if not found
  output_dir: "output"     # Optional, defaults to {project_name}/output_{version}

optimization:
  n_runs: 100              # Number of iterations
  mineff_index: 0.0        # Minimum efficiency code memorizes
  n_particles: 50          # Number of particles (PSO only)
  c1: 2.0                  # (PSO only)
  c2: 2.0                  # (PSO only)
  wmax: 0.9                # (PSO only)
  wmin: 0.4                # (PSO only)

parameter_bounds:
  min: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  max: [10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0]

parameters_forward: [1.2, 0.3, 0.2, 0.5, 0.1, 1.5, 0.4, 0.1] # Used only for FORWARD mode

gap_tolerant: true         # Enable gap-tolerant mode (default: false)
Qmedia: 15.3               # User-supplied external Qmedia (optional)
warmup_drop_days: 15       # Days dropped after restart (optional, default: 15)
min_segment_days: 30       # Minimum length of segments (optional, default: 30)
```

## 3. Data Format (CSVs)

pyair2stream now reads time-series inputs from standard CSV files with headers. This replaces the rigid space-separated formats in the original Fortran version.

Your input CSV files (`input_data` and `validation_data`) **must** contain a `Date` column (parseable by pandas, e.g., `YYYY-MM-DD`).

The standard required columns are:
*   `Date`: The timestamp (e.g., `2020-01-01`).
*   `T_air`: Air temperature.
*   `T_water`: Observed water temperature.
*   `Discharge`: River discharge (Q).

**Handling Missing Data:**
If `T_water` is missing for a specific day, you can explicitly use the legacy `-999.0` sentinel value or leave it blank (pandas will read as `NaN`).

### Gap-Tolerant Mode

By default, the model requires `T_air` and `Discharge` (Q) time series to be completely gap-free.
If your data has missing `T_air` or `Discharge` values, you can set `gap_tolerant: true` in your configuration file. In this mode, the model restarts integration at each contiguous block of valid data and pieces them together.

**Limitations to consider when using gap-tolerant mode:**
*   **MNAR (Missing Not At Random) Bias**: Gaps often correspond to extreme flood or freeze events. Calibrating on gapped data excludes these extremes, which reduces observed variance and **artificially inflates performance metrics** like NSE and KGE. Therefore, metrics obtained in gap-tolerant mode cannot be directly compared against continuous-record benchmarks. NSE and RMS are generally more robust than KGE under these conditions.
*   **External Qmedia**: When large chunks of high-discharge periods are missing, the internally computed `Qmedia` will be biased low. It is highly recommended to supply an external, historical `Qmedia` value in your configuration file.
*   **Excluded Observations**: Any valid `T_water` observations falling inside a forcing gap (`T_air` or `Q`) are excluded from calibration and evaluation.
*   **Warm-up buffer (`warmup_drop_days`)**: Restarting integration segments relies on a Day-Of-Year (DOY) climatology for the initial condition. A buffer period (default 15 days) at the start of every segment is ignored during calibration to avoid penalizing the initial condition transient.

### Example CSV (`calibration_data.csv`)

```csv
Date,T_air,T_water,Discharge
2020-01-01,5.2,4.1,12.5
2020-01-02,4.8,4.0,11.8
2020-01-03,6.1,-999.0,10.2
...
```

## 4. Running the Model

To run the model, use the `main.py` entry point. It will automatically load the configuration, run the calibration or forward simulation, and generate visualization plots.

```bash
# Run with the default config.yaml
python -m pyair2stream.main

# Run with a specific configuration file
python -m pyair2stream.main --config my_project_config.yaml
```

**What Happens When You Run:**
1.  **Calibration:** If `run_mode` is `PSO` or `LATHYP`, the model will optimize parameters based on your `input_data`.
2.  **Forward/Validation:** It will evaluate the best parameters against the `validation_data` (if provided).
3.  **Outputs:** Results are saved in standard CSV format in the designated output directory (e.g., `0_*.csv` for optimization history, `2_*.csv` for calibration time-series, `3_*.csv` for validation).
4.  **Post-Processing:** Dotty plots (parameter exploration) and time-series plots (comparing observed vs. simulated temperatures) are automatically generated as high-quality PDFs and PNGs using a colorblind-friendly palette.

## 5. Migrating from Fortran

If you are accustomed to the Fortran `air2stream`, note these key improvements:
*   **No Recompilation:** Change equations or routines directly in Python; no need to compile `.f90` files.
*   **Multiprocessing:** Optimization routines (`PSO`) use multiprocessing out-of-the-box, dramatically speeding up calibration.
*   **Direct Visuals:** No need for MATLAB (`post_processing.m`); Python generates the plots automatically at the end of the run.
*   **Bug Fixes:** Known legacy bugs (like dead PSO convergence checks and incorrect version 8 parameter initializations) have been fixed.