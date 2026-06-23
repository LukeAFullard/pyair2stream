import os
import yaml
import numpy as np
import pandas as pd
from typing import Tuple

from .config import CommonData

def read_calibration(config_file: str = 'config.yaml') -> CommonData:
    """
    Reads the calibration configuration from a YAML file and initializes the CommonData.
    """
    data = CommonData()

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Note: Using config.get() with defaults where appropriate, but strict mapping
    # to original inputs if they must be present.
    data.name = config.get('project_name', 'pyair2stream_project')
    data.air_station = config.get('station_name', 'AirStation')
    data.water_station = config.get('water_station', config.get('station_name', 'WaterStation'))
    data.series = config.get('series', 'series')
    data.time_res = config.get('time_resolution', '1d')
    data.version = int(config.get('version', 8))
    data.Tice_cover = np.float64(config.get('Tice_cover', 0.0))
    data.fun_obj = config.get('objective_function', 'NSE')
    data.mod_num = config.get('integrator', 'RK4')
    data.runmode = config.get('run_mode', 'PSO')
    data.prc = np.float64(config.get('prc', 1.0))

    # Paths mapping
    paths = config.get('paths', {})

    opt_config = config.get('optimization', {})
    data.n_run = int(opt_config.get('n_runs', 100))
    data.mineff_index = np.float64(opt_config.get('mineff_index', 0.0))

    data.station = data.air_station
    if data.air_station != data.water_station:
        data.station = f"{data.air_station}_{data.water_station}"

    data.folder = paths.get('output_dir', os.path.join(data.name, f"output_{data.version}"))
    os.makedirs(data.folder, exist_ok=True)

    # Fortran module hardcodes n_par = 8
    n_par = 8

    data.par = np.zeros(n_par, dtype=np.float64)
    data.parmin = np.zeros(n_par, dtype=np.float64)
    data.parmax = np.zeros(n_par, dtype=np.float64)
    data.flag_par = np.ones(n_par, dtype=np.bool_)

    if data.runmode == 'FORWARD':
        forward_params = config.get('parameters_forward', [])
        if len(forward_params) > 0:
            vals = [np.float64(x) for x in forward_params]
            data.par[:min(len(vals), n_par)] = vals[:min(len(vals), n_par)]
    elif data.runmode == 'PSO':
        data.n_particles = int(opt_config.get('n_particles', 50))
        data.c1 = np.float64(opt_config.get('c1', 2.0))
        data.c2 = np.float64(opt_config.get('c2', 2.0))
        data.wmax = np.float64(opt_config.get('wmax', 0.9))
        data.wmin = np.float64(opt_config.get('wmin', 0.4))

    if data.runmode in ['PSO', 'LATHYP']:
        bounds = config.get('parameter_bounds', {})
        vals_min = bounds.get('min', [])
        vals_max = bounds.get('max', [])

        if len(vals_min) > 0:
            data.parmin[:min(len(vals_min), n_par)] = [np.float64(x) for x in vals_min[:min(len(vals_min), n_par)]]
        if len(vals_max) > 0:
            data.parmax[:min(len(vals_max), n_par)] = [np.float64(x) for x in vals_max[:min(len(vals_max), n_par)]]

        # NOTE: 0-indexed in Python vs 1-indexed in Fortran
        # Fortran: parmin(4)=0 -> Python: parmin[3]=0

        if data.version == 3:
            data.parmin[3:8] = 0.0
            data.parmax[3:8] = 0.0
            data.flag_par[3:8] = False
        elif data.version == 4:
            data.parmin[4:8] = 0.0
            data.parmax[4:8] = 0.0
            data.flag_par[4:8] = False
        elif data.version == 5:
            data.parmin[3] = 0.0; data.parmax[3] = 0.0; data.flag_par[3] = False
            data.parmin[4] = 0.0; data.parmax[4] = 0.0; data.flag_par[4] = False
            data.parmin[7] = 0.0; data.parmax[7] = 0.0; data.flag_par[7] = False
        elif data.version == 7:
            data.parmin[3] = 0.0; data.parmax[3] = 0.0; data.flag_par[3] = False
        # Bug fix: Fortran had 'IF (version == 4)' twice.
        # The second one was clearly meant for version 8.
        elif data.version == 8:
            data.parmin[4:8] = 0.0
            data.parmax[4:8] = 0.0
            data.flag_par[4:8] = False

        out_param_path = os.path.join(data.folder, 'parameters.txt')
        with open(out_param_path, 'w') as f:
            f.write(f"{n_par}   !numero parametri\n")
            f.write(" ".join(f"{x:.5f}" for x in data.parmin) + "\n")
            f.write(" ".join(f"{x:.5f}" for x in data.parmax) + "\n")

    # Store paths in data to pass to read_Tseries
    data._input_data_path_cal = paths.get('input_data', None)
    data._input_data_path_val = paths.get('validation_data', None)

    return data

def read_Tseries(data: CommonData, p: str) -> None:
    """
    Reads the time series data from a CSV file and replicates the first year.
    Args:
        data: CommonData instance to update.
        p: 'c' for calibration, 'v' for validation.
    """
    if p == 'c':
        period = 'calibration'
        filename = getattr(data, '_input_data_path_cal', None)
    else:
        period = 'validation'
        filename = getattr(data, '_input_data_path_val', None)

    if not filename or not os.path.exists(filename):
        if p == 'v':
            print(f'Validation file not found --> validation is skipped')
            data.n_tot = 0
            return
        else:
            raise FileNotFoundError(f"Missing required {period} data file: {filename}")

    # Read the data using pandas. Expecting columns Date, T_air, T_water, Discharge
    df = pd.read_csv(filename)

    # Ensure Date is parsed
    if 'Date' not in df.columns:
        raise ValueError(f"Missing 'Date' column in {filename}")

    date_col = pd.to_datetime(df['Date'])

    # Map headers to internal names and handle missing data with -999.0 for backward compatibility
    # If not provided, assume T_water is just missing (-999.0)
    Tair = df.get('T_air', pd.Series(np.full(len(df), -999.0))).fillna(-999.0).astype(np.float64).values
    Twat_obs = df.get('T_water', pd.Series(np.full(len(df), -999.0))).fillna(-999.0).astype(np.float64).values
    Q = df.get('Discharge', pd.Series(np.full(len(df), -999.0))).fillna(-999.0).astype(np.float64).values

    n_tot_raw = len(df)

    if p == 'v' and n_tot_raw < 365:
        print('Validation period < 1 year --> validation is skipped')
        return

    n_year = int(np.ceil(n_tot_raw / 365.25))
    n_tot = n_tot_raw + 365

    data.n_tot = n_tot
    data.n_dat = n_tot  # Based on other parts of codebase, usually n_dat starts as n_tot. Aggregation changes n_dat.

    # Calculate Qmedia ignoring -999.0
    valid_Q_mask = Q != -999.0
    if np.any(valid_Q_mask):
        data.Qmedia = np.float64(np.mean(Q[valid_Q_mask]))
        data.n_Q = np.sum(valid_Q_mask)
    else:
        data.Qmedia = np.float64(0.0)
        data.n_Q = 0

    # Allocate arrays
    data.date = np.zeros((n_tot, 3), dtype=np.int32)
    data.Tair = np.zeros(n_tot, dtype=np.float64)
    data.Twat_obs = np.zeros(n_tot, dtype=np.float64)
    data.Q = np.zeros(n_tot, dtype=np.float64)
    data.tt = np.zeros(n_tot, dtype=np.float64)

    # Also allocate others for later
    data.Twat_obs_agg = np.zeros(n_tot, dtype=np.float64)
    data.Twat_mod = np.zeros(n_tot, dtype=np.float64)
    data.Twat_mod_agg = np.zeros(n_tot, dtype=np.float64)

    # Replicate the first year (first 365 days of data) at the beginning
    data.date[365:n_tot, 0] = date_col.dt.year.values
    data.date[365:n_tot, 1] = date_col.dt.month.values
    data.date[365:n_tot, 2] = date_col.dt.day.values

    data.Tair[365:n_tot] = Tair
    data.Twat_obs[365:n_tot] = Twat_obs
    data.Q[365:n_tot] = Q

    data.date[0:365, :] = -999
    data.Tair[0:365] = Tair[:365]
    data.Twat_obs[0:365] = Twat_obs[:365]
    data.Q[0:365] = Q[:365]

    # Check leap years and define tt
    # The first 365 days (warm-up) are always non-leap in tt
    for j in range(365):
        data.tt[j] = np.float64((j + 1) / 365.0)

    k = 365
    year_ini = data.date[365, 0]

    for i in range(n_year):
        year = year_ini + i
        # Simple leap year logic like Fortran
        is_leap = False
        if year % 4 == 0:
            if year % 100 != 0 or year % 400 == 0:
                is_leap = True

        days_in_year = 366 if is_leap else 365

        for j in range(days_in_year):
            if k + j >= n_tot:
                break
            data.tt[k + j] = np.float64((j + 1) / float(days_in_year))

        k += days_in_year
        if k >= n_tot:
            break
