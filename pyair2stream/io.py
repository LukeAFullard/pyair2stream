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
        raise FileNotFoundError(f"Configuration file not found: {config_file}\nPlease refer to USER_GUIDE.md for instructions on how to create a configuration file.")

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

    # Gap-tolerant mode configuration
    data.gap_tolerant = bool(config.get('gap_tolerant', False))
    qmedia_user = config.get('Qmedia')
    if qmedia_user is not None:
        data.Qmedia_user = float(qmedia_user)
    data.warmup_drop_days = int(config.get('warmup_drop_days', 15))
    data.min_segment_days = int(config.get('min_segment_days', 30))
    data.sensitivity_analysis = config.get('sensitivity_analysis', False)

    sens_pert = config.get('sensitivity_perturbations', [1.0])
    data.sensitivity_perturbations = [float(x) for x in sens_pert] if isinstance(sens_pert, list) else [float(sens_pert)]

    # Paths mapping
    paths = config.get('paths', {})

    data.forward_options = config.get('forward_options', {})

    opt_config = config.get('optimization', {})
    data.n_run = int(opt_config.get('n_runs', 100))
    data.mineff_index = np.float64(config.get('mineff_index', 0.0))

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
    elif data.runmode == 'DE':
        data.n_particles = int(opt_config.get('n_particles', 50)) # Using n_particles as population size
        # c1, c2, wmax, wmin not used for DE
    elif data.runmode == 'DE-MCMC':
        data.n_particles = int(opt_config.get('n_particles', 50)) # Using n_particles as population size for initial DE
        data.mcmc_walkers = int(opt_config.get('mcmc_walkers', 32))
        data.mcmc_steps = int(opt_config.get('mcmc_steps', 1000))

    bounds = config.get('parameter_bounds', {})
    vals_min = bounds.get('min', [])
    vals_max = bounds.get('max', [])

    if len(vals_min) > 0:
        data.parmin[:min(len(vals_min), n_par)] = [np.float64(x) for x in vals_min[:min(len(vals_min), n_par)]]
    if len(vals_max) > 0:
        data.parmax[:min(len(vals_max), n_par)] = [np.float64(x) for x in vals_max[:min(len(vals_max), n_par)]]

    if data.runmode in ['PSO', 'LATHYP', 'DE', 'DE-MCMC', 'FORWARD']:
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
        # The second one was clearly meant for version 8, but version 8 uses all 8 parameters.
        # We do not zero out any parameters for version 8.

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

    # Validate Start Date
    if not data.gap_tolerant and len(date_col) > 0 and (date_col.iloc[0].month != 1 or date_col.iloc[0].day != 1):
        raise ValueError(f"The time series in {filename} must start on January 1st.")

    # Validate Daily Scale (no gaps)
    expected_dates = pd.date_range(start=date_col.iloc[0], end=date_col.iloc[-1], freq='D')
    if len(date_col) != len(expected_dates) or not date_col.equals(pd.Series(expected_dates)):
        raise ValueError(f"The time series in {filename} must be continuous at a daily time scale with no missing dates. Fill missing rows with NaN or -999.")

    # Validate completeness of T_air and Discharge
    if 'T_air' not in df.columns:
        raise ValueError(f"Missing 'T_air' column in {filename}")
    if 'Discharge' not in df.columns:
        raise ValueError(f"Missing 'Discharge' column in {filename}")

    if not data.gap_tolerant:
        if df['T_air'].isnull().any():
            raise ValueError(f"The series of observed air temperature in {filename} must be complete. It cannot have gaps or missing data.")
        if df['Discharge'].isnull().any():
            raise ValueError(f"The series of discharge in {filename} must be complete. It cannot have gaps or missing data.")

    # Handle missing data via -999.0
    Tair = df['T_air'].fillna(-999.0).astype(np.float64).values
    Q = df['Discharge'].fillna(-999.0).astype(np.float64).values
    Twat_obs = df.get('T_water', pd.Series(np.full(len(df), -999.0))).fillna(-999.0).astype(np.float64).values

    n_tot_raw = len(df)

    if p == 'v' and n_tot_raw < 365:
        print('Validation period < 1 year --> validation is skipped')
        return

    n_year = int(np.ceil(n_tot_raw / 365.25))
    n_tot = n_tot_raw + 365

    data.n_tot = n_tot
    data.n_dat = n_tot  # Based on other parts of codebase, usually n_dat starts as n_tot. Aggregation changes n_dat.

    # Calculate Qmedia ignoring -999.0 and <= 0.0
    valid_Q_mask = (Q != -999.0) & (Q > 0.0)
    computed_qmedia = np.float64(0.0)
    if np.any(valid_Q_mask):
        computed_qmedia = np.float64(np.mean(Q[valid_Q_mask]))
        data.n_Q = np.sum(valid_Q_mask)
    else:
        computed_qmedia = np.float64(0.0)
        data.n_Q = 0

    if data.Qmedia_user is not None:
        data.Qmedia = np.float64(data.Qmedia_user)
        print(f"Using user-supplied Qmedia: {data.Qmedia:.5f} (computed was {computed_qmedia:.5f})")
    else:
        data.Qmedia = computed_qmedia

    if data.gap_tolerant:
        if data.Qmedia <= 0 and data.version not in [3, 5]:
            raise ValueError("Qmedia is zero or negative. Please supply Qmedia in the configuration file if the data is mostly empty.")
        if (data.n_Q / n_tot_raw) < 0.5 and data.Qmedia_user is None:
            print("Warning: More than 50% of Discharge values are missing. Consider supplying Qmedia_user in the configuration file.")
        if data.version in [3, 5]:
            print(f"Info: Q gaps are ignored because version {data.version} does not use Discharge in the ODE.")
        if data.Qmedia_user is not None and computed_qmedia > 0 and abs(computed_qmedia - data.Qmedia_user) / computed_qmedia > 0.3:
            print(f"Warning: Computed Qmedia ({computed_qmedia:.5f}) differs from user-supplied Qmedia ({data.Qmedia_user:.5f}) by more than 30%.")

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

    # Rewrite tt calculation to use calendar dates.
    # The first 365 days (warm-up) keep existing logic: 1..365 / 365.0
    for j in range(365):
        data.tt[j] = np.float64((j + 1) / 365.0)

    for i in range(365, n_tot):
        year = data.date[i, 0]
        month = data.date[i, 1]
        day = data.date[i, 2]
        is_leap = False
        if year % 4 == 0:
            if year % 100 != 0 or year % 400 == 0:
                is_leap = True
        days_in_year = 366 if is_leap else 365

        # Calculate day of year
        doy = (pd.Timestamp(year, month, day) - pd.Timestamp(year, 1, 1)).days + 1
        data.tt[i] = np.float64(doy / float(days_in_year))

    # Calculate DOY climatology (calibration pass only)
    if data.gap_tolerant and p == 'c':
        data.doy_climatology = np.zeros(366, dtype=np.float64)
        doy_sums = np.zeros(366, dtype=np.float64)
        doy_counts = np.zeros(366, dtype=int)

        for i in range(365, n_tot):
            if data.Twat_obs[i] != -999.0:
                year = data.date[i, 0]
                month = data.date[i, 1]
                day = data.date[i, 2]
                doy = (pd.Timestamp(year, month, day) - pd.Timestamp(year, 1, 1)).days
                doy_sums[doy] += data.Twat_obs[i]
                doy_counts[doy] += 1

        if np.sum(doy_counts) == 0:
            raise ValueError("Zero T_water observations found during calibration. Calibration is impossible.")

        for i in range(366):
            if doy_counts[i] > 0:
                data.doy_climatology[i] = doy_sums[i] / doy_counts[i]
            else:
                data.doy_climatology[i] = np.nan

        # Interpolate missing DOYs
        if np.isnan(data.doy_climatology).any():
            df_clim = pd.Series(data.doy_climatology)
            # Duplicate to handle wrapping around year
            df_clim_extended = pd.concat([df_clim, df_clim, df_clim]).reset_index(drop=True)
            df_clim_extended = df_clim_extended.interpolate(method='linear')
            data.doy_climatology = df_clim_extended.iloc[366:2*366].values
