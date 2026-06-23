import os
import numpy as np
import pandas as pd
from typing import Tuple

from .config import CommonData

def read_calibration(input_file: str = 'input.txt', pso_file: str = 'PSO.txt', parameters_forward: str = 'parameters_forward.txt', parameters: str = 'parameters.txt') -> CommonData:
    """
    Reads the calibration configuration and initializes the CommonData.
    """
    data = CommonData()

    # Read input information
    with open(input_file, 'r') as f:
        lines = f.readlines()

    data.name = lines[1].strip()
    data.air_station = lines[2].strip()
    data.water_station = lines[3].strip()
    data.series = lines[4].strip()
    data.time_res = lines[5].strip()
    data.version = int(lines[6].strip())
    data.Tice_cover = np.float64(lines[7].strip())
    data.fun_obj = lines[8].strip()
    data.mod_num = lines[9].strip()
    data.runmode = lines[10].strip()
    data.prc = np.float64(lines[11].strip())
    data.n_run = int(lines[12].strip())
    data.mineff_index = np.float64(lines[13].strip())

    data.station = f"{data.air_station}_{data.water_station}"
    data.folder = os.path.join(data.name, f"output_{data.version}")
    os.makedirs(data.folder, exist_ok=True)

    # Fortran module hardcodes n_par = 8
    n_par = 8

    data.par = np.zeros(n_par, dtype=np.float64)
    data.parmin = np.zeros(n_par, dtype=np.float64)
    data.parmax = np.zeros(n_par, dtype=np.float64)
    data.flag_par = np.ones(n_par, dtype=np.bool_)

    if data.runmode == 'FORWARD':
        forward_path = os.path.join(data.name, 'parameters_forward.txt')
        with open(forward_path, 'r') as f:
            line = f.readline()
            vals = [np.float64(x) for x in line.split()]
            data.par[:len(vals)] = vals
    elif data.runmode == 'PSO':
        with open(pso_file, 'r') as f:
            lines = f.readlines()
            data.n_particles = int(lines[1].strip())
            c1_c2 = lines[2].split()
            data.c1 = np.float64(c1_c2[0])
            data.c2 = np.float64(c1_c2[1])
            wmax_wmin = lines[3].split()
            data.wmax = np.float64(wmax_wmin[0])
            data.wmin = np.float64(wmax_wmin[1])

    if data.runmode in ['PSO', 'LATHYP']:
        param_path = os.path.join(data.name, 'parameters.txt')
        with open(param_path, 'r') as f:
            lines = f.readlines()
            vals_min = [np.float64(x) for x in lines[0].split()]
            vals_max = [np.float64(x) for x in lines[1].split()]
            data.parmin[:len(vals_min)] = vals_min
            data.parmax[:len(vals_max)] = vals_max

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

    return data

def read_Tseries(data: CommonData, p: str) -> None:
    """
    Reads the time series data and replicates the first year.
    Args:
        data: CommonData instance to update.
        p: 'c' for calibration, 'v' for validation.
    """
    if p == 'c':
        period = 'calibration'
    else:
        period = 'validation'

    filename = os.path.join(data.name, f"{data.station}_{data.series}{p}.txt")

    if not os.path.exists(filename):
        if p == 'v':
            print(f'Validation file {filename} not found --> validation is skipped')
            data.n_tot = 0
            return
        else:
            raise FileNotFoundError(f"Missing required data file: {filename}")

    # Read the data using pandas
    # Assuming columns are Year, Month, Day, Tair, Twat_obs, Q
    # We will use delim_whitespace=True or sep=r'\s+'
    df = pd.read_csv(filename, sep=r'\s+', header=None, names=['Year', 'Month', 'Day', 'Tair', 'Twat_obs', 'Q'])

    n_tot_raw = len(df)

    if p == 'v' and n_tot_raw < 365:
        print('Validation period < 1 year --> validation is skipped')
        return

    n_year = int(np.ceil(n_tot_raw / 365.25))
    n_tot = n_tot_raw + 365

    data.n_tot = n_tot
    data.n_dat = n_tot  # Based on other parts of codebase, usually n_dat starts as n_tot. Aggregation changes n_dat.

    # Calculate Qmedia ignoring -999
    valid_Q = df['Q'][df['Q'] != -999]
    if len(valid_Q) > 0:
        data.Qmedia = np.float64(valid_Q.mean())
        data.n_Q = len(valid_Q)
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
    data.date[365:n_tot, 0] = df['Year'].values
    data.date[365:n_tot, 1] = df['Month'].values
    data.date[365:n_tot, 2] = df['Day'].values

    data.Tair[365:n_tot] = df['Tair'].values
    data.Twat_obs[365:n_tot] = df['Twat_obs'].values
    data.Q[365:n_tot] = df['Q'].values

    data.date[0:365, :] = -999
    data.Tair[0:365] = df['Tair'].values[:365]
    data.Twat_obs[0:365] = df['Twat_obs'].values[:365]
    data.Q[0:365] = df['Q'].values[:365]

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
