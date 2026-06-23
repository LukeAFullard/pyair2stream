import numpy as np
import math
from .config import CommonData, PI, TTT

def call_model(data: CommonData) -> None:
    """
    Core air2stream simulation loop.
    Replicates SUBROUTINE call_model in AIR2STREAM_SUBROUTINES.f90
    """
    if data.Twat_obs[0] == -999.0:
        data.Twat_mod[0] = 4.0
    else:
        data.Twat_mod[0] = data.Twat_obs[0]

    # Convert par from 0-indexed to 1-indexed for the formula to match Fortran
    # We will just pad a 0 at index 0 for easy translation of par(1), par(2)...
    # Since in python, arrays are 0 indexed, we can map par[0] to par(1), etc.
    p = np.zeros(9, dtype=np.float64)
    p[1:9] = data.par[0:8]

    for j in range(data.n_tot - 1):
        if data.mod_num == 'RK2':
            K1 = RK4_air2stream(data, data.Tair[j], data.Q[j], data.Twat_mod[j], data.tt[j])
            K2 = RK4_air2stream(data, data.Tair[j+1], data.Q[j+1], data.Twat_mod[j] + K1, data.tt[j] + TTT)

        elif data.mod_num == 'RK4':
            K1 = RK4_air2stream(data, data.Tair[j], data.Q[j], data.Twat_mod[j], data.tt[j])
            K2 = RK4_air2stream(data, 0.5 * (data.Tair[j] + data.Tair[j+1]), 0.5 * (data.Q[j] + data.Q[j+1]), data.Twat_mod[j] + 0.5 * K1, data.tt[j] + 0.5 * TTT)
            K3 = RK4_air2stream(data, 0.5 * (data.Tair[j] + data.Tair[j+1]), 0.5 * (data.Q[j] + data.Q[j+1]), data.Twat_mod[j] + 0.5 * K2, data.tt[j] + 0.5 * TTT)
            K4 = RK4_air2stream(data, data.Tair[j+1], data.Q[j+1], data.Twat_mod[j] + K3, data.tt[j] + TTT)

        elif data.mod_num == 'EUL':
            K1 = RK4_air2stream(data, data.Tair[j+1], data.Q[j+1], data.Twat_mod[j], data.tt[j+1])

        elif data.mod_num == 'CRN':
            if data.version in [8, 7, 4]:
                theta_j = data.Q[j] / data.Qmedia
                theta_j1 = data.Q[j+1] / data.Qmedia
                DD_j = theta_j ** p[4]
                DD_j1 = theta_j1 ** p[4]

                data.Twat_mod[j+1] = (data.Twat_mod[j] + 0.5 / DD_j * (p[1] + p[2]*data.Tair[j] - p[3]*data.Twat_mod[j] + theta_j * (p[5] + p[6]*np.cos(2.0*PI*(data.tt[j] - p[7])) - p[8]*data.Twat_mod[j])) + \
                                      0.5 / DD_j1 * (p[1] + p[2]*data.Tair[j+1] + theta_j1 * (p[5] + p[6]*np.cos(2.0*PI*(data.tt[j+1] - p[7]))))) / \
                                     (1.0 + 0.5 * p[8] * theta_j1 / DD_j1 + 0.5 * p[3] / DD_j1)

            elif data.version in [5, 3]:
                data.Twat_mod[j+1] = (data.Twat_mod[j] * (1.0 - 0.5 * p[3]) + p[1] + 0.5 * p[2] * (data.Tair[j] + data.Tair[j+1]) + \
                                      0.5 * p[6] * np.cos(2.0*PI*(data.tt[j] - p[7])) + 0.5 * p[6] * np.cos(2.0*PI*(data.tt[j+1] - p[7]))) / \
                                     (1.0 + 0.5 * p[3])

        if data.mod_num == 'RK2':
            data.Twat_mod[j+1] = data.Twat_mod[j] + 0.5 * (K1 + K2)
        elif data.mod_num == 'RK4':
            data.Twat_mod[j+1] = data.Twat_mod[j] + (1.0 / 6.0) * (K1 + 2.0*K2 + 2.0*K3 + K4)
        elif data.mod_num == 'EUL':
            data.Twat_mod[j+1] = data.Twat_mod[j] + K1

        data.Twat_mod[j+1] = max(data.Twat_mod[j+1], data.Tice_cover)

def RK4_air2stream(data: CommonData, Ta: np.float64, QQ: np.float64, Tw: np.float64, time: np.float64) -> np.float64:
    """
    Subroutine RK4_air2stream from AIR2STREAM_SUBROUTINES.f90
    """
    p = np.zeros(9, dtype=np.float64)
    p[1:9] = data.par[0:8]

    if data.version in [8, 4]:
        DD = (QQ / data.Qmedia) ** p[4]
    elif data.version in [5, 3]:
        DD = 0.0
    else:
        DD = 1.0

    K = 0.0

    if data.version == 3:
        K = (p[1] + p[2]*Ta - p[3]*Tw)

    if data.version == 5:
        K = p[1] + p[2]*Ta - p[3]*Tw + p[6]*np.cos(2.0*PI*(time - p[7]))

    if data.version in [8, 7]:
        K = p[1] + p[2]*Ta - p[3]*Tw + (QQ / data.Qmedia) * (p[5] + p[6]*np.cos(2.0*PI*(time - p[7])) - p[8]*Tw)
        K = K / DD

    if data.version == 4:
        K = (p[1] + p[2]*Ta - p[3]*Tw) / DD

    return np.float64(K)

def aggregation(data: CommonData) -> None:
    """
    Aggregation (to calibrate the model with different time scale: daily, weekly, monthly)
    """
    pp = len(data.time_res)
    if pp == 2:
        unit = data.time_res[1]
        qty = int(data.time_res[0])
    elif pp == 3:
        unit = data.time_res[2]
        qty = int(data.time_res[0:2])

    data.I_pos = np.full(data.n_tot, -999, dtype=np.int32)
    data.Twat_obs_agg = np.full(data.n_tot, -999.0, dtype=np.float64)

    n_inf = 1
    n_pos = 1

    if data.time_res == '1d':
        n_units = data.n_tot - 365
        data.I_inf = np.full((n_units, 3), -999, dtype=np.int32)

        for i in range(365, data.n_tot):
            if data.Twat_obs[i] != -999.0:
                # 0-indexed I_inf and I_pos.
                # Fortran I_inf(n_inf, 2) -> Python I_inf[n_inf-1, 1]
                data.I_inf[n_inf - 1, 1] = n_pos - 1
                data.I_inf[n_inf - 1, 2] = i
                data.I_pos[n_pos - 1] = i
                data.Twat_obs_agg[i] = data.Twat_obs[i]
                n_inf += 1
                n_pos += 1

    elif unit == 'w':
        n_days = qty * 7
        n_units = int(np.ceil((data.n_tot - 365) / n_days))
        data.I_inf = np.full((n_units, 3), -999, dtype=np.int32)

        for i in range(365, data.n_tot, n_days):
            tmp = 0.0
            count = 0
            pos_tmp = i + int(np.ceil(0.5 * n_days)) - 1

            for j in range(n_days):
                k = i + j
                if k >= data.n_tot:
                    break
                if data.Twat_obs[k] != -999.0:
                    tmp += data.Twat_obs[k]
                    data.I_pos[n_pos - 1] = k
                    n_pos += 1
                    count += 1

            if count >= n_days * data.prc:
                data.I_inf[n_inf - 1, 1] = n_pos - 2 # n_pos-1 in Fortran (which is last idx added), in Python it's n_pos-2 because we do n_pos += 1
                data.I_inf[n_inf - 1, 2] = pos_tmp
                data.Twat_obs_agg[pos_tmp] = tmp / count
                n_inf += 1
            else:
                data.I_pos[n_pos - 1 - count : n_pos - 1] = -999
                n_pos = n_pos - count

    elif unit == 'm':
        n_units = int(np.ceil(data.n_tot / 30.5))
        data.I_inf = np.full((n_units, 3), -999, dtype=np.int32)
        n_days = 0
        month_curr = -999
        count = 0
        tmp = 0.0

        for i in range(365, data.n_tot):
            month = data.date[i, 1]
            if month != month_curr:
                if count >= n_days * data.prc and i != 365:
                    data.I_inf[n_inf - 1, 1] = n_pos - 2
                    data.I_inf[n_inf - 1, 2] = i - int(np.floor(0.5 * n_days)) - 1
                    data.Twat_obs_agg[data.I_inf[n_inf - 1, 2]] = tmp / count
                    n_inf += 1
                else:
                    if count > 0:
                        data.I_pos[n_pos - 1 - count : n_pos - 1] = -999
                        n_pos = n_pos - count
                month_curr = month
                count = 0
                n_days = 1
                tmp = 0.0
            else:
                n_days += 1

            if data.Twat_obs[i] != -999.0:
                tmp += data.Twat_obs[i]
                data.I_pos[n_pos - 1] = i
                n_pos += 1
                count += 1

        # Last month
        if count >= n_days * data.prc:
            data.I_inf[n_inf - 1, 1] = n_pos - 2
            data.I_inf[n_inf - 1, 2] = data.n_tot - 1 - int(np.floor(0.5 * n_days)) # using data.n_tot - 1 as the last i
            data.Twat_obs_agg[data.I_inf[n_inf - 1, 2]] = tmp / count
            n_inf += 1
        else:
            if count > 0:
                data.I_pos[n_pos - 1 - count : n_pos - 1] = -999
                n_pos = n_pos - count
    else:
        print("Error: variable time_res")

    data.n_dat = n_inf - 1
    n_pos = n_pos - 1

    if data.n_dat > 0:
        data.I_inf[0, 0] = 0
        for i in range(1, data.n_dat):
            data.I_inf[i, 0] = data.I_inf[i - 1, 1] + 1

    # Resize arrays
    data.I_inf = data.I_inf[:data.n_dat, :]
    data.I_pos = data.I_pos[:n_pos]

def statis(data: CommonData) -> None:
    """
    Statis (to calculate errors)
    """
    data.mean_obs = np.float64(0.0)
    data.TSS_obs = np.float64(0.0)

    for i in range(data.n_dat):
        data.mean_obs += data.Twat_obs_agg[data.I_inf[i, 2]]

    data.mean_obs /= np.float64(data.n_dat)

    for i in range(data.n_dat):
        data.TSS_obs += (data.Twat_obs_agg[data.I_inf[i, 2]] - data.mean_obs) ** 2

    if data.n_dat > 1:
        data.std_obs = np.sqrt(data.TSS_obs / np.float64(data.n_dat - 1))
    else:
        data.std_obs = np.float64(0.0)

def funcobj(data: CommonData) -> float:
    """
    Calculation of the objective function.
    Returns the objective value.
    """
    data.Twat_mod_agg = np.full(data.n_tot, -999.0, dtype=np.float64)

    for i in range(data.n_dat):
        tmp = 0.0
        start_idx = data.I_inf[i, 0]
        end_idx = data.I_inf[i, 1]

        for j in range(start_idx, end_idx + 1):
            tmp += data.Twat_mod[data.I_pos[j]]

        count = end_idx - start_idx + 1
        data.Twat_mod_agg[data.I_inf[i, 2]] = tmp / np.float64(count)

    ind = 0.0

    if data.fun_obj == 'NSE':
        TSS = 0.0
        for i in range(data.n_dat):
            TSS += (data.Twat_mod_agg[data.I_inf[i, 2]] - data.Twat_obs_agg[data.I_inf[i, 2]]) ** 2
        ind = 1.0 - TSS / data.TSS_obs

    elif data.fun_obj == 'KGE':
        mean_mod = 0.0
        for i in range(data.n_dat):
            mean_mod += data.Twat_mod_agg[data.I_inf[i, 2]]
        mean_mod /= np.float64(data.n_dat)

        covar_mod = 0.0
        TSS_mod = 0.0
        for i in range(data.n_dat):
            TSS_mod += (data.Twat_mod_agg[data.I_inf[i, 2]] - mean_mod) ** 2
            covar_mod += (data.Twat_mod_agg[data.I_inf[i, 2]] - mean_mod) * (data.Twat_obs_agg[data.I_inf[i, 2]] - data.mean_obs)

        std_mod = np.sqrt(TSS_mod / np.float64(data.n_dat - 1))
        covar_mod /= np.float64(data.n_dat - 1)

        ind = 1.0 - np.sqrt((std_mod / data.std_obs - 1.0)**2 + (mean_mod / data.mean_obs - 1.0)**2 + (covar_mod / (std_mod * data.std_obs) - 1.0)**2)

    elif data.fun_obj == 'RMS':
        TSS = 0.0
        for i in range(data.n_dat):
            TSS += (data.Twat_mod_agg[data.I_inf[i, 2]] - data.Twat_obs_agg[data.I_inf[i, 2]]) ** 2
        ind = -np.sqrt(TSS / np.float64(data.n_dat))

    else:
        print("Errore nella scelta della f. obiettivo")

    return ind
