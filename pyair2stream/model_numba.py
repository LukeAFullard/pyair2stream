import numpy as np
import math
from numba import njit

PI = np.pi
TTT = 1.0 / 365.0

@njit
def fast_funcobj(n_dat, n_tot, I_inf, I_pos, Twat_mod, Twat_obs_agg, eval_mask, fun_obj_type, mean_obs, TSS_obs, std_obs):
    # fun_obj_type: 0 for NSE, 1 for KGE, 2 for RMS

    Twat_mod_agg = np.full(n_tot, -999.0, dtype=np.float64)

    valid_n_dat = 0

    # Pre-allocate valid arrays for KGE
    agg_mod_valid = np.zeros(n_dat, dtype=np.float64)
    agg_obs_valid = np.zeros(n_dat, dtype=np.float64)

    for i in range(n_dat):
        start = I_inf[i, 0]
        end = I_inf[i, 1]
        target_idx = I_inf[i, 2]

        sum_val = 0.0
        count = 0

        for j in range(start, end + 1):
            pos = I_pos[j]
            if Twat_mod[pos] != -999.0 and eval_mask[pos]:
                sum_val += Twat_mod[pos]
                count += 1

        if count > 0:
            mod_agg = sum_val / count
            Twat_mod_agg[target_idx] = mod_agg

            agg_mod_valid[valid_n_dat] = mod_agg
            agg_obs_valid[valid_n_dat] = Twat_obs_agg[target_idx]

            valid_n_dat += 1

    if valid_n_dat == 0:
        return -999.0, Twat_mod_agg

    if fun_obj_type == 0: # NSE
        TSS = 0.0
        for k in range(valid_n_dat):
            TSS += (agg_mod_valid[k] - agg_obs_valid[k]) ** 2

        if TSS_obs == 0.0:
            if TSS == 0.0:
                return 1.0, Twat_mod_agg
            else:
                return -1e30, Twat_mod_agg
        return 1.0 - TSS / TSS_obs, Twat_mod_agg

    elif fun_obj_type == 1: # KGE
        if valid_n_dat < 2:
            return -999.0, Twat_mod_agg

        sum_mod = 0.0
        for k in range(valid_n_dat):
            sum_mod += agg_mod_valid[k]
        mean_mod = sum_mod / valid_n_dat

        TSS_mod = 0.0
        covar_mod = 0.0

        for k in range(valid_n_dat):
            TSS_mod += (agg_mod_valid[k] - mean_mod) ** 2
            covar_mod += (agg_mod_valid[k] - mean_mod) * (agg_obs_valid[k] - mean_obs)

        std_mod = np.sqrt(TSS_mod / (valid_n_dat - 1))
        covar_mod = covar_mod / (valid_n_dat - 1)

        if std_obs == 0 or std_mod == 0:
            return -999.0, Twat_mod_agg

        return 1.0 - np.sqrt((std_mod / std_obs - 1.0)**2 + (mean_mod / mean_obs - 1.0)**2 + (covar_mod / (std_mod * std_obs) - 1.0)**2), Twat_mod_agg

    elif fun_obj_type == 2: # RMS
        TSS = 0.0
        for k in range(valid_n_dat):
            TSS += (agg_mod_valid[k] - agg_obs_valid[k]) ** 2
        return -np.sqrt(TSS / valid_n_dat), Twat_mod_agg

    return -999.0, Twat_mod_agg

@njit
def fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta, QQ, Tw, time, Qmedia):
    if version == 3:
        return p1 + p2 * Ta - p3 * Tw
    elif version == 5:
        return p1 + p2 * Ta - p3 * Tw + p6 * math.cos(2.0 * PI * (time - p7))
    elif version == 8:
        theta = QQ / Qmedia
        if theta <= 0.0:
            DD = 1e-10
        else:
            DD = theta ** p4
        return (p1 + p2 * Ta - p3 * Tw + theta * (p5 + p6 * math.cos(2.0 * PI * (time - p7)) - p8 * Tw)) / DD
    elif version == 7:
        theta = QQ / Qmedia
        return p1 + p2 * Ta - p3 * Tw + theta * (p5 + p6 * math.cos(2.0 * PI * (time - p7)) - p8 * Tw)
    elif version == 4:
        theta = QQ / Qmedia
        if theta <= 0.0:
            DD = 1e-10
        else:
            DD = theta ** p4
        return (p1 + p2 * Ta - p3 * Tw) / DD
    else:
        return 0.0

@njit
def fast_run_integration(Tair, Q, tt, Twat_mod, Tice_cover, Qmedia, version, mod_num_idx, segments, p1, p2, p3, p4, p5, p6, p7, p8):
    # mod_num_idx: 0=CRN, 1=RK2, 2=RK4, 3=EUL

    if mod_num_idx == 0: # CRN
        if version in (8, 7, 4):
            theta = Q / Qmedia
            DD = np.empty_like(theta)
            for i in range(len(theta)):
                if theta[i] <= 0.0:
                    DD[i] = 1e-10
                else:
                    DD[i] = theta[i] ** p4
            denom_term = 1.0 + 0.5 * p8 * theta / DD + 0.5 * p3 / DD

            for s in range(len(segments)):
                start = segments[s, 0]
                end = segments[s, 1]
                for j in range(start, end):
                    Tw_j = Twat_mod[j]
                    theta_j = theta[j]
                    theta_j1 = theta[j+1]
                    DD_j = DD[j]
                    DD_j1 = DD[j+1]

                    num1 = 0.5 / DD_j * (p1 + p2*Tair[j] - p3*Tw_j + theta_j * (p5 + p6*math.cos(2.0*PI*(tt[j] - p7)) - p8*Tw_j))
                    num2 = 0.5 / DD_j1 * (p1 + p2*Tair[j+1] + theta_j1 * (p5 + p6*math.cos(2.0*PI*(tt[j+1] - p7))))
                    Tw_j1 = (Tw_j + num1 + num2) / denom_term[j+1]
                    if Tw_j1 < Tice_cover:
                        Tw_j1 = Tice_cover
                    Twat_mod[j+1] = Tw_j1
        elif version in (5, 3):
            denom = 1.0 + 0.5 * p3
            mult = 1.0 - 0.5 * p3
            for s in range(len(segments)):
                start = segments[s, 0]
                end = segments[s, 1]
                for j in range(start, end):
                    Tw_j1 = (Twat_mod[j] * mult + p1 + 0.5 * p2 * (Tair[j] + Tair[j+1]) + \
                             0.5 * p6 * math.cos(2.0*PI*(tt[j] - p7)) + 0.5 * p6 * math.cos(2.0*PI*(tt[j+1] - p7))) / denom
                    if Tw_j1 < Tice_cover:
                        Tw_j1 = Tice_cover
                    Twat_mod[j+1] = Tw_j1
        return

    if mod_num_idx == 1: # RK2
        for s in range(len(segments)):
            start = segments[s, 0]
            end = segments[s, 1]
            for j in range(start, end):
                Ta_j = Tair[j]
                Ta_j1 = Tair[j+1]
                Q_j = Q[j]
                Q_j1 = Q[j+1]
                Tw_j = Twat_mod[j]
                tt_j = tt[j]

                K1 = fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta_j, Q_j, Tw_j, tt_j, Qmedia)
                K2 = fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta_j1, Q_j1, Tw_j + K1, tt_j + TTT, Qmedia)

                Tw_j1 = Tw_j + 0.5 * (K1 + K2)
                if Tw_j1 < Tice_cover:
                    Tw_j1 = Tice_cover
                Twat_mod[j+1] = Tw_j1

    elif mod_num_idx == 2: # RK4
        for s in range(len(segments)):
            start = segments[s, 0]
            end = segments[s, 1]
            for j in range(start, end):
                Ta_j = Tair[j]
                Ta_j1 = Tair[j+1]
                Q_j = Q[j]
                Q_j1 = Q[j+1]
                Tw_j = Twat_mod[j]
                tt_j = tt[j]

                Ta_mid = 0.5 * (Ta_j + Ta_j1)
                Q_mid = 0.5 * (Q_j + Q_j1)
                tt_mid = tt_j + 0.5 * TTT

                K1 = fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta_j, Q_j, Tw_j, tt_j, Qmedia)
                K2 = fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta_mid, Q_mid, Tw_j + 0.5 * K1, tt_mid, Qmedia)
                K3 = fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta_mid, Q_mid, Tw_j + 0.5 * K2, tt_mid, Qmedia)
                K4 = fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta_j1, Q_j1, Tw_j + K3, tt_j + TTT, Qmedia)

                Tw_j1 = Tw_j + (1.0 / 6.0) * (K1 + 2.0*K2 + 2.0*K3 + K4)
                if Tw_j1 < Tice_cover:
                    Tw_j1 = Tice_cover
                Twat_mod[j+1] = Tw_j1

    elif mod_num_idx == 3: # EUL
        for s in range(len(segments)):
            start = segments[s, 0]
            end = segments[s, 1]
            for j in range(start, end):
                K1 = fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Tair[j+1], Q[j+1], Twat_mod[j], tt[j+1], Qmedia)
                Tw_j1 = Twat_mod[j] + K1
                if Tw_j1 < Tice_cover:
                    Tw_j1 = Tice_cover
                Twat_mod[j+1] = Tw_j1
