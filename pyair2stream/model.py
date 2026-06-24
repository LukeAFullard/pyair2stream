import numpy as np
import math
import pandas as pd
from .config import CommonData, PI, TTT

def detect_segments(data: CommonData) -> None:
    """
    Detect valid segments, handling gap-tolerant mode.
    Builds data.segments and data.eval_mask.
    """
    data.segments = []
    # In legacy mode or not gap_tolerant, the segment is the whole data (starting from 365)
    # and the mask covers everything.
    if not data.gap_tolerant:
        data.eval_mask = np.zeros(data.n_tot, dtype=np.bool_)
        if data.n_tot > 365:
            data.eval_mask[365:] = True
        return

    data.eval_mask = np.zeros(data.n_tot, dtype=np.bool_)

    in_segment = False
    seg_start = -1

    # We only care about data from index 365 onwards (no warm-up)
    for i in range(365, data.n_tot):
        is_valid = True
        if data.Tair[i] == -999.0:
            is_valid = False
        if data.version not in [3, 5] and data.Q[i] == -999.0:
            is_valid = False

        if is_valid:
            if not in_segment:
                in_segment = True
                seg_start = i
        else:
            if in_segment:
                in_segment = False
                seg_end = i - 1
                length = seg_end - seg_start + 1
                if length >= data.min_segment_days:
                    data.segments.append((seg_start, seg_end))
                else:
                    print(f"Warning: Dropped segment ({seg_start}, {seg_end}) of length {length} days (min_segment_days={data.min_segment_days})")

    # Handle segment extending to end of array
    if in_segment:
        seg_end = data.n_tot - 1
        length = seg_end - seg_start + 1
        if length >= data.min_segment_days:
            data.segments.append((seg_start, seg_end))
        else:
            print(f"Warning: Dropped segment ({seg_start}, {seg_end}) of length {length} days (min_segment_days={data.min_segment_days})")

    if not data.segments:
        raise ValueError("No valid segments found after gap detection and filtering.")

    total_valid_days = sum(end - start + 1 for start, end in data.segments)
    if total_valid_days == 0:
        raise ValueError("Total valid forcing days across all segments is zero.")

    # Optional diagnostics (avoid spamming in optimization loops)
    if not hasattr(data, '_segment_warned'):
        if total_valid_days < 365:
            print(f"Warning: Total valid forcing days is {total_valid_days} (< 365). Calibration results may be unreliable.")
        if len(data.segments) > 2:
            print(f"Warning: Data is highly fragmented ({len(data.segments)} segments).")
        data._segment_warned = True

    # Build eval_mask based on segments and warmup_drop_days
    for start, end in data.segments:
        # Exclude the first warmup_drop_days of each segment
        eval_start = min(start + data.warmup_drop_days, end + 1)
        if eval_start <= end:
            data.eval_mask[eval_start:end + 1] = True


def _step(data: CommonData, j: int, p: np.ndarray) -> None:
    raise NotImplementedError("_step has been removed for performance reasons. Use _run_integration instead.")

def _get_RK_func(version, Qmedia, p):
    """
    Returns a fast RK4_air2stream derivative function depending on version.
    Avoids conditionals during ODE integration.
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]

    if version == 3:
        def RK(Ta, QQ, Tw, time):
            return p1 + p2 * Ta - p3 * Tw

    elif version == 5:
        def RK(Ta, QQ, Tw, time):
            return p1 + p2 * Ta - p3 * Tw + p6 * math.cos(2.0 * PI * (time - p7))

    elif version in [8, 7]:
        if version == 8:
            def RK(Ta, QQ, Tw, time):
                theta = QQ / Qmedia
                DD = theta ** p4
                return (p1 + p2 * Ta - p3 * Tw + theta * (p5 + p6 * math.cos(2.0 * PI * (time - p7)) - p8 * Tw)) / DD
        else:
            def RK(Ta, QQ, Tw, time):
                theta = QQ / Qmedia
                return p1 + p2 * Ta - p3 * Tw + theta * (p5 + p6 * math.cos(2.0 * PI * (time - p7)) - p8 * Tw)

    elif version == 4:
        def RK(Ta, QQ, Tw, time):
            DD = (QQ / Qmedia) ** p4
            return (p1 + p2 * Ta - p3 * Tw) / DD

    else:
        def RK(Ta, QQ, Tw, time):
            return 0.0

    return RK

def _run_integration(data: CommonData, segments, p):
    """
    Core integration loop. Inlines _step and RK4_air2stream for maximum speed.
    """
    RK = _get_RK_func(data.version, data.Qmedia, p)
    mod_num = data.mod_num

    # Pre-extract arrays for speed
    Tair = data.Tair
    Q = data.Q
    tt = data.tt
    Twat_mod = data.Twat_mod
    Tice_cover = data.Tice_cover

    # Specific precomputations for CRN
    if mod_num == 'CRN':
        p1, p2, p3, p4, p5, p6, p7, p8 = p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
        if data.version in [8, 7, 4]:
            theta = Q / data.Qmedia
            DD = theta ** p4
            denom_term = 1.0 + 0.5 * p8 * theta / DD + 0.5 * p3 / DD

            for start, end in segments:
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

        elif data.version in [5, 3]:
            denom = 1.0 + 0.5 * p3
            mult = 1.0 - 0.5 * p3
            for start, end in segments:
                for j in range(start, end):
                    Tw_j1 = (Twat_mod[j] * mult + p1 + 0.5 * p2 * (Tair[j] + Tair[j+1]) + \
                             0.5 * p6 * math.cos(2.0*PI*(tt[j] - p7)) + 0.5 * p6 * math.cos(2.0*PI*(tt[j+1] - p7))) / denom
                    if Tw_j1 < Tice_cover:
                        Tw_j1 = Tice_cover
                    Twat_mod[j+1] = Tw_j1
        return

    # Non-CRN integration methods
    if mod_num == 'RK2':
        for start, end in segments:
            for j in range(start, end):
                Ta_j = Tair[j]
                Ta_j1 = Tair[j+1]
                Q_j = Q[j]
                Q_j1 = Q[j+1]
                Tw_j = Twat_mod[j]
                tt_j = tt[j]

                K1 = RK(Ta_j, Q_j, Tw_j, tt_j)
                K2 = RK(Ta_j1, Q_j1, Tw_j + K1, tt_j + TTT)
                Tw_j1 = Tw_j + 0.5 * (K1 + K2)
                if Tw_j1 < Tice_cover:
                    Tw_j1 = Tice_cover
                Twat_mod[j+1] = Tw_j1

    elif mod_num == 'RK4':
        for start, end in segments:
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

                K1 = RK(Ta_j, Q_j, Tw_j, tt_j)
                K2 = RK(Ta_mid, Q_mid, Tw_j + 0.5 * K1, tt_mid)
                K3 = RK(Ta_mid, Q_mid, Tw_j + 0.5 * K2, tt_mid)
                K4 = RK(Ta_j1, Q_j1, Tw_j + K3, tt_j + TTT)

                Tw_j1 = Tw_j + (1.0 / 6.0) * (K1 + 2.0*K2 + 2.0*K3 + K4)
                if Tw_j1 < Tice_cover:
                    Tw_j1 = Tice_cover
                Twat_mod[j+1] = Tw_j1

    elif mod_num == 'EUL':
        for start, end in segments:
            for j in range(start, end):
                K1 = RK(Tair[j+1], Q[j+1], Twat_mod[j], tt[j+1])
                Tw_j1 = Twat_mod[j] + K1
                if Tw_j1 < Tice_cover:
                    Tw_j1 = Tice_cover
                Twat_mod[j+1] = Tw_j1

def call_model_segmented(data: CommonData) -> None:
    """
    Segmented ODE integration for gap-tolerant mode.
    """
    data.Twat_mod[:] = -999.0

    p = np.zeros(9, dtype=np.float64)
    p[1:9] = data.par[0:8]

    for start, end in data.segments:
        # Initial Condition
        if data.Twat_obs[start] != -999.0:
            data.Twat_mod[start] = data.Twat_obs[start]
        else:
            # DOY is 0-indexed in array but 1-366 in reality
            year = data.date[start, 0]
            month = data.date[start, 1]
            day = data.date[start, 2]
            doy = (pd.Timestamp(year, month, day) - pd.Timestamp(year, 1, 1)).days
            data.Twat_mod[start] = data.doy_climatology[doy]

    _run_integration(data, data.segments, p)

def call_model(data: CommonData) -> None:
    """
    Core air2stream simulation loop.
    Replicates SUBROUTINE call_model in AIR2STREAM_SUBROUTINES.f90
    """
    if data.gap_tolerant:
        call_model_segmented(data)
        return

    if data.Twat_obs[0] == -999.0:
        data.Twat_mod[0] = 4.0
    else:
        data.Twat_mod[0] = data.Twat_obs[0]

    # Convert par from 0-indexed to 1-indexed for the formula to match Fortran
    p = np.zeros(9, dtype=np.float64)
    p[1:9] = data.par[0:8]

    segments = [(0, data.n_tot - 1)]
    _run_integration(data, segments, p)

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
    if data.n_dat == 0:
        raise ValueError("n_dat is 0 after aggregation. No T_water observations survived.")

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

    valid_mask = data.Twat_mod[data.I_pos] != -999.0
    if data.eval_mask is not None:
        valid_mask &= data.eval_mask[data.I_pos]

    # Valid mask for I_pos length
    pos_mask = np.zeros(len(data.I_pos), dtype=bool)
    for i in range(data.n_dat):
        start = data.I_inf[i, 0]
        end = data.I_inf[i, 1]
        pos_mask[start:end+1] = True

    # The group_indices size must exactly match len(data.I_pos) to avoid IndexError
    group_indices = np.full(len(data.I_pos), -1, dtype=np.int32)

    starts = data.I_inf[:, 0]
    ends = data.I_inf[:, 1]
    for i in range(data.n_dat):
        group_indices[starts[i]:ends[i]+1] = i

    # Combine pos_mask and valid_mask
    combined_mask = pos_mask & valid_mask

    valid_group_indices = group_indices[combined_mask]
    valid_mod_vals = data.Twat_mod[data.I_pos][combined_mask]

    # Sum up valid elements per group
    sums = np.bincount(valid_group_indices, weights=valid_mod_vals, minlength=data.n_dat)
    counts = np.bincount(valid_group_indices, minlength=data.n_dat)

    has_count = counts > 0
    valid_n_dat = np.sum(has_count)

    if valid_n_dat > 0:
        agg_indices = data.I_inf[has_count, 2]
        data.Twat_mod_agg[agg_indices] = sums[has_count] / counts[has_count]

    ind = 0.0

    if data.fun_obj == 'NSE':
        if valid_n_dat > 0:
            agg_mod = data.Twat_mod_agg[data.I_inf[has_count, 2]]
            agg_obs = data.Twat_obs_agg[data.I_inf[has_count, 2]]
            TSS = np.sum((agg_mod - agg_obs) ** 2)
            ind = 1.0 - TSS / data.TSS_obs
        else:
            ind = -999.0

    elif data.fun_obj == 'KGE':
        if valid_n_dat < 2:
            print("Warning: KGE undefined for n_dat < 2. Returning -999.0")
            return -999.0

        agg_mod = data.Twat_mod_agg[data.I_inf[has_count, 2]]
        agg_obs = data.Twat_obs_agg[data.I_inf[has_count, 2]]

        mean_mod = np.mean(agg_mod)
        TSS_mod = np.sum((agg_mod - mean_mod) ** 2)
        covar_mod = np.sum((agg_mod - mean_mod) * (agg_obs - data.mean_obs))

        std_mod = np.sqrt(TSS_mod / np.float64(valid_n_dat - 1))
        covar_mod /= np.float64(valid_n_dat - 1)

        if data.std_obs == 0 or std_mod == 0:
            print("Warning: KGE undefined because std is zero. Returning -999.0")
            return -999.0

        ind = 1.0 - np.sqrt((std_mod / data.std_obs - 1.0)**2 + (mean_mod / data.mean_obs - 1.0)**2 + (covar_mod / (std_mod * data.std_obs) - 1.0)**2)

    elif data.fun_obj == 'RMS':
        if valid_n_dat > 0:
            agg_mod = data.Twat_mod_agg[data.I_inf[has_count, 2]]
            agg_obs = data.Twat_obs_agg[data.I_inf[has_count, 2]]
            TSS = np.sum((agg_mod - agg_obs) ** 2)
            ind = -np.sqrt(TSS / np.float64(valid_n_dat))
        else:
            ind = -999.0

    else:
        print("Errore nella scelta della f. obiettivo")

    return np.float64(ind)
