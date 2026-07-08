"""
High-level ODE integration and simulation orchestration for air2stream.

This module provides the main entry points for running the air2stream model,
including handling missing data segments (gap-tolerant mode) and delegating
the heavy numeric lifting to the Numba-compiled functions.
"""

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
        if data.version not in [3, 5] and (data.Q[i] == -999.0 or data.Q[i] <= 0.0):
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
    """
    Execute a single time step of the ODE integration (deprecated).

    Parameters
    ----------
    data : CommonData
        The common data object.
    j : int
        The time step index.
    p : ndarray
        The parameters array.

    Raises
    ------
    NotImplementedError
        Always raises, as `_step` has been removed and replaced by Numba-compiled functions.
    """
    raise NotImplementedError("_step has been removed for performance reasons. Use _run_integration instead.")

def _get_RK_func(version, Qmedia, p):
    """
    Construct a fast derivative function closure for the chosen model version.

    This returns a pure Python function tailored to the specific model version
    to avoid conditional checks inside the integration loop.
    Note: This is mostly unused in the fast Numba path but retained for legacy/reference.

    Parameters
    ----------
    version : int
        The model version (3, 4, 5, 7, or 8).
    Qmedia : float
        Mean discharge used for normalization.
    p : ndarray
        Array of model parameters (1-indexed mapping to p1..p8).

    Returns
    -------
    callable
        A function `RK(Ta, QQ, Tw, time)` that computes the derivative dT_w/dt.
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]

    if version == 3:
        def RK(Ta, QQ, Tw, time):
            """Version 3: Uses air temperature and water temperature (no discharge, no seasonality)."""
            return p1 + p2 * Ta - p3 * Tw

    elif version == 5:
        def RK(Ta, QQ, Tw, time):
            """Version 5: Uses air temp, water temp, and seasonal cosine term (no discharge)."""
            return p1 + p2 * Ta - p3 * Tw + p6 * math.cos(2.0 * PI * (time - p7))

    elif version in [8, 7]:
        if version == 8:
            def RK(Ta, QQ, Tw, time):
                """Version 8: Full equation. Uses air temp, water temp, discharge, and seasonal term."""
                theta = QQ / Qmedia
                DD = theta ** p4
                return (p1 + p2 * Ta - p3 * Tw + theta * (p5 + p6 * math.cos(2.0 * PI * (time - p7)) - p8 * Tw)) / DD
        else:
            def RK(Ta, QQ, Tw, time):
                """Version 7: Similar to v8, but discharge only affects the numerator terms, not the denominator thermal volume."""
                theta = QQ / Qmedia
                return p1 + p2 * Ta - p3 * Tw + theta * (p5 + p6 * math.cos(2.0 * PI * (time - p7)) - p8 * Tw)

    elif version == 4:
        def RK(Ta, QQ, Tw, time):
            """Version 4: Uses air temp, water temp, and discharge (no seasonal term)."""
            DD = (QQ / Qmedia) ** p4
            return (p1 + p2 * Ta - p3 * Tw) / DD

    else:
        def RK(Ta, QQ, Tw, time):
            """Fallback: Returns zero derivative."""
            return 0.0

    return RK

def _run_integration(data: CommonData, segments, p):
    """
    Orchestrate the core numerical integration loop over specified segments.

    Delegates the actual computation to the Numba-compiled `fast_run_integration`
    function to maximize performance.

    Parameters
    ----------
    data : CommonData
        The common data object containing forcing data (Tair, Q), time arrays,
        and settings. The `Twat_mod` array will be mutated in-place.
    segments : list of tuple
        A list of (start_idx, end_idx) tuples defining contiguous blocks of valid data.
    p : ndarray
        Array containing the 8 model parameters (1-indexed: p[1] to p[8]).

    Returns
    -------
    None
    """
    from .model_numba import fast_run_integration

    mod_num = data.mod_num
    mod_num_idx = -1
    if mod_num == 'CRN': mod_num_idx = 0
    elif mod_num == 'RK2': mod_num_idx = 1
    elif mod_num == 'RK4': mod_num_idx = 2
    elif mod_num == 'EUL': mod_num_idx = 3
    else: raise ValueError(f"Unknown mod_num {mod_num}")

    segments_arr = np.array(segments, dtype=np.int32)

    # Numba will mutate Twat_mod in place
    fast_run_integration(
        data.Tair, data.Q, data.tt, data.Twat_mod, data.Tice_cover, data.Qmedia,
        data.version, mod_num_idx, segments_arr,
        p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
    )

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
    from .model_numba import fast_funcobj

    fun_obj_type = -1
    if data.fun_obj == 'NSE': fun_obj_type = 0
    elif data.fun_obj == 'KGE': fun_obj_type = 1
    elif data.fun_obj == 'RMS': fun_obj_type = 2
    else: print("Errore nella scelta della f. obiettivo")

    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=np.bool_)

    ind, Twat_mod_agg, current_nse, current_r2, current_mae = fast_funcobj(
        data.n_dat, data.n_tot, data.I_inf, data.I_pos, data.Twat_mod, data.Twat_obs_agg,
        eval_mask, fun_obj_type, data.mean_obs, data.TSS_obs, data.std_obs
    )

    data.Twat_mod_agg = Twat_mod_agg
    data.current_nse = current_nse
    data.current_r2 = current_r2
    data.current_mae = current_mae

    # Handle print warning consistency from original Python port
    if ind == -999.0 and fun_obj_type == 1:
        pass # The python version used to print "Warning: KGE undefined"

    return np.float64(ind)
