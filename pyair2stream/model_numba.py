"""
Core numerical methods for the air2stream model, compiled with Numba for performance.

This module contains the heavy-lifting functions for the objective calculation
and the ODE integrators. They are decorated with @njit and thus must remain
free of Python objects and dynamic typing.
"""

import numpy as np
import math
from numba import njit

PI = np.pi
TTT = 1.0 / 365.0

@njit
def fast_funcobj(n_dat, n_tot, I_inf, I_pos, Twat_mod, Twat_obs_agg, eval_mask, fun_obj_type, mean_obs, TSS_obs, std_obs):
    """
    Compute the calibration objective function from simulated vs
    observed water temperature.

    Aggregates the daily simulated series onto the observation time
    resolution defined by I_inf/I_pos, then evaluates the objective
    function selected by fun_obj_type against Twat_obs_agg.

    Parameters
    ----------
    n_dat : int
        Number of data points (after aggregation) used for evaluation.
    n_tot : int
        Total number of simulation days.
    I_inf : ndarray
        Integer array of shape (n_dat, 3) defining aggregation windows.
    I_pos : ndarray
        Integer array mapping indices to actual valid observation days.
    Twat_mod : ndarray
        Array of daily simulated water temperatures.
    Twat_obs_agg : ndarray
        Array of observed water temperatures (aggregated).
    eval_mask : ndarray
        Boolean mask array indicating which days are valid for evaluation.
    fun_obj_type : int
        The objective function to calculate: 0 = NSE, 1 = KGE, 2 = RMS.
    mean_obs : float
        Mean of the aggregated observations.
    TSS_obs : float
        Total sum of squares for the aggregated observations.
    std_obs : float
        Standard deviation of the aggregated observations.

    Returns
    -------
    tuple
        A tuple containing:
        - The calculated objective value (NSE, KGE, or RMS).
        - Twat_mod_agg (ndarray): The aggregated simulated temperatures.
        - nse (float): Nash-Sutcliffe Efficiency.
        - r2 (float): Coefficient of Determination (R^2).
        - mae (float): Mean Absolute Error.

    Notes
    -----
    This function is @njit-compiled, so it must stay free of
    Python objects and dynamic typing.
    """

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
        return -999.0, Twat_mod_agg, -999.0, -999.0, -999.0

    # Calculate NSE, R2, MAE for all evaluations
    sum_mod = 0.0
    sum_abs_err = 0.0
    TSS = 0.0
    for k in range(valid_n_dat):
        sum_mod += agg_mod_valid[k]
        sum_abs_err += abs(agg_mod_valid[k] - agg_obs_valid[k])
        TSS += (agg_mod_valid[k] - agg_obs_valid[k]) ** 2

    mean_mod = sum_mod / valid_n_dat
    mae = sum_abs_err / valid_n_dat

    if TSS_obs == 0.0:
        if TSS == 0.0:
            nse = 1.0
        else:
            nse = -1e30
    else:
        nse = 1.0 - TSS / TSS_obs

    # R2
    TSS_mod = 0.0
    covar_mod = 0.0
    for k in range(valid_n_dat):
        TSS_mod += (agg_mod_valid[k] - mean_mod) ** 2
        covar_mod += (agg_mod_valid[k] - mean_mod) * (agg_obs_valid[k] - mean_obs)

    if TSS_mod == 0.0 or TSS_obs == 0.0:
        r2 = -999.0
    else:
        r = covar_mod / np.sqrt(TSS_mod * TSS_obs)
        r2 = r ** 2

    if fun_obj_type == 0: # NSE
        return nse, Twat_mod_agg, nse, r2, mae

    elif fun_obj_type == 1: # KGE
        if valid_n_dat < 2:
            return -999.0, Twat_mod_agg, nse, r2, mae

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
            return -999.0, Twat_mod_agg, nse, r2, mae

        kge = 1.0 - np.sqrt((std_mod / std_obs - 1.0)**2 + (mean_mod / mean_obs - 1.0)**2 + (covar_mod / (std_mod * std_obs) - 1.0)**2)
        return kge, Twat_mod_agg, nse, r2, mae

    elif fun_obj_type == 2: # RMS
        TSS = 0.0
        for k in range(valid_n_dat):
            TSS += (agg_mod_valid[k] - agg_obs_valid[k]) ** 2
        rms = -np.sqrt(TSS / valid_n_dat)
        return rms, Twat_mod_agg, nse, r2, mae

    return -999.0, Twat_mod_agg, nse, r2, mae

@njit
def fast_rk_version(version, p1, p2, p3, p4, p5, p6, p7, p8, Ta, QQ, Tw, time, Qmedia):
    """
    Evaluate the right-hand side of the air2stream ODE for a specific model version.

    Parameters
    ----------
    version : int
        The model version (3, 4, 5, 7, or 8) determining the equation form.
    p1, p2, p3, p4, p5, p6, p7, p8 : float
        The eight model parameters. Unused parameters for a given version
        will be ignored.
    Ta : float
        Air temperature at the current time step.
    QQ : float
        River discharge at the current time step.
    Tw : float
        Water temperature at the current time step.
    time : float
        Current time represented as a fraction of the year (DOY / days_in_year).
    Qmedia : float
        Mean discharge used to normalize QQ (creates dimensionless term theta).

    Returns
    -------
    float
        The computed derivative (rate of change of water temperature) dT_w/dt.
    """
    if version == 3:
        return p1 + p2 * Ta - p3 * Tw
    elif version == 5:
        return p1 + p2 * Ta - p3 * Tw + p6 * math.cos(2.0 * PI * (time - p7))
    elif version == 8:
        theta = QQ / Qmedia
        DD = theta ** p4
        return (p1 + p2 * Ta - p3 * Tw + theta * (p5 + p6 * math.cos(2.0 * PI * (time - p7)) - p8 * Tw)) / DD
    elif version == 7:
        theta = QQ / Qmedia
        return p1 + p2 * Ta - p3 * Tw + theta * (p5 + p6 * math.cos(2.0 * PI * (time - p7)) - p8 * Tw)
    elif version == 4:
        DD = (QQ / Qmedia) ** p4
        return (p1 + p2 * Ta - p3 * Tw) / DD
    else:
        return 0.0

@njit
def fast_run_integration(Tair, Q, tt, Twat_mod, Tice_cover, Qmedia, version, mod_num_idx, segments, p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Execute the numerical integration of the air2stream ODE over given segments.

    Parameters
    ----------
    Tair : ndarray
        Array of daily air temperatures.
    Q : ndarray
        Array of daily river discharges.
    tt : ndarray
        Array of fractional times of the year corresponding to each day.
    Twat_mod : ndarray
        Array to store the simulated water temperatures. This array is
        updated in-place. Initial conditions should be set at segment start indices.
    Tice_cover : float
        Minimum allowable water temperature (ice cover threshold).
    Qmedia : float
        Mean historical discharge used for normalization.
    version : int
        The model version (3, 4, 5, 7, or 8) determining the governing ODE.
    mod_num_idx : int
        The integrator to use: 0 = Crank-Nicolson (CRN), 1 = Heun/RK2,
        2 = Runge-Kutta 4 (RK4), 3 = Explicit Euler (EUL).
    segments : ndarray
        Integer array of shape (N, 2) defining start and end indices of
        contiguous valid data segments to integrate over.
    p1, p2, p3, p4, p5, p6, p7, p8 : float
        The eight model parameters to use during integration.

    Returns
    -------
    None
        Results are written directly into the `Twat_mod` array.
    """

    if mod_num_idx == 0: # CRN
        if version in (8, 7, 4):
            theta = Q / Qmedia
            DD = theta ** p4
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
