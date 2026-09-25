"""
Input/output operations for pyair2stream.

This module handles parsing user YAML configurations, allocating internal arrays,
and reading/validating the input CSV time series data (forcing and observations).
"""

import os
import json
import re
import yaml
import numpy as np
import pandas as pd
from typing import Tuple

from .config import (
    CommonData, ACTIVE_PARAMS, VALID_VERSIONS, VALID_RUN_MODES, VALID_INTEGRATORS,
    VALID_OBJECTIVES,
)
from .model import prepare_evaluation, check_nonpositive_discharge


def _check_choice(name: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid {name} {value!r}. Must be one of: {', '.join(map(str, allowed))}.")


def _read_8_values(values, name: str) -> np.ndarray:
    """Return a list of exactly 8 numbers from the config as a float array."""
    if not isinstance(values, (list, tuple)) or len(values) != 8:
        raise ValueError(
            f"{name} must be a list of exactly 8 numbers (a1..a8), got {values!r}. "
            "Give a value for every parameter; the ones your model version does not "
            "use are set to zero automatically."
        )
    return np.array([np.float64(x) for x in values], dtype=np.float64)

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
    data.time_res = str(config.get('time_resolution', '1d'))
    # `aggregation()` understands exactly '1d' (daily), 'Nw' (N = 1-99 weeks) and
    # '1m' (calendar months). Like the Fortran, the monthly branch ignores N, so
    # e.g. '2m' would silently be scored as '1m' -- it is rejected instead.
    if not re.fullmatch(r'1d|\d{1,2}w|1m', data.time_res) or data.time_res in ('0w', '00w'):
        raise ValueError(
            f"Invalid time_resolution '{data.time_res}'. Must be '1d' (daily), "
            "'Nw' (N weeks, e.g. '1w', '2w') or '1m' (monthly)."
        )
    data.version = int(config.get('version', 8))
    _check_choice('version', data.version, VALID_VERSIONS)
    data.Tice_cover = np.float64(config.get('Tice_cover', 0.0))
    data.fun_obj = config.get('objective_function', 'NSE')
    _check_choice('objective_function', data.fun_obj, VALID_OBJECTIVES)
    # CRN is unconditionally stable; the explicit schemes (RK4/RK2/EUL) can diverge
    # silently on discharge different from the calibration record (USER_GUIDE §9.1).
    data.mod_num = config.get('integrator', 'CRN')
    _check_choice('integrator', data.mod_num, VALID_INTEGRATORS)
    if data.mod_num in ('RK4', 'RK2', 'EUL'):
        print(f"Note: integrator {data.mod_num} is kept to reproduce the original Fortran. With a "
              "one-day step it can be inaccurate even when stable (by up to about 1 degC for EUL); "
              "use CRN (the default) unless you need Fortran-identical results (USER_GUIDE §9.1).")
    data.runmode = config.get('run_mode', 'DE')
    _check_choice('run_mode', data.runmode, VALID_RUN_MODES)
    data.prc = np.float64(config.get('prc', 1.0))
    # Top-level calibration seed: threaded through to whichever optimizer `run_optimizer` dispatches to.
    # Without it, two runs of the same config produce different `par_best` (DE's
    # own global-state RNG, PSO/LATHYP's global `np.random`) with no way to
    # reproduce a published result.
    random_seed = config.get('random_seed', None)
    data.random_seed = int(random_seed) if random_seed is not None else None
    data.max_plausible_twat = np.float64(config.get('max_plausible_twat', 60.0))
    data.stability_error_fraction = np.float64(config.get('stability_error_fraction', 0.10))

    # Opt-in escape hatch for zero/negative discharge in versions 4/7/8 (which evaluate
    # theta = Q/Qmedia and theta**a4); see `min_theta_floor` on CommonData and
    # `model.check_nonpositive_discharge`. None (default) leaves the guard active.
    min_theta_floor = config.get('min_theta_floor', None)
    if min_theta_floor is not None:
        min_theta_floor = float(min_theta_floor)
        if min_theta_floor <= 0.0:
            raise ValueError(
                f"min_theta_floor must be a positive float if set, got {min_theta_floor}."
            )
    data.min_theta_floor = min_theta_floor

    data.calendar = config.get('calendar', 'standard')
    if data.calendar not in ('standard', 'noleap', '360_day'):
        raise ValueError(
            f"Invalid calendar '{data.calendar}'. Must be one of: 'standard', 'noleap', "
            "'360_day'. GCM output on a non-standard calendar (no leap days, or 12 "
            "uniform 30-day months) must declare it explicitly rather than being padded "
            "to fake Gregorian dates -- see USER_GUIDE.md §5."
        )

    # Paths mapping
    paths = config.get('paths', {})

    # Gap-tolerant mode configuration
    data.gap_tolerant = bool(config.get('gap_tolerant', False))
    qmedia_user = config.get('Qmedia')
    if qmedia_user is not None:
        data.Qmedia_user = float(qmedia_user)

    # `paths.calibration_metadata` pins Qmedia (and cross-checks version/integrator)
    # to the values a prior calibration run was fitted under (USER_GUIDE.md §6):
    # recomputing Qmedia from scenario discharge rescales theta and silently cancels
    # the discharge signal, which is fatal for scenario studies (abstraction,
    # naturalised flow, climate projection).
    calib_metadata_path = paths.get('calibration_metadata')
    calib_par_best = None
    if calib_metadata_path is not None:
        if not os.path.exists(calib_metadata_path):
            raise FileNotFoundError(f"calibration_metadata file not found: {calib_metadata_path}")
        with open(calib_metadata_path, 'r') as f:
            calib_metadata = json.load(f)

        meta_version = int(calib_metadata['version'])
        if meta_version != data.version:
            raise ValueError(
                f"calibration_metadata version ({meta_version}) does not match "
                f"the configured version ({data.version})."
            )
        meta_integrator = calib_metadata['integrator']
        if meta_integrator != data.mod_num:
            raise ValueError(
                f"calibration_metadata integrator ('{meta_integrator}') does not match "
                f"the configured integrator ('{data.mod_num}')."
            )
        meta_qmedia = float(calib_metadata['qmedia'])
        if qmedia_user is not None and abs(float(qmedia_user) - meta_qmedia) > 1e-9:
            raise ValueError(
                f"Both `Qmedia:` ({qmedia_user}) and `paths.calibration_metadata` "
                f"(qmedia={meta_qmedia}) were supplied and disagree. Provide only one, "
                f"or make sure they match."
            )
        data.Qmedia_user = meta_qmedia
        calib_par_best = calib_metadata.get('par_best')
        data.calib_theta_min = calib_metadata.get('theta_min')
        data.calib_theta_max = calib_metadata.get('theta_max')

    data.warmup_drop_days = int(config.get('warmup_drop_days', 15))
    data.min_segment_days = int(config.get('min_segment_days', 30))
    data.sensitivity_analysis = config.get('sensitivity_analysis', False)

    sens_pert = config.get('sensitivity_perturbations', [1.0])
    data.sensitivity_perturbations = [float(x) for x in sens_pert] if isinstance(sens_pert, list) else [float(sens_pert)]

    data.sensitivity_perturbation_mode = config.get('sensitivity_perturbation_mode', 'value')
    if data.sensitivity_perturbation_mode not in ('value', 'range'):
        raise ValueError(
            f"Invalid sensitivity_perturbation_mode: '{data.sensitivity_perturbation_mode}'. "
            "Must be 'value' or 'range'."
        )

    data.forward_options = config.get('forward_options', {})

    # Parse uncertainty_options
    uncertainty_options = config.get('uncertainty_options', {})
    noise_model = uncertainty_options.get('noise_model', 'iid')
    ar1_rho = uncertainty_options.get('ar1_rho', None)

    if noise_model not in ["iid", "ar1"]:
        raise ValueError(f"Invalid noise_model: '{noise_model}'. Must be 'iid' or 'ar1'.")

    if ar1_rho is not None:
        if not (-1.0 < float(ar1_rho) < 1.0):
            raise ValueError(f"ar1_rho must be strictly between -1.0 and 1.0, got {ar1_rho}")
        ar1_rho = float(ar1_rho)

    prediction_interval = float(uncertainty_options.get('prediction_interval', 90.0))
    if not (0.0 < prediction_interval < 100.0):
        raise ValueError(f"prediction_interval must be strictly between 0 and 100, got {prediction_interval}")

    save_ensemble = bool(uncertainty_options.get('save_ensemble', False))
    strict_convergence = bool(uncertainty_options.get('strict_convergence', True))

    # Burn-in override for DE-MCMC/DE-CV-MCMC. Left unset (None), burn-in defaults to
    # max(0.3*mcmc_steps, 5*max(tau)), where tau is the autocorrelation time.
    burnin_fraction = uncertainty_options.get('burnin_fraction', None)
    if burnin_fraction is not None:
        burnin_fraction = float(burnin_fraction)
        if not (0.0 < burnin_fraction < 1.0):
            raise ValueError(f"burnin_fraction must be strictly between 0 and 1, got {burnin_fraction}")

    # Per-draw divergence handling for the posterior/prediction-interval ensemble
    # loops (forward_mode's prediction-interval block, _run_mcmc_uncertainty's
    # envelope loop)
    # 'drop' (default): exclude a divergent draw from the ensemble and report the
    # count; 'raise': fail loudly on the first divergent draw instead.
    on_divergent_draw = uncertainty_options.get('on_divergent_draw', 'drop')
    if on_divergent_draw not in ('drop', 'raise'):
        raise ValueError(
            f"Invalid on_divergent_draw: '{on_divergent_draw}'. Must be 'drop' or 'raise'."
        )

    # If more than this fraction of draws are excluded as divergent, raise rather than
    # silently proceeding with a depleted ensemble -- mirrors the `stability_error_fraction`
    # pattern (model.warn_on_stability).
    max_divergent_fraction = float(uncertainty_options.get('max_divergent_fraction', 0.10))
    if not (0.0 < max_divergent_fraction <= 1.0):
        raise ValueError(
            f"max_divergent_fraction must be in (0, 1], got {max_divergent_fraction}"
        )

    data.uncertainty_options = {
        "noise_model": noise_model,
        "ar1_rho": ar1_rho,
        "prediction_interval": prediction_interval,
        "save_ensemble": save_ensemble,
        "strict_convergence": strict_convergence,
        "burnin_fraction": burnin_fraction,
        "on_divergent_draw": on_divergent_draw,
        "max_divergent_fraction": max_divergent_fraction,
    }

    cv_config_dict = config.get('cross_validation', {})
    if cv_config_dict and cv_config_dict.get('enabled', False):
        from .cross_validation import CVConfig
        data.cross_validation = CVConfig(
            unit=cv_config_dict.get('unit', 'year'),
            n_years_per_fold=int(cv_config_dict.get('n_years_per_fold', 1)),
            water_year_start_month=int(cv_config_dict.get('water_year_start_month', 1)),
            min_train_years=int(cv_config_dict.get('min_train_years', 1)),
            skip_first_year=bool(cv_config_dict.get('skip_first_year', True)),
            min_valid_obs=int(cv_config_dict.get('min_valid_obs', 10)),
            optimizer_overrides=cv_config_dict.get('optimizer_overrides', None)
        )

    opt_config = config.get('optimization', {})
    data.n_run = int(opt_config.get('n_run', opt_config.get('n_runs', 100)))
    # Accepted for compatibility with Fortran-style configs but not used: the
    # 0_*.csv history always records every evaluated parameter set.
    data.mineff_index = np.float64(config.get('mineff_index', 0.0))

    data.station = data.air_station
    if data.air_station != data.water_station:
        data.station = f"{data.air_station}_{data.water_station}"

    data.folder = paths.get('output_dir', os.path.join(data.name, f"output_{data.version}"))
    os.makedirs(data.folder, exist_ok=True)

    # Fortran module hardcodes n_par = 8
    n_par = 8

    data.par = np.zeros(n_par, dtype=np.float64)
    data.par_best = np.zeros(n_par, dtype=np.float64)
    data.parmin = np.zeros(n_par, dtype=np.float64)
    data.parmax = np.zeros(n_par, dtype=np.float64)
    data.flag_par = np.ones(n_par, dtype=np.bool_)

    if data.runmode == 'FORWARD':
        # Without parameters_forward, use the calibrated parameters recorded with the
        # calibration (no copying by hand).
        par_forward = config.get('parameters_forward')
        if par_forward is None and calib_par_best is not None:
            par_forward = calib_par_best
            print(f"Using the calibrated parameters recorded in {calib_metadata_path}.")
        if par_forward is None:
            raise ValueError("FORWARD mode needs parameters: give parameters_forward (8 numbers), or "
                             "paths.calibration_metadata pointing at a calibration's "
                             "calibration_metadata.json to use its calibrated parameters.")
        data.par[:] = _read_8_values(par_forward, 'parameters_forward')
    elif data.runmode == 'PSO':
        data.n_particles = int(opt_config.get('n_particles', 50))
        data.c1 = np.float64(opt_config.get('c1', 2.0))
        data.c2 = np.float64(opt_config.get('c2', 2.0))
        data.wmax = np.float64(opt_config.get('wmax', 0.9))
        data.wmin = np.float64(opt_config.get('wmin', 0.4))
    elif data.runmode == 'DE':
        data.n_particles = int(opt_config.get('n_particles', 50)) # Using n_particles as population size
        # c1, c2, wmax, wmin not used for DE
    elif data.runmode in ['DE-MCMC', 'DE-CV-MCMC']:
        data.n_particles = int(opt_config.get('n_particles', 50)) # Using n_particles as population size for initial DE
        data.mcmc_walkers = int(opt_config.get('mcmc_walkers', 32))
        data.mcmc_steps = int(opt_config.get('mcmc_steps', 20000))  # maximum; stops earlier once converged

    # Bounds must list all 8 parameters when given. The optimizers refuse to run
    # if no parameter is free (e.g. bounds omitted), rather than "calibrating" zeros.
    bounds = config.get('parameter_bounds') or {}
    if bounds:
        data.parmin[:] = _read_8_values(bounds.get('min'), 'parameter_bounds.min')
        data.parmax[:] = _read_8_values(bounds.get('max'), 'parameter_bounds.max')
        bad = [j + 1 for j in range(n_par) if data.parmin[j] > data.parmax[j]]
        if bad:
            raise ValueError(f"parameter_bounds: min > max for parameter(s) a{bad}.")

    # Parameters the chosen version does not use are fixed at zero (as in the
    # Fortran). This includes `parameters_forward`: the CRN integrator evaluates
    # the full 8-parameter equation and relies on unused parameters being zero,
    # so a non-zero value there would otherwise change the physics.
    inactive = [j for j in range(n_par) if j not in ACTIVE_PARAMS[data.version]]
    data.parmin[inactive] = 0.0
    data.parmax[inactive] = 0.0
    data.flag_par[inactive] = False
    if data.runmode == 'FORWARD':
        ignored = [j + 1 for j in inactive if data.par[j] != 0.0]
        if ignored:
            print(f"Warning: version {data.version} does not use parameter(s) a{ignored}; "
                  "their values in parameters_forward are ignored (set to 0).")
        data.par[inactive] = 0.0

    out_param_path = os.path.join(data.folder, 'parameters.txt')
    with open(out_param_path, 'w') as f:
        f.write(f"{n_par}   !number of parameters\n")
        f.write(" ".join(f"{x:.5f}" for x in data.parmin) + "\n")
        f.write(" ".join(f"{x:.5f}" for x in data.parmax) + "\n")

    # Store paths in data to pass to read_Tseries
    data._input_data_path_cal = paths.get('input_data', None)
    data._input_data_path_val = paths.get('validation_data', None)

    return data

def compute_qmedia(data: CommonData, verbose: bool = False) -> None:
    """
    Calculate Qmedia from the currently valid (unmasked) Q array.
    Recalculating this per fold ensures held-out data doesn't leak into ODE physics.
    """
    Q = data.Q[365:data.n_tot]
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
        if verbose:
            print(f"Using user-supplied Qmedia: {data.Qmedia:.5f} (computed was {computed_qmedia:.5f})")
    else:
        data.Qmedia = computed_qmedia

    if data.gap_tolerant:
        n_tot_raw = data._n_tot_raw if data._n_tot_raw is not None else data.n_tot - 365
        if data.Qmedia <= 0 and data.version not in [3, 5]:
            raise ValueError("Qmedia is zero or negative. Please supply Qmedia in the configuration file if the data is mostly empty.")
        if verbose and (data.n_Q / n_tot_raw) < 0.5 and data.Qmedia_user is None:
            print("Warning: More than 50% of Discharge values are missing. Consider supplying Qmedia_user in the configuration file.")
        if verbose and data.version in [3, 5]:
            print(f"Info: Q gaps are ignored because version {data.version} does not use Discharge in the ODE.")
        if verbose and data.Qmedia_user is not None and computed_qmedia > 0 and abs(computed_qmedia - data.Qmedia_user) / computed_qmedia > 0.3:
            print(f"Warning: Computed Qmedia ({computed_qmedia:.5f}) differs from user-supplied Qmedia ({data.Qmedia_user:.5f}) by more than 30%.")


def compute_doy_climatology(data: CommonData) -> None:
    """
    Calculate DOY climatology from the currently valid (unmasked) Twat_obs.
    Recalculating this per fold ensures held-out data doesn't leak into segment initial conditions.
    """
    data.doy_climatology = np.zeros(366, dtype=np.float64)
    doy_sums = np.zeros(366, dtype=np.float64)
    doy_counts = np.zeros(366, dtype=int)

    for i in range(365, data.n_tot):
        if data.Twat_obs[i] != -999.0:
            if data.calendar == 'standard':
                year = data.date[i, 0]
                month = data.date[i, 1]
                day = data.date[i, 2]
                doy = (pd.Timestamp(year, month, day) - pd.Timestamp(year, 1, 1)).days
            else:
                days_in_year = 365 if data.calendar == 'noleap' else 360
                doy = (i - 365) % days_in_year
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


def read_Tseries(data: CommonData, p: str, recompute_qmedia: bool = True) -> None:
    """
    Reads the time series data from a CSV file and replicates the first year.
    Args:
        data: CommonData instance to update.
        p: 'c' for calibration, 'v' for validation.
        recompute_qmedia: whether to (re)compute Qmedia (and the DOY climatology)
            from the data just loaded. Pass False to freeze `data.Qmedia` at its
            current value, e.g. when loading the validation period so it is
            scored under the same normalisation the parameters were fitted with
.
    """
    # Invalidate segments/eval_mask from any previous load up front: they must never
    # be silently reused against data they were not built from.
    data.segments = None
    data.eval_mask = None

    if p == 'c':
        period = 'calibration'
        filename = data._input_data_path_cal
    else:
        period = 'validation'
        filename = data._input_data_path_val
        # Pessimistic default: only set True once validation has been fully and
        # successfully loaded below. Every early-return path in this function
        # must leave this False rather than overload data.n_tot, which stays at
        # the calibration value on some of those paths.
        data.validation_available = False

    if not filename and p == 'v':
        print('No validation_data given: validation is skipped.')
        data.n_tot = 0
        return
    if not filename or not os.path.exists(filename):
        # A validation file that was asked for but is missing (e.g. a typo in the
        # path) must not silently skip validation.
        raise FileNotFoundError(f"Missing {period} data file: {filename}")

    # Read the data using pandas. Expecting columns Date, T_air, T_water, Discharge
    df = pd.read_csv(filename)

    # Ensure Date is parsed
    if 'Date' not in df.columns:
        raise ValueError(f"Missing 'Date' column in {filename}")

    date_col = pd.to_datetime(df['Date'])

    # Calibration/validation records must start on 1 January (as in the Fortran).
    # FORWARD runs may start on any date; the warm-up block's seasonal phase is then
    # taken from the rows it copies (see the tt construction below).
    if (
        not data.gap_tolerant
        and data.runmode != 'FORWARD'
        and len(date_col) > 0
        and (date_col.iloc[0].month != 1 or date_col.iloc[0].day != 1)
    ):
        raise ValueError(f"The time series in {filename} must start on January 1st.")

    if data.calendar == 'standard':
        # Validate Daily Scale (no gaps). Only meaningful for the real Gregorian
        # calendar -- a genuine noleap/360_day series will never satisfy this by
        # construction (see the `calendar` branch below).
        expected_dates = pd.date_range(start=date_col.iloc[0], end=date_col.iloc[-1], freq='D')
        if len(date_col) != len(expected_dates) or not date_col.equals(pd.Series(expected_dates)):
            raise ValueError(f"The time series in {filename} must be continuous at a daily time scale with no missing dates. Fill missing rows with NaN or -999.")
    else:
        # Non-standard calendar: dates are informational only (used for the
        # output Year/Month/Day columns), not for physics. Do not validate them
        # against real Gregorian day-spacing -- only that they are in order, so
        # a genuinely mis-ordered file is still caught. tt is computed from row
        # position against the declared calendar below, not from these dates,
        # so a "padded" Gregorian date column cannot silently misalign it.
        if not date_col.is_monotonic_increasing:
            raise ValueError(
                f"The time series in {filename} must have non-decreasing dates."
            )

    # Validate completeness of T_air and Discharge
    if 'T_air' not in df.columns:
        raise ValueError(f"Missing 'T_air' column in {filename}")

    if 'Discharge' not in df.columns:
        if data.version in [3, 5]:
            df['Discharge'] = -999.0
        else:
            raise ValueError(f"Missing 'Discharge' column in {filename}")

    # A blank cell and the legacy -999 marker both mean "missing". Normalise to
    # NaN first so the completeness checks below catch both; otherwise a -999 in
    # T_air would be simulated as an air temperature of -999 degC.
    for col in ('T_air', 'T_water', 'Discharge'):
        if col in df.columns:
            df[col] = df[col].replace(-999.0, np.nan)

    if not data.gap_tolerant:
        if df['T_air'].isnull().any():
            raise ValueError(f"The series of observed air temperature in {filename} must be complete. It cannot have gaps or missing data.")
        if data.version not in [3, 5] and df['Discharge'].isnull().any():
            raise ValueError(f"The series of discharge in {filename} must be complete. It cannot have gaps or missing data.")

    # Handle missing data via -999.0
    Tair = df['T_air'].fillna(-999.0).astype(np.float64).values
    Q = df['Discharge'].fillna(-999.0).astype(np.float64).values
    Twat_obs = df.get('T_water', pd.Series(np.full(len(df), -999.0))).fillna(-999.0).astype(np.float64).values

    n_tot_raw = len(df)

    if p == 'v' and n_tot_raw < 365:
        print('Validation period < 1 year --> validation is skipped')
        return

    if p == 'c' and n_tot_raw < 365:
        # The warm-up block replicates the first 365 rows of the real record; a
        # shorter calibration series would otherwise fail later with an opaque
        # numpy broadcast error instead of an actionable message.
        raise ValueError(
            f"The {period} time series in {filename} has only {n_tot_raw} day(s); "
            "at least 365 are required (the warm-up block replicates the first "
            "year of data)."
        )

    n_year = int(np.ceil(n_tot_raw / 365.25))
    n_tot = n_tot_raw + 365

    data.n_tot = n_tot
    data.n_dat = n_tot  # Based on other parts of codebase, usually n_dat starts as n_tot. Aggregation changes n_dat.

    # Store n_tot_raw on data object for use in compute_qmedia
    data._n_tot_raw = n_tot_raw

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

    # Seasonal phase tt = day-of-year / days-in-year. Warm-up block: (j+1)/365,
    # as in the Fortran (re-aligned below if the record does not start on 1 Jan).
    for j in range(365):
        data.tt[j] = np.float64((j + 1) / 365.0)

    if data.calendar == 'standard':
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
    else:
        # noleap / 360_day: compute tt from ROW POSITION against the declared
        # calendar's fixed day-count, not from the (possibly padded/fake)
        # Gregorian dates in `Date` -- those would silently misalign the
        # seasonal cosine term against the true day of year. Row 365 (the first real day) restarts the annual cycle at day 1,
        # matching the warm-up block's own convention above.
        days_in_year = 365 if data.calendar == 'noleap' else 360
        for i in range(365, n_tot):
            doy = ((i - 365) % days_in_year) + 1
            data.tt[i] = np.float64(doy / float(days_in_year))

    # The warm-up block copies the first 365 rows of forcing, so it must also copy
    # their seasonal phase. The Fortran's (j+1)/365 is only correct for a record
    # that starts on 1 January of the standard calendar (kept in that case for
    # exact equivalence); otherwise the seasonal term would be out of phase with
    # the copied forcing and bias the first weeks of the simulation.
    starts_jan1 = date_col.iloc[0].month == 1 and date_col.iloc[0].day == 1
    if data.calendar != 'standard' or not starts_jan1:
        data.tt[0:365] = data.tt[365:730]

    # Initial Qmedia and DOY climatology calculations
    if recompute_qmedia:
        # FORWARD mode runs externally-supplied (already fitted) parameters, often
        # on different discharge than the run that fitted them (a naturalised-flow
        # or climate-projection scenario). theta = Q / Qmedia is the model's only
        # window onto discharge, so silently recomputing Qmedia here rescales theta
        # and cancels the scenario signal. Versions 3 and 5 pin
        # every discharge-related parameter to zero and never evaluate theta, so
        # the guard does not apply to them.
        if (
            data.runmode == 'FORWARD'
            and data.Qmedia_user is None
            and data.version not in (3, 5)
        ):
            raise ValueError(
                "FORWARD mode requires an explicit `Qmedia:` in the config (or a "
                "`calibration_metadata.json` via `paths.calibration_metadata`). "
                "Recomputing Qmedia from scenario discharge rescales theta and cancels "
                "the discharge signal. See USER_GUIDE.md §6 (Qmedia)."
            )
        compute_qmedia(data, verbose=True)
        if data.gap_tolerant and p == 'c':
            compute_doy_climatology(data)

        if (
            data.runmode == 'FORWARD'
            and p == 'c'
            and data.calib_theta_min is not None
            and data.calib_theta_max is not None
            and data.Qmedia > 0
        ):
            Q_period = data.Q[365:data.n_tot]
            valid_Q = (Q_period != -999.0) & (Q_period > 0.0)
            if np.any(valid_Q):
                theta = Q_period[valid_Q] / data.Qmedia
                frac_outside = float(np.mean((theta < data.calib_theta_min) | (theta > data.calib_theta_max)))
                if frac_outside > 0.01:
                    print(
                        f"Warning: {frac_outside:.1%} of days in this run have theta = Q/Qmedia "
                        f"outside the calibrated range [{data.calib_theta_min:.5f}, "
                        f"{data.calib_theta_max:.5f}]. The model is being extrapolated beyond the "
                        f"calibrated regime for these days."
                    )

    # Guard against non-positive discharge for theta-using versions (4/7/8) in the
    # non-gap-tolerant path -- applies identically to the calibration record and to
    # a FORWARD-mode scenario record (naturalised flow, climate projection): see
    # `check_nonpositive_discharge`. Runs once per data load rather than per
    # calibration evaluation, since it does not depend on the currently loaded a4.
    check_nonpositive_discharge(data)

    # Rebuild segments/eval_mask for the data just loaded: this must run
    # unconditionally, not only in gap-tolerant mode, so eval_mask is never left None
    # and never stale against data it was not built from. For the
    # validation period, a gap-tolerant record with no valid segments is not a hard
    # error -- it means validation is skipped, exactly like the other validation-only
    # early-returns above.
    if p == 'v':
        try:
            prepare_evaluation(data)
        except ValueError as e:
            print(f"Validation skipped: {e}")
            data.n_tot = 0
            return
        data.validation_available = True
    else:
        prepare_evaluation(data)
