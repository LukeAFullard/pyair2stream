"""
Input/output operations for pyair2stream.

This module handles parsing user YAML configurations, allocating internal arrays,
and reading/validating the input CSV time series data (forcing and observations).
"""

import copy
import os
import json
import re
import yaml
import numpy as np
import pandas as pd
from typing import Tuple

from .config import (
    CommonData, DEFAULT_NOISE_MODEL, DEFAULT_LIKELIHOOD, VALID_LIKELIHOODS, DEFAULT_RHO_TIMESCALE, VALID_RHO_TIMESCALES, ACTIVE_PARAMS, VALID_VERSIONS, VALID_RUN_MODES, VALID_INTEGRATORS,
    VALID_OBJECTIVES,
)
from .model import prepare_evaluation, check_nonpositive_discharge, STABILITY_MAX_GROWTH
from .data_checks import CALENDARS, NO_360_DAY, check_table, PLAUSIBLE_RANGES as _PLAUSIBLE_RANGES


# Day of the year on which each month starts, minus one, in a year without 29 February.
NOLEAP_MONTH_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def calendar_day_index(data: CommonData, i: int) -> int:
    """0-based day of the year of row `i` for the noleap calendar, from the seasonal phase
    `data.tt` that `read_Tseries` set (tt = day of the year / 365)."""
    return int(round(data.tt[i] * 365)) - 1


def _check_choice(name: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid {name} {value!r}. Must be one of: {', '.join(map(str, allowed))}.")


# Kept here for backward compatibility; the checks themselves are in data_checks.
PLAUSIBLE_RANGES = _PLAUSIBLE_RANGES


def _read_8_values(values, name: str) -> np.ndarray:
    """Return a list of exactly 8 numbers from the config as a float array."""
    if not isinstance(values, (list, tuple)) or len(values) != 8:
        raise ValueError(
            f"{name} must be a list of exactly 8 numbers (a1..a8), got {values!r}. "
            "Give a value for every parameter; the ones your model version does not "
            "use are set to zero automatically."
        )
    return np.array([np.float64(x) for x in values], dtype=np.float64)

def _season_months(values):
    """`cross_validation.season_months`: None, or a list of calendar months (1-12)."""
    if values is None:
        return None
    months = [int(m) for m in (values if isinstance(values, (list, tuple)) else [values])]
    if not months or any(m < 1 or m > 12 for m in months):
        raise ValueError(f"cross_validation.season_months must list calendar months 1-12, got {values!r}.")
    return sorted(set(months))


class SettingsFileNotFoundError(FileNotFoundError):
    """The settings (YAML) file does not exist."""


def read_calibration(config_file='config.yaml') -> CommonData:
    """
    Reads the calibration configuration from a YAML file (or a dict with the same
    keys) and initializes the CommonData.
    """
    data = CommonData()

    if isinstance(config_file, dict):
        config = copy.deepcopy(config_file)
    else:
        if not os.path.exists(config_file):
            raise SettingsFileNotFoundError(f"Configuration file not found: {config_file}\nPlease refer to USER_GUIDE.md for instructions on how to create a configuration file.")
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
    if data.runmode == 'DE-CV-MCMC':
        raise ValueError(
            "run_mode 'DE-CV-MCMC' has been removed: it gave the same intervals as 'DE-MCMC' "
            "(the cross-validation only set the sampler's starting points). Use run_mode: "
            "'DE-MCMC', and cross-validation with run_mode 'DE' to see how stable the "
            "parameters are from year to year (USER_GUIDE §13)."
        )
    _check_choice('run_mode', data.runmode, VALID_RUN_MODES)
    data.prc = np.float64(config.get('prc', 1.0))
    if not (0.0 < data.prc <= 1.0):
        raise ValueError(
            f"prc must be a fraction above 0 and at most 1 (the share of days in a week or "
            f"month that must have an observation), got {data.prc}."
        )
    # Top-level calibration seed: threaded through to whichever optimizer `run_optimizer` dispatches to.
    # Without it, two runs of the same config produce different `par_best` (DE's
    # own global-state RNG, PSO/LATHYP's global `np.random`) with no way to
    # reproduce a published result.
    random_seed = config.get('random_seed', None)
    data.random_seed = int(random_seed) if random_seed is not None else None
    data.max_plausible_twat = np.float64(config.get('max_plausible_twat', 60.0))
    data.stability_error_fraction = np.float64(config.get('stability_error_fraction', 0.10))
    data.stability_max_growth = np.float64(config.get('stability_max_growth', STABILITY_MAX_GROWTH))
    if not data.stability_max_growth >= 1.0:
        raise ValueError(
            f"stability_max_growth must be at least 1 (the most that a difference between two "
            f"simulations may grow over a stretch of days), got {data.stability_max_growth}."
        )

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
    if data.calendar == '360_day':
        raise ValueError(NO_360_DAY)
    if data.calendar not in CALENDARS:
        raise ValueError(
            f"Invalid calendar '{data.calendar}'. Must be one of: 'standard', 'noleap'. "
            "Climate-model output without leap days must declare calendar: 'noleap' rather "
            "than being padded with invented dates -- see USER_GUIDE.md §5."
        )

    # Paths mapping
    paths = config.get('paths') or {}

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

    data.forward_options = config.get('forward_options') or {}

    # Parse uncertainty_options
    uncertainty_options = config.get('uncertainty_options') or {}
    noise_model = uncertainty_options.get('noise_model', DEFAULT_NOISE_MODEL)
    ar1_rho = uncertainty_options.get('ar1_rho', None)

    if noise_model not in ["iid", "ar1"]:
        raise ValueError(f"Invalid noise_model: '{noise_model}'. Must be 'iid' or 'ar1'.")
    likelihood = uncertainty_options.get('likelihood', DEFAULT_LIKELIHOOD)
    _check_choice('uncertainty_options.likelihood', likelihood, VALID_LIKELIHOODS)
    rho_timescale = uncertainty_options.get('rho_timescale', DEFAULT_RHO_TIMESCALE)
    _check_choice('uncertainty_options.rho_timescale', rho_timescale, VALID_RHO_TIMESCALES)

    if ar1_rho is not None:
        if not (-1.0 < float(ar1_rho) < 1.0):
            raise ValueError(f"ar1_rho must be strictly between -1.0 and 1.0, got {ar1_rho}")
        ar1_rho = float(ar1_rho)

    # Central level (%) of every range the package reports: predictions (daily bands, the
    # cross-validation check of yearly statistics) and parameters (MCMC summary, jackknife).
    prediction_interval = float(uncertainty_options.get('prediction_interval', 90.0))
    if not (0.0 < prediction_interval < 100.0):
        raise ValueError(f"prediction_interval must be strictly between 0 and 100, got {prediction_interval}")
    parameter_interval = float(uncertainty_options.get('parameter_interval', 90.0))
    if not (0.0 < parameter_interval < 100.0):
        raise ValueError(f"parameter_interval must be strictly between 0 and 100, got {parameter_interval}")

    save_ensemble = bool(uncertainty_options.get('save_ensemble', False))
    strict_convergence = bool(uncertainty_options.get('strict_convergence', True))

    # Burn-in override for DE-MCMC. Left unset (None), burn-in defaults to
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
        # Whether the settings name the error model: a FORWARD run otherwise takes the chain's.
        "noise_model_set": 'noise_model' in uncertainty_options,
        "likelihood": likelihood,
        "rho_timescale": rho_timescale,
        "ar1_rho": ar1_rho,
        "prediction_interval": prediction_interval,
        "parameter_interval": parameter_interval,
        "save_ensemble": save_ensemble,
        "strict_convergence": strict_convergence,
        "burnin_fraction": burnin_fraction,
        "on_divergent_draw": on_divergent_draw,
        "max_divergent_fraction": max_divergent_fraction,
    }

    cv_config_dict = config.get('cross_validation') or {}
    if cv_config_dict and cv_config_dict.get('enabled', False):
        from .cross_validation import CVConfig
        data.cross_validation = CVConfig(
            unit=cv_config_dict.get('unit', 'year'),
            n_years_per_fold=int(cv_config_dict.get('n_years_per_fold', 1)),
            water_year_start_month=int(cv_config_dict.get('water_year_start_month', 1)),
            min_train_years=int(cv_config_dict.get('min_train_years', 1)),
            skip_first_year=bool(cv_config_dict.get('skip_first_year', True)),
            min_valid_obs=int(cv_config_dict.get('min_valid_obs', 10)),
            optimizer_overrides=cv_config_dict.get('optimizer_overrides', None),
            threshold=(float(cv_config_dict['threshold'])
                       if cv_config_dict.get('threshold') is not None else None),
            season_months=_season_months(cv_config_dict.get('season_months')),
        )

    opt_config = config.get('optimization') or {}
    data.n_run = int(opt_config.get('n_run', opt_config.get('n_runs', 100)))
    # DE stops once its population's scores agree to within `tol` (relative). SciPy's default
    # (0.01) stopped some calibrations a third of the way through, on a worse fit.
    data.de_tol = float(opt_config.get('tol', 1e-3))
    if not data.de_tol >= 0:
        raise ValueError(f"optimization.tol must be zero or positive, got {data.de_tol}")
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
    elif data.runmode == 'DE-MCMC':
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
                doy = calendar_day_index(data, i)
            doy_sums[doy] += data.Twat_obs[i]
            doy_counts[doy] += 1

    if np.sum(doy_counts) == 0:
        raise ValueError(
            "No T_water observations in this file. Gap-tolerant mode starts each segment from "
            "the observed water temperature, or from its day-of-year average in this file, so "
            "it needs some observations (docs/METHODS.md §10)."
        )

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


def precheck_validation(data: CommonData) -> None:
    """
    Check the validation file before any calibration, with exactly the checks its
    later load makes (`data_checks.check_table`), so that a problem in it stops the
    run at once instead of after a calibration that may take hours. Call it after the
    calibration file is loaded. It also warns when measured days of the validation
    file are measured days of the calibration file too, and keeps the checked table
    for the later load, so the file is not read or reported twice.
    """
    filename = getattr(data, '_input_data_path_val', None)
    if not filename:
        return
    if getattr(data, 'cross_validation', None) and data.runmode in ('PSO', 'DE', 'LATHYP'):
        print("Note: paths.validation_data is not used by a cross-validation run; each held-out year is its test.")
        return
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing validation data file: {filename}")
    checked = check_table(pd.read_csv(filename), filename, period='validation', version=data.version,
                          gap_tolerant=data.gap_tolerant, calendar=data.calendar,
                          min_theta_floor=data.min_theta_floor)
    checked.raise_first_error()
    checked.print_warnings()

    if data.runmode != 'FORWARD' and checked.dates is not None and data.date is not None and data.n_tot > 365:
        real = slice(365, data.n_tot)
        measured = data.Twat_obs[real] != -999.0
        cal_days = pd.to_datetime(pd.DataFrame({'year': data.date[real, 0], 'month': data.date[real, 1],
                                                'day': data.date[real, 2]})[measured], errors='coerce')
        shared = checked.dates[checked.df['T_water'].notna() & checked.dates.isin(cal_days)]
        if len(shared):
            print(f"Warning: {len(shared)} measured day(s) of the validation file {filename} (first: "
                  f"{shared.iloc[0].date()}) are also measured days of the calibration file "
                  f"{data._input_data_path_cal}. On those days the validation score does not test the model on "
                  "data it was not fitted to. Use separate years for calibration and validation.")
    data._prechecked_validation = (filename, checked)


def take_prechecked(data: CommonData, filename) -> object:
    """The table `precheck_validation` checked for `filename`, once; None otherwise."""
    pre = getattr(data, '_prechecked_validation', None)
    if pre is not None and pre[0] == filename:
        data._prechecked_validation = None
        return pre[1]
    return None


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

    # Every check of the file's content (columns, dates, values, gaps, length) is in
    # data_checks.check_table, shared with the pre-analysis report. A validation file
    # that main() already checked before calibrating is not read or reported twice.
    checked = take_prechecked(data, filename) if p == 'v' else None
    if checked is None:
        check_period = 'validation' if p == 'v' else ('scenario' if data.runmode == 'FORWARD' else 'calibration')
        checked = check_table(pd.read_csv(filename), filename, period=check_period, version=data.version,
                              gap_tolerant=data.gap_tolerant, calendar=data.calendar,
                              min_theta_floor=data.min_theta_floor)
        checked.raise_first_error()
        checked.print_warnings()
    df, date_col = checked.df, checked.dates

    # Missing values (a blank cell or -999 in the file) are -999.0 from here on.
    Tair = df['T_air'].fillna(-999.0).to_numpy(dtype=np.float64)
    Q = df['Discharge'].fillna(-999.0).to_numpy(dtype=np.float64)
    Twat_obs = df['T_water'].fillna(-999.0).to_numpy(dtype=np.float64)

    n_tot_raw = len(df)

    if p == 'v' and n_tot_raw < 365:
        print('Validation period < 1 year --> validation is skipped')
        return

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
        # noleap: real dates without 29 February. The day of the year follows from each
        # row's month and day, with the month lengths of a year that is never a leap year.
        doy = NOLEAP_MONTH_START[data.date[365:n_tot, 1] - 1] + data.date[365:n_tot, 2]
        data.tt[365:n_tot] = doy / 365.0

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
