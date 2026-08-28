"""
Configuration and central data structures for pyair2stream.

This module defines the primary data transfer object (`CommonData`) that
carries model configuration, physical boundaries, forcing arrays, and state
variables throughout the calibration and simulation processes.
"""

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt

# Constants from Fortran module
N_PAR: int = 8
PI: np.float64 = np.float64(math.pi) # ACOS(0.d0)*2.d0 is math.pi
TTT: np.float64 = np.float64(1.0 / 365.0)
MISSING_DATA_SENTINEL: np.float64 = np.float64(-999.0)

@dataclass
class CommonData:
    """
    Data class representing the `commondata` module in AIR2STREAM_MODULES.f90.
    """
    # Scalars - Integers
    n_Q: int = 0
    n_tot: int = 0
    n_dat: int = 0
    version: int = 0
    qty: int = 0
    n_run: int = 0
    n_particles: int = 0

    # Gap-tolerant mode fields
    gap_tolerant: bool = False
    Qmedia_user: Optional[float] = None
    calib_theta_min: Optional[float] = None
    calib_theta_max: Optional[float] = None
    warmup_drop_days: int = 15
    min_segment_days: int = 30
    segments: Optional[list] = None
    sensitivity_analysis: bool = False
    sensitivity_perturbations: Optional[list] = None
    # 'value' (default, unchanged behaviour): perturb by delta_pct% of the parameter's
    # own calibrated value. 'range': perturb by delta_pct% of (parmax - parmin) instead,
    # which is comparable across parameters and immune to the near-zero-value problem
    # but is not what earlier releases computed -- see docs/audit/06_diagnostics_and_plots.md,
    # Defect E.
    sensitivity_perturbation_mode: str = 'value'
    mcmc_walkers: int = 32
    mcmc_steps: int = 2000

    # Set True by read_Tseries only once a validation period has been fully and
    # successfully loaded (file present, >= 1 year, valid gap-tolerant segments if
    # applicable). main.forward() must gate the validation block on this flag, not
    # on data.n_tot -- a too-short validation period returns before data.n_tot is
    # overwritten, so it stays at the calibration value (see
    # docs/audit/05_cli_and_io_correctness.md, Defect B).
    validation_available: bool = False

    # Forward options
    forward_options: Optional[dict] = None

    # Uncertainty-quantification options parsed from `uncertainty_options:` in the
    # config (noise_model, ar1_rho, prediction_interval, save_ensemble,
    # strict_convergence, burnin_fraction). Declared explicitly (rather than set
    # only by assignment in io.py) so it is visible to type checkers and callers
    # don't need `getattr(data, 'uncertainty_options', {})` (docs/audit/07, Defect G).
    uncertainty_options: Optional[dict] = None

    # Top-level calibration seed (docs/audit/07_reproducibility_and_provenance.md,
    # 7.1). None reproduces the previous unseeded behaviour.
    random_seed: Optional[int] = None

    # Cross Validation
    cross_validation: Optional['CVConfig'] = None  # Expected to be Optional[CVConfig]

    # Scalars - Floats (np.float64 to enforce 64-bit precision)
    Qmedia: np.float64 = np.float64(0.0)
    theta_j: np.float64 = np.float64(0.0)
    theta_j1: np.float64 = np.float64(0.0)
    DD_j: np.float64 = np.float64(0.0)
    DD_j1: np.float64 = np.float64(0.0)
    Tice_cover: np.float64 = np.float64(0.0)
    prc: np.float64 = np.float64(0.0)
    mean_obs: np.float64 = np.float64(0.0)
    TSS_obs: np.float64 = np.float64(0.0)
    std_obs: np.float64 = np.float64(0.0)
    mineff_index: np.float64 = np.float64(0.0)
    finalfit: np.float64 = np.float64(0.0)
    c1: np.float64 = np.float64(0.0)
    c2: np.float64 = np.float64(0.0)
    # Numerical-stability guard settings (see docs/audit/02_numerical_integration.md)
    max_plausible_twat: np.float64 = np.float64(60.0)
    stability_error_fraction: np.float64 = np.float64(0.10)

    # Opt-in escape hatch for a legitimate zero-flow (or negative, e.g. sensor fault)
    # discharge day in a version that evaluates theta = Q/Qmedia (4, 7, 8). None
    # (the default) means `check_nonpositive_discharge` raises instead -- see
    # docs/audit/10_zero_discharge_handling.md. When set (a small positive float,
    # e.g. 1e-6), `theta` is clamped to at least this value before `theta ** a4` is
    # evaluated in every integrator, so the same non-gap-tolerant run can proceed
    # instead of hitting a `ZeroDivisionError` (a4 > 0) or a silent `inf` (a4 < 0).
    min_theta_floor: Optional[float] = None

    # Declared calendar for the forcing series: 'standard' (real Gregorian dates,
    # the only calendar the daily-continuity check validates against), 'noleap'
    # (365 days every year, no Feb 29), or '360_day' (12 uniform 30-day months).
    # See docs/audit/05_cli_and_io_correctness.md, Defect D.
    calendar: str = 'standard'
    wmin: np.float64 = np.float64(0.0)
    wmax: np.float64 = np.float64(0.0)

    # Input data file paths, stashed by `read_calibration` for `read_Tseries` to
    # consume. Declared explicitly rather than set only by assignment
    # (docs/audit/07_reproducibility_and_provenance.md, Defect G).
    _input_data_path_cal: Optional[str] = None
    _input_data_path_val: Optional[str] = None

    # Raw (pre-warm-up-padding) row count of the most recently loaded series, used by
    # `compute_qmedia` to report the fraction of missing discharge. Declared
    # explicitly rather than set only by assignment (docs/audit/07, Defect G).
    _n_tot_raw: Optional[int] = None

    # Set True once `detect_segments` has printed its one-time fragmentation
    # diagnostics for the current data load, so repeated calls inside the
    # optimizer hot loop don't spam the same warning. Declared explicitly rather
    # than tested via `hasattr` (docs/audit/07_reproducibility_and_provenance.md,
    # Defect G).
    _segment_warned: bool = False

    # Strings
    folder: str = ""
    name: str = ""
    air_station: str = ""
    water_station: str = ""
    station: str = ""
    model: str = ""
    runmode: str = ""
    series: str = ""
    unit: str = ""
    time_res: str = ""
    fun_obj: str = ""
    mod_num: str = ""

    # Allocatable arrays - Integer
    I_pos: Optional[npt.NDArray[np.int32]] = None
    I_inf: Optional[npt.NDArray[np.int32]] = None
    date: Optional[npt.NDArray[np.int32]] = None

    # Tracking metrics across evaluations
    current_nse: np.float64 = np.float64(-999.0)
    current_r2: np.float64 = np.float64(-999.0)
    current_mae: np.float64 = np.float64(-999.0)

    # Allocatable arrays - Float (np.float64)
    tt: Optional[npt.NDArray[np.float64]] = None
    Tair: Optional[npt.NDArray[np.float64]] = None
    Twat_obs_agg: Optional[npt.NDArray[np.float64]] = None
    Twat_obs: Optional[npt.NDArray[np.float64]] = None
    Q: Optional[npt.NDArray[np.float64]] = None
    Twat_mod: Optional[npt.NDArray[np.float64]] = None
    Twat_mod_agg: Optional[npt.NDArray[np.float64]] = None
    parmin: Optional[npt.NDArray[np.float64]] = None
    parmax: Optional[npt.NDArray[np.float64]] = None
    par: Optional[npt.NDArray[np.float64]] = None
    par_best: Optional[npt.NDArray[np.float64]] = None

    # Allocatable arrays - Logical (Bool)
    flag_par: Optional[npt.NDArray[np.bool_]] = None
    eval_mask: Optional[npt.NDArray[np.bool_]] = None
    doy_climatology: Optional[npt.NDArray[np.float64]] = None
