import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt

# Constants from Fortran module
N_PAR: int = 8
PI: np.float64 = np.float64(math.pi) # ACOS(0.d0)*2.d0 is math.pi
TTT: np.float64 = np.float64(1.0 / 365.0)

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

    # Differential Evolution Fields
    maxiter: int = 1000
    n_jobs: int = -1
    polish: bool = True
    seed: Optional[int] = None

    # Gap-tolerant mode fields
    gap_tolerant: bool = False
    Qmedia_user: Optional[float] = None
    warmup_drop_days: int = 15
    min_segment_days: int = 30
    segments: Optional[list] = None
    sensitivity_analysis: bool = False
    sensitivity_perturbations: Optional[list] = None

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
    wmin: np.float64 = np.float64(0.0)
    wmax: np.float64 = np.float64(0.0)

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
