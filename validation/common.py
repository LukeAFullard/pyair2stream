"""
Shared helpers for the validation suite: data locations, running pyair2stream
from a config dictionary, published values, and metrics.

Every check runs the package through its public entry points (read_calibration,
read_Tseries, the run modes), i.e. the same code a user runs.
"""

import contextlib
import io
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import openpyxl
import pandas as pd
import yaml

from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis, call_model, funcobj
from pyair2stream.optimization import DE_mode

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "switzerland")
VALIDATION = os.path.join(REPO, "validation")
WORK = os.path.join(VALIDATION, "work")          # scratch, not committed
FIGURES = os.path.join(VALIDATION, "figures")
RESULTS = os.path.join(VALIDATION, "results")

RIVERS = {"MAH_2369": "Mentue", "SIO_2011": "Rhône", "DAV_2327": "Dischmabach"}
WATER_ID = {"MAH_2369": 2369, "SIO_2011": 2011, "DAV_2327": 2327}
VERSIONS = (3, 4, 5, 7, 8)

# Parameter ranges used by the original model's authors (air2stream Switzerland example).
AUTHORS_BOUNDS = {"min": [-5, -5, -5, -1, 0, 0, 0, -1], "max": [15, 1.5, 5, 1, 20, 10, 1, 5]}

# DE settings used for every calibration in the suite (population 15 x 8 = 120).
DE_SETTINGS = {"n_run": 300, "n_particles": 15}


@dataclass
class Result:
    """Outcome of one validation check, rendered into REPORT.md by run_all.py."""
    code: str
    title: str
    question: str
    method: str
    criterion: str
    passed: Optional[bool] = None          # None = descriptive, no pass/fail
    summary: str = ""
    tables: List[Tuple[str, pd.DataFrame]] = field(default_factory=list)
    figures: List[Tuple[str, str]] = field(default_factory=list)   # (file name, caption)
    notes: List[str] = field(default_factory=list)
    seconds: float = 0.0
    figure_data: object = None             # raw data for run_all.py's figures


def river_csv(station: str, period: str) -> str:
    """Path of a Swiss data file; period is 'calibration' or 'validation'."""
    return os.path.join(DATA, f"{station}_{period}.csv")


def mean_discharge(csv: str) -> float:
    q = pd.read_csv(csv)["Discharge"]
    return float(q[q > 0].mean())


@contextlib.contextmanager
def quiet():
    """Silence the package's console output (it is chatty by design)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load(cfg: dict, name: str, period: str = "c"):
    """Write `cfg` to a YAML file under validation/work/<name>/ and load it as a user would."""
    folder = os.path.join(WORK, name)
    os.makedirs(folder, exist_ok=True)
    cfg = dict(cfg)
    cfg.setdefault("station_name", "S")
    cfg.setdefault("series", "c")
    cfg.setdefault("paths", {})
    cfg["paths"] = {**cfg["paths"], "output_dir": cfg["paths"].get("output_dir", os.path.join(folder, "out"))}
    path = os.path.join(folder, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    with quiet():
        data = read_calibration(path)
        read_Tseries(data, period)
        if (data.Twat_obs[365:] != -999.0).any():
            aggregation(data)
            statis(data)
    return data


def simulate(csv: str, version: int, par, integrator: str = "CRN", qmedia: float = None,
             objective: str = "RMS", name: str = "sim"):
    """FORWARD run of `par` on `csv`; returns the loaded data object after simulation."""
    cfg = {"version": version, "integrator": integrator, "run_mode": "FORWARD",
           "objective_function": objective, "parameters_forward": [float(x) for x in par],
           "Qmedia": float(qmedia if qmedia is not None else mean_discharge(csv)),
           "paths": {"input_data": csv}}
    data = load(cfg, name)
    with quiet():
        call_model(data)
        if (data.Twat_obs[365:] != -999.0).any():
            funcobj(data)
    return data


def calibrate(csv: str, version: int, objective: str = "NSE", integrator: str = "CRN",
              seed: int = 1, name: str = "cal", **extra):
    """DE calibration of `version` on `csv`; returns the data object (par_best, finalfit set)."""
    cfg = {"version": version, "integrator": integrator, "run_mode": "DE", "objective_function": objective,
           "random_seed": seed, "optimization": dict(DE_SETTINGS), "parameter_bounds": AUTHORS_BOUNDS,
           "paths": {"input_data": csv}}
    cfg.update(extra)
    data = load(cfg, name)
    with quiet():
        DE_mode(data, seed=seed)
    return data


def daily(data) -> Tuple[np.ndarray, np.ndarray]:
    """Observed and simulated daily water temperature of the real record (warm-up removed)."""
    obs = data.Twat_obs[365:].astype(float)
    sim = data.Twat_mod[365:].astype(float)
    obs = np.where(obs == -999.0, np.nan, obs)
    sim = np.where(sim == -999.0, np.nan, sim)
    return obs, sim


def metrics(obs, sim) -> dict:
    """RMSE, NSE, KGE and mean bias over days where both are available."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[m], sim[m]
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    nse = float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))
    r = np.corrcoef(o, s)[0, 1]
    kge = float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2))
    return {"n": int(m.sum()), "RMSE": rmse, "NSE": nse, "KGE": kge, "bias": float(np.mean(s - o))}


def params_at_bounds(par, version: int) -> List[str]:
    from pyair2stream.config import ACTIVE_PARAMS
    out = []
    for j in ACTIVE_PARAMS[version]:
        lo, hi = AUTHORS_BOUNDS["min"][j], AUTHORS_BOUNDS["max"][j]
        if min(abs(par[j] - lo), abs(par[j] - hi)) < 1e-3 * (hi - lo):
            out.append(f"a{j + 1}")
    return out


# --- Published values (Piccolroaz et al., 2016) -----------------------------

_PARAM_BOOK = os.path.join(DATA, "published", "Piccolroaz_etal_HP2016-Parameter_values.xlsx")
_RMSE_BOOK = os.path.join(DATA, "published", "Piccolroaz_etal_HP2016-AIC.xlsx")
_RMSE_COL = {3: 9, 4: 10, 5: 11, 7: 12, 8: 13}   # a2s-N columns in the calibration/validation sheets


def published_params(version: int, station: str) -> List[float]:
    """Published daily-resolution parameters (8 values, unused ones 0)."""
    rows = list(openpyxl.load_workbook(_PARAM_BOOK, data_only=True)[f"a2s_{version}"].iter_rows(values_only=True))
    header = rows[1]
    for r in rows[2:]:
        if r[1] == WATER_ID[station]:
            par = [0.0] * 8
            c = 3
            while c < len(header) and header[c] is not None:     # first block = daily ("RMSE_1d")
                par[int(header[c][1]) - 1] = float(r[c])
                c += 1
            return par
    raise KeyError(station)


def published_rmse(version: int, station: str, period: str) -> float:
    """Published RMSE (°C); period is 'calibration' or 'validation'."""
    for r in openpyxl.load_workbook(_RMSE_BOOK, data_only=True)[period].iter_rows(values_only=True):
        if r and r[1] == WATER_ID[station]:
            return float(r[_RMSE_COL[version]])
    raise KeyError(station)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
