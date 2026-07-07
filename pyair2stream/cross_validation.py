"""
Leave-one-year-out (and leave-N-years-out) cross-validation for pyair2stream.

See CROSS_VALIDATION_PLAN.md for full rationale. Summary of the design:

- Folds are built from *calendar dates* (data.date), never row counts, so
  leap years, gap-tolerant segments, and the seasonal cosine term's phase
  all stay aligned with fold boundaries.
- This module does NOT touch the ODE integrator (model.py) or optimizer
  internals (optimization.py). It reuses the existing missing-observation
  pathway (Twat_obs == -999.0, already consumed by aggregation()/funcobj())
  to hide a fold's T_water targets from calibration while leaving T_air/Q
  forcing untouched -- so the ODE state integrates continuously through and
  past the held-out period, with no re-spin-up required.
- Only data.Twat_obs is ever mutated, and only transiently (masked, then
  restored via try/finally before the next fold or on error).

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import CommonData
from .model import aggregation, statis, call_model, detect_segments


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class CVConfig:
    """
    Settings for a leave-one-year-out (or leave-N-years-out) CV run.
    Populated from the `cross_validation:` block in config.yaml.
    """
    unit: str = "year"                  # "year" or "n_years"
    n_years_per_fold: int = 1           # only used if unit == "n_years"
    water_year_start_month: int = 1     # 1 = calendar year; e.g. 10 = Oct-Sep water year
    min_train_years: int = 1            # extra full years (beyond the mandatory
                                         # spin-up year) required before a fold
                                         # is eligible to be held out
    skip_first_year: bool = True        # first calendar/water year is spin-up-only,
                                         # never a candidate fold (nothing precedes
                                         # it to spin up from)
    optimizer_overrides: Optional[dict] = None  # e.g. {"n_runs": 20, "n_particles": 20}
                                                 # to cut per-fold cost vs. the
                                                 # production calibration


@dataclass
class FoldResult:
    fold_id: int
    label: str                          # e.g. "2014" or "2014-2016"
    held_out_start: pd.Timestamp
    held_out_end: pd.Timestamp
    n_obs_held_out: int                 # count of *actual* (non-missing) T_water
                                         # observations in the held-out window
    par_best: np.ndarray
    nse: float
    kge: float
    rmse: float


# --------------------------------------------------------------------------
# Fold construction (date-based)
# --------------------------------------------------------------------------

def assign_year_groups(data: CommonData, water_year_start_month: int = 1) -> np.ndarray:
    """
    Return an int array (length data.n_tot) labelling every row with its
    water year, derived from data.date[:, 0] (year) and data.date[:, 1]
    (month) -- NOT from row position.

    water_year_start_month=1 recovers plain calendar years. For any other
    value, rows in or after that month belong to the *next* labelled year
    (e.g. water_year_start_month=10 means Oct 2013 - Sep 2014 is all
    labelled "2014").
    """
    years = data.date[:, 0]
    months = data.date[:, 1]
    if water_year_start_month == 1:
        return years.copy()
    return np.where(months >= water_year_start_month, years + 1, years)


def build_folds(data: CommonData, cv_config: CVConfig) -> list[tuple[str, np.ndarray]]:
    """
    Returns a list of (fold_label, row_indices) tuples -- one per eligible
    fold -- built strictly from calendar dates via assign_year_groups.

    - Drops the earliest (min_train_years + int(skip_first_year)) labelled
      years entirely: they exist only to spin up / train, never to be held
      out (there's nothing before the record start to spin up a first-year
      fold correctly).
    - unit="n_years": groups the remaining eligible years into consecutive
      non-overlapping blocks of n_years_per_fold; a short trailing block
      (fewer than n_years_per_fold years) is dropped rather than yielded as
      a partial fold.
    """
    wy = assign_year_groups(data, cv_config.water_year_start_month)
    unique_years = sorted(int(y) for y in np.unique(wy))

    first_eligible = cv_config.min_train_years + (1 if cv_config.skip_first_year else 0)
    eligible_years = unique_years[first_eligible:]

    if cv_config.unit == "year":
        blocks = [[y] for y in eligible_years]
    elif cv_config.unit == "n_years":
        n = cv_config.n_years_per_fold
        blocks = [eligible_years[i:i + n] for i in range(0, len(eligible_years), n)]
        blocks = [b for b in blocks if len(b) == n]
    else:
        raise ValueError(f"Unknown CVConfig.unit: {cv_config.unit!r} (expected 'year' or 'n_years')")

    folds = []
    for block in blocks:
        mask = np.isin(wy, block)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        label = str(block[0]) if len(block) == 1 else f"{block[0]}-{block[-1]}"
        folds.append((label, idx))
    return folds


# --------------------------------------------------------------------------
# Masking helpers
# --------------------------------------------------------------------------

def _mask_fold(data: CommonData, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Set Twat_obs, Tair, and Q to the configured missing value for the given rows and return the original
    values (for scoring + restoration). By masking the forcing data as well, gap_tolerant
    mode will correctly segment the ODE integration, preventing catastrophic state drift
    over long missing target windows.
    """
    orig_twat = data.Twat_obs[idx].copy()
    orig_tair = data.Tair[idx].copy()
    orig_q = data.Q[idx].copy()

    data.Twat_obs[idx] = data.mineff_index

    if data.gap_tolerant:
        data.Tair[idx] = -999.0
        data.Q[idx] = -999.0

    return orig_twat, orig_tair, orig_q


def _restore_fold(data: CommonData, idx: np.ndarray, orig_twat: np.ndarray, orig_tair: np.ndarray, orig_q: np.ndarray) -> None:
    data.Twat_obs[idx] = orig_twat
    data.Tair[idx] = orig_tair
    data.Q[idx] = orig_q


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _compute_fold_metrics(obs: np.ndarray, sim: np.ndarray, missing_val: float) -> tuple[float, float, float]:
    """
    NSE, KGE, RMSE for one fold, computed only over rows where `obs`
    (the pre-mask backup) is an actual observation, i.e. genuinely-missing
    T_water inside the held-out window is correctly excluded too.
    """
    valid = obs != missing_val
    o, s = obs[valid], sim[valid]
    if o.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean_o = o.mean()
    denom = np.sum((o - mean_o) ** 2)
    nse = 1.0 - np.sum((o - s) ** 2) / denom if denom > 0 else float("nan")

    if o.size > 1 and o.std() > 0 and s.std() > 0:
        r = np.corrcoef(o, s)[0, 1]
        alpha = s.std() / o.std()
        beta = s.mean() / mean_o if mean_o != 0 else float("nan")
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    else:
        kge = float("nan")

    rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    return float(nse), float(kge), rmse


# --------------------------------------------------------------------------
# Optimizer dispatch
# --------------------------------------------------------------------------

def _run_optimizer(data: CommonData, run_mode: str, overrides: Optional[dict]) -> None:
    """
    Run the configured optimizer for one fold, applying `overrides` (e.g.
    reduced n_runs/n_particles for cheaper per-fold CV calibration) for the
    duration of the call only, then restoring data's original settings.
    """
    from .main import run_optimizer as _dispatch

    saved = {}
    if overrides:
        for key, val in overrides.items():
            saved[key] = getattr(data, key)
            setattr(data, key, val)
    try:
        original_runmode, data.runmode = data.runmode, run_mode
        try:
            _dispatch(data)
        finally:
            data.runmode = original_runmode
    finally:
        for key, val in saved.items():
            setattr(data, key, val)


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

def run_leave_one_year_out_cv(
    data: CommonData,
    cv_config: CVConfig,
    run_mode: str,
) -> list[FoldResult]:
    """
    Full leave-one-year-out (or leave-N-years-out) CV driver.

    Per fold: mask -> aggregate/statis -> (rebuild segments if
    gap_tolerant) -> optimize -> simulate full series -> score held-out
    window against the original obs -> restore. See module docstring and
    CROSS_VALIDATION_PLAN.md section 6 for the full rationale.

    Every fold reuses the same data.Tair/data.Q; only data.Twat_obs is ever
    mutated, and only for the duration of that fold's block below.
    """
    folds = build_folds(data, cv_config)
    if not folds:
        raise ValueError(
            "No eligible folds found. Check cv_config.min_train_years / "
            "skip_first_year against the length of the record."
        )

    results: list[FoldResult] = []

    for label, idx in folds:
        orig_twat, orig_tair, orig_q = _mask_fold(data, idx)
        try:
            aggregation(data)
            statis(data)

            if data.gap_tolerant:
                data.segments = None
                detect_segments(data)

            _run_optimizer(data, run_mode, cv_config.optimizer_overrides)
            data.par[:] = data.par_best[:]

            # Restore forcing variables (Tair, Q) before forward simulation
            # so the model integrates through the held-out window properly
            data.Tair[idx] = orig_tair
            data.Q[idx] = orig_q

            if data.gap_tolerant:
                data.segments = None
                detect_segments(data)

            # Re-aggregate and statis now that forcing data is restored to allow
            # potential evaluation at non-daily time-scales
            aggregation(data)
            statis(data)

            call_model(data)

            sim = data.Twat_mod
            nse, kge, rmse = _compute_fold_metrics(orig_twat, sim[idx], data.mineff_index)

            start_date = pd.Timestamp(*data.date[idx[0]])
            end_date = pd.Timestamp(*data.date[idx[-1]])

            results.append(FoldResult(
                fold_id=len(results),
                label=label,
                held_out_start=start_date,
                held_out_end=end_date,
                n_obs_held_out=int(np.sum(orig_twat != data.mineff_index)),
                par_best=data.par_best.copy(),
                nse=nse,
                kge=kge,
                rmse=rmse,
            ))
        finally:
            _restore_fold(data, idx, orig_twat, orig_tair, orig_q)

    return results


def summarize(results: list[FoldResult]) -> pd.DataFrame:
    """
    One row per fold: metrics + calibrated parameter columns (p1..pN), for
    easy mean/std reporting and for checking whether par_best is stable
    across folds -- a useful equifinality diagnostic alongside the
    DE-vs-PSO comparison already discussed in the README.
    """
    rows = []
    for r in results:
        row = {
            "fold": r.label,
            "held_out_start": r.held_out_start.date().isoformat(),
            "held_out_end": r.held_out_end.date().isoformat(),
            "n_obs_held_out": r.n_obs_held_out,
            "NSE": r.nse,
            "KGE": r.kge,
            "RMSE": r.rmse,
        }
        row.update({f"p{i + 1}": v for i, v in enumerate(r.par_best)})
        rows.append(row)
    return pd.DataFrame(rows)
