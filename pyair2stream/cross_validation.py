"""
Leave-one-year-out (and leave-N-years-out) cross-validation for pyair2stream.

This module implements the block cross-validation routines used to evaluate
the out-of-sample parameter stability and predictive performance of the
calibrated air2stream models.

Summary of the design:

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
- The first eligible calendar year is strictly enforced to never be a
  candidate fold, as there is no prior year of valid data to use for
  model spin-up. If `skip_first_year` is False, `min_train_years` must
  be > 0.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import CommonData, MISSING_DATA_SENTINEL
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
    min_train_years: int = 1            # Skip the first N eligible years (beyond the mandatory
                                         # spin-up year) to ensure they are always used for
                                         # training. This ONLY gates the start of the fold
                                         # sequence, it is NOT an ongoing per-fold minimum.
    skip_first_year: bool = True        # first calendar/water year is spin-up-only,
                                         # never a candidate fold (nothing precedes
                                         # it to spin up from). If set to False, you
                                         # MUST set min_train_years > 0 to skip it.
    min_valid_obs: int = 10              # minimum number of valid T_water observations
                                         # required for a block to be considered a fold
    optimizer_overrides: Optional[dict] = None  # e.g. {"n_run": 20, "n_particles": 20}
                                                 # to cut per-fold cost vs. the
                                                 # production calibration


@dataclass
class FoldResult:
    """
    Data container for the outcome of a single cross-validation fold.

    Fields:
    - fold_id: Integer index of the fold.
    - label: Human-readable string identifier for the held-out period (e.g. "2014" or "2014-2016").
    - held_out_start: The first date of the held-out window.
    - held_out_end: The final date of the held-out window.
    - n_obs_held_out: Count of *actual* (non-missing) T_water observations in the held-out window.
    - par_best: The optimized parameters (1-indexed numpy array) calibrated on the remaining data.
    - nse, kge, rmse: Goodness-of-fit metrics evaluated strictly on the held-out block.
    - obs_held_out: Array of true T_water values inside the block (includes missing sentinels).
    - sim_held_out: Array of simulated T_water values for the corresponding dates.
    """
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

    # Raw validation arrays, used for computing the pooled out-of-sample metrics
    obs_held_out: np.ndarray = field(repr=False)
    sim_held_out: np.ndarray = field(repr=False)


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
    if not cv_config.skip_first_year and cv_config.min_train_years == 0:
        raise ValueError(
            "The first year cannot be a candidate fold. You must set skip_first_year=True "
            "or min_train_years > 0 to ensure the model has a prior year to spin up from."
        )

    wy = assign_year_groups(data, cv_config.water_year_start_month)
    # Exclude the synthetic -999 year from the warm-up block
    unique_years = sorted(int(y) for y in np.unique(wy) if y != -999)

    first_eligible = cv_config.min_train_years + int(cv_config.skip_first_year)
    eligible_years = unique_years[first_eligible:]

    if cv_config.unit == "year":
        blocks = [[y] for y in eligible_years]
    elif cv_config.unit == "n_years":
        n = cv_config.n_years_per_fold
        blocks = [eligible_years[i:i + n] for i in range(0, len(eligible_years), n)]
        valid_blocks = []
        for b in blocks:
            if len(b) == n:
                valid_blocks.append(b)
            else:
                import warnings
                warnings.warn(f"Dropping short trailing cross-validation block of {len(b)} years (requires {n}).")
        blocks = valid_blocks
    else:
        raise ValueError(f"Unknown CVConfig.unit: {cv_config.unit!r} (expected 'year' or 'n_years')")

    folds = []
    for block in blocks:
        mask = np.isin(wy, block)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        valid_obs_count = np.sum(data.Twat_obs[idx] != MISSING_DATA_SENTINEL)
        if valid_obs_count < cv_config.min_valid_obs:
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

    Note: In default whole-series mode (gap_tolerant=False), a held-out year is properly
    excluded from the objective (Twat_obs = -999.0), but the ODE still free-integrates
    through the held-out window using actual forcing data with no restart. This is generally
    fine as the model is fairly mean-reverting, but represents an asymmetry compared to
    the segmented restart behavior of gap-tolerant mode.
    """
    orig_twat = data.Twat_obs[idx].copy()
    orig_tair = data.Tair[idx].copy()
    orig_q = data.Q[idx].copy()

    data.Twat_obs[idx] = MISSING_DATA_SENTINEL

    if data.gap_tolerant:
        data.Tair[idx] = MISSING_DATA_SENTINEL
        data.Q[idx] = MISSING_DATA_SENTINEL

    return orig_twat, orig_tair, orig_q


def _restore_fold(data: CommonData, idx: np.ndarray, orig_twat: np.ndarray, orig_tair: np.ndarray, orig_q: np.ndarray) -> None:
    """
    Restore the original forcing data and observations to the global CommonData
    arrays after a cross-validation fold has finished.

    This ensures subsequent folds start with a clean slate without reloading
    the datasets from disk.
    """
    data.Twat_obs[idx] = orig_twat

    if data.gap_tolerant:
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

    In gap-tolerant mode, `sim` will contain `missing_val` for days outside
    of any valid segment (e.g., due to forcing data gaps). We must exclude
    these from scoring as well to prevent -999.0 from destroying the metric.
    """
    valid = (obs != missing_val) & (sim != missing_val)
    o, s = obs[valid], sim[valid]
    if o.size == 0:
        return float("nan"), float("nan"), float("nan")

    if o.size < 10:
        import warnings
        warnings.warn(f"Computing metrics with very few observations ({o.size}). Metrics may not be statistically meaningful.")

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

    if o.size > 1:
        rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    else:
        rmse = float("nan")

    return float(nse), float(kge), rmse


# --------------------------------------------------------------------------
# Optimizer dispatch
# --------------------------------------------------------------------------

def _run_optimizer(data: CommonData, run_mode: str, overrides: Optional[dict]) -> None:
    """
    Run the configured optimizer for one fold, applying `overrides` (e.g.
    reduced n_run/n_particles for cheaper per-fold CV calibration) for the
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
    window against the original obs -> restore. See module docstring for the full rationale.

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

    from .io import compute_doy_climatology, compute_qmedia

    orig_par = data.par.copy() if data.par is not None else None
    orig_par_best = data.par_best.copy() if data.par_best is not None else None

    try:
        for label, idx in folds:
            orig_twat, orig_tair, orig_q = _mask_fold(data, idx)
            try:
                # To prevent Qmedia data leakage in non-gap-tolerant mode (V6/V8),
                # we must mask Q during compute_qmedia so the held-out fold is excluded.
                if not data.gap_tolerant:
                    data.Q[idx] = MISSING_DATA_SENTINEL

                compute_qmedia(data, verbose=True)
                if data.gap_tolerant:
                    compute_doy_climatology(data)

                # Restore Q if we masked it solely for Qmedia computation in non-gap-tolerant mode
                if not data.gap_tolerant:
                    data.Q[idx] = orig_q

                aggregation(data)
                statis(data)

                if data.gap_tolerant:
                    data.segments = None
                    detect_segments(data)

                _run_optimizer(data, run_mode, cv_config.optimizer_overrides)
                data.par[:] = data.par_best[:]

                # Restore forcing variables (Tair, Q) before forward simulation
                # so the model integrates through the held-out window properly.
                # (Note: _restore_fold also restores forcing, but doing it here is required
                # for the forward simulation. _restore_fold is idempotent.)
                if data.gap_tolerant:
                    data.Tair[idx] = orig_tair
                    data.Q[idx] = orig_q

                if data.gap_tolerant:
                    data.segments = None
                    detect_segments(data)

                call_model(data)

                sim = data.Twat_mod
                nse, kge, rmse = _compute_fold_metrics(orig_twat, sim[idx], MISSING_DATA_SENTINEL)

                start_date = pd.Timestamp(*data.date[idx[0]])
                end_date = pd.Timestamp(*data.date[idx[-1]])

                results.append(FoldResult(
                    fold_id=len(results),
                    label=label,
                    held_out_start=start_date,
                    held_out_end=end_date,
                    n_obs_held_out=int(np.sum(orig_twat != MISSING_DATA_SENTINEL)),
                    par_best=data.par_best.copy(),
                    nse=nse,
                    kge=kge,
                    rmse=rmse,
                    obs_held_out=orig_twat.copy(),
                    sim_held_out=sim[idx].copy(),
                ))
            finally:
                _restore_fold(data, idx, orig_twat, orig_tair, orig_q)
    finally:
        if orig_par is not None:
            data.par[:] = orig_par[:]
        if orig_par_best is not None:
            data.par_best[:] = orig_par_best[:]

    compute_qmedia(data)
    if data.gap_tolerant:
        compute_doy_climatology(data)

    aggregation(data)
    statis(data)

    return results


def summarize(results: list[FoldResult]) -> pd.DataFrame:
    """
    One row per fold: metrics + calibrated parameter columns (p1..pN), for
    easy mean/std reporting and for checking whether par_best is stable
    across folds -- a useful equifinality diagnostic alongside the
    DE-vs-PSO comparison already discussed in the README.

    The final rows include 'mean', 'std', and 'pooled' (which computes
    NSE/KGE/RMSE on the concatenated held-out predictions from all folds,
    weighting all out-of-sample days equally).
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

    df = pd.DataFrame(rows)

    if not df.empty:
        # Calculate mean/std of metrics across folds
        mean_row = {
            "fold": "mean",
            "held_out_start": "",
            "held_out_end": "",
            "n_obs_held_out": df["n_obs_held_out"].sum(),
            "NSE": df["NSE"].mean(),
            "KGE": df["KGE"].mean(),
            "RMSE": df["RMSE"].mean(),
        }
        std_row = {
            "fold": "std",
            "held_out_start": "",
            "held_out_end": "",
            "n_obs_held_out": "",
            "NSE": df["NSE"].std(ddof=1) if len(df) > 1 else 0.0,
            "KGE": df["KGE"].std(ddof=1) if len(df) > 1 else 0.0,
            "RMSE": df["RMSE"].std(ddof=1) if len(df) > 1 else 0.0,
        }

        # Calculate pooled metrics using the raw arrays
        pooled_obs = np.concatenate([r.obs_held_out for r in results])
        pooled_sim = np.concatenate([r.sim_held_out for r in results])
        p_nse, p_kge, p_rmse = _compute_fold_metrics(pooled_obs, pooled_sim, MISSING_DATA_SENTINEL)

        pooled_row = {
            "fold": "pooled",
            "held_out_start": "",
            "held_out_end": "",
            "n_obs_held_out": len(pooled_obs[pooled_obs != MISSING_DATA_SENTINEL]),
            "NSE": p_nse,
            "KGE": p_kge,
            "RMSE": p_rmse,
        }

        # Parameter means/stds
        for col in df.columns:
            if col.startswith('p'):
                mean_row[col] = df[col].mean()
                std_row[col] = df[col].std(ddof=1) if len(df) > 1 else 0.0
                pooled_row[col] = float('nan') # not applicable for pooled metric row

        # Append summary rows
        summary_df = pd.DataFrame([mean_row, std_row, pooled_row])
        df = pd.concat([df, summary_df], ignore_index=True)

    return df
