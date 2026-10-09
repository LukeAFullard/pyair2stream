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
  to hide a fold's T_water targets from calibration. In the default mode
  T_air/Q forcing is left untouched, so the ODE integrates continuously
  through the held-out period. In gap-tolerant mode the fold's forcing is
  also masked during calibration, so it becomes a gap in the segmentation.
- Only data.Twat_obs (plus data.Tair/data.Q in gap-tolerant mode) is ever
  mutated, and only transiently (masked, then restored via try/finally
  before the next fold or on error).
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
    # Yearly-statistics check (check_yearly_statistics): the threshold for "days above
    # threshold" (default: 90th percentile of the measured water temperatures) and the
    # season (calendar months) that must be at least 80% measured for a year to count
    # (default: the four warmest months).
    threshold: Optional[float] = None
    season_months: Optional[list] = None


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
    dates_held_out: Optional[pd.DatetimeIndex] = field(default=None, repr=False)
    # Error model of this fold's calibration, from its training years only (as DE-MCMC
    # estimates it): daily residual SD and AR(1) rho (0 for the iid noise model).
    sigma: float = float("nan")
    rho: float = float("nan")
    # Year label (calendar or water year) of each held-out day.
    years_held_out: Optional[np.ndarray] = field(default=None, repr=False)


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


def _fold_error_model(data: CommonData) -> tuple[float, float]:
    """Daily residual SD and AR(1) rho on the currently observed, scored days (the
    fold's training days while its observations are hidden), estimated as DE-MCMC
    estimates them (docs/METHODS.md §12): rho at `uncertainty_options.rho_timescale`,
    and 0 for the iid noise model."""
    from .config import DEFAULT_NOISE_MODEL, DEFAULT_RHO_TIMESCALE
    from .optimization import _daily_residual_sigma, _segments_for
    from .uncertainty import estimate_rho
    options = data.uncertainty_options or {}
    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=bool)
    sigma = _daily_residual_sigma(data, eval_mask)
    rho = 0.0
    if options.get('noise_model', DEFAULT_NOISE_MODEL) == 'ar1':
        rho = estimate_rho(data.Twat_mod, data.Twat_obs, eval_mask, _segments_for(data),
                           options.get('rho_timescale', DEFAULT_RHO_TIMESCALE))
    return float(sigma), float(rho)


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

    Per fold: mask -> recompute Qmedia (and climatology) without the fold ->
    (rebuild segments if gap_tolerant) -> aggregate/statis -> optimize ->
    simulate full series -> score held-out window against the original obs ->
    restore. See module docstring for the full rationale.

    Only data.Twat_obs (and, in gap-tolerant mode, data.Tair/data.Q) is ever
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

                # Re-segment BEFORE aggregating: aggregation()/statis() must use the
                # eval_mask of the masked record (including the warm-up days after
                # the held-out gap), the same one funcobj() scores against.
                if data.gap_tolerant:
                    data.segments = None
                    detect_segments(data)

                aggregation(data)
                statis(data)

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
                # The fold's observations are still hidden here, so these are the
                # residuals of its training years: its error model, as DE-MCMC would
                # estimate it from a calibration on those years.
                sigma, rho = _fold_error_model(data)

                # Score the held-out days the calibration would score: in
                # gap-tolerant mode not the unscored start of a segment (§10),
                # which begins from an approximate temperature.
                scored = data.eval_mask[idx] if data.eval_mask is not None else True
                obs_scored = np.where(scored, orig_twat, MISSING_DATA_SENTINEL)
                sim = data.Twat_mod
                nse, kge, rmse = _compute_fold_metrics(obs_scored, sim[idx], MISSING_DATA_SENTINEL)

                start_date = pd.Timestamp(*data.date[idx[0]])
                end_date = pd.Timestamp(*data.date[idx[-1]])

                results.append(FoldResult(
                    fold_id=len(results),
                    label=label,
                    held_out_start=start_date,
                    held_out_end=end_date,
                    n_obs_held_out=int(np.sum(obs_scored != MISSING_DATA_SENTINEL)),
                    par_best=data.par_best.copy(),
                    nse=nse,
                    kge=kge,
                    rmse=rmse,
                    obs_held_out=obs_scored,
                    sim_held_out=sim[idx].copy(),
                    dates_held_out=pd.DatetimeIndex(pd.to_datetime(
                        {'year': data.date[idx, 0], 'month': data.date[idx, 1], 'day': data.date[idx, 2]})),
                    sigma=sigma,
                    rho=rho,
                    years_held_out=assign_year_groups(data, cv_config.water_year_start_month)[idx],
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


JACKKNIFE_LEVEL = 0.90


def count_blocks(data: CommonData, cv_config: CVConfig) -> int:
    """Number of blocks (years, or groups of `n_years_per_fold` years) in the whole record,
    including the leading years that are never held out."""
    wy = assign_year_groups(data, cv_config.water_year_start_month)
    n_years = len([y for y in np.unique(wy) if y != -999])
    size = cv_config.n_years_per_fold if cv_config.unit == "n_years" else 1
    return n_years // size


def jackknife_rows(par: np.ndarray, n_blocks: int, level: float = JACKKNIFE_LEVEL) -> list[dict]:
    """
    Delete-one-block jackknife intervals for the parameters, from the m folds' fitted
    parameters (rows of `par`). The standard jackknife deletes each of the n blocks once:
    SE^2 = (n-1)/n * sum over n of (theta_i - mean)^2. Cross-validation never holds out the
    first years, so only m < n deletions are available; the sum over n is estimated as n/m
    times the sum over the m folds, giving SE^2 = (n-1)/m * sum over m. The interval is the
    mean of the folds +- t(level, m-1) x SE.
    """
    from scipy import stats
    par = np.asarray(par, dtype=np.float64)
    m = par.shape[0]
    centre = par.mean(axis=0)
    se = np.sqrt((n_blocks - 1) / m * np.sum((par - centre) ** 2, axis=0))
    half = stats.t.ppf(0.5 + level / 2, m - 1) * se
    pct = f"{level * 100:g}"
    return [{"fold": "jackknife_se", **{f"p{i + 1}": v for i, v in enumerate(se)}},
            {"fold": f"jackknife_{pct}_lower", **{f"p{i + 1}": v for i, v in enumerate(centre - half)}},
            {"fold": f"jackknife_{pct}_upper", **{f"p{i + 1}": v for i, v in enumerate(centre + half)}}]


def cross_validate(data: CommonData, run_mode: str, return_folds: bool = False):
    """Run the cross-validation configured in `data.cross_validation` and return the
    `cv_results.csv` table, including jackknife parameter intervals at
    `uncertainty_options.parameter_interval` (see `summarize`). With `return_folds`,
    also return the list of FoldResult (held-out series per fold)."""
    cv_config = data.cross_validation
    results = run_leave_one_year_out_cv(data, cv_config, run_mode)
    level = float((data.uncertainty_options or {}).get('parameter_interval', JACKKNIFE_LEVEL * 100)) / 100
    table = summarize(results, n_blocks=count_blocks(data, cv_config), level=level)
    return (table, results) if return_folds else table


def held_out_series(results: list[FoldResult]) -> tuple:
    """Dates, observed and simulated daily values of every fold's held-out days, concatenated
    (missing or unscored days as NaN): the out-of-sample predictions of the cross-validation."""
    dates, obs, sim = [], [], []
    for r in results:
        if r.dates_held_out is None:
            continue
        dates.append(r.dates_held_out)
        obs.append(np.where(r.obs_held_out == MISSING_DATA_SENTINEL, np.nan, r.obs_held_out))
        sim.append(np.where(r.sim_held_out == MISSING_DATA_SENTINEL, np.nan, r.sim_held_out))
    if not dates:
        return pd.DatetimeIndex([]), np.array([]), np.array([])
    return dates[0].append(dates[1:]), np.concatenate(obs), np.concatenate(sim)


def summarize(results: list[FoldResult], n_blocks: Optional[int] = None,
              level: float = JACKKNIFE_LEVEL) -> pd.DataFrame:
    """
    One row per fold: metrics + calibrated parameter columns (p1..pN), for
    easy mean/std reporting and for checking whether par_best is stable
    across folds -- a useful equifinality diagnostic alongside the
    DE-vs-PSO comparison already discussed in the README.

    The final rows include 'mean', 'std', and 'pooled' (which computes
    NSE/KGE/RMSE on the concatenated held-out predictions from all folds,
    weighting all out-of-sample days equally). Given `n_blocks` (the number of
    blocks in the whole record) and at least two folds, three more rows give
    jackknife standard errors and intervals for the parameters at `level` (a fraction)
    (`jackknife_rows`). The 'std' row is only the spread between folds: the folds
    share most of their data, so it understates the parameters' uncertainty.
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
        extra = []
        if n_blocks is not None and len(results) >= 2:
            extra = jackknife_rows(np.array([r.par_best for r in results]), n_blocks, level=level)
        summary_df = pd.DataFrame([mean_row, std_row, pooled_row, *extra])
        df = pd.concat([df, summary_df], ignore_index=True)

    return df


# --------------------------------------------------------------------------
# Out-of-sample check of yearly statistics (docs/METHODS.md §11, §13)
# --------------------------------------------------------------------------

CHECK_SIMULATIONS = 1000        # simulations per held-out year
MIN_SEASON_OBSERVED = 0.8       # a year counts if at least this share of its season was measured
DEFAULT_THRESHOLD_QUANTILE = 0.9
COVERAGE_LEVELS = (50.0, 80.0, 90.0, 95.0)   # central ranges (%) whose coverage is always reported
CHANCE_CONFIDENCE = 0.95        # range of coverage expected by chance (binomial)


def warmest_months(dates, values, n: int = 4) -> list:
    """The `n` calendar months with the highest mean of `values` (NaN ignored)."""
    s = pd.Series(np.asarray(values, dtype=np.float64), index=pd.DatetimeIndex(dates)).dropna()
    return sorted(int(m) for m in s.groupby(s.index.month).mean().nlargest(n).index)


def _fold_ensemble(r: FoldResult, noise_model: str, n_simulations: int, rng: np.random.Generator):
    """`n_simulations` series of a fold's held-out window: its simulation plus random error from its
    own error model (sigma, rho), as a FORWARD run makes them. Returns (observed, ensemble), with NaN
    where there is no measurement or no simulation."""
    from .uncertainty import generate_ar1_noise
    obs = np.where(r.obs_held_out == MISSING_DATA_SENTINEL, np.nan, r.obs_held_out)
    sim = np.where(r.sim_held_out == MISSING_DATA_SENTINEL, np.nan, r.sim_held_out)
    n_days = len(sim)
    if noise_model == "ar1":
        noise = np.array([generate_ar1_noise(n_days, r.sigma, r.rho, [(0, n_days - 1)], rng)
                          for _ in range(n_simulations)])
    else:
        noise = rng.normal(0.0, r.sigma, (n_simulations, n_days))
    measured = np.isfinite(obs) & np.isfinite(sim)
    return np.where(measured, obs, np.nan), np.where(measured[None, :], sim[None, :] + noise, np.nan)


def check_interval_coverage(results: list[FoldResult], levels=COVERAGE_LEVELS, extra_level: Optional[float] = None,
                            noise_model: str = "ar1", n_simulations: int = CHECK_SIMULATIONS,
                            seed: Optional[int] = None) -> pd.DataFrame:
    """
    Did prediction intervals of each level hold in the held-out years? For each level (%), the share
    of measured held-out days inside the central range of the fold's simulations (as in
    `check_yearly_statistics`), and the same for 7-day moving means (all 7 days measured). Days are
    not independent, so no range expected by chance is given; differences of a few tenths of a
    percentage point mean nothing. Parameter uncertainty is not included, so the intervals are slightly
    narrower than a FORWARD run's: coverage here errs low, not high.
    """
    levels = sorted(set(levels) | ({float(extra_level)} if extra_level is not None else set()))
    inside = {(what, lev): [0, 0] for what in ("daily", "7-day") for lev in levels}
    for r in results:
        if r.dates_held_out is None or not np.isfinite(r.sigma):
            continue
        rng = np.random.default_rng(None if seed is None else [int(seed), int(r.fold_id), 7])
        obs, ens = _fold_ensemble(r, noise_model, n_simulations, rng)
        week_obs = pd.Series(obs).rolling(7).mean().to_numpy()
        week_ens = pd.DataFrame(ens.T).rolling(7).mean().to_numpy().T
        for what, o, e in (("daily", obs, ens), ("7-day", week_obs, week_ens)):
            ok = np.isfinite(o)
            if not ok.any():
                continue
            for lev in levels:
                lo, hi = np.percentile(e[:, ok], [50 - lev / 2, 50 + lev / 2], axis=0)
                inside[(what, lev)][0] += int(np.sum((o[ok] >= lo) & (o[ok] <= hi)))
                inside[(what, lev)][1] += int(ok.sum())
    rows = [{"level": lev, **{f"{what} inside": (inside[(what, lev)][0] / inside[(what, lev)][1]
                                                if inside[(what, lev)][1] else np.nan) for what in ("daily", "7-day")},
             "days": inside[("daily", lev)][1], "7-day means": inside[("7-day", lev)][1]} for lev in levels]
    return pd.DataFrame(rows)


def check_yearly_statistics(results: list[FoldResult], threshold: Optional[float] = None,
                            season_months: Optional[list] = None, noise_model: str = "ar1",
                            level: float = 90.0, n_simulations: int = CHECK_SIMULATIONS,
                            seed: Optional[int] = None, return_simulations: bool = False):
    """
    Did the predicted ranges of yearly statistics hold in years the model was not
    calibrated on?

    For every held-out year of a cross-validation, `n_simulations` series are made
    from that fold's simulation plus random error from the fold's own error model
    (`FoldResult.sigma`, `.rho`, estimated on its training years only), as a FORWARD
    run makes them. In each series and in the measurements, the yearly statistics of
    `scenario.year_statistics` are computed over the days that were measured. A year
    counts if at least MIN_SEASON_OBSERVED of the days in `season_months` were
    measured. Parameter uncertainty is not included (each fold has one calibrated
    parameter set), so the ranges are slightly narrower than those of a FORWARD run
    from a DE-MCMC chain.

    `level` (%) is the central range reported for each year (normally
    `uncertainty_options.prediction_interval`); the summary also gives the coverage
    of the ranges in COVERAGE_LEVELS, from the same PIT values.

    Returns
    -------
    per_year : DataFrame
        One row per held-out year and statistic: the measured value, the median and
        the central `level`% range of the simulations, the PIT (share of simulations
        below the measured value), whether the measured value lies in that range,
        and the deviation (measured minus median), the input of
        `scenario.correct_statistic`.
    summary : DataFrame
        One row per statistic: number of years; for each level, the share of years
        inside the central range and the shares expected by chance (95% binomial
        range); and the mean deviation with its 95% confidence interval. A
        confidence interval that excludes zero means the model is biased in that
        statistic.
    simulations : dict, only with `return_simulations`
        {(year, statistic): simulated values}.
    """
    from scipy.stats import binom, t as student_t
    from .scenario import YEARLY_STATISTICS, year_statistics, pit as pit_of, central_range, inside_range
    level = float(level)
    if not (0.0 < level < 100.0):
        raise ValueError(f"level must be strictly between 0 and 100 (per cent), got {level}")
    levels = sorted(set(COVERAGE_LEVELS) | {level})
    from .uncertainty import generate_ar1_noise

    all_dates = pd.DatetimeIndex([]).append([r.dates_held_out for r in results if r.dates_held_out is not None])
    all_obs = np.concatenate([np.where(r.obs_held_out == MISSING_DATA_SENTINEL, np.nan, r.obs_held_out)
                              for r in results if r.dates_held_out is not None])
    if threshold is None:
        threshold = float(np.nanquantile(all_obs, DEFAULT_THRESHOLD_QUANTILE))
    if season_months is None:
        season_months = warmest_months(all_dates, all_obs)

    rows, simulations = [], {}
    for r in results:
        if r.dates_held_out is None or not np.isfinite(r.sigma):
            continue
        rng = np.random.default_rng(None if seed is None else [int(seed), int(r.fold_id)])
        obs_used, ens = _fold_ensemble(r, noise_model, n_simulations, rng)
        measured = np.isfinite(obs_used)
        years = r.years_held_out if r.years_held_out is not None else r.dates_held_out.year.to_numpy()
        in_season = np.isin(r.dates_held_out.month, season_months)
        # Partial years are kept: a year counts below if its season was measured.
        sim_stats = year_statistics(ens, r.dates_held_out, threshold, years=years, partial_years="keep")
        obs_stats = year_statistics(obs_used, r.dates_held_out, threshold, years=years, partial_years="keep")
        for year in sorted(sim_stats):
            season = (years == year) & in_season
            if not season.any() or measured[season].mean() < MIN_SEASON_OBSERVED:
                continue
            for name in YEARLY_STATISTICS:
                value = float(obs_stats[year][name][0])
                sims = sim_stats[year][name]
                sims = sims[np.isfinite(sims)]
                if not np.isfinite(value) or len(sims) == 0:
                    continue
                lower, upper = central_range(sims, level)
                median = float(np.median(sims))
                p = pit_of(sims, value, rng)
                rows.append({"fold": r.label, "year": int(year), "statistic": name, "measured": value,
                             "median": median, "level": level, "lower": float(lower), "upper": float(upper),
                             "pit": p, "inside": bool(inside_range(p, level)), "deviation": value - median,
                             "sigma": r.sigma, "rho": r.rho, "threshold": threshold,
                             "season_months": " ".join(map(str, season_months))})
                if return_simulations:
                    simulations[(int(year), name)] = sims
    per_year = pd.DataFrame(rows, columns=["fold", "year", "statistic", "measured", "median", "level", "lower",
                                           "upper", "pit", "inside", "deviation", "sigma", "rho", "threshold",
                                           "season_months"])
    summary = []
    for name in YEARLY_STATISTICS:
        sub = per_year[per_year.statistic == name]
        n = len(sub)
        row = {"statistic": name, "n_years": n}
        for lev in levels:
            col = f"inside_{lev:g}"
            row[f"share_{col}"] = float(np.mean(inside_range(sub.pit, lev))) if n else np.nan
            lo, hi = binom.interval(CHANCE_CONFIDENCE, n, lev / 100) if n else (np.nan, np.nan)
            row[f"expected_{col}_low"] = lo / n if n else np.nan
            row[f"expected_{col}_high"] = hi / n if n else np.nan
        d = sub.deviation.to_numpy()
        row["mean_deviation"] = float(d.mean()) if n else np.nan
        half = float(student_t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
        row["mean_deviation_ci95_lower"] = row["mean_deviation"] - half
        row["mean_deviation_ci95_upper"] = row["mean_deviation"] + half
        summary.append(row)
    summary = pd.DataFrame(summary)
    return (per_year, summary, simulations) if return_simulations else (per_year, summary)
