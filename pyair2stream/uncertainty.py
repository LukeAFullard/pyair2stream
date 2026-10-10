"""
Uncertainty quantification and noise modeling for pyair2stream.

This module provides tools for estimating autoregressive (AR1) properties
from model residuals and generating structurally consistent noise envelopes
for probabilistic forward predictions.
"""

import numpy as np
import scipy.signal
from scipy.optimize import brentq

MIN_PAIRS_FOR_RHO_ESTIMATE = 30
WEEK = 7
MIN_WEEK_PAIRS_FOR_RHO_ESTIMATE = 20
MAX_RHO = 0.99

def estimate_ar1_rho(Twat_mod: np.ndarray, Twat_obs: np.ndarray, eval_mask: np.ndarray, segments: list) -> float:
    """
    Estimate the lag-1 autocorrelation coefficient (rho) of the daily residuals.

    Pairs are collected only where both elements are in the same segment,
    are unmasked in eval_mask, and have valid Twat_obs (!= -999.0).
    """
    valid_mask = eval_mask & (Twat_obs != -999.0)
    residuals = Twat_mod - Twat_obs

    pairs_t0 = []
    pairs_t1 = []

    for start, end in segments:
        for t in range(start + 1, end + 1):
            if valid_mask[t - 1] and valid_mask[t]:
                pairs_t0.append(residuals[t - 1])
                pairs_t1.append(residuals[t])

    n_valid_pairs = len(pairs_t0)

    if n_valid_pairs < MIN_PAIRS_FOR_RHO_ESTIMATE:
        print(f"Warning: Only {n_valid_pairs} valid residual pairs available for AR(1) estimation (need >= {MIN_PAIRS_FOR_RHO_ESTIMATE}). Falling back to rho=0.0.")
        return 0.0

    pairs_t0 = np.array(pairs_t0)
    pairs_t1 = np.array(pairs_t1)

    # Calculate sample Pearson correlation coefficient
    # np.corrcoef returns a 2x2 matrix, we want the off-diagonal element
    rho = np.corrcoef(pairs_t0, pairs_t1)[0, 1]

    if np.isnan(rho):
        print("Warning: AR(1) rho estimation resulted in NaN. Falling back to rho=0.0.")
        return 0.0

    # Clip strictly to [0.0, 0.99]. Note: the lower bound of 0.0 enforces non-negative serial correlation.
    # Hydrological water temperature residuals are typically positively autocorrelated (persistence);
    # negative serial correlation is disallowed by design in empirical noise estimation.
    return float(np.clip(rho, 0.0, 0.99))


def weekly_mean_correlation(rho: float, week: int = WEEK) -> float:
    """
    Correlation between the means of two consecutive, non-overlapping blocks of
    `week` days (7 by default; any block length) of a stationary AR(1) process
    with lag-1 correlation `rho`.

    With c_d the number of day pairs d days apart across the two blocks
    (c_d = week - |d - week|, d = 1 .. 2*week - 1), the correlation is
    sum(c_d * rho**d) / (week + 2 * sum_{k<week} (week - k) * rho**k).
    It rises from 0 (rho = 0) towards 1 (rho -> 1).
    """
    if rho <= 0.0:
        return 0.0
    d = np.arange(1, 2 * week)
    cov = np.sum((week - np.abs(d - week)) * rho ** d)
    k = np.arange(1, week)
    var = week + 2.0 * np.sum((week - k) * rho ** k)
    return float(cov / var)


def scoring_block_days(time_res: str) -> int:
    """
    Days behind each scored value at a `time_resolution`: 1 for '1d', 7N for
    'Nw', and 30 for '1m' (calendar months have 28-31 days).
    """
    if time_res == '1d':
        return 1
    if time_res == '1m':
        return 30
    if time_res.endswith('w') and time_res[:-1].isdigit() and int(time_res[:-1]) > 0:
        return WEEK * int(time_res[:-1])
    raise ValueError(f"Invalid time_resolution '{time_res}'. Must be '1d', 'Nw' or '1m'.")


def mean_error_variance_factor(rho: float, block_days: int = 1) -> float:
    """
    Variance of the mean of n consecutive scored errors relative to n independent
    errors, when daily errors are AR(1) with lag-1 correlation `rho` and each
    scored value is the mean of a block of `block_days` days.

    Block means k >= 1 blocks apart have correlation r_b * rho**(m*(k-1)), with
    m = block_days and r_b = `weekly_mean_correlation(rho, m)` the correlation
    of adjacent blocks, so the factor is 1 + 2 r_b / (1 - rho**m) (for large n).
    For daily values (m = 1, r_b = rho) this is (1 + rho) / (1 - rho). The
    least-squares likelihood uses n divided by this factor as the effective
    number of independent values.
    """
    if rho <= 0.0:
        return 1.0
    m = int(block_days)
    if m == 1:
        return (1.0 + rho) / (1.0 - rho)
    return 1.0 + 2.0 * weekly_mean_correlation(rho, m) / (1.0 - rho ** m)


def scored_error_variance_factor(positions: np.ndarray, rho: float, block_days: int = 1) -> float:
    """
    Variance of the mean of the scored errors relative to independent ones, for scored
    values on the rows (days) `positions`, when daily errors are AR(1) with lag-1
    correlation `rho`: 1 + (2/n) * the sum, over pairs, of their correlation. Values d days
    apart have correlation rho**d; block means (`block_days` = m > 1) k blocks apart,
    k = round(d / m), have r_b * rho**(m*(k-1)), r_b = `weekly_mean_correlation(rho, m)`.
    So the spacing of the scored values counts: two values a gap apart are less related
    than consecutive ones. For n consecutive values it tends to `mean_error_variance_factor`
    as n grows (for a record of a few years it is within about 0.3%).
    """
    pos = np.sort(np.asarray(positions, dtype=np.int64))
    n = len(pos)
    if rho <= 0.0 or n < 2:
        return 1.0
    gaps = np.diff(pos)
    m = int(block_days)
    if m == 1:
        step, scale = rho ** gaps.astype(np.float64), 1.0
    else:
        blocks = np.maximum(np.rint(gaps / m), 1.0)
        step, scale = rho ** (m * blocks), weekly_mean_correlation(rho, m) / rho ** m
    # Sum over pairs i < j of the product of the steps between them: S_j = step_j * (1 + S_{j-1}).
    s = total = 0.0
    for x in step.tolist():
        s = x * (1.0 + s)
        total += s
    return 1.0 + 2.0 * scale * total / n


def estimate_ar1_rho_weekly(Twat_mod: np.ndarray, Twat_obs: np.ndarray, eval_mask: np.ndarray,
                            segments: list) -> float:
    """
    Estimate rho from the persistence of the errors from one week to the next.

    For every 7-day window whose days are all valid (the same days
    `estimate_ar1_rho` uses) and that is followed, within the same segment, by
    another complete 7-day window starting 7 days later, the two windows' mean
    errors form a pair. r is the correlation over all such pairs (every
    starting day, so the result does not depend on where weeks are taken to
    begin), and rho is the AR(1) coefficient whose consecutive weekly means
    have that correlation (`weekly_mean_correlation(rho) = r`). For errors
    that really are AR(1) this estimates the same rho as consecutive days;
    real model errors also persist for weeks, which the lag-1 correlation of
    consecutive days does not show. With fewer pairs than
    MIN_WEEK_PAIRS_FOR_RHO_ESTIMATE non-overlapping weeks would give, it falls
    back to `estimate_ar1_rho`.
    """
    valid_mask = (eval_mask & (Twat_obs != -999.0)).astype(bool)
    residuals = np.where(valid_mask, Twat_mod - Twat_obs, 0.0)
    first, second = [], []
    for start, end in segments:
        n = end - start + 1
        if n < 2 * WEEK:
            continue
        valid_cum = np.concatenate([[0], np.cumsum(valid_mask[start:end + 1])])
        resid_cum = np.concatenate([[0.0], np.cumsum(residuals[start:end + 1])])
        n_windows = n - WEEK + 1
        complete = (valid_cum[WEEK:WEEK + n_windows] - valid_cum[:n_windows]) == WEEK
        means = (resid_cum[WEEK:WEEK + n_windows] - resid_cum[:n_windows]) / WEEK
        i = np.arange(n_windows - WEEK)
        ok = complete[i] & complete[i + WEEK]
        first.append(means[i][ok])
        second.append(means[i + WEEK][ok])
    first = np.concatenate(first) if first else np.empty(0)
    second = np.concatenate(second) if second else np.empty(0)
    if len(first) < WEEK * MIN_WEEK_PAIRS_FOR_RHO_ESTIMATE:
        print(f"Warning: Only {len(first)} pairs of complete 7-day windows a week apart are available for the "
              f"weekly rho estimate (need >= {WEEK * MIN_WEEK_PAIRS_FOR_RHO_ESTIMATE}, about "
              f"{MIN_WEEK_PAIRS_FOR_RHO_ESTIMATE + 1} complete weeks). Using the lag-1 correlation of "
              "consecutive days instead.")
        return estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments)
    # Errors that do not vary (to rounding of the running sums) carry no persistence to measure.
    scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))), 1e-300)
    if np.std(first) <= 1e-9 * scale or np.std(second) <= 1e-9 * scale:
        return 0.0
    r = np.corrcoef(first, second)[0, 1]
    if not np.isfinite(r) or r <= 0.0:
        return 0.0
    if r >= weekly_mean_correlation(MAX_RHO):
        return MAX_RHO
    return float(brentq(lambda x: weekly_mean_correlation(x) - r, 0.0, MAX_RHO))


def estimate_rho(Twat_mod: np.ndarray, Twat_obs: np.ndarray, eval_mask: np.ndarray, segments: list,
                 timescale: str = 'weekly') -> float:
    """
    rho at the chosen time scale.

    'daily': the lag-1 correlation of consecutive days (`estimate_ar1_rho`).
    'weekly': the larger of that and the week-to-week estimate
    (`estimate_ar1_rho_weekly`). The week-to-week estimate captures errors that
    persist for weeks; taking the larger of the two means the result is never
    less persistent than the day-to-day correlation shows. For AR(1) errors
    both estimate the same rho; the week-to-week estimate alone is imprecise
    when the errors are only weakly correlated, and taking the larger then errs
    towards wider intervals.
    """
    if timescale == 'daily':
        rho = estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments)
    elif timescale == 'weekly':
        rho = max(estimate_ar1_rho(Twat_mod, Twat_obs, eval_mask, segments),
                  estimate_ar1_rho_weekly(Twat_mod, Twat_obs, eval_mask, segments))
    else:
        raise ValueError(f"Invalid rho_timescale '{timescale}'. Must be 'weekly' or 'daily'.")
    if rho >= MAX_RHO:
        print(f"Warning: rho reached its upper limit of {MAX_RHO}: the model's errors persist for months. This "
              "usually means a systematic error, such as a bias in one season (see the "
              "bias_by_month output). Intervals for multi-week quantities will be wide, and the "
              "model may not suit this river.")
    return rho


def build_ar1_runs(valid_mask: np.ndarray, segments: list) -> list:
    """
    Partition valid, in-segment time indices into maximal runs of temporally
    contiguous days.

    A "run" is what the AR(1) whitening transform (see `ar1_whitened_stats`)
    treats as a single uninterrupted realization of the process: consecutive
    valid days within the same segment are chained together, while a missing
    day (invalid in `valid_mask`) or a segment boundary starts a new run. This
    reuses exactly the adjacency test in `estimate_ar1_rho` above, so the same
    pairs that inform the `rho` estimate are the ones treated as correlated by
    the likelihood that consumes it.
    """
    runs = []
    for start, end in segments:
        current = []
        for t in range(start, end + 1):
            if valid_mask[t]:
                current.append(t)
            else:
                if current:
                    runs.append(np.array(current, dtype=np.int64))
                current = []
        if current:
            runs.append(np.array(current, dtype=np.int64))
    return runs


def ar1_whitened_stats(residuals: np.ndarray, rho: float, runs: list, independent: bool = False) -> tuple:
    """
    Whiten the scored `residuals` (on the rows of `runs`, taken together in date order)
    with the AR(1) transform and return the sufficient statistics for the concentrated
    AR(1) log-likelihood.

    Values d days apart have correlation rho**d, so after a gap the step is
    `u = (e[t] - rho**d * e[prev]) * sqrt((1 - rho**2) / (1 - rho**(2d)))`, and the first
    value is `u = e[0] * sqrt(1 - rho**2)`. For consecutive days (d = 1) this is the usual
    `e[t] - rho * e[t-1]`; across a long gap it tends to a fresh start. The u are iid under
    the AR(1) model with the innovation variance, so their sum of squares is the AR(1)
    analogue of the iid SSE. (A gap used to start a new, independent run, which treated
    values on either side of a short gap as unrelated.) With `independent` every value is
    a fresh start (block means with weekly or monthly scoring).

    Returns
    -------
    sse_u : float
        Sum of squared whitened residuals.
    n : int
        Number of residuals.
    log_scale : float
        Sum of the log of the scale factors, `0.5 * log(1 - rho**2)` for the first value
        and `0.5 * log((1 - rho**2) / (1 - rho**(2d)))` for each later one (0 for d = 1):
        the Jacobian term of the concentrated log-likelihood.
    """
    if not runs:
        return 0.0, 0, 0.0
    pos = np.sort(np.concatenate(runs).astype(np.int64))
    e = residuals[pos].astype(np.float64)
    n = len(e)
    r2 = 1.0 - rho ** 2
    if independent or n == 1:
        scale = np.full(n, np.sqrt(r2))
        u = e * scale
    else:
        decay = rho ** np.diff(pos).astype(np.float64)
        scale = np.concatenate([[np.sqrt(r2)], np.sqrt(r2 / (1.0 - decay ** 2))])
        u = np.concatenate([[e[0]], e[1:] - decay * e[:-1]]) * scale
    return float(np.sum(u ** 2)), n, float(np.sum(np.log(scale)))


def generate_ar1_noise(n_tot: int, sigma: float, rho: float, segments: list, rng: np.random.Generator) -> np.ndarray:
    """
    Generate exact stationary AR(1) noise over the specified segments.

    Indices outside the specified segments remain 0.0.
    """
    noise = np.zeros(n_tot)

    for start, end in segments:
        L = end - start + 1
        eps = rng.standard_normal(L)
        epsilon = np.empty(L)

        epsilon[0] = sigma * eps[0]
        if L > 1:
            epsilon[1:] = sigma * np.sqrt(1 - rho**2) * eps[1:]

        noise[start:end+1] = scipy.signal.lfilter([1.0], [1.0, -rho], epsilon)

    return noise


# --- Seasonal error size ---------------------------------------------------------------------------

SIGMA_MONTH_MIN_DAYS = 30        # a month's error size is measured if it has at least this many scored days
_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.float64)
_MONTH_START = np.concatenate([[0.0], np.cumsum(_DAYS_IN_MONTH)[:-1]])
_MONTH_MIDDLE = _MONTH_START + _DAYS_IN_MONTH / 2.0


def _year_position(months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Position of each date in a 365-day year (0 at the start of 1 January), at the middle of the day;
    29 February counts as the end of 28 February."""
    months = np.asarray(months, dtype=np.int64)
    days = np.minimum(np.asarray(days, dtype=np.float64), _DAYS_IN_MONTH[months - 1])
    return _MONTH_START[months - 1] + days - 0.5


def monthly_sigma_factors(residuals: np.ndarray, months: np.ndarray, days: np.ndarray,
                          min_days: int = SIGMA_MONTH_MIN_DAYS):
    """
    The size of the model's daily errors in each calendar month, relative to its size over the
    whole year, from the residuals of the scored days (with their months and days of the month).
    Each month's factor is its root-mean-square residual; a month with fewer than `min_days`
    scored days is interpolated from the nearest measured months on either side (around the
    year). The factors are scaled so that `daily_sigma_factor` has a mean square of 1 over the
    scored days: the errors then have the same overall size as with a constant sigma, spread
    differently over the year. Returns 12 factors (all 1.0 when fewer than two months are
    measured) and the number of scored days in each month.
    """
    e = np.asarray(residuals, dtype=np.float64)
    months = np.asarray(months, dtype=np.int64)
    counts = np.bincount(months - 1, minlength=12)[:12]
    measured = counts >= min_days
    if len(e) == 0 or not np.any(e != 0.0) or measured.sum() < 2:
        return np.ones(12), counts
    rms = np.full(12, np.nan)
    for m in np.flatnonzero(measured):
        rms[m] = np.sqrt(np.mean(e[months == m + 1] ** 2))
    known = np.flatnonzero(measured)
    factors = np.interp(np.arange(12), known, rms[known], period=12)
    scale = np.sqrt(np.mean(daily_sigma_factor(factors, months, days) ** 2))
    return factors / scale, counts


def daily_sigma_factor(factors: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """The error-size factor of each date: the monthly factors placed at the middle of their months
    and interpolated linearly between them, around the year, so the error size changes smoothly."""
    pos = _year_position(months, days)
    return np.interp(pos, _MONTH_MIDDLE, np.asarray(factors, dtype=np.float64), period=365.0)
