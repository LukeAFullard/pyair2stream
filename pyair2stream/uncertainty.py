"""
Uncertainty quantification and noise modeling for pyair2stream.

This module provides tools for estimating autoregressive (AR1) properties
from model residuals and generating structurally consistent noise envelopes
for probabilistic forward predictions.
"""

import numpy as np
import scipy.signal
import logging

MIN_PAIRS_FOR_RHO_ESTIMATE = 30

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
        logging.warning(f"Only {n_valid_pairs} valid residual pairs available for AR(1) estimation (need >= {MIN_PAIRS_FOR_RHO_ESTIMATE}). Falling back to rho=0.0.")
        return 0.0

    pairs_t0 = np.array(pairs_t0)
    pairs_t1 = np.array(pairs_t1)

    # Calculate sample Pearson correlation coefficient
    # np.corrcoef returns a 2x2 matrix, we want the off-diagonal element
    rho = np.corrcoef(pairs_t0, pairs_t1)[0, 1]

    if np.isnan(rho):
        logging.warning("AR(1) rho estimation resulted in NaN. Falling back to rho=0.0.")
        return 0.0

    # Clip strictly to [0.0, 0.99]. Note: the lower bound of 0.0 enforces non-negative serial correlation.
    # Hydrological water temperature residuals are typically positively autocorrelated (persistence);
    # negative serial correlation is disallowed by design in empirical noise estimation.
    return float(np.clip(rho, 0.0, 0.99))


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


def ar1_whitened_stats(residuals: np.ndarray, rho: float, runs: list) -> tuple:
    """
    Whiten `residuals` within each run of `runs` using the AR(1) transform and
    return the sufficient statistics for the concentrated AR(1) log-likelihood.

    For a run `e[0..L-1]`, the whitened residuals are
    `u[0] = e[0] * sqrt(1 - rho**2)` and `u[t] = e[t] - rho * e[t-1]` for
    `t >= 1`; these are iid under the AR(1) model, so their sum of squares is
    the AR(1) analogue of the iid SSE.

    Returns
    -------
    sse_u : float
        Sum of squared whitened residuals across all runs.
    n : int
        Total number of residuals (sum of run lengths).
    n_runs : int
        Number of runs. Each run independently contributes one
        `0.5 * log(1 - rho**2)` term to the concentrated log-likelihood (see
        `docs/audit/04_uncertainty_and_mcmc.md`, 4.1), so the total correction
        scales with the number of runs, not just with N.
    """
    sse_u = 0.0
    n = 0
    for run in runs:
        e = residuals[run]
        u = np.empty_like(e, dtype=np.float64)
        u[0] = e[0] * np.sqrt(1.0 - rho ** 2)
        if len(e) > 1:
            u[1:] = e[1:] - rho * e[:-1]
        sse_u += float(np.sum(u ** 2))
        n += len(e)
    return sse_u, n, len(runs)


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
