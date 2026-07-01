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

    # Clip strictly to [0.0, 0.99] as required
    return float(np.clip(rho, 0.0, 0.99))


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
