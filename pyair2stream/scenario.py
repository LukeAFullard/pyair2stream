"""
Helpers for working with saved MCMC/forward-prediction ensembles.

`optimization.py`'s `uncertainty_options.save_ensemble` option writes the full
(n_samples, n_days) matrix of noisy simulated trajectories that back a percentile
prediction envelope, alongside the calendar dates of its columns, as a compressed
`.npz` file. Percentile bands alone cannot produce aggregate statistics -- the p5 of
a 7-day rolling mean is not the 7-day rolling mean of the p5 series -- so anything
downstream that needs degree-days, threshold-exceedance counts, or a paired
scenario comparison must operate on the raw ensemble instead.

See docs/audit/04_uncertainty_and_mcmc.md, Defect B / 4.2.
"""

import numpy as np
import pandas as pd


def load_ensemble(path: str):
    """
    Load a saved ensemble `.npz` file.

    Parameters
    ----------
    path : str
        Path to a `.npz` file written by `optimization.py` (e.g.
        `MCMC_ensemble_<station>_<series>_<time_res>.npz` or
        `Forward_Prediction_Ensemble_<station>_<series>_<time_res>.npz`).

    Returns
    -------
    ensemble : ndarray, shape (n_samples, n_days)
        Raw noisy simulated trajectories, one row per posterior/parameter draw.
    dates : pandas.DatetimeIndex, length n_days
        Calendar dates of the ensemble's columns.
    """
    with np.load(path) as npz:
        ensemble = npz['simulations']
        year = npz['year']
        month = npz['month']
        day = npz['day']
    dates = pd.DatetimeIndex(pd.to_datetime({'year': year, 'month': month, 'day': day}))
    return ensemble, dates


def aggregate(ensemble: np.ndarray, dates, how: str = 'mean', freq: str = '7D') -> np.ndarray:
    """
    Resample each ensemble member (row) over `freq`, independently.

    This is the correct way to build, e.g., a 7-day rolling/blocked mean prediction
    interval: aggregate first, per draw, then take percentiles across draws --
    never the reverse.

    Parameters
    ----------
    ensemble : ndarray, shape (n_samples, n_days)
    dates : array-like of datetime-like, length n_days
    how : str
        Any reduction name supported by `pandas.Resampler` (e.g. 'mean', 'sum', 'max').
    freq : str
        Any pandas offset alias (e.g. '7D', 'MS').

    Returns
    -------
    ndarray, shape (n_samples, n_periods)
    """
    ensemble = np.asarray(ensemble)
    dates = pd.DatetimeIndex(dates)
    aggregated_rows = [
        getattr(pd.Series(row, index=dates).resample(freq), how)().to_numpy()
        for row in ensemble
    ]
    return np.array(aggregated_rows)


def exceedance(ensemble: np.ndarray, threshold: float, consecutive_days: int = 1) -> np.ndarray:
    """
    Per ensemble member, count days exceeding `threshold`.

    Parameters
    ----------
    ensemble : ndarray, shape (n_samples, n_days)
    threshold : float
    consecutive_days : int
        If 1 (default), returns the total number of days above `threshold`. If > 1,
        returns the number of days that are part of a run of at least
        `consecutive_days` consecutive days above `threshold` -- a simple measure of
        sustained-exceedance duration (e.g. a multi-day thermal-stress event).

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    ensemble = np.asarray(ensemble)
    above = ensemble > threshold
    if consecutive_days <= 1:
        return above.sum(axis=1)

    counts = np.zeros(ensemble.shape[0], dtype=np.int64)
    for i, row in enumerate(above):
        run_len = 0
        total = 0
        for val in row:
            if val:
                run_len += 1
            else:
                if run_len >= consecutive_days:
                    total += run_len
                run_len = 0
        if run_len >= consecutive_days:
            total += run_len
        counts[i] = total
    return counts


def paired_difference(ens_a: np.ndarray, ens_b: np.ndarray) -> np.ndarray:
    """
    Row-aligned difference between two ensembles.

    Requires both ensembles to have been generated from the SAME parameter draws in
    the SAME order (e.g. two `forward_mode` runs against the same
    `mcmc_chain_path`/`n_samples`/`random_seed`, one on observed/naturalised
    discharge and one on an abstraction scenario) -- see
    docs/audit/09_study_design_notes.md. This is the function both the water
    abstraction and climate projection studies actually need: a credible interval on
    a *difference*, not just on each scenario separately.

    Raises
    ------
    ValueError
        If the two ensembles do not have identical shape.
    """
    ens_a = np.asarray(ens_a)
    ens_b = np.asarray(ens_b)
    if ens_a.shape != ens_b.shape:
        raise ValueError(
            f"paired_difference requires both ensembles to have identical shape "
            f"(same parameter draws in the same order); got {ens_a.shape} and {ens_b.shape}."
        )
    return ens_a - ens_b
