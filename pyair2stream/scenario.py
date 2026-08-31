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

import json
import os

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

    This only checks `.shape` -- it has no way to detect two ensembles that happen
    to have the same shape but were drawn from different (or differently-ordered,
    or differently-seeded) posterior samples, which silently produces a
    plausible-shaped but statistically meaningless "paired" difference. **Prefer
    `paired_difference_from_files()`**, which additionally verifies the two runs'
    saved provenance (source chain, requested sample indices, and the indices that
    actually survived per-draw divergence filtering) before differencing --
    see docs/audit/12_ensemble_provenance_and_pairing.md and
    docs/MCMC_uncertainty.md. Use this shape-only function directly only when you
    are certain both arrays came from the same in-process draw (e.g. you built both
    yourself in the same script from the same `sample_indices` array).

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


def _load_ensemble_provenance(ensemble_path: str) -> dict:
    """
    Load the provenance sidecar JSON alongside a saved ensemble `.npz`, written by
    `optimization.forward_mode()`'s prediction-interval block (or
    `optimization._run_mcmc_uncertainty()`) next to the ensemble file, e.g.
    `Forward_Prediction_Ensemble_<station>_<series>_<time_res>_meta.json` for
    `Forward_Prediction_Ensemble_<station>_<series>_<time_res>.npz`. See
    docs/audit/12_ensemble_provenance_and_pairing.md.
    """
    meta_path = ensemble_path.replace('.npz', '_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No provenance sidecar found for ensemble '{ensemble_path}' (expected "
            f"'{meta_path}'). paired_difference_from_files() requires the sidecar "
            "written alongside a forward_mode() prediction-interval ensemble (or an "
            "MCMC envelope ensemble); use the shape-only paired_difference() instead "
            "if you are certain both ensembles were drawn from the same parameter "
            "draws in the same order."
        )
    with open(meta_path, 'r') as f:
        return json.load(f)


def paired_difference_from_files(path_a: str, path_b: str) -> np.ndarray:
    """
    Provenance-checked, recommended alternative to `paired_difference()`.

    Loads both raw ensembles (`load_ensemble`) and the provenance sidecar JSON
    saved alongside each `.npz` (see `_load_ensemble_provenance`), and verifies the
    two runs actually drew the SAME posterior parameter samples in the SAME order
    -- not just that the two arrays happen to have the same shape -- before
    differencing. This is the workflow both the water-abstraction and
    climate-projection studies need: calibrate once, run `forward_mode()` on
    scenario A with `uncertainty_options.save_ensemble: true`, then run it again on
    scenario B with `forward_options.reuse_sample_indices_from` pointing at
    scenario A's saved sidecar (rather than relying on matching `random_seed`
    across two separate config files/processes), then pair the two saved ensembles
    with this function. See docs/audit/12_ensemble_provenance_and_pairing.md and
    docs/MCMC_uncertainty.md for the full workflow.

    Checks, in order: the two runs' source MCMC/posterior chain (content hash and
    row count), the number of samples requested, the exact `sample_indices` drawn,
    and `valid_draw_indices` -- the subset of those indices that actually survived
    per-draw divergence filtering (docs/audit/11_ensemble_divergence_handling.md)
    and therefore ended up as rows in the saved ensemble. `valid_draw_indices` is
    the authoritative check: two runs can request identical `sample_indices` and
    still end up with differently-excluded (and therefore misaligned) rows if one
    scenario's discharge diverges on draws the other's does not.

    Raises
    ------
    FileNotFoundError
        If either ensemble has no provenance sidecar alongside it.
    ValueError
        If the two runs' source chain, requested sample indices, or the indices
        that actually survived divergence filtering do not match exactly -- any of
        which means the two ensembles are not row-aligned, even if their shapes
        happen to match.
    """
    meta_a = _load_ensemble_provenance(path_a)
    meta_b = _load_ensemble_provenance(path_b)

    for key, label in (
        ('chain_content_sha256', 'source chain content'),
        ('chain_n_rows', 'source chain row count'),
        ('n_draws_requested', 'requested sample count'),
    ):
        if meta_a.get(key) != meta_b.get(key):
            raise ValueError(
                f"paired_difference_from_files: {label} differs between '{path_a}' "
                f"({meta_a.get(key)!r}) and '{path_b}' ({meta_b.get(key)!r}). Both runs "
                "must be forward_mode() (or DE-MCMC/DE-CV-MCMC envelope) calls against "
                "the SAME posterior chain -- see docs/MCMC_uncertainty.md."
            )

    if meta_a.get('sample_indices') != meta_b.get('sample_indices'):
        raise ValueError(
            "paired_difference_from_files: the requested `sample_indices` differ "
            f"between '{path_a}' and '{path_b}'. Use "
            "`forward_options.reuse_sample_indices_from` on the second run to reuse "
            "the exact indices drawn by the first, rather than drawing a fresh, "
            "independent sample."
        )

    if meta_a.get('valid_draw_indices') != meta_b.get('valid_draw_indices'):
        raise ValueError(
            "paired_difference_from_files: the SAME sample_indices were requested for "
            f"both runs, but a different subset survived per-draw divergence filtering "
            f"in '{path_a}' vs '{path_b}' (see each sidecar's `excluded_draws`). The two "
            "saved ensembles are therefore not row-aligned even though they may have "
            "the same shape -- a paired difference between them would silently compare "
            "unrelated parameter draws."
        )

    ens_a, dates_a = load_ensemble(path_a)
    ens_b, dates_b = load_ensemble(path_b)

    if not dates_a.equals(dates_b):
        raise ValueError(
            f"paired_difference_from_files: dates do not match between '{path_a}' "
            f"and '{path_b}'. Both ensemble files must cover the exact same date range."
        )

    return paired_difference(ens_a, ens_b)
