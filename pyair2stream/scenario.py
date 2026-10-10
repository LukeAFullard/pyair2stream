"""
Helpers for working with saved MCMC/forward-prediction ensembles.

`optimization.py`'s `uncertainty_options.save_ensemble` option writes the full
(n_samples, n_days) matrix of noisy simulated trajectories that back a percentile
prediction envelope, alongside the calendar dates of its columns, as a compressed
`.npz` file. Percentile bands alone cannot produce aggregate statistics -- the p5 of
a 7-day rolling mean is not the 7-day rolling mean of the p5 series -- so anything
downstream that needs degree-days, threshold-exceedance counts, or a paired
scenario comparison must operate on the raw ensemble instead.

See docs/METHODS.md §13.
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


def _warn_days_without_value(ensemble: np.ndarray, consequence: str) -> None:
    """Print a warning when some days of an ensemble have no simulated value (NaN)."""
    missing = np.isnan(np.atleast_2d(ensemble)).any(axis=0)
    if missing.any():
        print(f"Warning: {int(missing.sum())} day(s) have no simulated value (for example gaps in gap-tolerant "
              f"mode); {consequence}.")


def aggregate(ensemble: np.ndarray, dates, how: str = 'mean', freq: str = '7D',
              min_days=None, return_periods: bool = False):
    """
    Resample each ensemble member (row) over `freq`, independently.

    This is the correct way to build, e.g., a 7-day blocked mean prediction
    interval: aggregate first, per draw, then take percentiles across draws --
    never the reverse.

    A period counts only if enough of its days have a simulated value: by default
    all of them (`min_days=None`), so a week or month that the dates cover only in
    part (at either end of the file) or that includes days without a value (gaps
    in gap-tolerant mode) gets no value (NaN), with a warning. A 3-day mean is not
    a 7-day mean, and a sum over part of a month understates it. With
    `min_days=N`, a period with at least N days with a value is computed from
    those days.

    Parameters
    ----------
    ensemble : ndarray, shape (n_samples, n_days)
    dates : array-like of datetime-like, length n_days (consecutive or not, one per day)
    how : str
        Any reduction name supported by `pandas.Resampler` (e.g. 'mean', 'sum', 'max').
    freq : str
        Any pandas offset alias of a day or longer (e.g. '7D', 'MS'). '7D' blocks start
        on the first date.
    min_days : int, optional
        The fewest days with a value a period needs. Default: every day of the period.
    return_periods : bool
        Also return the label of each period (pandas' resample label: the first day of
        a '7D' block or an 'MS' month).

    Returns
    -------
    ndarray, shape (n_samples, n_periods); with `return_periods`, also a
    pandas.DatetimeIndex of length n_periods.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    dates = pd.DatetimeIndex(dates)
    if min_days is not None and (int(min_days) != min_days or min_days < 1):
        raise ValueError(f"min_days must be a whole number of days, at least 1, got {min_days!r}.")
    offset = pd.tseries.frequencies.to_offset(freq)
    if isinstance(offset, (pd.offsets.Day, pd.offsets.Tick)):     # Day is not a Tick from pandas 3
        # Fixed-length blocks ('7D'), starting on the first date: every block has freq's days.
        days_per_block = (offset.n if isinstance(offset, pd.offsets.Day)
                          else pd.Timedelta(offset) / pd.Timedelta(days=1))
        if days_per_block < 1 or days_per_block != int(days_per_block):
            raise ValueError(f"freq must be a whole number of days or a calendar period, got {freq!r}.")
        length = None
    else:
        # Calendar periods (weeks, months, years): their lengths from a daily calendar that
        # covers whole periods beyond both ends of the file.
        margin = pd.Timedelta(days=1100)
        calendar = pd.date_range(dates[0].normalize() - margin, dates[-1].normalize() + margin, freq='D')
        length = pd.Series(1.0, index=calendar).resample(freq).count()

    def resample(series):
        return series.resample(freq)

    aggregated_rows, enough = [], None
    for row in ensemble:
        series = pd.Series(row, index=dates)
        values = getattr(resample(series), how)()
        days = resample(series).count()
        if min_days is not None:
            need = float(min_days)
        elif length is None:
            need = days_per_block
        else:
            need = length.reindex(values.index).to_numpy()
        ok = days.to_numpy() >= need
        aggregated_rows.append(np.where(ok, values.to_numpy(dtype=np.float64), np.nan))
        enough = ok if enough is None else enough & ok
    result = np.array(aggregated_rows)
    periods = values.index if len(ensemble) else pd.DatetimeIndex([])

    if enough is not None and not enough.all():
        rule = ("fewer than all their days" if min_days is None
                else f"fewer than min_days={int(min_days)} days")
        print(f"Warning: {int((~enough).sum())} of {len(enough)} period(s) ({freq}) have {rule} with a "
              "simulated value (the dates cover them only in part, or some days have no value); "
              f"they have no {how} (NaN)." + (" Set min_days to accept partial periods." if min_days is None else ""))
    if min_days is not None:
        _warn_days_without_value(ensemble, f"each period's {how} uses only the days that have one")
    return (result, periods) if return_periods else result


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
    ensemble = np.asarray(ensemble, dtype=np.float64)
    _warn_days_without_value(ensemble, "they are not counted as above the threshold")
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


YEARLY_STATISTICS = ("highest daily mean", "highest 7-day mean", "days above threshold")


def year_statistics(ensemble: np.ndarray, dates, threshold: float, window: int = 7, years=None,
                    partial_years: str = "skip") -> dict:
    """
    Three statistics of each year, for each ensemble member (row): the highest daily
    mean, the highest `window`-day moving mean (the day and the `window - 1` days
    before it, all in the same year), and the number of days above `threshold`.

    Days that are NaN are not used: a moving mean counts only if all its days are
    present, and a NaN day is not counted as above the threshold. To compare with
    measurements, set the simulations to NaN on the days that were not measured, so
    both are computed over the same days.

    Parameters
    ----------
    ensemble : ndarray, shape (n_samples, n_days), or (n_days,) for one series
    dates : array-like of datetime-like, length n_days (consecutive days)
    threshold : float
    window : int
    years : array-like of int, length n_days, optional
        Year label of each day (for example water years); default: calendar year.
    partial_years : "skip" (default) or "keep"
        A record that starts or ends part-way through a year covers only part of
        its first or last year. Its highest values and its count of days above
        the threshold would then describe only those days, not the year, so such
        years are left out, with a warning. "keep" includes them.

    Returns
    -------
    dict
        {year: {statistic name: ndarray of shape (n_samples,)}}, names as in
        YEARLY_STATISTICS (the moving mean is named "highest 7-day mean" whatever
        `window` is). A statistic with no usable day is NaN.
    """
    if partial_years not in ("skip", "keep"):
        raise ValueError(f"partial_years must be 'skip' or 'keep', got {partial_years!r}")
    ens = np.atleast_2d(np.asarray(ensemble, dtype=np.float64))
    labels = np.asarray(pd.DatetimeIndex(dates).year if years is None else years)
    skipped = partial_year_labels(dates, labels, calendar_years=years is None) if partial_years == "skip" else []
    if skipped:
        print(f"Warning: year(s) {', '.join(str(int(y)) for y in skipped)} are only partly covered by the dates, "
              "so their yearly statistics are left out (partial_years='keep' includes them).")
    out = {}
    for year in np.unique(labels):
        if year in skipped:
            continue
        x = ens[:, labels == year]
        ok = np.isfinite(x)
        with np.errstate(invalid="ignore"):
            highest = np.where(ok.any(axis=1), np.max(np.where(ok, x, -np.inf), axis=1), np.nan)
        # Moving means from running sums; a window counts only if all its days are present.
        csum = np.concatenate([np.zeros((len(x), 1)), np.cumsum(np.where(ok, x, 0.0), axis=1)], axis=1)
        cnt = np.concatenate([np.zeros((len(x), 1)), np.cumsum(ok, axis=1)], axis=1)
        if x.shape[1] >= window:
            full = (cnt[:, window:] - cnt[:, :-window]) == window
            means = (csum[:, window:] - csum[:, :-window]) / window
            week = np.where(full.any(axis=1), np.max(np.where(full, means, -np.inf), axis=1), np.nan)
        else:
            week = np.full(len(x), np.nan)
        above = np.sum(ok & (np.where(ok, x, -np.inf) > threshold), axis=1).astype(np.float64)
        out[int(year)] = {YEARLY_STATISTICS[0]: highest, YEARLY_STATISTICS[1]: week,
                          YEARLY_STATISTICS[2]: np.where(ok.any(axis=1), above, np.nan)}
    return out


def partial_year_labels(dates, labels, calendar_years: bool = True) -> list:
    """
    The year labels that the dates cover only in part. The dates are consecutive days,
    so only the first and the last year can be partial. A year is whole if the record
    starts on its first day and ends on its last. The first day of a year is 1 January
    for calendar years; for other labels (water years) it is the day on which the label
    changes inside the record. With no such change (one label, not a calendar year), a
    year is taken as whole if it has at least 365 days.
    """
    idx = pd.DatetimeIndex(dates)
    labels = np.asarray(labels)
    if len(idx) == 0:
        return []
    if calendar_years:
        start = (1, 1)
    else:
        change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        start = (idx[change[0]].month, idx[change[0]].day) if len(change) else None
    if start is None:
        return [labels[0]] if len(idx) < 365 else []
    partial = []
    if (idx[0].month, idx[0].day) != start:
        partial.append(labels[0])
    after = idx[-1] + pd.Timedelta(days=1)
    if (after.month, after.day) != start and labels[-1] not in partial:
        partial.append(labels[-1])
    return partial


def pit(simulated: np.ndarray, value: float, rng: np.random.Generator) -> float:
    """
    Share of simulated values below `value` (the probability integral transform).
    Ties, which occur for day counts, are split at random. If the predicted
    distribution is right, this is uniform between 0 and 1, so for any level L the
    measured value lies inside the central L% range (PIT between 0.5 - L/200 and
    0.5 + L/200) L% of the time (`inside_range`).
    """
    s = np.asarray(simulated, dtype=np.float64)
    s = s[np.isfinite(s)]
    return float((np.sum(s < value) + rng.random() * np.sum(s == value)) / len(s))


def central_range(values: np.ndarray, level: float, axis=None):
    """
    The central `level`% range of `values` (e.g. 90: the 5th and 95th percentiles),
    along `axis`. NaN values are ignored. Returns (lower, upper).
    """
    if not (0.0 < level < 100.0):
        raise ValueError(f"level must be strictly between 0 and 100 (per cent), got {level}")
    lo, hi = np.nanpercentile(np.asarray(values, dtype=np.float64), [50.0 - level / 2, 50.0 + level / 2], axis=axis)
    return lo, hi


def inside_range(pit_value, level: float):
    """Whether a PIT value (`pit`) lies inside the central `level`% range."""
    return np.abs(np.asarray(pit_value, dtype=np.float64) - 0.5) <= level / 200.0


def correct_statistic(values: np.ndarray, deviations, seed=None) -> np.ndarray:
    """
    Correct the simulated values of a yearly statistic for the model's typical error
    in that statistic, measured by cross-validation (docs/METHODS.md §13).

    `deviations` are, for each year held out in a cross-validation, the measured
    value minus the median of the simulations (`deviation` in
    `cv_yearly_statistics.csv`, for the same statistic, threshold and season). Each
    simulated value is shifted by their mean, plus a random draw of the uncertainty
    of that mean (standard error times a Student t variable with n - 1 degrees of
    freedom, n the number of years). With few years the shift is uncertain, and the
    corrected range is wider.

    Parameters
    ----------
    values : ndarray
        The statistic in each simulation (e.g. each simulation's highest 7-day mean).
    deviations : array-like
        At least 3 cross-validated deviations of the same statistic.
    seed : int, optional

    Returns
    -------
    ndarray, same shape as `values`
    """
    d = np.asarray(deviations, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        raise ValueError(f"correct_statistic needs the deviations of at least 3 held-out years, got {n}.")
    values = np.asarray(values, dtype=np.float64)
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    return values + float(np.mean(d)) + se * rng.standard_t(n - 1, size=values.shape)


def paired_difference(ens_a: np.ndarray, ens_b: np.ndarray) -> np.ndarray:
    """
    Row-aligned difference between two ensembles.

    Requires both ensembles to have been generated from the SAME parameter draws in
    the SAME order (e.g. two `forward_mode` runs against the same
    `mcmc_chain_path`/`n_samples`/`random_seed`, one on observed/naturalised
    discharge and one on an abstraction scenario). This is the function both the water
    abstraction and climate projection studies actually need: a credible interval on
    a *difference*, not just on each scenario separately.

    The residual noise added to each draw is fixed by the draw's chain row, so the
    same draw carries the same noise in both runs and it cancels here: the spread of
    the difference is the parameter uncertainty of the effect. This assumes the
    model's error on a given day would be the same under both scenarios. On days a
    member is held at the ice floor (`Tice_cover`) in one run only, the noise does
    not cancel exactly.

    This only checks `.shape` -- it has no way to detect two ensembles that happen
    to have the same shape but were drawn from different (or differently-ordered,
    or differently-seeded) posterior samples, which silently produces a
    plausible-shaped but statistically meaningless "paired" difference. **Prefer
    `paired_difference_from_files()`**, which additionally verifies the two runs'
    saved provenance (source chain, requested sample indices, and the indices that
    actually survived per-draw divergence filtering) before differencing --
    see docs/METHODS.md §13. Use this shape-only function directly only when you
    are certain both arrays came from the same in-process draw (e.g. you built both
    yourself in the same script from the same `sample_indices` array).

    Raises
    ------
    ValueError
        If the two ensembles do not have identical shape, or do not have values on the
        same days (for example zero-flow days left out as gaps in only one of the runs).
    """
    ens_a = np.asarray(ens_a, dtype=np.float64)
    ens_b = np.asarray(ens_b, dtype=np.float64)
    if ens_a.shape != ens_b.shape:
        raise ValueError(
            f"paired_difference requires both ensembles to have identical shape "
            f"(same parameter draws in the same order); got {ens_a.shape} and {ens_b.shape}."
        )
    differ = (np.isnan(ens_a) != np.isnan(ens_b)).any(axis=0)
    if differ.any():
        first = int(np.argmax(differ))
        raise ValueError(
            f"The two runs do not have values on the same days: {int(differ.sum())} day(s) are simulated in one "
            f"and not the other (first: day {first} of the series). A difference there would compare a value with "
            "nothing. This happens, for example, when zero-flow days are left out as gaps in gap-tolerant mode in "
            "only one run; make both runs simulate the same days (USER_GUIDE.md §9.2)."
        )
    return ens_a - ens_b


def _load_ensemble_provenance(ensemble_path: str) -> dict:
    """
    Load the provenance sidecar JSON alongside a saved ensemble `.npz`, written by
    `optimization.forward_mode()`'s prediction-interval block (or
    `optimization._run_mcmc_uncertainty()`) next to the ensemble file, e.g.
    `Forward_Prediction_Ensemble_<station>_<series>_<time_res>_meta.json` for
    `Forward_Prediction_Ensemble_<station>_<series>_<time_res>.npz`.
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
    with this function. See docs/METHODS.md §13 for the full workflow.

    Checks, in order: the two runs' source MCMC/posterior chain (content hash and
    row count), the number of samples requested, the error settings (noise model,
    σ and ρ: the added error cancels only if they match), the exact `sample_indices` drawn,
    and `valid_draw_indices` -- the subset of those indices that actually survived
    per-draw divergence filtering
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
                "must be forward_mode() (or DE-MCMC envelope) calls against "
                "the SAME posterior chain -- see docs/METHODS.md §13."
            )

    # The daily error added to a draw cancels in the difference only if both runs used the
    # same error model, size and persistence.
    for key, label in (('noise_model', 'error model (noise_model)'), ('residual_sigma', 'error size σ'),
                       ('rho', 'error persistence ρ')):
        a, b = meta_a.get(key), meta_b.get(key)
        same = a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and np.isclose(a, b))
        if not same:
            raise ValueError(
                f"paired_difference_from_files: the {label} differs between '{path_a}' ({a!r}) and "
                f"'{path_b}' ({b!r}). The daily error added to each draw cancels in a paired difference only "
                "if both runs used the same error settings; otherwise the difference's spread would include "
                "that error, not only the parameter uncertainty. Run both scenarios with the same "
                "uncertainty_options and forward_options.residual_sigma (docs/METHODS.md §13)."
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
