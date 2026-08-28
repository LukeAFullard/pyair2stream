"""
High-level ODE integration and simulation orchestration for air2stream.

This module provides the main entry points for running the air2stream model,
including handling missing data segments (gap-tolerant mode) and delegating
the heavy numeric lifting to the Numba-compiled functions.
"""

import numpy as np
import math
import pandas as pd
from .config import CommonData, PI, TTT

# Sanity bound on simulated water temperature (degC). See docs/audit/02_numerical_integration.md:
# explicit integrators (RK4/RK2/EUL) can diverge silently on scenario discharge that differs
# from the calibration record, producing either huge or plausible-but-wrong numbers with no
# error/NaN. This is the default for `max_plausible_twat`.
TWAT_SANITY_MAX = 60.0

# One-step amplification-factor stability limits for B = (a3 + a8*theta)/theta**a4, i.e. the
# ODE's linear decay rate (1/day) times the fixed dt=1 day step. CRN and EXP are unconditionally
# stable. See docs/audit/02_numerical_integration.md for the derivation.
STABILITY_LIMITS = {'EUL': 2.0, 'RK2': 2.0, 'RK4': 2.785, 'CRN': np.inf, 'EXP': np.inf}

# "Error if more than a small fraction of days exceed [the stability limit]" (report 02, 2.2).
# This screening criterion is conservative, not exact (isolated high-theta days often simulate
# fine); it is a companion to the divergence guard (check_numerical_divergence), not a
# replacement for it.
STABILITY_ERROR_FRACTION = 0.10


class NumericalDivergenceError(RuntimeError):
    """
    Raised when a simulated water-temperature series is non-finite or exceeds a
    physically implausible bound.

    The air2stream ODE is linear in Tw with a discharge-dependent decay rate B; an
    explicit integrator (RK4/RK2/EUL) stable at the calibration discharge can be
    unstable at a different scenario discharge, producing either astronomical or
    plausible-but-wrong output with no NaN and no warning. See
    docs/audit/02_numerical_integration.md.
    """


def compute_B_series(data: CommonData) -> np.ndarray:
    """
    Compute B(t), the ODE's linear decay rate (1/day), for every day in `data.Q`
    using the current `data.par`. `B * dt` (dt is fixed at 1 day) governs the
    stability of the explicit integrators (see docs/audit/02_numerical_integration.md).

    Entries where discharge is invalid (missing sentinel, non-positive) or where the
    computation is undefined (e.g. a negative theta raised to a non-integer power)
    come back as NaN; callers should mask on validity/finiteness. If
    `data.min_theta_floor` is set, theta is floored the same way the integrators
    themselves floor it (see `check_nonpositive_discharge`), so this reports the B
    that will actually be used, not the raw (possibly non-finite) one.
    """
    p = data.par
    a3, a4, a8 = p[2], p[3], p[7]

    if data.version in (3, 5):
        return np.full(data.Q.shape, a3, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        theta = data.Q / data.Qmedia
        theta_floor = getattr(data, 'min_theta_floor', None)
        if theta_floor is not None:
            theta = np.where(theta < theta_floor, theta_floor, theta)
        if data.version == 7:
            B = a3 + a8 * theta
        else:  # versions 4 and 8
            DD = theta ** a4
            if data.version == 4:
                B = a3 / DD
            else:
                B = (a3 + a8 * theta) / DD
    return B


def stability_report(data: CommonData) -> dict:
    """
    Pre-flight stability screening for the explicit integrators (report 02, 2.2).

    Computes B = (a3 + a8*theta)/theta**a4 (version-dependent) over the whole
    forcing series and compares it against the current integrator's one-step
    stability limit. This is a conservative screening heuristic, not an exact
    verdict: isolated high-theta days often simulate fine because the transient
    decays before it compounds. Pair it with `check_numerical_divergence`, which
    catches actual divergence.
    """
    B = compute_B_series(data)

    if data.version in (3, 5):
        valid = np.ones(data.n_tot, dtype=np.bool_)
    else:
        valid = (data.Q != -999.0) & (data.Q > 0.0)
    valid &= np.isfinite(B)

    limit = STABILITY_LIMITS.get(data.mod_num, np.inf)

    if not np.any(valid):
        return {
            'mod_num': data.mod_num, 'limit': limit, 'max_B': None,
            'frac_exceeding': 0.0, 'n_exceeding': 0, 'n_valid': 0, 'worst': [],
        }

    idx_valid = np.nonzero(valid)[0]
    B_valid = B[valid]
    max_B = float(np.max(B_valid))
    exceed = B_valid > limit
    n_exceeding = int(np.sum(exceed))
    n_valid = int(B_valid.shape[0])
    frac_exceeding = n_exceeding / n_valid

    n_worst = min(5, n_valid)
    worst_order = np.argsort(-B_valid)[:n_worst]
    worst = []
    for k in worst_order:
        i = int(idx_valid[k])
        date = tuple(int(x) for x in data.date[i]) if data.date is not None else None
        worst.append({'index': i, 'date': date, 'B': float(B_valid[k])})

    return {
        'mod_num': data.mod_num,
        'limit': limit,
        'max_B': max_B,
        'frac_exceeding': frac_exceeding,
        'n_exceeding': n_exceeding,
        'n_valid': n_valid,
        'worst': worst,
    }


def warn_on_stability(data: CommonData, error_fraction: float = STABILITY_ERROR_FRACTION) -> dict:
    """
    Run `stability_report` and print a warning (or raise `NumericalDivergenceError`
    if too large a fraction of days exceed the limit) before a user-facing
    simulation. See docs/audit/02_numerical_integration.md, 2.2.
    """
    report = stability_report(data)

    if report['max_B'] is None or not np.isfinite(report['limit']):
        return report

    if report['max_B'] > report['limit']:
        worst = report['worst'][0] if report['worst'] else None
        worst_str = f" Worst day: {worst['date']} (B={worst['B']:.3f})." if worst else ""
        print(
            f"Warning: {report['n_exceeding']}/{report['n_valid']} days "
            f"({report['frac_exceeding']:.1%}) exceed the {report['mod_num']} stability limit "
            f"(B > {report['limit']:.3f}); max B = {report['max_B']:.3f}.{worst_str} "
            f"This is a screening heuristic, not a verdict (see "
            f"docs/audit/02_numerical_integration.md) -- consider CRN or EXP, especially for "
            f"scenario runs on discharge different from the calibration record."
        )
        if report['frac_exceeding'] > error_fraction:
            raise NumericalDivergenceError(
                f"{report['frac_exceeding']:.1%} of days exceed the {report['mod_num']} "
                f"stability limit (B > {report['limit']:.3f}), above the "
                f"error_fraction={error_fraction:.0%} threshold. Use CRN or EXP for this run, "
                f"or raise `stability_error_fraction` in the config if you have verified the "
                f"simulation is stable (see docs/audit/02_numerical_integration.md)."
            )

    return report


def _divergence_bad_mask(Twat_mod: np.ndarray, max_plausible_twat: float) -> np.ndarray:
    """Shared "bad" definition for `check_numerical_divergence`/`is_numerically_divergent`:
    a present (not the -999.0 missing sentinel) value that is non-finite or exceeds the
    sanity bound."""
    present = Twat_mod != -999.0
    non_finite = ~np.isfinite(Twat_mod)
    too_hot = np.isfinite(Twat_mod) & (Twat_mod > max_plausible_twat)
    return present & (non_finite | too_hot)


def is_numerically_divergent(data: CommonData, max_plausible_twat: float = None) -> bool:
    """
    Lightweight, non-raising sibling of `check_numerical_divergence`: True if
    `data.Twat_mod` currently contains any non-finite or implausibly large (present)
    value. Intended for a per-draw check inside an ensemble/posterior-sample loop
    (`optimization.forward_mode`'s prediction-interval loop,
    `optimization._run_mcmc_uncertainty`'s envelope loop), where a single bad draw
    should be excluded (or the batch aborted, per `on_divergent_draw`) rather than
    raising and losing the rest of the ensemble -- see
    docs/audit/11_ensemble_divergence_handling.md.
    """
    if max_plausible_twat is None:
        max_plausible_twat = getattr(data, 'max_plausible_twat', TWAT_SANITY_MAX)
    return bool(np.any(_divergence_bad_mask(data.Twat_mod, max_plausible_twat)))


def check_numerical_divergence(data: CommonData, max_plausible_twat: float = None) -> None:
    """
    Raise `NumericalDivergenceError` if `data.Twat_mod` contains non-finite values
    or exceeds a physically implausible sanity bound. See
    docs/audit/02_numerical_integration.md, 2.1.

    Intended for user-facing simulation paths (main.forward(), optimization.forward_mode(),
    sensitivity_analysis()) -- NOT the optimizer hot loop, where a diverged trial parameter
    set is a normal occurrence already handled via the NaN/penalty path in funcobj.
    """
    if max_plausible_twat is None:
        max_plausible_twat = getattr(data, 'max_plausible_twat', TWAT_SANITY_MAX)

    Twat_mod = data.Twat_mod
    bad = _divergence_bad_mask(Twat_mod, max_plausible_twat)

    if not np.any(bad):
        return

    idx = int(np.argmax(bad))
    date = tuple(int(x) for x in data.date[idx]) if data.date is not None else None

    theta = None
    B = None
    if data.Q is not None and data.Qmedia and data.Qmedia > 0 and data.Q[idx] not in (-999.0,):
        theta = float(data.Q[idx] / data.Qmedia)
        B_series = compute_B_series(data)
        if np.isfinite(B_series[idx]):
            B = float(B_series[idx])

    raise NumericalDivergenceError(
        f"Numerical divergence detected in the simulated water temperature at index {idx} "
        f"(date={date}): Twat_mod={Twat_mod[idx]!r} is non-finite or exceeds the sanity bound "
        f"max_plausible_twat={max_plausible_twat} (theta={theta}, B={B}), using integrator "
        f"'{data.mod_num}'. Explicit schemes (RK4/RK2/EUL) can be unstable at discharge "
        f"different from the calibration record even when stable at calibration. Use CRN "
        f"(the default) or EXP for scenario runs. See docs/audit/02_numerical_integration.md."
    )


def check_nonpositive_discharge(data: CommonData) -> None:
    """
    Raise `ValueError` if any non-positive discharge day (`Q <= 0`) is present in a
    non-gap-tolerant record for a model version that evaluates `theta = Q/Qmedia`
    (4, 7, 8): `theta ** a4` divides by zero if the currently loaded `a4 > 0`, and
    silently evaluates to `inf` (no NaN, no error) if `a4 < 0` -- see
    docs/audit/10_zero_discharge_handling.md. The check does not depend on the sign
    of `a4` (or on `a4` at all) since it must hold for every parameter vector a
    calibration search might sample, not just the one currently loaded.

    Skipped when:
    - `data.gap_tolerant` is True -- gap-tolerant mode already excludes `Q <= 0`
      days from every integrated segment via a different (heavier) mechanism
      (`detect_segments`); that behaviour is unchanged.
    - `data.version` is 3 or 5 -- these never evaluate `theta`, so a non-positive
      `Q` there is a data-quality question, not a numerical one.
    - `data.min_theta_floor` is set -- the opt-in escape hatch clamps `theta` away
      from zero instead of raising (applied inside the integrators themselves).

    Called from `read_Tseries` for both the calibration and validation/FORWARD-mode
    scenario record, so a naturally-occurring zero-flow day is caught once at data
    load rather than crashing calibration on whichever DE trial first samples a
    positive `a4`, and applies identically to a naturalised-flow/climate-projection
    FORWARD run.
    """
    if data.gap_tolerant or data.version not in (4, 7, 8):
        return
    if data.min_theta_floor is not None:
        return
    if data.Q is None or data.n_tot <= 365:
        return

    # The warm-up block (indices 0..364) is a verbatim copy of the real record's
    # first 365 rows, so checking the real record (365..n_tot) is sufficient -- any
    # zero-flow day within the first year would already be flagged there.
    Q = data.Q[365:data.n_tot]
    bad = Q <= 0.0
    n_bad = int(np.sum(bad))
    if n_bad == 0:
        return

    first_idx = 365 + int(np.argmax(bad))
    date = tuple(int(x) for x in data.date[first_idx]) if data.date is not None else None

    raise ValueError(
        f"Non-positive discharge (Q <= 0) found at index {first_idx} (date={date}); "
        f"{n_bad} day(s) in total across the record. Model version {data.version} "
        "evaluates theta = Q/Qmedia and theta**a4, which is undefined at Q=0 "
        "regardless of the currently loaded parameter vector: a4 > 0 divides by "
        "zero (ZeroDivisionError), a4 < 0 silently evaluates to inf with no error "
        "or warning. Options: (1) fix or remove the offending day(s) in the input "
        "data, (2) set `gap_tolerant: true` to exclude them via segment restart "
        "(changes calibration semantics broadly, not just for this case), or (3) "
        "set `min_theta_floor: <small positive epsilon>` in the config to clamp "
        "theta away from zero instead of raising. See "
        "docs/audit/10_zero_discharge_handling.md."
    )


def detect_segments(data: CommonData) -> None:
    """
    Detect valid segments, handling gap-tolerant mode.
    Builds data.segments and data.eval_mask.
    """
    data.segments = []
    # In legacy mode or not gap_tolerant, the segment is the whole data (starting from 365)
    # and the mask covers everything.
    if not data.gap_tolerant:
        data.eval_mask = np.zeros(data.n_tot, dtype=np.bool_)
        if data.n_tot > 365:
            data.eval_mask[365:] = True
        return

    data.eval_mask = np.zeros(data.n_tot, dtype=np.bool_)

    in_segment = False
    seg_start = -1

    # We only care about data from index 365 onwards (no warm-up)
    for i in range(365, data.n_tot):
        is_valid = True
        if data.Tair[i] == -999.0:
            is_valid = False
        if data.version not in [3, 5] and (data.Q[i] == -999.0 or data.Q[i] <= 0.0):
            is_valid = False

        if is_valid:
            if not in_segment:
                in_segment = True
                seg_start = i
        else:
            if in_segment:
                in_segment = False
                seg_end = i - 1
                length = seg_end - seg_start + 1
                if length >= data.min_segment_days:
                    data.segments.append((seg_start, seg_end))
                else:
                    print(f"Warning: Dropped segment ({seg_start}, {seg_end}) of length {length} days (min_segment_days={data.min_segment_days})")

    # Handle segment extending to end of array
    if in_segment:
        seg_end = data.n_tot - 1
        length = seg_end - seg_start + 1
        if length >= data.min_segment_days:
            data.segments.append((seg_start, seg_end))
        else:
            print(f"Warning: Dropped segment ({seg_start}, {seg_end}) of length {length} days (min_segment_days={data.min_segment_days})")

    if not data.segments:
        raise ValueError("No valid segments found after gap detection and filtering.")

    total_valid_days = sum(end - start + 1 for start, end in data.segments)
    if total_valid_days == 0:
        raise ValueError("Total valid forcing days across all segments is zero.")

    # Optional diagnostics (avoid spamming in optimization loops)
    if not data._segment_warned:
        if total_valid_days < 365:
            print(f"Warning: Total valid forcing days is {total_valid_days} (< 365). Calibration results may be unreliable.")
        if len(data.segments) > 2:
            print(f"Warning: Data is highly fragmented ({len(data.segments)} segments).")
        data._segment_warned = True

    # Build eval_mask based on segments and warmup_drop_days
    for start, end in data.segments:
        # Exclude the first warmup_drop_days of each segment
        eval_start = min(start + data.warmup_drop_days, end + 1)
        if eval_start <= end:
            data.eval_mask[eval_start:end + 1] = True


def prepare_evaluation(data: CommonData) -> None:
    """
    (Re)build `data.segments` and `data.eval_mask` for the currently loaded data.

    This must run after every load of Tair/Q/Twat_obs/n_tot (handled by
    `read_Tseries`), and after any later in-place mutation of those arrays (e.g.
    cross-validation folds) -- not only in gap-tolerant mode. Report 03 found
    `eval_mask` was previously left `None` for the whole non-gap-tolerant
    workflow, and that a `data.segments is None` staleness check let stale
    segments survive a later mutation of the underlying data. Idempotent and
    cheap enough to call unconditionally rather than cached with an `is None`
    check. See docs/audit/03_objective_function_and_masks.md.
    """
    detect_segments(data)


def _run_integration(data: CommonData, segments, p):
    """
    Orchestrate the core numerical integration loop over specified segments.

    Delegates the actual computation to the Numba-compiled `fast_run_integration`
    function to maximize performance.

    Parameters
    ----------
    data : CommonData
        The common data object containing forcing data (Tair, Q), time arrays,
        and settings. The `Twat_mod` array will be mutated in-place.
    segments : list of tuple
        A list of (start_idx, end_idx) tuples defining contiguous blocks of valid data.
    p : ndarray
        Array containing the 8 model parameters (1-indexed: p[1] to p[8]).

    Returns
    -------
    None
    """
    from .model_numba import fast_run_integration

    mod_num = data.mod_num
    mod_num_idx = -1
    if mod_num == 'CRN': mod_num_idx = 0
    elif mod_num == 'RK2': mod_num_idx = 1
    elif mod_num == 'RK4': mod_num_idx = 2
    elif mod_num == 'EUL': mod_num_idx = 3
    elif mod_num == 'EXP': mod_num_idx = 4
    else: raise ValueError(f"Unknown mod_num {mod_num}")

    segments_arr = np.array(segments, dtype=np.int32)
    theta_floor = data.min_theta_floor if data.min_theta_floor is not None else 0.0

    # Numba will mutate Twat_mod in place
    fast_run_integration(
        data.Tair, data.Q, data.tt, data.Twat_mod, data.Tice_cover, data.Qmedia,
        data.version, mod_num_idx, segments_arr,
        p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], theta_floor
    )

def call_model_segmented(data: CommonData) -> None:
    """
    Segmented ODE integration for gap-tolerant mode.
    """
    data.Twat_mod[:] = -999.0

    p = np.zeros(9, dtype=np.float64)
    p[1:9] = data.par[0:8]

    for start, end in data.segments:
        # Initial Condition
        if data.Twat_obs[start] != -999.0:
            data.Twat_mod[start] = data.Twat_obs[start]
        else:
            # DOY is 0-indexed in array but 1-366 in reality
            year = data.date[start, 0]
            month = data.date[start, 1]
            day = data.date[start, 2]
            doy = (pd.Timestamp(year, month, day) - pd.Timestamp(year, 1, 1)).days
            data.Twat_mod[start] = data.doy_climatology[doy]

    _run_integration(data, data.segments, p)

def call_model(data: CommonData) -> None:
    """
    Core air2stream simulation loop.
    Replicates SUBROUTINE call_model in AIR2STREAM_SUBROUTINES.f90
    """
    if data.gap_tolerant:
        call_model_segmented(data)
        return

    if data.Twat_obs[0] == -999.0:
        data.Twat_mod[0] = 4.0
    else:
        data.Twat_mod[0] = data.Twat_obs[0]

    # Convert par from 0-indexed to 1-indexed for the formula to match Fortran
    p = np.zeros(9, dtype=np.float64)
    p[1:9] = data.par[0:8]

    segments = [(0, data.n_tot - 1)]
    _run_integration(data, segments, p)

def aggregation(data: CommonData) -> None:
    """
    Aggregation (to calibrate the model with different time scale: daily, weekly, monthly)

    A day only contributes to a window if it also passes `data.eval_mask` (warm-up
    and, in gap-tolerant mode, each segment's `warmup_drop_days`). Without this,
    `statis()` (which sums every emitted window) and `funcobj()` (which additionally
    skips days failing `eval_mask`) score different samples -- see
    docs/audit/03_objective_function_and_masks.md, Defect A.
    """
    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=np.bool_)

    pp = len(data.time_res)
    if pp == 2:
        unit = data.time_res[1]
        qty = int(data.time_res[0])
    elif pp == 3:
        unit = data.time_res[2]
        qty = int(data.time_res[0:2])

    data.I_pos = np.full(data.n_tot, -999, dtype=np.int32)
    data.Twat_obs_agg = np.full(data.n_tot, -999.0, dtype=np.float64)

    n_inf = 1
    n_pos = 1

    if data.time_res == '1d':
        n_units = data.n_tot - 365
        data.I_inf = np.full((n_units, 3), -999, dtype=np.int32)

        for i in range(365, data.n_tot):
            if data.Twat_obs[i] != -999.0 and eval_mask[i]:
                # 0-indexed I_inf and I_pos.
                # Fortran I_inf(n_inf, 2) -> Python I_inf[n_inf-1, 1]
                data.I_inf[n_inf - 1, 1] = n_pos - 1
                data.I_inf[n_inf - 1, 2] = i
                data.I_pos[n_pos - 1] = i
                data.Twat_obs_agg[i] = data.Twat_obs[i]
                n_inf += 1
                n_pos += 1

    elif unit == 'w':
        n_days = qty * 7
        n_units = int(np.ceil((data.n_tot - 365) / n_days))
        data.I_inf = np.full((n_units, 3), -999, dtype=np.int32)

        for i in range(365, data.n_tot, n_days):
            tmp = 0.0
            count = 0
            # The nominal window-midpoint position assumes a full n_days-day
            # window. The trailing window is shorter whenever (n_tot - 365) is
            # not an exact multiple of n_days (the common case for any real
            # dataset), so this can point past the end of the array -- an
            # out-of-bounds write in the original Fortran (`AIR2STREAM_
            # SUBROUTINES.f90`'s `pos_tmp=i+CEILING(0.5*n_days)-1`, unguarded
            # there too) that silently corrupts memory instead of the IndexError
            # Python raises. Clamped to the last valid index, which only
            # changes behaviour for that trailing partial window -- a full
            # window's own `pos_tmp` is never affected (see docs/audit/
            # 08_testing_gaps.md, 8.3, where extending aggregation test
            # coverage to '1w'/'2w' surfaced this).
            pos_tmp = min(i + int(np.ceil(0.5 * n_days)) - 1, data.n_tot - 1)

            for j in range(n_days):
                k = i + j
                if k >= data.n_tot:
                    break
                if data.Twat_obs[k] != -999.0 and eval_mask[k]:
                    tmp += data.Twat_obs[k]
                    data.I_pos[n_pos - 1] = k
                    n_pos += 1
                    count += 1

            if count >= n_days * data.prc:
                data.I_inf[n_inf - 1, 1] = n_pos - 2 # n_pos-1 in Fortran (which is last idx added), in Python it's n_pos-2 because we do n_pos += 1
                data.I_inf[n_inf - 1, 2] = pos_tmp
                data.Twat_obs_agg[pos_tmp] = tmp / count
                n_inf += 1
            else:
                data.I_pos[n_pos - 1 - count : n_pos - 1] = -999
                n_pos = n_pos - count

    elif unit == 'm':
        n_units = int(np.ceil(data.n_tot / 30.5))
        data.I_inf = np.full((n_units, 3), -999, dtype=np.int32)
        n_days = 0
        month_curr = -999
        count = 0
        tmp = 0.0

        for i in range(365, data.n_tot):
            month = data.date[i, 1]
            if month != month_curr:
                if count >= n_days * data.prc and i != 365:
                    data.I_inf[n_inf - 1, 1] = n_pos - 2
                    data.I_inf[n_inf - 1, 2] = i - int(np.floor(0.5 * n_days)) - 1
                    data.Twat_obs_agg[data.I_inf[n_inf - 1, 2]] = tmp / count
                    n_inf += 1
                else:
                    if count > 0:
                        data.I_pos[n_pos - 1 - count : n_pos - 1] = -999
                        n_pos = n_pos - count
                month_curr = month
                count = 0
                n_days = 1
                tmp = 0.0
            else:
                n_days += 1

            if data.Twat_obs[i] != -999.0 and eval_mask[i]:
                tmp += data.Twat_obs[i]
                data.I_pos[n_pos - 1] = i
                n_pos += 1
                count += 1

        # Last month
        if count >= n_days * data.prc:
            data.I_inf[n_inf - 1, 1] = n_pos - 2
            data.I_inf[n_inf - 1, 2] = data.n_tot - 1 - int(np.floor(0.5 * n_days)) # using data.n_tot - 1 as the last i
            data.Twat_obs_agg[data.I_inf[n_inf - 1, 2]] = tmp / count
            n_inf += 1
        else:
            if count > 0:
                data.I_pos[n_pos - 1 - count : n_pos - 1] = -999
                n_pos = n_pos - count
    else:
        print("Error: variable time_res")

    data.n_dat = n_inf - 1
    n_pos = n_pos - 1

    if data.n_dat > 0:
        data.I_inf[0, 0] = 0
        for i in range(1, data.n_dat):
            data.I_inf[i, 0] = data.I_inf[i - 1, 1] + 1

    # Resize arrays
    data.I_inf = data.I_inf[:data.n_dat, :]
    data.I_pos = data.I_pos[:n_pos]

def statis(data: CommonData) -> None:
    """
    Statis (to calculate errors)
    """
    if data.n_dat == 0:
        raise ValueError("n_dat is 0 after aggregation. No T_water observations survived.")

    data.mean_obs = np.float64(0.0)
    data.TSS_obs = np.float64(0.0)

    for i in range(data.n_dat):
        data.mean_obs += data.Twat_obs_agg[data.I_inf[i, 2]]

    data.mean_obs /= np.float64(data.n_dat)

    for i in range(data.n_dat):
        data.TSS_obs += (data.Twat_obs_agg[data.I_inf[i, 2]] - data.mean_obs) ** 2

    if data.n_dat > 1:
        data.std_obs = np.sqrt(data.TSS_obs / np.float64(data.n_dat - 1))
    else:
        data.std_obs = np.float64(0.0)

def funcobj(data: CommonData) -> float:
    """
    Calculation of the objective function.
    Returns the objective value.
    """
    from .model_numba import fast_funcobj

    fun_obj_type = -1
    if data.fun_obj == 'NSE': fun_obj_type = 0
    elif data.fun_obj == 'KGE': fun_obj_type = 1
    elif data.fun_obj == 'RMS': fun_obj_type = 2
    else:
        raise ValueError(
            f"Invalid objective_function '{data.fun_obj}'. Must be one of: NSE, KGE, RMS."
        )

    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=np.bool_)

    ind, Twat_mod_agg, current_nse, current_r2, current_mae = fast_funcobj(
        data.n_dat, data.n_tot, data.I_inf, data.I_pos, data.Twat_mod, data.Twat_obs_agg,
        eval_mask, fun_obj_type, data.mean_obs, data.TSS_obs, data.std_obs
    )

    data.Twat_mod_agg = Twat_mod_agg
    data.current_nse = current_nse
    data.current_r2 = current_r2
    data.current_mae = current_mae

    # Handle print warning consistency from original Python port
    if ind == -999.0 and fun_obj_type == 1:
        pass # The python version used to print "Warning: KGE undefined"

    return np.float64(ind)
