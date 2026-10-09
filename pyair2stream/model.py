"""
High-level ODE integration and simulation orchestration for air2stream.

This module provides the main entry points for running the air2stream model,
including handling missing data segments (gap-tolerant mode) and delegating
the heavy numeric lifting to the Numba-compiled functions.
"""

import numpy as np
import math
import pandas as pd
from .config import CommonData, PI, TTT, ACTIVE_PARAMS

# Sanity bound on simulated water temperature (degC) -- see USER_GUIDE.md §9.1:
# explicit integrators (RK4/RK2/EUL) can diverge silently on scenario discharge that differs
# from the calibration record, producing either huge or plausible-but-wrong numbers with no
# error/NaN. This is the default for `max_plausible_twat`.
TWAT_SANITY_MAX = 60.0

# One-step amplification-factor stability limits for B = (a3 + a8*theta)/theta**a4, i.e. the
# ODE's linear decay rate (1/day) times the fixed dt=1 day step. CRN and EXP are unconditionally
# stable. See USER_GUIDE.md §9.1.
STABILITY_LIMITS = {'EUL': 2.0, 'RK2': 2.0, 'RK4': 2.785, 'CRN': np.inf, 'EXP': np.inf}

# "Error if more than a small fraction of days exceed [the stability limit]".
# This screening criterion is conservative, not exact (isolated high-theta days often simulate
# fine); it is a companion to the divergence guard (check_numerical_divergence), not a
# replacement for it.
STABILITY_ERROR_FRACTION = 0.10

# The most that a difference between two simulations (in the start value, by rounding, or from
# an error in the inputs) may grow over a stretch of consecutive days before a run with an
# explicit integrator is stopped (`stability_max_growth`; see `largest_growth`). Unlike the share
# of days above the limit, the growth follows exactly from the B series; examples/
# 09_integrator_stability compares the two on the Swiss rivers. See USER_GUIDE.md §9.1.
STABILITY_MAX_GROWTH = 100.0


class NumericalDivergenceError(RuntimeError):
    """
    Raised when a simulated water-temperature series is non-finite or exceeds a
    physically implausible bound.

    The air2stream ODE is linear in Tw with a discharge-dependent decay rate B; an
    explicit integrator (RK4/RK2/EUL) stable at the calibration discharge can be
    unstable at a different scenario discharge, producing either astronomical or
    plausible-but-wrong output with no NaN and no warning. See
    USER_GUIDE.md §9.1.
    """


def compute_B_series(data: CommonData) -> np.ndarray:
    """
    Compute B(t), the ODE's linear decay rate (1/day), for every day in `data.Q`
    using the current `data.par`. `B * dt` (dt is fixed at 1 day) governs the
    stability of the explicit integrators.

    Entries where discharge is invalid (missing sentinel, non-positive) or where the
    computation is undefined (e.g. a negative theta raised to a non-integer power)
    come back as NaN; callers should mask on validity/finiteness. If
    `data.min_theta_floor` is set, theta is floored the same way the integrators
    themselves floor it (see `check_nonpositive_discharge`), so this reports the B
    that will actually be used, not the raw (possibly non-finite) one.
    """
    return _relaxation_rate(data, data.Q)


def _relaxation_rate(data: CommonData, Q) -> np.ndarray:
    """B (1/day) for the discharge series `Q`, with the current `data.par`: the formula of
    `compute_B_series`, which passes `data.Q`. RK4's midpoint stages use the mean discharge
    of two days (`step_amplification`)."""
    p = data.par
    a3, a4, a8 = p[2], p[3], p[7]

    if data.version in (3, 5):
        return np.full(np.shape(Q), a3, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        theta = Q / data.Qmedia
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


def step_amplification(data: CommonData) -> np.ndarray:
    """
    R_j for j = 0 .. n_tot-2: one step of the current integrator (`data.mod_num`), from
    day j to day j+1, multiplies a difference between two simulations (in the start
    value, by rounding, or from an error in the inputs) by R_j. The equation is linear
    in water temperature, so this is exact, apart from the ice floor (`Tice_cover`).

    Each integrator takes B (`compute_B_series`) from particular days: CRN B_j (its
    explicit half) and B_j+1 (its implicit half); EXP their mean; EUL B_j+1 (like the
    Fortran, it takes the inputs of the next day); RK2 B_j and B_j+1; RK4 also B at the
    mean discharge of the two days (its two midpoint stages). NaN where B is undefined
    on a day the step uses. With the same B on every day, R is the integrator's
    stability function at z = -B: |R| <= 1 for every B >= 0 with CRN and EXP, but only
    up to B = 2 with EUL and RK2, and up to 2.785 with RK4.
    """
    B = compute_B_series(data)
    b0, b1 = B[:-1], B[1:]
    with np.errstate(all='ignore'):
        if data.mod_num == 'CRN':
            return (1.0 - b0 / 2.0) / (1.0 + b1 / 2.0)
        if data.mod_num == 'EXP':
            return np.exp(-(b0 + b1) / 2.0)
        if data.mod_num == 'EUL':
            return 1.0 - b1
        if data.mod_num == 'RK2':
            return 1.0 - b0 / 2.0 - b1 * (1.0 - b0) / 2.0
        if data.mod_num == 'RK4':
            if data.version in (3, 5):
                bm = b0
            else:
                bm = _relaxation_rate(data, 0.5 * (data.Q[:-1] + data.Q[1:]))
            k1 = -b0
            k2 = -bm * (1.0 + k1 / 2.0)
            k3 = -bm * (1.0 + k2 / 2.0)
            k4 = -b1 * (1.0 + k3)
            return 1.0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    raise ValueError(f"Unknown mod_num {data.mod_num}")


def largest_growth(data: CommonData) -> dict:
    """
    The largest factor by which a difference between two simulations (in the start
    value, by rounding, or from an error in the inputs) can grow over a stretch of
    consecutive days: the largest |product of R_j| (`step_amplification`) over the steps
    of a stretch that is integrated in one go (the whole record, or each segment in
    gap-tolerant mode). 1.0 means that no difference ever grows.

    The equation is linear in water temperature, so this follows from the B series
    exactly (apart from the ice floor), before anything is simulated. It separates
    stable from unstable runs of the explicit integrators much better than the share of
    days on which B is above their limit, because a difference grows only while B stays
    above the limit, and it decays again afterwards (examples/09_integrator_stability).
    A step whose factor is undefined (NaN B) ends a stretch.

    Returns a dict: 'growth' (inf if it overflows), 'log10_growth', and 'start' and
    'end', the indices of the first and last day of the stretch (None if nothing grows).
    """
    R = step_amplification(data)
    if data.gap_tolerant and data.segments:
        segments = [(int(a), int(b)) for a, b in data.segments]
    else:
        segments = [(0, data.n_tot - 1)]
    best, start, end = 0.0, None, None
    with np.errstate(divide='ignore', invalid='ignore'):
        for first, last in segments:
            log_r = np.log10(np.abs(R[first:last]))       # the steps from day `first` to day `last`
            if log_r.size == 0:
                continue
            # A factor of 0 wipes a difference out, and an undefined one ends the stretch.
            log_r = np.where(np.isfinite(log_r), log_r, -300.0)
            # The largest sum of log10|R_j| over a run of consecutive steps.
            s = np.concatenate(([0.0], np.cumsum(log_r)))
            gain = s - np.minimum.accumulate(s)
            k = int(np.argmax(gain))
            if gain[k] > best:
                best = float(gain[k])
                start, end = first + int(np.argmin(s[:k + 1])), first + k
    growth = float('inf') if best > 308.0 else float(10.0 ** best)
    return {'growth': growth, 'log10_growth': best, 'start': start, 'end': end}


def stability_report(data: CommonData) -> dict:
    """
    Pre-flight stability check of the current integrator, from the B series.

    Computes B = (a3 + a8*theta)/theta**a4 (version-dependent) over the whole
    forcing series and compares it against the current integrator's one-step
    stability limit: the share of days above it is a screen (isolated high-theta
    days often simulate fine because the transient decays before it compounds).
    `max_growth` (`largest_growth`) is exact for this linear equation: how much a
    difference between two simulations can grow over a stretch of days. Pair them
    with `check_numerical_divergence`, which catches actual divergence after the run.
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
            'mod_num': data.mod_num, 'limit': limit, 'max_B': None, 'min_B': None,
            'frac_exceeding': 0.0, 'n_exceeding': 0, 'n_valid': 0, 'worst': [],
            'max_growth': 1.0, 'log10_max_growth': 0.0, 'growth_stretch': (None, None),
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

    # An unknown integrator has no limit (above) and is not checked.
    growth = (largest_growth(data) if data.mod_num in STABILITY_LIMITS else
              {'growth': 1.0, 'log10_growth': 0.0, 'start': None, 'end': None})
    return {
        'mod_num': data.mod_num,
        'limit': limit,
        'max_B': max_B,
        'min_B': float(np.min(B_valid)),
        'frac_exceeding': frac_exceeding,
        'n_exceeding': n_exceeding,
        'n_valid': n_valid,
        'worst': worst,
        'max_growth': growth['growth'],
        'log10_max_growth': growth['log10_growth'],
        'growth_stretch': (growth['start'], growth['end']),
    }


def _day_label(data: CommonData, i) -> str:
    """A day of the simulation for a message: its date, or its place in the warm-up year."""
    if i is None:
        return "?"
    if data.date is not None and data.date[i, 0] != -999:
        y, m, d = (int(x) for x in data.date[i])
        return f"{y:04d}-{m:02d}-{d:02d}"
    return f"day {i + 1} of the warm-up year" if i < 365 else f"day {i + 1}"


def _growth_label(report: dict) -> str:
    """The largest growth of a stability report, readable at any size."""
    g, log_g = report['max_growth'], report['log10_max_growth']
    if not np.isfinite(g) or g >= 1e6:
        return f"10^{log_g:.0f}"
    return f"{g:,.0f}" if g >= 100 else f"{g:.1f}"


def warn_on_stability(data: CommonData, error_fraction: float = STABILITY_ERROR_FRACTION,
                      max_growth: float = None) -> dict:
    """
    Run `stability_report` before a user-facing simulation. With an explicit integrator
    (RK4/RK2/EUL), print a warning if B is above its stability limit on some days, or if a
    difference can grow over some stretch of days; raise `NumericalDivergenceError` if B is
    above the limit on more than `error_fraction` of the days, or if a difference can grow
    more than `max_growth` times over a stretch of days (default:
    `data.stability_max_growth`). CRN and EXP are not checked: for B >= 0 they are stable.
    """
    if max_growth is None:
        max_growth = getattr(data, 'stability_max_growth', STABILITY_MAX_GROWTH)
    report = stability_report(data)

    if report['max_B'] is None or not np.isfinite(report['limit']):
        return report

    grows = report['max_growth'] > 1.0 + 1e-9
    if report['max_B'] > report['limit'] or grows:
        worst = report['worst'][0] if report['worst'] else None
        worst_str = f" Worst day: {_day_label(data, worst['index'])} (B={worst['B']:.3f})." if worst else ""
        first, last = (_day_label(data, i) for i in report['growth_stretch'])
        growth_str = (f" From {first} to {last}, a difference in the simulated temperature (from the "
                      f"start value, rounding or the inputs) can grow {_growth_label(report)} times; the "
                      f"run stops above stability_max_growth={max_growth:g}." if grows else
                      " No difference can grow over any stretch of days.")
        negative = report['min_B'] < 0.0
        if negative:
            growth_str += (f" B is negative on some days (lowest {report['min_B']:.3f}), so the "
                           f"equation itself is unstable there.")
        print(
            f"Warning: {report['n_exceeding']}/{report['n_valid']} days "
            f"({report['frac_exceeding']:.1%}) exceed the {report['mod_num']} stability limit "
            f"(B > {report['limit']:.3f}); max B = {report['max_B']:.3f}.{worst_str}{growth_str} "
            f"Consider CRN or EXP, especially for scenario runs on discharge different from the "
            f"calibration record (USER_GUIDE.md §9.1)."
        )
        if report['frac_exceeding'] > error_fraction:
            raise NumericalDivergenceError(
                f"{report['frac_exceeding']:.1%} of days exceed the {report['mod_num']} "
                f"stability limit (B > {report['limit']:.3f}), above the "
                f"error_fraction={error_fraction:.0%} threshold. Use CRN or EXP for this run, "
                f"or raise `stability_error_fraction` in the config if you have verified the "
                f"simulation is stable (see USER_GUIDE.md §9.1)."
            )
        if report['max_growth'] > max_growth:
            if negative:
                reason = (f"B is negative on some days (lowest {report['min_B']:.3f}), so the equation "
                          f"itself is unstable. Narrow parameter_bounds so that B stays positive")
            else:
                reason = (f"B stays above the stability limit (B > {report['limit']:.3f}) for too long. "
                          f"Use CRN or EXP for this run, or raise `stability_max_growth` in the config "
                          f"if you have verified the simulation")
            raise NumericalDivergenceError(
                f"With {report['mod_num']}, a difference in the simulated water temperature (from "
                f"the start value, rounding or the inputs) can grow {_growth_label(report)} times "
                f"from {first} to {last}, more than stability_max_growth={max_growth:g}: {reason} "
                f"(see USER_GUIDE.md §9.1)."
            )

    return report


# Successive daily changes of a simulated water temperature are positively correlated (about
# +0.5 on the Swiss rivers); a simulation that zigzags from one day to the next gives about -1.
OSCILLATION_CHANGE_CORR = -0.5


def check_daily_plausibility(data: CommonData) -> dict:
    """
    Warn if the simulation at the current `data.par` is physically implausible in a
    way its score may not show (run after `call_model`):

    - the relaxation rate B (`compute_B_series`) is negative on some day, so water
      temperature moves away from equilibrium instead of towards it;
    - the simulated daily temperature zigzags from one day to the next (the
      correlation of successive daily changes is below OSCILLATION_CHANGE_CORR).

    With weekly or monthly scoring a zigzag averages out within each block, so such
    a parameter set can score as well as a sensible one; in validation V4 (case J),
    weekly-scored calibrations with bounds that allow a negative a2 and a3 often
    ended on one. Returns the numbers behind the warnings.
    """
    B = compute_B_series(data)
    valid = np.isfinite(B)
    if data.version not in (3, 5):
        valid &= (data.Q != -999.0) & (data.Q > 0.0)
    n_negative = int(np.sum(valid & (B < 0.0)))
    min_B = float(np.min(B[valid])) if np.any(valid) else None

    if data.gap_tolerant and data.segments:
        segments = [(int(a), int(b)) for a, b in data.segments]
    else:
        segments = [(0, data.n_tot - 1)]
    first, second = [], []
    for start, end in segments:
        sim = np.asarray(data.Twat_mod[max(start, 365):end + 1], dtype=np.float64)
        change = np.diff(sim)
        ok = np.isfinite(change[:-1]) & np.isfinite(change[1:])
        first.append(change[:-1][ok])
        second.append(change[1:][ok])
    first = np.concatenate(first) if first else np.empty(0)
    second = np.concatenate(second) if second else np.empty(0)
    change_corr = None
    if len(first) >= 30 and np.std(first) > 0 and np.std(second) > 0:
        change_corr = float(np.corrcoef(first, second)[0, 1])

    if n_negative:
        print(f"Warning: the relaxation rate B is negative on {n_negative} days (lowest {min_B:.3f} per day), "
              "so the simulated water temperature moves away from equilibrium instead of towards it. This is "
              "physically impossible. Narrow parameter_bounds so that B stays positive (for versions 3 and 5, "
              "a3 at least 0); USER_GUIDE.md §9.1 defines B.")
    if change_corr is not None and change_corr < OSCILLATION_CHANGE_CORR:
        print(f"Warning: the simulated daily temperature zigzags from one day to the next (correlation of "
              f"successive daily changes {change_corr:.2f}). This is a numerical artefact, not river behaviour; "
              "with weekly or monthly scoring it can still score well, because it averages out within each "
              "block. Narrow parameter_bounds (for example a2 and a3 at least 0) or score daily values.")
    return {"min_B": min_B, "n_negative_B": n_negative, "change_corr": change_corr}


def check_segment_warmup(data: CommonData) -> None:
    """
    Gap-tolerant mode only: warn if `warmup_drop_days` is too short for the
    approximate restart temperature of each segment to be forgotten.

    A difference between the restart value and the "true" state decays roughly
    as exp(-B*t), where B (1/day) is the ODE's decay rate for the current
    parameters. After 3/B days about 95% of it has gone, so the unscored start
    of each segment should be at least that long.
    """
    if not data.gap_tolerant or not data.segments:
        return
    in_seg = np.zeros(data.n_tot, dtype=bool)
    for start, end in data.segments:
        in_seg[start:end + 1] = True
    B = compute_B_series(data)
    ok = in_seg & np.isfinite(B) & (B > 0)
    if not np.any(ok):
        return
    needed = int(np.ceil(3.0 / np.median(B[ok])))
    if data.warmup_drop_days < needed:
        print(
            f"Warning: warmup_drop_days={data.warmup_drop_days} is shorter than about three "
            f"relaxation times of the calibrated model ({needed} days). The start of each "
            f"segment may still reflect its approximate restart temperature; consider "
            f"warmup_drop_days: {needed}. See USER_GUIDE.md §10."
        )
        return
    # The opposite case: with many gaps, a warm-up much longer than the model needs throws
    # away measurements. Say how many a warm-up of `needed` days (and pieces of at least
    # twice that) would score.
    min_segment = min(2 * needed, data.min_segment_days)
    now = scored_days(data, data.segments, data.warmup_drop_days)
    shorter = scored_days(data, find_segments(data, min_segment)[0], needed)
    if shorter - now >= max(30, 0.05 * now):
        print(
            f"Note: the calibrated model forgets its restart within about {needed} days. "
            f"warmup_drop_days: {needed} and min_segment_days: {min_segment} would score "
            f"{shorter} measured days instead of {now}. Consider them and calibrate again; "
            f"see USER_GUIDE.md §10."
        )


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
    value, or if, with an explicit integrator (RK4/RK2/EUL), a difference can grow
    more than `data.stability_max_growth` times over a stretch of days at the current
    `data.par` (`largest_growth`): such a run can look plausible and still be wrong.
    Intended for a per-draw check inside an ensemble/posterior-sample loop
    (`optimization.forward_mode`'s prediction-interval loop,
    `optimization._run_mcmc_uncertainty`'s envelope loop), where a single bad draw
    should be excluded (or the batch aborted, per `on_divergent_draw`) rather than
    raising and losing the rest of the ensemble (docs/METHODS.md §12).
    """
    if max_plausible_twat is None:
        max_plausible_twat = getattr(data, 'max_plausible_twat', TWAT_SANITY_MAX)
    if np.any(_divergence_bad_mask(data.Twat_mod, max_plausible_twat)):
        return True
    if data.mod_num in ('RK4', 'RK2', 'EUL'):
        limit = getattr(data, 'stability_max_growth', STABILITY_MAX_GROWTH)
        return bool(largest_growth(data)['growth'] > limit)
    return False


def check_numerical_divergence(data: CommonData, max_plausible_twat: float = None) -> None:
    """
    Raise `NumericalDivergenceError` if `data.Twat_mod` contains non-finite values
    or exceeds a physically implausible sanity bound. See USER_GUIDE.md §9.1.

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
        f"(the default) or EXP for scenario runs. See USER_GUIDE.md §9.1."
    )


def check_nonpositive_discharge(data: CommonData) -> None:
    """
    Raise `ValueError` if any non-positive discharge day (`Q <= 0`) is present in a
    non-gap-tolerant record for a model version that evaluates `theta = Q/Qmedia`
    (4, 7, 8): `theta ** a4` divides by zero if the currently loaded `a4 > 0`, and
    silently evaluates to `inf` (no NaN, no error) if `a4 < 0` -- see
    USER_GUIDE.md §9.2. The check does not depend on the sign
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
        "USER_GUIDE.md §9.2."
    )


def find_segments(data: CommonData, min_segment_days: int):
    """
    Gap-tolerant mode: the stretches of consecutive days (from index 365 on) with valid air
    temperature and, for versions 4/7/8, positive discharge. Returns (kept, dropped): lists of
    (start, end) index pairs, inclusive, split by whether they are at least `min_segment_days` long.
    """
    valid = data.Tair[365:data.n_tot] != -999.0
    if data.version not in [3, 5]:
        valid &= (data.Q[365:data.n_tot] != -999.0) & (data.Q[365:data.n_tot] > 0.0)
    edges = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1) + 365
    ends = np.flatnonzero(edges == -1) + 364
    kept, dropped = [], []
    for start, end in zip(starts.tolist(), ends.tolist()):
        (kept if end - start + 1 >= min_segment_days else dropped).append((start, end))
    return kept, dropped


def scored_days(data: CommonData, segments, warmup_drop_days: int) -> int:
    """Number of measured water temperatures scored in `segments` after `warmup_drop_days`."""
    measured = data.Twat_obs != -999.0
    return int(sum(np.sum(measured[min(start + warmup_drop_days, end + 1):end + 1]) for start, end in segments))


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

    data.segments, dropped = find_segments(data, data.min_segment_days)
    for seg_start, seg_end in dropped:
        print(f"Warning: Dropped segment ({seg_start}, {seg_end}) of length {seg_end - seg_start + 1} days "
              f"(min_segment_days={data.min_segment_days})")

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
    check.
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

    # Zero the parameters this version does not use. The CRN branch evaluates the
    # full 8-parameter equation (as the Fortran does), so a stray non-zero unused
    # parameter would otherwise change the physics for CRN but not for RK4/EXP.
    active = ACTIVE_PARAMS[data.version]
    p = np.array([p[0]] + [p[j + 1] if j in active else 0.0 for j in range(8)], dtype=np.float64)

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
            if data.calendar == 'standard':
                year = data.date[start, 0]
                month = data.date[start, 1]
                day = data.date[start, 2]
                doy = (pd.Timestamp(year, month, day) - pd.Timestamp(year, 1, 1)).days
            else:
                days_in_year = 365 if data.calendar == 'noleap' else 360
                doy = (start - 365) % days_in_year
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
    skips days failing `eval_mask`) would score different samples.
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
            # window's own `pos_tmp` is never affected.
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

            # `count > 0` also guards prc <= 0 (rejected by read_calibration), where
            # the Fortran would divide by zero for a block with no observations.
            if count > 0 and count >= n_days * data.prc:
                data.I_inf[n_inf - 1, 1] = n_pos - 2 # n_pos-1 in Fortran (which is last idx added), in Python it's n_pos-2 because we do n_pos += 1
                data.I_inf[n_inf - 1, 2] = pos_tmp
                data.Twat_obs_agg[pos_tmp] = tmp / count
                n_inf += 1
            else:
                data.I_pos[n_pos - 1 - count : n_pos - 1] = -999
                n_pos = n_pos - count

    elif unit == 'm':
        # At most one month per 28 days, plus partial months at either end. (A count of
        # n_tot / 30.5 was too small for 360-day records longer than about 60 years.)
        n_units = (data.n_tot - 365) // 28 + 2
        data.I_inf = np.full((n_units, 3), -999, dtype=np.int32)
        n_days = 0
        month_curr = -999
        count = 0
        tmp = 0.0

        for i in range(365, data.n_tot):
            month = data.date[i, 1]
            if month != month_curr:
                if count > 0 and count >= n_days * data.prc and i != 365:
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
        if count > 0 and count >= n_days * data.prc:
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
