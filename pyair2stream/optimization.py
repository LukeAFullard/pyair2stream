"""
Optimization and calibration routines for pyair2stream.

This module implements the calibration algorithms (PSO, DE, LATHYP)
used to fit the air2stream model to observed data, as well as the
MCMC sampling routines for uncertainty quantification.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from typing import Optional
import concurrent.futures
from scipy.optimize import differential_evolution, minimize
import emcee

import json
from .config import CommonData, DEFAULT_NOISE_MODEL, DEFAULT_LIKELIHOOD, DEFAULT_RHO_TIMESCALE
from .io import fit_settings, settings_differences, SETTINGS_NOT_RECORDED
from .model import (
    call_model, funcobj, aggregation, statis, warn_on_stability, check_numerical_divergence,
    is_numerically_divergent, NumericalDivergenceError, check_daily_plausibility,
)
from .uncertainty import (estimate_rho, estimate_ar1_rho, generate_ar1_noise, ar1_whitened_stats,
                          scored_error_variance_factor, scoring_block_days)

# A near-perfect-fit MCMC log-likelihood is capped at this large but finite value rather
# than returned as a literal np.inf, which poisons emcee's acceptance-ratio arithmetic
# (inf - inf = nan).
MCMC_MAX_LOG_LIKELIHOOD = 1e10

# Number of physical model parameters (a1..a8 in the Fortran reference).
N_PAR = 8

# MCMC convergence rule (docs/METHODS.md §12): the chain is extended in blocks of
# MCMC_CHECK_INTERVAL steps (after at least MCMC_MIN_STEPS) until it is at least
# MCMC_TAU_FACTOR times the autocorrelation time long and split-Rhat is below
# MCMC_MAX_RHAT, or `mcmc_steps` (the maximum) is reached.
MCMC_CHECK_INTERVAL = 1000
MCMC_MIN_STEPS = 2000
MCMC_TAU_FACTOR = 50
MCMC_MAX_RHAT = 1.01


def _active_params(data: CommonData, n_par: int = N_PAR) -> list:
    """Indices of parameters that are both flagged active and non-degenerate (parmin != parmax)."""
    return [j for j in range(n_par) if data.flag_par[j] and data.parmin[j] != data.parmax[j]]


def _require_free_parameters(data: CommonData) -> None:
    """Refuse to calibrate when no parameter can vary (e.g. `parameter_bounds` missing)."""
    if not _active_params(data):
        raise ValueError(
            "No parameter is free to calibrate (every parameter has min == max). Set "
            "`parameter_bounds` (min and max, 8 values each) in the config."
        )


def _segments_for(data: CommonData) -> list:
    """Segments to treat as independent adjacency/AR(1) runs: gap-tolerant segments, or the whole series."""
    return data.segments if data.gap_tolerant else [(0, data.n_tot - 1)]


def _iid_log_likelihood(mod_valid: np.ndarray, obs_valid: np.ndarray) -> float:
    """Concentrated Gaussian log-likelihood assuming iid residuals."""
    N = len(obs_valid)
    if N == 0:
        return -np.inf
    SSE = np.sum((mod_valid - obs_valid) ** 2)
    if SSE == 0:
        return MCMC_MAX_LOG_LIKELIHOOD
    return -0.5 * N * np.log(SSE / N)


def _ar1_log_likelihood(residuals: np.ndarray, rho: float, runs: list, block_days: int = 1) -> float:
    """
    Concentrated Gaussian log-likelihood accounting for AR(1)-correlated residuals on the
    scored rows `runs` (taken together in date order): values d days apart have
    correlation rho**d, so values on either side of a gap stay related
    (`ar1_whitened_stats`). `rho` is treated as fixed (estimated once at the DE optimum,
    not sampled). With weekly or monthly scoring (`block_days` > 1) the block means are
    treated as independent (a warning says so where this is used).
    """
    sse_u, N, log_scale = ar1_whitened_stats(residuals, rho, runs, independent=block_days > 1)
    if N == 0:
        return -np.inf
    if sse_u == 0:
        return MCMC_MAX_LOG_LIKELIHOOD
    return -0.5 * N * np.log(sse_u / N) + log_scale


def _least_squares_log_likelihood(residuals: np.ndarray, rho: float, runs: list, block_days: int = 1,
                                  factor: Optional[float] = None) -> float:
    """
    Concentrated least-squares (iid Gaussian) log-likelihood with the effective number of
    independent observations n_eff = n / factor in place of n. Its maximum is the
    least-squares fit; its spread is widened for the autocorrelation of the errors: the
    factor is the variance of a mean of the scored errors relative to independent ones,
    for daily AR(1) errors with lag-1 correlation rho, given how far apart the scored values
    are (`scored_error_variance_factor`; with weekly or monthly scoring each scored value is
    a block mean of `block_days` days). For n consecutive days it is close to
    (1 + rho) / (1 - rho). Pass `factor` to reuse one computed for the same scored rows.
    Equal to `_iid_log_likelihood` when rho = 0.
    """
    if not runs:
        return -np.inf
    rows = np.concatenate(runs)
    e = residuals[rows]
    n = len(e)
    sse = float(np.sum(e ** 2))
    if sse == 0:
        return MCMC_MAX_LOG_LIKELIHOOD
    if factor is None:
        factor = scored_error_variance_factor(rows, rho, block_days)
    return -0.5 * (n / factor) * np.log(sse / n)


def _reflected_walker_init(initial: np.ndarray, scale: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                            nwalkers: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build the initial emcee walker ball around `initial`, reflecting draws back inside
    `[lo, hi]` instead of clipping them to the bound. Clipping collapses the ensemble's
    spread in any dimension where the DE optimum sits exactly on a bound -- emcee's
    stretch move cannot generate spread from a degenerate ensemble. Raises if any dimension still ends up with zero spread.
    """
    ndim = len(initial)
    raw = initial[None, :] + scale[None, :] * rng.standard_normal((nwalkers, ndim))
    span = hi - lo
    rel = (raw - lo) % (2.0 * span)
    reflected = np.where(rel > span, 2.0 * span - rel, rel)
    p0 = lo + reflected

    variances = np.var(p0, axis=0)
    collapsed = np.where(variances <= 0.0)[0]
    if len(collapsed) > 0:
        raise ValueError(
            "MCMC walker initialisation collapsed to zero spread in active-parameter "
            f"position(s) {list(collapsed)}; emcee's stretch move cannot explore from a "
            "degenerate ensemble."
        )
    return p0


def _split_rhat(chain: np.ndarray) -> np.ndarray:
    """
    Gelman-Rubin split-Rhat per parameter, from a raw (non-flattened) emcee chain of
    shape (n_iter, n_walkers, n_dim). Splitting each walker's chain in half along the
    iteration axis also flags within-walker non-stationarity, not just between-walker
    disagreement.
    """
    n_iter, n_chains, n_dim = chain.shape
    n = n_iter // 2
    if n < 2:
        return np.full(n_dim, np.nan)
    split = np.concatenate([chain[:n], chain[n:2 * n]], axis=1)
    chain_means = split.mean(axis=0)
    chain_vars = split.var(axis=0, ddof=1)
    W = chain_vars.mean(axis=0)
    B = n * chain_means.var(axis=0, ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    with np.errstate(divide='ignore', invalid='ignore'):
        rhat = np.sqrt(var_hat / W)
    return rhat


def _estimate_autocorr(sampler, discard: int):
    """
    Estimate the per-parameter integrated autocorrelation time on the chain with the
    first `discard` steps removed. Returns `(tau_array_or_None, mean_tau_or_None)`.
    emcee's own "chain too short" check is switched off (tol=0): whether the chain is
    long enough is decided by `_mcmc_diagnostics` against MCMC_TAU_FACTOR.
    """
    try:
        tau = sampler.get_autocorr_time(discard=discard, tol=0)
        if not np.all(np.isfinite(tau)):
            return None, None
        return tau, float(np.mean(tau))
    except Exception:
        return None, None


def _make_sampler(nwalkers: int, ndim: int, log_prob, seed: Optional[int]) -> emcee.EnsembleSampler:
    """
    The ensemble sampler used for DE-MCMC.

    Proposals use emcee's differential-evolution move (ter Braak, 2006), which
    mixes 3-4x faster than emcee's default stretch move on the strongly
    correlated air2stream posteriors and samples a known correlated Gaussian
    correctly (tests/test_mcmc_sampler.py). emcee's DESnookerMove is deliberately
    not used: in emcee 3.1.6 it fails that known-answer test.
    """
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, moves=emcee.moves.DEMove())
    if seed is not None:
        # emcee draws its moves from a private RandomState copied from numpy's
        # *global* state (seeded from system entropy in every new process), so it
        # must be seeded explicitly for `random_seed` to make the chain reproducible.
        sampler.random_state = np.random.RandomState(seed).get_state()
    return sampler


def _mcmc_diagnostics(sampler, uncertainty_options: dict) -> dict:
    """Burn-in, autocorrelation time, split-Rhat and the convergence verdict for the chain so far."""
    n = sampler.iteration
    tau_rough, _ = _estimate_autocorr(sampler, discard=0)
    burnin = _resolve_burnin(n, tau_rough, uncertainty_options)
    tau, mean_tau = _estimate_autocorr(sampler, discard=burnin)
    rhat = _split_rhat(sampler.get_chain(discard=burnin))
    finite = rhat[np.isfinite(rhat)]
    max_rhat = float(np.max(finite)) if len(finite) == len(rhat) and len(rhat) > 0 else None
    max_tau = float(np.max(tau)) if tau is not None else None
    converged = (max_tau is not None and max_rhat is not None
                 and n >= MCMC_TAU_FACTOR * max_tau and max_rhat < MCMC_MAX_RHAT)
    return {"steps": n, "burnin": burnin, "tau": tau, "mean_tau": mean_tau, "max_tau": max_tau,
            "max_rhat": max_rhat, "converged": bool(converged)}


def _run_until_converged(sampler, p0, max_steps: int, uncertainty_options: dict) -> dict:
    """
    Run `sampler` in blocks of MCMC_CHECK_INTERVAL steps until `_mcmc_diagnostics`
    reports convergence or `max_steps` is reached. Returns the final diagnostics.
    """
    state = p0
    diag = None
    while sampler.iteration < max_steps:
        state = sampler.run_mcmc(state, min(MCMC_CHECK_INTERVAL, max_steps - sampler.iteration), progress=False)
        if sampler.iteration < min(MCMC_MIN_STEPS, max_steps):
            continue
        diag = _mcmc_diagnostics(sampler, uncertainty_options)
        tau_txt = f"{diag['max_tau']:.1f}" if diag['max_tau'] is not None else "n/a"
        rhat_txt = f"{diag['max_rhat']:.4f}" if diag['max_rhat'] is not None else "n/a"
        print(f"  {diag['steps']} steps: max autocorrelation time {tau_txt}, max split-Rhat {rhat_txt}")
        if diag["converged"]:
            break
    if diag is None:
        diag = _mcmc_diagnostics(sampler, uncertainty_options)
    return diag


def _resolve_burnin(nsteps: int, tau_rough, uncertainty_options: dict) -> int:
    """
    Burn-in length in steps. An explicit `uncertainty_options.burnin_fraction` overrides
    everything; otherwise default to `max(0.3*nsteps, 5*max(tau))` when a rough
    autocorrelation estimate is available, else the historical flat 30%
.
    """
    frac = uncertainty_options.get('burnin_fraction')
    if frac is not None:
        burnin = int(round(float(frac) * nsteps))
    elif tau_rough is not None:
        burnin = max(int(0.3 * nsteps), int(5 * np.max(tau_rough)))
    else:
        burnin = int(0.3 * nsteps)
    return int(np.clip(burnin, 0, max(nsteps - 1, 0)))


def _daily_residual_sigma(data: CommonData, eval_mask: np.ndarray) -> float:
    """
    Root-mean-square of the daily residuals (simulated - observed) on scored days
    that have an observation. Used as the standard deviation of the daily noise
    added to prediction intervals, which are produced at daily resolution even
    when calibration scored weekly/monthly means.
    """
    m = eval_mask & (data.Twat_obs != -999.0) & (data.Twat_mod != -999.0)
    if not np.any(m):
        return 0.0
    return float(np.sqrt(np.mean((data.Twat_mod[m] - data.Twat_obs[m]) ** 2)))


def _draw_rng(chain_hash: str, chain_row: int) -> np.random.Generator:
    """Random generator for the residual noise of one posterior draw in a FORWARD run,
    determined by the chain's content hash and the draw's row in the chain."""
    return np.random.default_rng([int(chain_hash[:16], 16), int(chain_row)])


def _noisy_member(Twat_mod: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """One ensemble member: simulation plus residual noise, NaN where not simulated (gaps)."""
    member = Twat_mod + noise
    member[Twat_mod == -999.0] = np.nan
    return member


def _percentile_envelope(data: CommonData, ensemble_simulations: np.ndarray, prediction_interval: float) -> pd.DataFrame:
    lower_perc = (100.0 - prediction_interval) / 2.0
    upper_perc = 100.0 - lower_perc

    perc_lower = np.percentile(ensemble_simulations, lower_perc, axis=0)
    perc_50 = np.percentile(ensemble_simulations, 50, axis=0)
    perc_upper = np.percentile(ensemble_simulations, upper_perc, axis=0)

    # Replace calculated percentiles with NaN where the base model has missing data gaps
    perc_lower = np.where(data.Twat_mod == -999.0, np.nan, perc_lower)
    perc_50 = np.where(data.Twat_mod == -999.0, np.nan, perc_50)
    perc_upper = np.where(data.Twat_mod == -999.0, np.nan, perc_upper)

    return pd.DataFrame({
        'Year': data.date[:, 0],
        'Month': data.date[:, 1],
        'Day': data.date[:, 2],
        'Twat_mod_lower': perc_lower,
        'Twat_mod_p50': perc_50,
        'Twat_mod_upper': perc_upper
    })


def _save_ensemble_npz(data: CommonData, ensemble_simulations: np.ndarray, filename: str) -> None:
    """
    Write the raw (n_samples, n_days) ensemble matrix, post-warm-up, as compressed
    npz. Percentile bands alone
    cannot produce aggregate statistics (a rolling mean of the p5 series is not the p5
    of the rolling mean), so the raw ensemble is needed for degree-days, threshold
    exceedance, and paired scenario differences -- see `pyair2stream/scenario.py`.
    """
    dates = data.date[365:]
    np.savez_compressed(
        filename,
        simulations=ensemble_simulations[:, 365:],
        year=dates[:, 0],
        month=dates[:, 1],
        day=dates[:, 2],
    )
    print(f"Saved raw ensemble matrix ({ensemble_simulations.shape[0]} samples x "
          f"{ensemble_simulations.shape[1] - 365} days) to {filename}")


def _export_ensemble_outputs(data: CommonData, ensemble_simulations: np.ndarray, prediction_interval: float,
                              env_filename: str, ensemble_filename: Optional[str] = None,
                              save_ensemble: bool = False) -> dict:
    """
    Write the percentile envelope CSV and, if requested, the raw ensemble npz (both
    post-warm-up). Returns the interval's empirical coverage (see `_interval_coverage`).
    """
    env_df = _percentile_envelope(data, ensemble_simulations, prediction_interval)
    env_df.iloc[365:].to_csv(env_filename, index=False)  # drop the warm-up block
    print(f"Saved predictive uncertainty envelopes to {env_filename}")

    if save_ensemble:
        if ensemble_filename is None:
            raise ValueError("save_ensemble is True but no ensemble_filename was provided.")
        _save_ensemble_npz(data, ensemble_simulations, ensemble_filename)

    return _interval_coverage(data, env_df, prediction_interval)


def _interval_coverage(data: CommonData, env_df: pd.DataFrame, prediction_interval: float) -> dict:
    """
    Share of observed, scored days whose observation lies inside the prediction
    interval. For a well-calibrated X% interval this should be close to X%; a much
    lower value means the interval is too narrow. None when there are no observations.
    """
    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=bool)
    lo = env_df['Twat_mod_lower'].to_numpy()
    hi = env_df['Twat_mod_upper'].to_numpy()
    m = eval_mask & (data.Twat_obs != -999.0) & np.isfinite(lo) & np.isfinite(hi)
    n = int(np.sum(m))
    if n == 0:
        return {"interval_coverage": None, "interval_coverage_n_days": 0}
    obs = data.Twat_obs[m]
    coverage = float(np.mean((obs >= lo[m]) & (obs <= hi[m])))
    print(f"Interval check: {coverage:.1%} of {n} observed days lie inside the "
          f"{prediction_interval:g}% prediction interval.")
    return {"interval_coverage": coverage, "interval_coverage_n_days": n}


def _hash_file(path: str) -> str:
    """SHA-256 hex digest of a file's raw bytes -- used as a chain-identity check
 so a paired-scenario pairing
    check can detect "this is a different chain" even if the path string reused is
    identical (e.g. the file was regenerated) or different (e.g. a copy)."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _check_chain_provenance(data: CommonData, sidecar_path: str) -> None:
    """
    Refuse a FORWARD run whose model version, integrator, Qmedia, Tice_cover or
    min_theta_floor differ from those the chain's parameters were fitted under
    (recorded in the chain's `_meta.json`): the parameters mean something only with
    them (docs/METHODS.md §4, §6). Chains written before these were recorded (0.4.2
    and earlier for the first three, 0.5.0 and earlier for the last two) cannot be
    checked; a note says so.
    """
    meta = {}
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path, 'r') as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = {}
    if not all(k in meta for k in ("version", "integrator", "qmedia")):
        print(f"Note: {sidecar_path} does not record the model version, integrator and Qmedia the chain was "
              "fitted with (chains from version 0.4.2 or earlier), so they cannot be checked against this run. "
              "Make sure they match.")
        return
    problems = []
    if int(meta["version"]) != int(data.version):
        problems.append(f"model version {meta['version']} (this run: {data.version})")
    if meta["integrator"] != data.mod_num:
        problems.append(f"integrator {meta['integrator']} (this run: {data.mod_num})")
    # 0.1%: a Qmedia typed with a few significant digits is the same one.
    if data.version not in (3, 5) and not np.isclose(float(meta["qmedia"]), float(data.Qmedia), rtol=1e-3, atol=0.0):
        problems.append(f"Qmedia {meta['qmedia']} (this run: {float(data.Qmedia)}; they differ by more than 0.1%)")
    differences = settings_differences(meta, data)
    if differences is None:
        print(SETTINGS_NOT_RECORDED.format(path=sidecar_path))
    else:
        problems += differences
    if problems:
        raise ValueError(
            f"The MCMC chain {sidecar_path.replace('_meta.json', '.csv')} was fitted with "
            + "; ".join(problems) + ". Its parameters mean something only with the settings they were "
            "fitted with: use the calibration's settings (paths.calibration_metadata) or the chain of a "
            "calibration with this run's settings."
        )


def _check_ensemble_divergence(n_total: int, excluded: list, on_divergent_draw: str,
                                max_divergent_fraction: float, label: str,
                                sample_indices=None) -> dict:
    """
    Summarize per-draw divergence (see `model.is_numerically_divergent`) after an
    ensemble-generation loop (`forward_mode`'s prediction-interval block,
    `_run_mcmc_uncertainty`'s envelope loop) has finished appending its valid draws
    and skipping/recording its divergent ones in `excluded`.

    `excluded` is a list of `{"draw_index", "chain_row", "params"}` dicts, one per
    draw excluded as numerically divergent (non-finite, above `max_plausible_twat`,
    or unstable with an explicit integrator; `model.is_numerically_divergent`).
    `sample_indices` (if given) is the full array of chain rows requested for this
    batch, in order; used to compute `valid_draw_indices` -- the subset that actually survived
    filtering, which is what determines the row alignment of the saved ensemble
    and is the authoritative check for a paired-scenario comparison.

    Prints a console warning if any draws were excluded, then raises
    `NumericalDivergenceError` if either no draws were requested/available in the
    first place, every requested draw diverged (an empty ensemble is never
    silently returned as if it had succeeded), or the excluded fraction exceeds
    `max_divergent_fraction` (mirroring `model.warn_on_stability`'s
    `stability_error_fraction` pattern: a small, expected fraction of bad draws is
    tolerated and reported, a large one is refused rather than silently proceeding
    with a depleted ensemble).

    Returns a dict of the same summary, meant to be merged into the run's sidecar
    metadata JSON (`MCMC_chain_*_meta.json` / the FORWARD prediction-interval
    equivalent) so the exclusion is visible without inspecting console logs.
    """
    if n_total == 0:
        raise NumericalDivergenceError(
            f"No {label} draws were requested/available (n_total=0); there is nothing "
            f"to build an ensemble/percentile envelope from. This is not itself a "
            f"divergence -- check `n_samples`/the source chain rather than "
            f"`max_divergent_fraction`."
        )

    n_excluded = len(excluded)
    frac_excluded = n_excluded / n_total

    if n_excluded > 0:
        print(
            f"Warning: {n_excluded}/{n_total} {label} draws ({frac_excluded:.1%}) were "
            f"excluded from the ensemble as numerically divergent (non-finite, above "
            f"max_plausible_twat, or unstable with RK4/RK2/EUL). uncertainty_options.on_divergent_draw='{on_divergent_draw}'."
        )

    n_valid = n_total - n_excluded
    if n_valid == 0:
        raise NumericalDivergenceError(
            f"All {n_total} {label} draws diverged (non-finite, above "
            f"max_plausible_twat, or unstable with RK4/RK2/EUL); no valid draws remain to build an ensemble/percentile "
            f"envelope. See docs/METHODS.md §12."
        )

    if frac_excluded > max_divergent_fraction:
        raise NumericalDivergenceError(
            f"{frac_excluded:.1%} of {label} draws ({n_excluded}/{n_total}) were excluded "
            f"as numerically divergent, above the "
            f"max_divergent_fraction={max_divergent_fraction:.0%} threshold. Investigate the "
            f"excluded parameter draws (see the console warning above and the sidecar "
            f"metadata), tighten `parameter_bounds`, or raise "
            f"`uncertainty_options.max_divergent_fraction` if you have verified this is "
            f"expected. See docs/METHODS.md §12."
        )

    summary = {
        "n_draws_requested": n_total,
        "n_divergent_draws_excluded": n_excluded,
        "divergent_draw_fraction": frac_excluded,
        "excluded_draws": excluded,
    }

    if sample_indices is not None:
        excluded_positions = {d["draw_index"] for d in excluded}
        summary["valid_draw_indices"] = [
            int(sample_indices[i]) for i in range(n_total) if i not in excluded_positions
        ]

    return summary

def sub_1(data: CommonData) -> np.float64:
    """
    Helper function to call model and evaluate the objective function.
    Replicates SUBROUTINE sub_1
    """
    call_model(data)
    return np.float64(funcobj(data))

# PSO evaluates its particles in parallel worker processes. Each worker receives the
# data once, when it starts; each task then carries only one particle's parameters.
# (Sending the whole data object with every task cost more than the model run itself.)
_worker_data: Optional[CommonData] = None


def _init_particle_worker(data: CommonData) -> None:
    global _worker_data
    _worker_data = data


def eval_particle_worker(p_vals: np.ndarray):
    """
    Evaluate one particle in a worker process started with `_init_particle_worker`.

    Returns `(eff_index, nse, r2, mae)`. `sub_1` sets `data.current_nse`/
    `current_r2`/`current_mae` as a side effect inside this (child) process; only
    the explicit return value crosses the process boundary back to the parent, so
    those metrics must be returned here rather than read from `data` afterward --
    the parent's own `data.current_*` are untouched defaults otherwise.
    """
    data = _worker_data
    data.par[:len(p_vals)] = p_vals
    eff_index = sub_1(data)
    return eff_index, data.current_nse, data.current_r2, data.current_mae


def forward_mode(data: CommonData) -> None:
    """
    Replicates SUBROUTINE forward_mode
    Adds optional probabilistic Prediction Intervals based on MCMC chains.
    """
    # FORWARD mode does not calibrate, so it may legitimately have no T_water
    # observations at all (a pure climate-projection/scenario run). main() no
    # longer calls aggregation()/statis() unconditionally before dispatching
    # here -- statis() raises when there are no
    # observations, so it must only run when there are some.
    has_obs = False
    for val in data.Twat_obs:
        if val != -999.0:
            has_obs = True
            break

    warn_on_stability(data, error_fraction=data.stability_error_fraction)

    # Always aggregate: this builds I_inf/I_pos (needed by funcobj) and
    # correctly re-initialises Twat_obs_agg to all -999 when there are no
    # observations, rather than leaving it at read_Tseries's all-zero
    # allocation. n_dat comes out 0 when has_obs is False, which funcobj()
    # already handles by returning -999.0 without touching mean_obs/TSS_obs.
    aggregation(data)

    if has_obs:
        statis(data)
        ei = sub_1(data)
    else:
        # It's a pure projection, we skip the objective evaluation.
        call_model(data)
        ei = -999.0

    check_numerical_divergence(data, max_plausible_twat=data.max_plausible_twat)

    data.par_best = data.par.copy()
    data.finalfit = ei
    if has_obs:
        print(f'Efficiency index of this run against its own T_water observations: {data.finalfit}')

    # Optional Probabilistic Forward Envelope
    if data.forward_options and data.forward_options.get('enable_prediction_intervals', False):
        chain_path = data.forward_options.get('mcmc_chain_path')
        if not chain_path or not os.path.exists(chain_path):
            print(f"Warning: Cannot generate prediction intervals. MCMC chain not found at {chain_path}")
            return

        print(f"Generating Forward Prediction Intervals from {chain_path}...")

        # The seed of the draw of parameter sets: forward_options.random_seed, else the
        # top-level random_seed (the one the guide documents for repeatable results).
        forward_seed = data.forward_options.get('random_seed', None)
        if forward_seed is not None:
            seed, seed_source = int(forward_seed), 'forward_options.random_seed'
            if data.random_seed is not None and int(data.random_seed) != seed:
                print(f"Note: random_seed is {data.random_seed} and forward_options.random_seed is {seed}; the "
                      f"parameter sets are drawn with forward_options.random_seed ({seed}).")
        elif data.random_seed is not None:
            seed, seed_source = int(data.random_seed), 'random_seed'
        else:
            seed, seed_source = None, None
        rng = np.random.default_rng(seed)
        data.forward_draw = {"seed": seed, "source": seed_source, "reused_from": None}

        chain_df = pd.read_csv(chain_path)
        chain = chain_df.values
        chain_hash = _hash_file(chain_path)
        chain_n_rows = len(chain)

        # A paired scenario comparison (water abstraction: observed vs. naturalised
        # flow; climate projection: historical vs. projected flow) requires two
        # forward_mode() runs to draw the exact SAME posterior samples in the SAME
        # order. Relying on a shared `random_seed` across two separate CLI
        # invocations/config files is fragile (easy to omit, or to typo two
        # different values) and gives no way to detect the mistake after the fact
        #. `reuse_sample_indices_from`
        # instead reuses the literal indices a prior run saved, skipping the random
        # draw (and therefore the global random state) entirely.
        reuse_path = data.forward_options.get('reuse_sample_indices_from')
        if reuse_path:
            if not os.path.exists(reuse_path):
                raise FileNotFoundError(
                    f"forward_options.reuse_sample_indices_from file not found: {reuse_path}"
                )
            with open(reuse_path, 'r') as f:
                prior_meta = json.load(f)
            prior_hash = prior_meta.get('chain_content_sha256')
            prior_n_rows = prior_meta.get('chain_n_rows')
            if prior_hash != chain_hash or prior_n_rows != chain_n_rows:
                raise ValueError(
                    f"forward_options.reuse_sample_indices_from='{reuse_path}' was drawn "
                    f"from a different MCMC chain (content hash={prior_hash}, "
                    f"{prior_n_rows} rows) than the one currently loaded from "
                    f"'{chain_path}' (content hash={chain_hash}, {chain_n_rows} rows). "
                    "Both forward_mode() runs in a paired scenario comparison must point "
                    "`forward_options.mcmc_chain_path` at the SAME chain file."
                )
            sample_indices = np.asarray(prior_meta['sample_indices'], dtype=np.int64)
            n_samples = len(sample_indices)
            data.forward_draw["reused_from"] = reuse_path
            print(
                f"Reusing {n_samples} sample indices from {reuse_path} "
                "(random_seed/global random state ignored for this draw)."
            )
        else:
            n_samples = data.forward_options.get('n_samples', 1000)
            n_samples = min(n_samples, len(chain))
            sample_indices = rng.choice(len(chain), size=n_samples, replace=False)

        samples = chain[sample_indices]

        uncertainty_options = data.uncertainty_options or {}
        sidecar_path = chain_path.replace('.csv', '_meta.json')

        source_chain_converged = None
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f:
                    source_chain_converged = json.load(f).get('converged')
            except Exception:
                source_chain_converged = None
        if source_chain_converged is False:
            print(f"Warning: the MCMC chain {chain_path} did NOT converge (see {sidecar_path}); "
                  "prediction intervals built from it are not reliable.")
        _check_chain_provenance(data, sidecar_path)

        # Resolve sigma: explicit config override first, then the sidecar written by
        # DE-MCMC (mirroring the `rho` resolution below), matching `rho`'s
        # existing carry-forward instead of silently defaulting to 0.0 behind a print.
        chain_meta = {}
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f:
                    chain_meta = json.load(f)
            except (OSError, ValueError):
                chain_meta = {}
        sigma_override = data.forward_options.get('residual_sigma')
        if sigma_override is not None and float(sigma_override) > 0.0:
            sigma = float(sigma_override)
            print(f"Note: this run uses residual_sigma = {sigma:g}, as set in forward_options, not the "
                  f"chain's {chain_meta.get('sigma', 'unrecorded value')}.")
        elif os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f:
                    sidecar_data = json.load(f)
                sigma = float(sidecar_data.get('sigma', 0.0))
                if sigma > 0.0:
                    print(f"Using sigma={sigma:.4f} carried from calibration run {sidecar_path}")
            except Exception as e:
                print(f"Warning: Failed to read sigma from sidecar {sidecar_path} ({e}).")
                sigma = 0.0
        else:
            sigma = 0.0

        if sigma <= 0.0:
            raise ValueError(
                "enable_prediction_intervals is True but residual_sigma is 0.0/unavailable "
                f"(no forward_options.residual_sigma override, and no usable 'sigma' in "
                f"sidecar {sidecar_path}). A prediction interval with no residual term is "
                "not a prediction interval (docs/METHODS.md §13)."
            )

        # The error model the chain was fitted with, unless this run's settings name another
        # (a deliberate choice, e.g. a sensitivity test, which is then noted in the summary).
        set_noise_model = uncertainty_options.get('noise_model', DEFAULT_NOISE_MODEL)
        chain_noise_model = chain_meta.get('noise_model_used_for_this_run')
        if chain_noise_model is None:
            noise_model = set_noise_model
            print(f"Note: {sidecar_path} does not record the error model (noise_model) the chain was fitted "
                  f"with, so this run uses '{noise_model}' from its settings. Make sure it is the calibration's.")
        elif uncertainty_options.get('noise_model_set') and set_noise_model != chain_noise_model:
            noise_model = set_noise_model
            print(f"Note: the MCMC chain was fitted with noise_model '{chain_noise_model}'; this run uses "
                  f"'{noise_model}', as set in uncertainty_options. Its prediction ranges are therefore not "
                  "those of the calibration's error model.")
        else:
            noise_model = chain_noise_model
            print(f"Using noise_model '{noise_model}' carried from calibration run {sidecar_path}")
        rho_used = 0.0

        if noise_model == 'ar1':
            ar1_rho_override = uncertainty_options.get('ar1_rho')

            # Like sigma, rho comes from the calibration run when it is known, so the
            # interval does not depend on the data being predicted and two scenario runs
            # use the same noise (their paired difference then cancels it exactly).
            sidecar_rho = None
            sidecar_timescale = None
            if os.path.exists(sidecar_path):
                try:
                    with open(sidecar_path, 'r') as f:
                        sidecar = json.load(f)
                    sidecar_rho = sidecar.get('rho')
                    # Chains written before rho_timescale existed (0.4.1 and earlier) used consecutive days.
                    sidecar_timescale = sidecar.get('rho_timescale', 'daily')
                except Exception as e:
                    print(f"Warning: Failed to read rho from sidecar {sidecar_path} ({e}).")
            if ar1_rho_override is not None:
                rho_used = ar1_rho_override
                print(f"Note: this run uses ar1_rho = {rho_used:g}, as set in uncertainty_options, not the "
                      f"chain's {sidecar_rho if sidecar_rho is not None else 'unrecorded value'}.")
            elif sidecar_rho is not None:
                rho_used = float(sidecar_rho)
                print(f"Using rho={rho_used:.4f} carried from calibration run {sidecar_path}")
                rho_timescale = uncertainty_options.get('rho_timescale', DEFAULT_RHO_TIMESCALE)
                if sidecar_timescale != rho_timescale:
                    print(f"Note: that rho was estimated at the '{sidecar_timescale}' time scale, not this run's "
                          f"rho_timescale '{rho_timescale}' (which applies only when the chain records no rho). "
                          "To change it, rerun DE-MCMC with the time scale wanted, or set "
                          "uncertainty_options.ar1_rho.")
            elif has_obs:
                eval_mask_for_rho = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=bool)
                segments_for_rho = _segments_for(data)
                rho_timescale = uncertainty_options.get('rho_timescale', DEFAULT_RHO_TIMESCALE)
                rho_used = estimate_rho(data.Twat_mod, data.Twat_obs, eval_mask_for_rho, segments_for_rho,
                                        rho_timescale)
                print(f"Using rho={rho_used:.4f} estimated ({rho_timescale}) from this run's own residuals "
                      "(no rho recorded with the chain).")
            else:
                print("Warning: No residuals available to estimate rho; falling back to rho=0.0 (equivalent to iid)")
                rho_used = 0.0

        ensemble_simulations = []
        excluded_draws = []
        n_par = N_PAR

        on_divergent_draw = uncertainty_options.get('on_divergent_draw', 'drop')
        max_divergent_fraction = uncertainty_options.get('max_divergent_fraction', 0.10)

        # Determine active params from dataframe columns
        active_cols = chain_df.columns
        active_params = [int(c.split('_')[1])-1 for c in active_cols]

        best_params_deterministic = data.par_best.copy()
        segments_for_noise = _segments_for(data)

        for i, theta in enumerate(samples):
            p_vals = best_params_deterministic.copy()
            for idx, j in enumerate(active_params):
                p_vals[j] = theta[idx]

            data.par[:n_par] = p_vals

            call_model(data)

            # A single bad posterior draw (e.g. a scenario discharge the chain was
            # never fitted under) must not crash the whole ensemble, nor be silently
            # written into the percentile envelope / raw ensemble.
            if is_numerically_divergent(data, data.max_plausible_twat):
                chain_row = int(sample_indices[i])
                params_dict = {f"par_{j+1}": float(p_vals[j]) for j in active_params}
                if on_divergent_draw == 'raise':
                    raise NumericalDivergenceError(
                        f"Forward prediction-interval draw {i} (chain row {chain_row}, "
                        f"params={params_dict}) diverged (non-finite, above "
                        f"max_plausible_twat, or unstable with RK4/RK2/EUL). uncertainty_options.on_divergent_draw='raise'; "
                        f"set 'drop' (the default) to exclude divergent draws instead. See "
                        f"docs/METHODS.md §12."
                    )
                excluded_draws.append({"draw_index": i, "chain_row": chain_row, "params": params_dict})
                continue

            # The residual noise of a draw comes from a random stream fixed by the chain
            # and the chain row, not from this run's generator: two scenario runs that
            # use the same draws then add the same noise on the same days, so it cancels
            # in their paired difference, which reflects parameter uncertainty only.
            draw_rng = _draw_rng(chain_hash, int(sample_indices[i]))
            if noise_model == 'ar1':
                noise = generate_ar1_noise(data.n_tot, sigma, rho_used, segments_for_noise, draw_rng)
            else:
                noise = draw_rng.normal(0, sigma, data.n_tot)

            ensemble_simulations.append(_noisy_member(data.Twat_mod, noise))

        # `sample_indices` is passed through so `valid_draw_indices` (the chain
        # rows that actually survived divergence filtering, in order -- not the
        # originally-requested `sample_indices`) ends up in the summary: that is
        # what determines the row alignment of the ensemble saved below, and is
        # therefore the authoritative check `scenario.paired_difference_from_files`
        # uses.
        divergence_summary = _check_ensemble_divergence(
            len(samples), excluded_draws, on_divergent_draw, max_divergent_fraction,
            "forward prediction-interval", sample_indices=sample_indices,
        )

        ensemble_simulations = np.array(ensemble_simulations)

        prediction_interval = uncertainty_options.get('prediction_interval', 90.0)
        env_filename = os.path.join(data.folder, f"Forward_Prediction_Envelopes_{data.station}_{data.series}_{data.time_res}.csv")
        ensemble_filename = os.path.join(data.folder, f"Forward_Prediction_Ensemble_{data.station}_{data.series}_{data.time_res}.npz")
        save_ensemble = bool(uncertainty_options.get('save_ensemble', False))
        coverage = _export_ensemble_outputs(data, ensemble_simulations, prediction_interval, env_filename, ensemble_filename, save_ensemble)

        # Sidecar metadata for the forward prediction-interval ensemble (the
        # FORWARD-mode equivalent of MCMC_chain_*_meta.json), named to pair with the
        # ensemble .npz (not the envelope CSV) since that is what `scenario.py`
        # consumes. Records the divergent-draw exclusion so it is visible without inspecting
        # console logs, plus the provenance (source chain identity, requested and
        # surviving sample indices) `scenario.paired_difference_from_files` needs to
        # detect a mismatched pairing between two scenario runs.
        meta_filename = ensemble_filename.replace('.npz', '_meta.json')
        meta_data = {
            **divergence_summary,
            **coverage,
            "on_divergent_draw": on_divergent_draw,
            "max_divergent_fraction": max_divergent_fraction,
            "chain_path": chain_path,
            "chain_content_sha256": chain_hash,
            "chain_n_rows": chain_n_rows,
            "requested_seed": seed,
            "seed_source": seed_source,
            "sample_indices": [int(x) for x in sample_indices],
            "reused_sample_indices_from": reuse_path if reuse_path else None,
            "source_chain_converged": source_chain_converged,
            "noise_model": noise_model,
            "residual_sigma": sigma,
            "rho": float(rho_used),
        }
        with open(meta_filename, 'w') as f:
            json.dump(meta_data, f, indent=2, allow_nan=False)
        print(f"Saved forward prediction-interval metadata sidecar to {meta_filename}")

        # Restore deterministic parameters
        data.par[:n_par] = best_params_deterministic
        call_model(data)

def PSO_mode(data: CommonData, seed: Optional[int] = None) -> None:
    """
    Replicates SUBROUTINE PSO_mode
    """
    _require_free_parameters(data)
    print(f'N. particles = {data.n_particles}, N. run = {data.n_run}')

    # A local Generator (rather than the legacy `np.random.seed()`, which mutates
    # global numpy random state and would interfere with any calling script)
    rng = np.random.default_rng(seed)

    n_par = 8
    n_particles = data.n_particles
    n_run = data.n_run

    x = np.zeros((n_par, n_particles), dtype=np.float64)
    v = np.zeros((n_par, n_particles), dtype=np.float64)
    pbest = np.zeros((n_par, n_particles), dtype=np.float64)
    gbest = np.zeros(n_par, dtype=np.float64)
    fit = np.zeros(n_particles, dtype=np.float64)
    # fitbest must NOT be initialized to zero: the objective function (e.g. NSE)
    # can be strongly negative for poor initial random parameter draws, so a
    # zero-initialized fitbest is never beaten and PSO silently returns the
    # all-zero initial parameters.
    fitbest = np.full(n_particles, -1e30, dtype=np.float64)

    # We output history to CSV instead of binary
    output_filename = os.path.join(data.folder, f"0_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.csv")
    history = []

    dw = (data.wmax - data.wmin) / n_run
    w = data.wmax

    x_rand = rng.random((n_par, n_particles))
    v_rand = rng.random((n_par, n_particles))

    for j in range(n_par):
        dxmax = data.parmax[j] - data.parmin[j]
        dvmax = 1.0 * dxmax
        x[j, :] = x_rand[j, :] * dxmax + data.parmin[j]
        v[j, :] = v_rand[j, :] * dvmax
        pbest[j, :] = x[j, :]

    n_workers = os.cpu_count() or 1
    chunk = max(1, n_particles // (4 * n_workers))   # a few batches per worker and iteration
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=_init_particle_worker,
                                                initargs=(data,)) as executor:
        results = list(executor.map(eval_particle_worker, [x[:, k].copy() for k in range(n_particles)],
                                    chunksize=chunk))

        for k in range(n_particles):
            eff_index, nse_k, r2_k, mae_k = results[k]
            if not np.isnan(eff_index):
                fitbest[k] = eff_index
            row = list(x[:, k]) + [eff_index, nse_k, r2_k, mae_k]
            history.append(row)

        # Fix: use fitbest to find initial global best instead of fit
        # Fix: use nanargmax to handle NaN efficiency values correctly
        best_idx = int(np.nanargmax(fitbest))
        foptim = fitbest[best_idx]
        gbest[:] = x[:, best_idx]

        for i in range(n_run):
            # We can also parallelize the updates in each run
            # Collect particles to evaluate
            particles_to_eval = []
            eval_indices = []
            for k in range(n_particles):
                r = rng.random(2 * n_par)
                status = 0

                for j in range(n_par):
                    v[j, k] = w * v[j, k] + data.c1 * r[j] * (pbest[j, k] - x[j, k]) + data.c2 * r[j + n_par] * (gbest[j] - x[j, k])
                    x[j, k] = x[j, k] + v[j, k]

                    # Absorbing wall
                    if x[j, k] > data.parmax[j]:
                        x[j, k] = data.parmax[j]
                        v[j, k] = 0.0
                        status = 1
                    elif x[j, k] < data.parmin[j]:
                        x[j, k] = data.parmin[j]
                        v[j, k] = 0.0
                        status = 1

                if status == 0:
                    particles_to_eval.append(x[:, k].copy())
                    eval_indices.append(k)
                else:
                    fit[k] = -1e30

            eval_results = list(executor.map(eval_particle_worker, particles_to_eval, chunksize=chunk))

            idx = 0
            for k in eval_indices:
                eff_index, nse_k, r2_k, mae_k = eval_results[idx]
                fit[k] = eff_index
                row = list(x[:, k]) + [eff_index, nse_k, r2_k, mae_k]
                history.append(row)
                idx += 1

            for k in range(n_particles):
                # Extreme initial parameter draws can cause solver arithmetic overflow,
                # producing NaN objective values. np.argmax over an array containing NaN
                # returns a NaN-adjacent/undefined index, so both the per-particle update
                # and the global-best lookup must explicitly exclude NaNs.
                if not np.isnan(fit[k]) and fit[k] > fitbest[k]:
                    fitbest[k] = fit[k]
                    pbest[:, k] = x[:, k]

            best_idx = int(np.nanargmax(fitbest))
            foptim = fitbest[best_idx]
            gbest[:] = pbest[:, best_idx]

            w = w - dw

            if i >= 9:
                if (i + 1) % max(1, int(n_run / 10)) == 0:
                    perc = float(i + 1) / float(n_run) * 100.0
                    print(f"Progress: {perc:.1f} %")

            count = 0
            for k in range(n_particles):
                norm = 0.0
                for j in range(n_par):
                    if data.flag_par[j]:
                        denom = data.parmax[j] - data.parmin[j]
                        if denom > 0:
                            diff = (pbest[j, k] - gbest[j]) / denom
                        else:
                            diff = 0.0
                        norm += diff ** 2
                norm = np.sqrt(norm)
                # Fix: meaningful tolerance instead of norm < 0.0
                if norm < 1e-4:
                    count += 1

            if count >= (0.9 * n_particles):
                print('- Warning: PSO has been stopped')
                break

    data.par_best = gbest.copy()
    data.finalfit = foptim
    print(f'Efficiency Index in calibration {data.finalfit}')

    # Save to CSV
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index", "NSE", "R2", "MAE"])
    df.to_csv(output_filename, index=False)


def LH_mode(data: CommonData, seed: Optional[int] = None) -> None:
    """
    Replicates SUBROUTINE LH_mode
    """
    _require_free_parameters(data)
    print(f'N. run = {data.n_run}')

    # A local Generator (rather than the legacy `np.random.seed()`, which mutates
    # global numpy random state and would interfere with any calling script)
    rng = np.random.default_rng(seed)

    n_par = 8
    n_run = data.n_run

    gbest = np.zeros(n_par, dtype=np.float64)
    # Any finite score beats the start value: a floor such as -999 would keep the all-zero
    # start if every sample scored below it (NSE can be far below -999 for a poor set).
    foptim = -np.inf

    output_filename = os.path.join(data.folder, f"0_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.csv")
    history = []

    permut = np.zeros((n_run, n_par), dtype=np.int32)
    for j in range(n_par):
        # Fix: Using numpy.random.permutation to avoid custom Shuffle
        permut[:, j] = rng.permutation(n_run) + 1

    for i in range(n_run):
        for j in range(n_par):
            r = rng.random()
            r = r + (float(permut[i, j]) - 1.0)
            r = r / float(n_run)

            data.par[j] = data.parmin[j] + (data.parmax[j] - data.parmin[j]) * r

        eff_index = sub_1(data)
        fit = eff_index

        row = list(data.par[:n_par]) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
        history.append(row)

        if np.isfinite(fit) and fit > foptim:
            foptim = fit
            gbest[:] = data.par[:n_par]

        if i >= 9:
            if (i + 1) % max(1, int(n_run / 10)) == 0:
                perc = float(i + 1) / float(n_run) * 100.0
                print(f"Progress: {perc:.1f} %")

    if not np.isfinite(foptim):
        raise RuntimeError(f"LATHYP: none of the {n_run} parameter sets gave a finite score. "
                           "Check parameter_bounds and the data.")
    data.par_best = gbest.copy()
    data.finalfit = foptim
    print(f'Calibration efficiency index: {data.finalfit}')

    # Save to CSV
    # Fix: Pandas handles closing the file handle automatically via to_csv
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index", "NSE", "R2", "MAE"])
    df.to_csv(output_filename, index=False)


def DE_mode(data: CommonData, seed: Optional[int] = None) -> None:
    """
    Differential Evolution + L-BFGS-B hybrid optimization.
    Replaces PSO for a more robust global search followed by a local polish.
    """
    _require_free_parameters(data)
    print(f'Pop. Size (particles) = {data.n_particles}, Max Generations (runs) = {data.n_run}')

    # `seed` is passed directly to `differential_evolution` below, which accepts an
    # explicit seed/Generator without mutating global numpy random state -- no
    # `np.random.seed()` call needed here. `minimize`'s L-BFGS-B polish phase is deterministic given its starting point.
    n_par = 8
    output_filename = os.path.join(data.folder, f"0_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.csv")
    history = []

    # SciPy minimizers expect an objective to MINIMIZE.
    # sub_1 returns the raw objective (which we want to maximize for NSE/KGE, but minimize for RMS).
    # Since our internal RMS is already negated (returns -RMS), we ALWAYS want to maximize the output of sub_1.
    # Therefore, we negate the output of sub_1 for SciPy to minimize.
    # To avoid multiprocessing pickling issues with local functions, we run single-threaded (workers=1)
    # The performance is still very fast because scipy DE converges quickly.
    def objective_wrapper(p_vals):
        """
        Evaluate the objective function for a given parameter set during DE optimization.

        Parameters
        ----------
        p_vals : ndarray
            Array of length `n_par` containing the parameter values to evaluate.

        Returns
        -------
        float
            The negated objective value (since scipy minimizes). Returns a large
            positive penalty if the parameters lead to invalid or NaN metric values.
        """
        # Update parameters (only the first n_par)
        data.par[:n_par] = p_vals

        # Evaluate
        eff_index = sub_1(data)

        # Record history if valid
        row = list(p_vals) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
        history.append(row)

        # Return negated efficiency so scipy minimizes. A NaN score, or an infinite one (a
        # simulation that runs away), gets a large finite penalty: with inf, L-BFGS-B's
        # finite-difference gradient is inf - inf = NaN and the polish stops.
        if not np.isfinite(eff_index):
            return 1e30
        return -eff_index

    # Prepare bounds for scipy
    bounds = []
    for j in range(n_par):
        # If a parameter is fixed (min == max), differential_evolution can struggle if lb == ub.
        # But we must respect the flag_par and bounds.
        # SciPy handles lb == ub by fixing the parameter if we're careful, but let's ensure it's exact.
        if not data.flag_par[j] or data.parmin[j] == data.parmax[j]:
            bounds.append((data.parmin[j], data.parmin[j] + 1e-12)) # Add tiny epsilon to prevent DE failure
        else:
            bounds.append((data.parmin[j], data.parmax[j]))

    # Phase 1: Differential Evolution (Global Search)
    # workers=1 to avoid unpicklable local function 'objective_wrapper'
    # Parameter sets whose simulation runs away (e.g. a3 < 0) score astronomically badly.
    # SciPy's convergence test takes the standard deviation of all scores, which then
    # overflows to inf and correctly reads as "not converged yet"; only the warning is
    # silenced, the search is unchanged.
    with np.errstate(over='ignore'):
        result_de = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=data.n_run,
            popsize=data.n_particles,
            tol=data.de_tol,
            workers=1,
            polish=False,
            seed=seed
        )

    print(f"DE Finished. Best internal negated objective: {result_de.fun:.6f}")

    # Phase 2: L-BFGS-B (Local Polish)
    # Re-use the same objective wrapper
    result_bfgs = minimize(
        objective_wrapper,
        result_de.x,
        method="L-BFGS-B",
        bounds=bounds
    )

    print(f"L-BFGS-B Finished. Best internal negated objective: {result_bfgs.fun:.6f}")
    if not result_bfgs.success:
        print(f"Warning: the local search after DE (L-BFGS-B) stopped early: {result_bfgs.message}. "
              "The better of its result and the DE result is kept.")

    # Keep the DE solution if the local polish did not improve on it (e.g. an
    # abnormal L-BFGS-B termination).
    best_params = result_bfgs.x if result_bfgs.fun <= result_de.fun else result_de.x.copy()

    # Ensure fixed parameters are exactly at their fixed values (removing the 1e-12 epsilon if it was added)
    for j in range(n_par):
        if not data.flag_par[j] or data.parmin[j] == data.parmax[j]:
            best_params[j] = data.parmin[j]

    data.par[:n_par] = best_params
    final_eff = sub_1(data)

    data.par_best = best_params.copy()
    data.finalfit = final_eff
    print(f'Efficiency Index in calibration {data.finalfit}')

    # Save history to CSV
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index", "NSE", "R2", "MAE"])
    df.to_csv(output_filename, index=False)

def _run_mcmc_uncertainty(data: CommonData, seed: Optional[int], best_params: np.ndarray,
                           active_params: list, n_par: int = N_PAR) -> None:
    """
    Phase 3 of `DE_MCMC_mode`: builds the walker ensemble (a small ball around the DE
    optimum, scaled to each active parameter's bound width), runs `emcee` until
    converged, computes the diagnostics, and writes the chain/sidecar/envelope (and
    optional raw ensemble) outputs.
    """
    nwalkers = data.mcmc_walkers
    nsteps = data.mcmc_steps
    ndim = len(active_params)

    uncertainty_options = data.uncertainty_options or {}
    noise_model = uncertainty_options.get('noise_model', DEFAULT_NOISE_MODEL)
    likelihood = uncertainty_options.get('likelihood', DEFAULT_LIKELIHOOD)

    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=np.bool_)
    segments = _segments_for(data)

    # Re-evaluate at the DE optimum so rho/sigma are estimated at the point the MCMC
    # ensemble is actually centred on.
    data.par[:n_par] = best_params.copy()
    call_model(data)
    funcobj(data)
    check_daily_plausibility(data)

    rho_timescale = uncertainty_options.get('rho_timescale', DEFAULT_RHO_TIMESCALE)
    best_rho = estimate_rho(data.Twat_mod, data.Twat_obs, eval_mask, segments, rho_timescale)
    # The exact AR(1) likelihood removes the day-to-day correlation (e_t - rho * e_{t-1}), a
    # day-scale operation, so it always uses the correlation of consecutive days. rho_timescale
    # sets the rho of the simulated prediction noise and of the least-squares effective sample size.
    rho_likelihood = (estimate_ar1_rho(data.Twat_mod, data.Twat_obs, eval_mask, segments)
                      if likelihood == 'exact' else best_rho)
    # With weekly or monthly scoring each scored value is the mean of a block of days.
    block_days = scoring_block_days(data.time_res)
    if noise_model == 'ar1' and likelihood == 'exact' and block_days > 1:
        print(f"Warning: with time_resolution '{data.time_res}' no two scored values are consecutive days, so "
              "the exact AR(1) likelihood treats the scored errors as independent and the parameter intervals "
              "are too narrow if the errors persist from one block to the next. The default likelihood, "
              "'least_squares', allows for that persistence.")

    # Exactly the values the objective scores (`aggregation`): with weekly or monthly
    # scoring a block is stored on its middle day, which need not itself be scored
    # (prc < 1, or a segment's unscored start in gap-tolerant mode).
    valid_mask_agg = np.zeros(data.n_tot, dtype=bool)
    valid_mask_agg[data.I_inf[:data.n_dat, 2]] = True
    N = int(np.sum(valid_mask_agg))
    # Daily residual SD at the best fit: the noise level carried to FORWARD runs.
    best_sigma = _daily_residual_sigma(data, eval_mask)

    # Reused across every likelihood evaluation below: observations (and therefore the
    # scored rows and their spacing) do not change while theta is being explored, only the
    # simulated series does. Every scored value counts, including a block whose middle day
    # lies in a gap (it was left out of the AR(1) likelihoods); the likelihoods use the
    # distance in days between the scored values.
    ar1_runs = [np.flatnonzero(valid_mask_agg)] if noise_model == 'ar1' else None
    variance_factor = (scored_error_variance_factor(ar1_runs[0], rho_likelihood, block_days)
                       if noise_model == 'ar1' and likelihood == 'least_squares' else None)

    def log_probability(theta):
        p_vals = best_params.copy()
        for idx, j in enumerate(active_params):
            p_vals[j] = theta[idx]
            if p_vals[j] < data.parmin[j] or p_vals[j] > data.parmax[j]:
                return -np.inf

        data.par[:n_par] = p_vals
        call_model(data)
        eff_index = funcobj(data)

        if np.isnan(eff_index):
            return -np.inf

        # Computed on the SAME (aggregated) series the objective function itself scores
        # -- daily and aggregated coincide at 1d resolution.
        if noise_model == 'ar1':
            residuals = data.Twat_mod_agg - data.Twat_obs_agg
            if likelihood == 'least_squares':
                return _least_squares_log_likelihood(residuals, rho_likelihood, ar1_runs, block_days,
                                                     variance_factor)
            return _ar1_log_likelihood(residuals, rho_likelihood, ar1_runs, block_days)
        else:
            mod = data.Twat_mod_agg[valid_mask_agg]
            obs = data.Twat_obs_agg[valid_mask_agg]
            return _iid_log_likelihood(mod, obs)

    initial = np.array([best_params[j] for j in active_params])
    lo = np.array([data.parmin[j] for j in active_params])
    hi = np.array([data.parmax[j] for j in active_params])
    scale = 1e-3 * (hi - lo)

    rng = np.random.default_rng(seed)
    p0 = _reflected_walker_init(initial, scale, lo, hi, nwalkers, rng)

    sampler = _make_sampler(nwalkers, ndim, log_probability, seed)

    print(f"Running MCMC with {nwalkers} walkers until converged (at most {nsteps} steps)...")
    diag = _run_until_converged(sampler, p0, nsteps, uncertainty_options)
    burnin = diag["burnin"]
    tau_final, mean_tau, max_rhat = diag["tau"], diag["mean_tau"], diag["max_rhat"]
    converged = diag["converged"]
    if converged:
        print(f"MCMC converged after {diag['steps']} steps (at least {MCMC_TAU_FACTOR} x the "
              f"autocorrelation time, split-Rhat below {MCMC_MAX_RHAT}).")
    mean_acc = float(np.mean(sampler.acceptance_fraction))
    print(f"Mean acceptance fraction: {mean_acc:.3f}")

    # Keep roughly every (tau/2)-th step after burn-in: consecutive steps are highly
    # correlated, so this loses no information and keeps the chain file small.
    thin = max(1, int(0.5 * np.min(tau_final))) if tau_final is not None else 1
    chain = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    chain_df = pd.DataFrame(chain, columns=[f"par_{j+1}" for j in active_params])

    chain_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}.csv")
    chain_df.to_csv(chain_filename, index=False)
    print(f"Saved MCMC chain ({diag['steps']} steps, {burnin} discarded as burn-in, "
          f"every {thin}th step kept) to {chain_filename}")
    chain_hash = _hash_file(chain_filename)
    chain_n_rows = len(chain)

    convergence = {
        "converged": converged,
        "steps_run": diag["steps"],
        "max_steps": nsteps,
        "burnin": burnin,
        "thin": thin,
        "mean_autocorr_time": mean_tau,
        "max_autocorr_time": diag["max_tau"],
        "max_split_rhat": max_rhat,
        "convergence_rule": f"steps >= {MCMC_TAU_FACTOR} x autocorrelation time and split-Rhat < {MCMC_MAX_RHAT}",
        "mean_acceptance_fraction": mean_acc,
        "sampler_move": "emcee.moves.DEMove",
    }
    sidecar_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}_meta.json")
    strict_convergence = bool(uncertainty_options.get('strict_convergence', True))
    if not converged:
        msg = (f"MCMC did not converge within {nsteps} steps (rule: "
               f"{convergence['convergence_rule']}; reached max autocorrelation time "
               f"{diag['max_tau']}, max split-Rhat {max_rhat}).")
        if strict_convergence:
            with open(sidecar_filename, 'w') as f:
                json.dump({**convergence, "chain_path": chain_filename}, f, indent=4, allow_nan=False)
            raise RuntimeError(
                f"{msg} No prediction interval was produced. Increase optimization.mcmc_steps, "
                "or use a simpler model version (a posterior that will not converge usually "
                "means the data cannot pin down all the parameters), or set "
                "uncertainty_options.strict_convergence: false to produce results marked as "
                f"not converged. Diagnostics: {sidecar_filename}"
            )
        print(f"Warning: {msg} Results are marked as NOT CONVERGED.")

    # Compute Predictive Uncertainty Envelopes
    print("Generating Predictive Uncertainty Envelopes...")
    n_samples = min(1000, len(chain))
    sample_indices = rng.choice(len(chain), size=n_samples, replace=False)
    samples = chain[sample_indices]

    on_divergent_draw = uncertainty_options.get('on_divergent_draw', 'drop')
    max_divergent_fraction = uncertainty_options.get('max_divergent_fraction', 0.10)

    ensemble_simulations = []
    excluded_draws = []

    for i, theta in enumerate(samples):
        p_vals = best_params.copy()
        for idx, j in enumerate(active_params):
            p_vals[j] = theta[idx]

        data.par[:n_par] = p_vals
        call_model(data)

        # A single bad posterior draw must not crash the whole envelope-generation
        # batch, nor be silently written into the percentile envelope / raw
        # ensemble
        if is_numerically_divergent(data, data.max_plausible_twat):
            chain_row = int(sample_indices[i])
            params_dict = {f"par_{j+1}": float(p_vals[j]) for j in active_params}
            if on_divergent_draw == 'raise':
                raise NumericalDivergenceError(
                    f"MCMC envelope draw {i} (chain row {chain_row}, params={params_dict}) "
                    f"diverged (non-finite, above max_plausible_twat, or unstable with "
                    f"RK4/RK2/EUL). "
                    f"uncertainty_options.on_divergent_draw='raise'; set 'drop' (the default) "
                    f"to exclude divergent draws instead. See "
                    f"docs/METHODS.md §12."
                )
            excluded_draws.append({"draw_index": i, "chain_row": chain_row, "params": params_dict})
            continue

        # Residual SD of this draw's own daily residuals: the envelope is daily, so
        # the noise added to it must be at daily scale (at time_resolution 1d this
        # equals the RMSE of the scored series).
        sigma_iter = _daily_residual_sigma(data, eval_mask)

        if noise_model == 'ar1':
            noise = generate_ar1_noise(data.n_tot, sigma_iter, best_rho, segments, rng)
        else:
            noise = rng.normal(0, sigma_iter, data.n_tot)

        ensemble_simulations.append(_noisy_member(data.Twat_mod, noise))

    # `sample_indices` is passed through so `valid_draw_indices` (the chain rows
    # that actually survived divergence filtering, in order) ends up in the
    # summary -- see the matching note in forward_mode(). DE-MCMC doesn't itself
    # support `reuse_sample_indices_from` (a paired scenario comparison pairs two
    # `forward_mode()` runs, not two calibration runs), but this provenance is
    # still persisted for consistency/auditability.
    divergence_summary = _check_ensemble_divergence(
        len(samples), excluded_draws, on_divergent_draw, max_divergent_fraction,
        "MCMC envelope", sample_indices=sample_indices,
    )

    ensemble_simulations = np.array(ensemble_simulations)

    prediction_interval = uncertainty_options.get('prediction_interval', 90.0)
    env_filename = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")
    ensemble_filename = os.path.join(data.folder, f"MCMC_ensemble_{data.station}_{data.series}_{data.time_res}.npz")
    save_ensemble = bool(uncertainty_options.get('save_ensemble', False))
    coverage = _export_ensemble_outputs(data, ensemble_simulations, prediction_interval, env_filename, ensemble_filename, save_ensemble)

    print("Writing metadata sidecar...")
    sidecar_data = {
        # What the parameters mean depends on these: a FORWARD run refuses a chain fitted
        # under different ones (_check_chain_provenance).
        "version": int(data.version),
        "integrator": data.mod_num,
        "qmedia": float(data.Qmedia),
        **fit_settings(data),
        "rho": best_rho,
        "rho_timescale": rho_timescale,
        "rho_likelihood": rho_likelihood,
        "scoring_block_days": block_days,
        "likelihood_variance_factor": variance_factor,
        "sigma": best_sigma,
        "n_valid_pairs": N,  # N valid points used for variance, proxy for pairs
        "noise_model_used_for_this_run": noise_model,
        "likelihood": likelihood if noise_model == 'ar1' else 'least_squares',
        "mcmc_walkers": nwalkers,
        "mcmc_seed": seed,
        **convergence,
        "strict_convergence": strict_convergence,
        "on_divergent_draw": on_divergent_draw,
        "max_divergent_fraction": max_divergent_fraction,
        "n_draws_requested": divergence_summary["n_draws_requested"],
        "n_divergent_draws_excluded": divergence_summary["n_divergent_draws_excluded"],
        "divergent_draw_fraction": divergence_summary["divergent_draw_fraction"],
        "excluded_draws": divergence_summary["excluded_draws"],
        "chain_path": chain_filename,
        "chain_content_sha256": chain_hash,
        "chain_n_rows": chain_n_rows,
        "envelope_sample_seed": seed,
        "sample_indices": [int(x) for x in sample_indices],
        "valid_draw_indices": divergence_summary["valid_draw_indices"],
        **coverage,
    }

    with open(sidecar_filename, 'w') as f:
        json.dump(sidecar_data, f, indent=4, allow_nan=False)
    print(f"Saved MCMC metadata sidecar to {sidecar_filename}")

    # Restore best parameters for forward pass and fix finalfit mismatch
    data.par[:n_par] = best_params.copy()
    if getattr(data, 'par_best', None) is None:
        data.par_best = np.zeros_like(data.par)
    data.par_best[:n_par] = best_params.copy()

    call_model(data)
    data.finalfit = funcobj(data)


def DE_MCMC_mode(data: CommonData, seed: Optional[int] = None) -> None:
    """
    Differential Evolution + L-BFGS-B followed by MCMC for uncertainty quantification.
    """
    print("Starting DE-MCMC Calibration Mode")

    n_par = N_PAR
    nwalkers = data.mcmc_walkers

    active_params = _active_params(data, n_par)
    ndim = len(active_params)

    if ndim > 0 and nwalkers < 2 * ndim:
        raise ValueError(
            f"mcmc_walkers ({nwalkers}) must be at least 2x the number of "
            f"active parameters ({ndim} active -> need >= {2*ndim}). "
            "Increase mcmc_walkers in your config."
        )

    print("Phase 1 & 2: Finding best parameters using DE + L-BFGS-B")

    # Run the standard DE mode first to find best parameters
    # DE_mode sets data.par_best and data.finalfit
    DE_mode(data, seed)

    if ndim == 0:
        print("Warning: No active parameters for MCMC. Skipping MCMC phase.")
        return

    print("Phase 3: MCMC Uncertainty Analysis")
    best_params = data.par_best[:n_par].copy()

    # Walker ball scaled to each active parameter's bound width,
    # rather than the previous fixed 1e-4, which is negligible for a wide parameter
    # and can collapse the ensemble's effective spread relative to the posterior.
    _run_mcmc_uncertainty(data, seed, best_params, active_params, n_par=n_par)

