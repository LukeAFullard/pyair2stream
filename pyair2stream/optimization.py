"""
Optimization and calibration routines for pyair2stream.

This module implements the calibration algorithms (PSO, DE, LATHYP)
used to fit the air2stream model to observed data, as well as the
MCMC sampling routines for uncertainty quantification.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
import concurrent.futures
from scipy.optimize import differential_evolution, minimize
import emcee

import json
from .config import CommonData
from .model import call_model, funcobj, aggregation, statis, warn_on_stability, check_numerical_divergence
from .uncertainty import estimate_ar1_rho, generate_ar1_noise, build_ar1_runs, ar1_whitened_stats

# A near-perfect-fit MCMC log-likelihood is capped at this large but finite value rather
# than returned as a literal np.inf, which poisons emcee's acceptance-ratio arithmetic
# (inf - inf = nan). See docs/audit/03_objective_function_and_masks.md, 3.4.
MCMC_MAX_LOG_LIKELIHOOD = 1e10

# Number of physical model parameters (a1..a8 in the Fortran reference).
N_PAR = 8


def _active_params(data: CommonData, n_par: int = N_PAR) -> list:
    """Indices of parameters that are both flagged active and non-degenerate (parmin != parmax)."""
    return [j for j in range(n_par) if data.flag_par[j] and data.parmin[j] != data.parmax[j]]


def _segments_for(data: CommonData) -> list:
    """Segments to treat as independent adjacency/AR(1) runs: gap-tolerant segments, or the whole series."""
    return data.segments if data.gap_tolerant else [(0, data.n_tot - 1)]


def _iid_log_likelihood(mod_valid: np.ndarray, obs_valid: np.ndarray) -> float:
    """Concentrated Gaussian log-likelihood assuming iid residuals (docs/audit/04, pre-existing form)."""
    N = len(obs_valid)
    if N == 0:
        return -np.inf
    SSE = np.sum((mod_valid - obs_valid) ** 2)
    if SSE == 0:
        return MCMC_MAX_LOG_LIKELIHOOD
    return -0.5 * N * np.log(SSE / N)


def _ar1_log_likelihood(residuals: np.ndarray, rho: float, runs: list) -> float:
    """
    Concentrated Gaussian log-likelihood accounting for AR(1)-correlated residuals
    (docs/audit/04_uncertainty_and_mcmc.md, Defect A / 4.1). `rho` is treated as fixed
    (estimated once at the DE optimum, not sampled). Each independent run contributes
    its own `0.5*log(1-rho**2)` term, so the correction scales with the number of runs.
    """
    sse_u, N, n_runs = ar1_whitened_stats(residuals, rho, runs)
    if N == 0:
        return -np.inf
    if sse_u == 0:
        return MCMC_MAX_LOG_LIKELIHOOD
    return -0.5 * N * np.log(sse_u / N) + 0.5 * n_runs * np.log(1.0 - rho ** 2)


def _reflected_walker_init(initial: np.ndarray, scale: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                            nwalkers: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build the initial emcee walker ball around `initial`, reflecting draws back inside
    `[lo, hi]` instead of clipping them to the bound. Clipping collapses the ensemble's
    spread in any dimension where the DE optimum sits exactly on a bound -- emcee's
    stretch move cannot generate spread from a degenerate ensemble (docs/audit/04,
    Defect D / 4.4). Raises if any dimension still ends up with zero spread.
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
            "degenerate ensemble (docs/audit/04_uncertainty_and_mcmc.md, Defect D)."
        )
    return p0


def _split_rhat(chain: np.ndarray) -> np.ndarray:
    """
    Gelman-Rubin split-Rhat per parameter, from a raw (non-flattened) emcee chain of
    shape (n_iter, n_walkers, n_dim). Splitting each walker's chain in half along the
    iteration axis also flags within-walker non-stationarity, not just between-walker
    disagreement (docs/audit/04_uncertainty_and_mcmc.md, 4.6).
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
    Failures (chain too short, non-finite estimate) are reported as a warning rather
    than raised; convergence is only fatal via the explicit `strict_convergence` gate.
    """
    try:
        tau = sampler.get_autocorr_time(discard=discard, quiet=True)
        if np.any(np.isnan(tau)) or np.any(~np.isfinite(tau)):
            raise ValueError("autocorrelation time not reliably estimated")
        return tau, float(np.mean(tau))
    except (ValueError, emcee.autocorr.AutocorrError):
        print("Warning: autocorrelation time could not be reliably estimated; "
              "chain may be too short to assess convergence.")
        return None, None
    except Exception as e:
        print(f"Warning: failed to compute autocorrelation time ({e}).")
        return None, None


def _resolve_burnin(nsteps: int, tau_rough, uncertainty_options: dict) -> int:
    """
    Burn-in length in steps. An explicit `uncertainty_options.burnin_fraction` overrides
    everything; otherwise default to `max(0.3*nsteps, 5*max(tau))` when a rough
    autocorrelation estimate is available, else the historical flat 30%
    (docs/audit/04_uncertainty_and_mcmc.md, 4.6).
    """
    frac = uncertainty_options.get('burnin_fraction')
    if frac is not None:
        burnin = int(round(float(frac) * nsteps))
    elif tau_rough is not None:
        burnin = max(int(0.3 * nsteps), int(5 * np.max(tau_rough)))
    else:
        burnin = int(0.3 * nsteps)
    return int(np.clip(burnin, 0, max(nsteps - 1, 0)))


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
    npz (docs/audit/04_uncertainty_and_mcmc.md, 4.2 / Defect B). Percentile bands alone
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
                              save_ensemble: bool = False) -> None:
    """Write the percentile envelope CSV and, if requested, the raw ensemble npz (both post-warm-up)."""
    env_df = _percentile_envelope(data, ensemble_simulations, prediction_interval)
    env_df.iloc[365:].to_csv(env_filename, index=False)  # drop the warm-up block (report 05, Defect C)
    print(f"Saved predictive uncertainty envelopes to {env_filename}")

    if save_ensemble:
        if ensemble_filename is None:
            raise ValueError("save_ensemble is True but no ensemble_filename was provided.")
        _save_ensemble_npz(data, ensemble_simulations, ensemble_filename)

def sub_1(data: CommonData) -> np.float64:
    """
    Helper function to call model and evaluate the objective function.
    Replicates SUBROUTINE sub_1
    """
    call_model(data)
    return np.float64(funcobj(data))

def eval_particle_worker(args):
    """
    Top-level helper for multiprocessing.
    Args should be a tuple of (CommonData, parameter_array, n_par).

    Returns `(eff_index, nse, r2, mae)`. `sub_1` sets `data.current_nse`/
    `current_r2`/`current_mae` as a side effect inside this (child) process; only
    the explicit return value crosses the process boundary back to the parent, so
    those metrics must be returned here rather than read from `data` afterward --
    the parent's own `data.current_*` are untouched defaults otherwise (audit
    report 06, Defect C).
    """
    data, p_vals, n_par = args
    # When passed to a new process via executor.map, 'data' is already a local deserialized copy.
    data.par[:n_par] = p_vals
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
    # here (report 05, Defect A) -- statis() raises when there are no
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
    print(f'Efficiency Index in calibration {data.finalfit}')

    # Optional Probabilistic Forward Envelope
    if data.forward_options and data.forward_options.get('enable_prediction_intervals', False):
        chain_path = data.forward_options.get('mcmc_chain_path')
        if not chain_path or not os.path.exists(chain_path):
            print(f"Warning: Cannot generate prediction intervals. MCMC chain not found at {chain_path}")
            return

        print(f"Generating Forward Prediction Intervals from {chain_path}...")

        seed = data.forward_options.get('random_seed', None)
        if seed is not None:
            np.random.seed(seed)

        chain_df = pd.read_csv(chain_path)
        chain = chain_df.values

        n_samples = data.forward_options.get('n_samples', 1000)
        n_samples = min(n_samples, len(chain))

        sample_indices = np.random.choice(len(chain), size=n_samples, replace=False)
        samples = chain[sample_indices]

        uncertainty_options = getattr(data, 'uncertainty_options', None) or {}
        sidecar_path = chain_path.replace('.csv', '_meta.json')

        # Resolve sigma: explicit config override first, then the sidecar written by
        # DE-MCMC/DE-CV-MCMC (mirroring the `rho` resolution below), matching `rho`'s
        # existing carry-forward instead of silently defaulting to 0.0 behind a print
        # (docs/audit/04_uncertainty_and_mcmc.md, Defect C / 4.3).
        sigma_override = data.forward_options.get('residual_sigma')
        if sigma_override is not None and float(sigma_override) > 0.0:
            sigma = float(sigma_override)
            print(f"Using explicit residual_sigma override: {sigma}")
        elif os.path.exists(sidecar_path):
            import json
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
                "not a prediction interval -- docs/audit/04_uncertainty_and_mcmc.md, Defect C."
            )

        noise_model = uncertainty_options.get('noise_model', 'iid')
        rho_used = 0.0

        if noise_model == 'ar1':
            ar1_rho_override = uncertainty_options.get('ar1_rho')

            if ar1_rho_override is not None:
                rho_used = ar1_rho_override
                print(f"Using explicit ar1_rho override: {rho_used}")
            elif has_obs:
                eval_mask_for_rho = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=bool)
                segments_for_rho = _segments_for(data)
                rho_used = estimate_ar1_rho(data.Twat_mod, data.Twat_obs, eval_mask_for_rho, segments_for_rho)
                print(f"Using rho={rho_used:.4f} estimated directly from this run's own residuals.")
            elif os.path.exists(sidecar_path):
                import json
                try:
                    with open(sidecar_path, 'r') as f:
                        sidecar_data = json.load(f)
                    rho_used = sidecar_data.get('rho', 0.0)
                    print(f"Using rho={rho_used:.4f} carried from calibration run {sidecar_path}")
                except Exception as e:
                    print(f"Warning: Failed to read sidecar {sidecar_path} ({e}). Falling back to rho=0.0.")
                    rho_used = 0.0
            else:
                print("Warning: No residuals available to estimate rho; falling back to rho=0.0 (equivalent to iid)")
                rho_used = 0.0

            rng = np.random.default_rng(seed)

        ensemble_simulations = []
        n_par = N_PAR

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

            if noise_model == 'ar1':
                noise = generate_ar1_noise(data.n_tot, sigma, rho_used, segments_for_noise, rng)
            else:
                noise = np.random.normal(0, sigma, data.n_tot)

            noisy_simulation = data.Twat_mod + noise

            ensemble_simulations.append(noisy_simulation)

        ensemble_simulations = np.array(ensemble_simulations)

        prediction_interval = uncertainty_options.get('prediction_interval', 90.0)
        env_filename = os.path.join(data.folder, f"Forward_Prediction_Envelopes_{data.station}_{data.series}_{data.time_res}.csv")
        ensemble_filename = os.path.join(data.folder, f"Forward_Prediction_Ensemble_{data.station}_{data.series}_{data.time_res}.npz")
        save_ensemble = bool(uncertainty_options.get('save_ensemble', False))
        _export_ensemble_outputs(data, ensemble_simulations, prediction_interval, env_filename, ensemble_filename, save_ensemble)

        # Restore deterministic parameters
        data.par[:n_par] = best_params_deterministic
        call_model(data)

def PSO_mode(data: CommonData, seed: Optional[int] = None) -> None:
    """
    Replicates SUBROUTINE PSO_mode
    """
    print(f'N. particles = {data.n_particles}, N. run = {data.n_run}')

    if seed is not None:
        np.random.seed(seed)

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
    # all-zero initial parameters (see examples/validation/Switzerland/README.md).
    fitbest = np.full(n_particles, -1e30, dtype=np.float64)

    # We output history to CSV instead of binary
    output_filename = os.path.join(data.folder, f"0_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.csv")
    history = []

    dw = (data.wmax - data.wmin) / n_run
    w = data.wmax

    x_rand = np.random.rand(n_par, n_particles)
    v_rand = np.random.rand(n_par, n_particles)

    for j in range(n_par):
        dxmax = data.parmax[j] - data.parmin[j]
        dvmax = 1.0 * dxmax
        x[j, :] = x_rand[j, :] * dxmax + data.parmin[j]
        v[j, :] = v_rand[j, :] * dvmax
        pbest[j, :] = x[j, :]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(eval_particle_worker, [(data, x[:, k], n_par) for k in range(n_particles)]))

        for k in range(n_particles):
            eff_index, nse_k, r2_k, mae_k = results[k]
            if not np.isnan(eff_index):
                fitbest[k] = eff_index
            if not np.isnan(eff_index) and eff_index >= data.mineff_index:
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
                r = np.random.rand(2 * n_par)
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
                    particles_to_eval.append((data, x[:, k], n_par))
                    eval_indices.append(k)
                else:
                    fit[k] = -1e30

            eval_results = list(executor.map(eval_particle_worker, particles_to_eval))

            idx = 0
            for k in eval_indices:
                eff_index, nse_k, r2_k, mae_k = eval_results[idx]
                fit[k] = eff_index
                if not np.isnan(eff_index) and eff_index >= data.mineff_index:
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
                        diff = (pbest[j, k] - gbest[j]) / (data.parmax[j] - data.parmin[j])
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
    print(f'N. run = {data.n_run}')

    if seed is not None:
        np.random.seed(seed)

    n_par = 8
    n_run = data.n_run

    gbest = np.zeros(n_par, dtype=np.float64)
    foptim = -999.0

    output_filename = os.path.join(data.folder, f"0_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.csv")
    history = []

    permut = np.zeros((n_run, n_par), dtype=np.int32)
    for j in range(n_par):
        # Fix: Using numpy.random.permutation to avoid custom Shuffle
        permut[:, j] = np.random.permutation(n_run) + 1

    for i in range(n_run):
        for j in range(n_par):
            r = np.random.rand()
            r = r + (float(permut[i, j]) - 1.0)
            r = r / float(n_run)

            data.par[j] = data.parmin[j] + (data.parmax[j] - data.parmin[j]) * r

        eff_index = sub_1(data)
        fit = eff_index

        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(data.par[:n_par]) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
            history.append(row)

        if fit > foptim:
            foptim = fit
            gbest[:] = data.par[:n_par]

        if i >= 9:
            if (i + 1) % max(1, int(n_run / 10)) == 0:
                perc = float(i + 1) / float(n_run) * 100.0
                print(f"Progress: {perc:.1f} %")

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
    print(f'Pop. Size (particles) = {data.n_particles}, Max Generations (runs) = {data.n_run}')

    if seed is not None:
        np.random.seed(seed)

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
        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(p_vals) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
            history.append(row)

        # Return negated efficiency so scipy minimizes
        # Handle NaN by returning a large positive number
        if np.isnan(eff_index):
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
    result_de = differential_evolution(
        objective_wrapper,
        bounds,
        maxiter=data.n_run,
        popsize=data.n_particles,
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

    # Finalize
    best_params = result_bfgs.x

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
                           active_params: list, init_scale, n_par: int = N_PAR) -> None:
    """
    Shared Phase 3+ implementation for `DE_MCMC_mode` and `DE_CV_MCMC_mode`: builds the
    walker ensemble, runs `emcee`, computes convergence diagnostics, and writes the
    chain/sidecar/envelope (and optional raw ensemble) outputs.

    `init_scale` is either `None` (the two-mode-agnostic default: a ball scaled to each
    active parameter's bound width) or an explicit per-dimension array of standard
    deviations (the `DE-CV-MCMC` cross-validation-informed spread) -- see
    docs/audit/04_uncertainty_and_mcmc.md, 3.3/4.4.
    """
    nwalkers = data.mcmc_walkers
    nsteps = data.mcmc_steps
    ndim = len(active_params)

    uncertainty_options = getattr(data, 'uncertainty_options', None) or {}
    noise_model = uncertainty_options.get('noise_model', 'iid')

    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=np.bool_)
    segments = _segments_for(data)

    # Re-evaluate at the DE optimum so rho/sigma are estimated at the point the MCMC
    # ensemble is actually centred on.
    data.par[:n_par] = best_params.copy()
    call_model(data)
    funcobj(data)

    best_rho = estimate_ar1_rho(data.Twat_mod, data.Twat_obs, eval_mask, segments)

    valid_mask_agg = (data.Twat_obs_agg != -999.0) & eval_mask
    mod_valid = data.Twat_mod_agg[valid_mask_agg]
    obs_valid = data.Twat_obs_agg[valid_mask_agg]
    N = len(obs_valid)
    best_sigma = float(np.sqrt(np.sum((mod_valid - obs_valid) ** 2) / N)) if N > 0 else 0.0

    # Reused across every likelihood evaluation below: observations (and therefore the
    # valid/AR(1)-run structure) do not change while theta is being explored, only the
    # simulated series does (docs/audit/04_uncertainty_and_mcmc.md, 4.1).
    ar1_runs = build_ar1_runs(valid_mask_agg, segments) if noise_model == 'ar1' else None

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
        # (report 03, 3.4) -- daily and aggregated coincide at 1d resolution.
        if noise_model == 'ar1':
            residuals = data.Twat_mod_agg - data.Twat_obs_agg
            return _ar1_log_likelihood(residuals, best_rho, ar1_runs)
        else:
            mod = data.Twat_mod_agg[valid_mask_agg]
            obs = data.Twat_obs_agg[valid_mask_agg]
            return _iid_log_likelihood(mod, obs)

    initial = np.array([best_params[j] for j in active_params])
    lo = np.array([data.parmin[j] for j in active_params])
    hi = np.array([data.parmax[j] for j in active_params])
    scale = (1e-3 * (hi - lo)) if init_scale is None else np.asarray(init_scale, dtype=np.float64)

    rng = np.random.default_rng(seed)
    p0 = _reflected_walker_init(initial, scale, lo, hi, nwalkers, rng)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

    print(f"Running MCMC for {nsteps} steps with {nwalkers} walkers...")
    sampler.run_mcmc(p0, nsteps, progress=True)

    # A rough, full-chain autocorrelation estimate sizes the burn-in; the diagnostic
    # actually reported is then recomputed on the post-burn-in chain below
    # (docs/audit/04_uncertainty_and_mcmc.md, 4.6).
    tau_rough, _ = _estimate_autocorr(sampler, discard=0)
    burnin = _resolve_burnin(nsteps, tau_rough, uncertainty_options)

    tau_final, mean_tau = _estimate_autocorr(sampler, discard=burnin)
    if tau_final is not None:
        print(f"Estimated autocorrelation time per parameter (post burn-in): {tau_final}")

    strict_convergence = bool(uncertainty_options.get('strict_convergence', False))
    if tau_final is not None and nsteps < 50 * np.max(tau_final):
        msg = (f"chain length ({nsteps}) is less than 50x the estimated post-burn-in "
               f"autocorrelation time ({np.max(tau_final):.1f}).")
        if strict_convergence:
            raise RuntimeError(
                f"MCMC did not converge: {msg} Increase mcmc_steps, or unset "
                "uncertainty_options.strict_convergence to downgrade this to a warning."
            )
        print(f"Warning: {msg} Consider increasing mcmc_steps.")

    mean_acc = float(np.mean(sampler.acceptance_fraction))
    print(f"Mean acceptance fraction: {mean_acc:.3f}")

    max_rhat = None
    try:
        raw_chain = sampler.get_chain(discard=burnin, flat=False)
        rhat = _split_rhat(raw_chain)
        finite_rhat = rhat[np.isfinite(rhat)]
        if len(finite_rhat) > 0:
            max_rhat = float(np.max(finite_rhat))
            print(f"Split-Rhat per parameter: {rhat}")
            if max_rhat >= 1.01:
                print(f"Warning: split-Rhat ({max_rhat:.4f}) exceeds 1.01; chain may not have converged.")
    except Exception as e:
        print(f"Warning: failed to compute split-Rhat ({e}).")

    # Save raw MCMC chain (flattened, removing burnin)
    chain = sampler.get_chain(discard=burnin, flat=True)
    chain_df = pd.DataFrame(chain, columns=[f"par_{j+1}" for j in active_params])

    chain_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}.csv")
    chain_df.to_csv(chain_filename, index=False)
    print(f"Saved MCMC chain (discarded {burnin} burn-in steps) to {chain_filename}")

    print("Writing metadata sidecar...")
    sidecar_data = {
        "rho": best_rho,
        "sigma": best_sigma,
        "n_valid_pairs": N,  # N valid points used for variance, proxy for pairs
        "noise_model_used_for_this_run": noise_model,
        "mcmc_walkers": nwalkers,
        "mcmc_steps": nsteps,
        "mcmc_seed": seed,
        "burnin": burnin,
        "mean_acceptance_fraction": mean_acc,
        "mean_autocorr_time": mean_tau,
        "max_split_rhat": max_rhat if (max_rhat is not None and np.isfinite(max_rhat)) else None,
        "strict_convergence": strict_convergence,
    }

    sidecar_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}_meta.json")
    with open(sidecar_filename, 'w') as f:
        json.dump(sidecar_data, f, indent=4, allow_nan=False)
    print(f"Saved MCMC metadata sidecar to {sidecar_filename}")

    # Compute Predictive Uncertainty Envelopes
    print("Generating Predictive Uncertainty Envelopes...")
    n_samples = min(1000, len(chain))
    sample_indices = rng.choice(len(chain), size=n_samples, replace=False)
    samples = chain[sample_indices]

    ensemble_simulations = []

    for theta in samples:
        p_vals = best_params.copy()
        for idx, j in enumerate(active_params):
            p_vals[j] = theta[idx]

        data.par[:n_par] = p_vals
        call_model(data)
        funcobj(data)  # populate Twat_mod_agg for the aggregated-residual sigma below

        # Estimate sigma from this sample's own residuals, on the same (aggregated)
        # series the objective scores (report 03, 3.4). The noise itself is still
        # injected at daily resolution, matching the exported daily envelope.
        mod_iter = data.Twat_mod_agg[valid_mask_agg]
        obs_iter = data.Twat_obs_agg[valid_mask_agg]
        N_iter = len(obs_iter)
        sigma_iter = float(np.sqrt(np.sum((mod_iter - obs_iter) ** 2) / N_iter)) if N_iter > 0 else 0.0

        if noise_model == 'ar1':
            noise = generate_ar1_noise(data.n_tot, sigma_iter, best_rho, segments, rng)
        else:
            noise = rng.normal(0, sigma_iter, data.n_tot)

        ensemble_simulations.append(data.Twat_mod + noise)

    ensemble_simulations = np.array(ensemble_simulations)

    prediction_interval = uncertainty_options.get('prediction_interval', 90.0)
    env_filename = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")
    ensemble_filename = os.path.join(data.folder, f"MCMC_ensemble_{data.station}_{data.series}_{data.time_res}.npz")
    save_ensemble = bool(uncertainty_options.get('save_ensemble', False))
    _export_ensemble_outputs(data, ensemble_simulations, prediction_interval, env_filename, ensemble_filename, save_ensemble)

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

    # Walker ball scaled to each active parameter's bound width (docs/audit/04, 4.4),
    # rather than the previous fixed 1e-4, which is negligible for a wide parameter
    # and can collapse the ensemble's effective spread relative to the posterior.
    _run_mcmc_uncertainty(data, seed, best_params, active_params, init_scale=None, n_par=n_par)


def DE_CV_MCMC_mode(data: CommonData, seed: Optional[int] = None) -> None:
    """
    Differential Evolution + L-BFGS-B followed by Cross-Validation to inform MCMC initialization.
    """
    print("Starting DE-CV-MCMC Calibration Mode")

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

    print("Phase 3: Cross-Validation to estimate parameter standard deviations")
    from .cross_validation import CVConfig, run_leave_one_year_out_cv

    cv_config = data.cross_validation
    if cv_config is None:
        cv_config = CVConfig()

    # Run CV using DE. We override n_run and n_particles to keep it fast
    # or just use whatever is in cv_config.optimizer_overrides
    results = run_leave_one_year_out_cv(data, cv_config, 'DE')

    # Extract standard deviations from CV folds
    if len(results) > 1:
        cv_params = np.array([r.par_best for r in results])
        stds = np.std(cv_params, axis=0, ddof=1)
    else:
        stds = np.full(n_par, np.nan)

    # Calculate active standard deviations for walkers
    std_active = np.zeros(ndim)
    for idx, j in enumerate(active_params):
        val = stds[j]
        if np.isnan(val) or val <= 0.0:
            std_active[idx] = 1e-4
        else:
            std_active[idx] = val

    print("Phase 4: MCMC Uncertainty Analysis (with CV-informed spread)")
    best_params = data.par_best[:n_par].copy()
    _run_mcmc_uncertainty(data, seed, best_params, active_params, init_scale=std_active, n_par=n_par)
