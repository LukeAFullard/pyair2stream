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
from .model import call_model, funcobj, detect_segments
from .uncertainty import estimate_ar1_rho, generate_ar1_noise

def sub_1(data: CommonData) -> np.float64:
    """
    Helper function to call model and evaluate the objective function.
    Replicates SUBROUTINE sub_1
    """
    if data.gap_tolerant and data.segments is None:
        detect_segments(data)
    call_model(data)
    return np.float64(funcobj(data))

def eval_particle_worker(args):
    """
    Top-level helper for multiprocessing.
    Args should be a tuple of (CommonData, parameter_array, n_par).
    """
    data, p_vals, n_par = args
    # When passed to a new process via executor.map, 'data' is already a local deserialized copy.
    data.par[:n_par] = p_vals
    return sub_1(data)

def forward_mode(data: CommonData) -> None:
    """
    Replicates SUBROUTINE forward_mode
    Adds optional probabilistic Prediction Intervals based on MCMC chains.
    """
    # We must aggregate the data so I_inf / I_pos are created for funcobj to work without throwing NoneType exceptions
    # But ONLY if Twat_obs is not entirely -999.0, else n_dat becomes 0 which we should skip or mock.
    has_obs = False
    for val in data.Twat_obs:
        if val != -999.0:
            has_obs = True
            break

    if has_obs:
        ei = sub_1(data)
    else:
        # It's a pure projection, we skip the objective evaluation.
        if data.gap_tolerant and data.segments is None:
            detect_segments(data)
        call_model(data)
        ei = -999.0

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

        import pandas as pd
        chain_df = pd.read_csv(chain_path)
        chain = chain_df.values

        n_samples = data.forward_options.get('n_samples', 1000)
        n_samples = min(n_samples, len(chain))

        sample_indices = np.random.choice(len(chain), size=n_samples, replace=False)
        samples = chain[sample_indices]

        sigma = float(data.forward_options.get('residual_sigma', 0.0))
        if sigma <= 0.0:
            print("Warning: residual_sigma is 0.0. Prediction interval will only reflect parameter uncertainty, not observation error.")

        noise_model = getattr(data, 'uncertainty_options', {}).get('noise_model', 'iid')
        rho_used = 0.0

        if noise_model == 'ar1':
            ar1_rho_override = data.uncertainty_options.get('ar1_rho')
            sidecar_path = chain_path.replace('.csv', '_meta.json')

            if ar1_rho_override is not None:
                rho_used = ar1_rho_override
                print(f"Using explicit ar1_rho override: {rho_used}")
            elif has_obs:
                eval_mask_for_rho = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=bool)
                segments_for_rho = data.segments if data.gap_tolerant else [(0, data.n_tot - 1)]
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
        n_par = 8

        # Determine active params from dataframe columns
        active_cols = chain_df.columns
        active_params = [int(c.split('_')[1])-1 for c in active_cols]

        best_params_deterministic = data.par_best.copy()
        segments_for_noise = data.segments if data.gap_tolerant else [(0, data.n_tot - 1)]

        for i, theta in enumerate(samples):
            p_vals = best_params_deterministic.copy()
            for idx, j in enumerate(active_params):
                p_vals[j] = theta[idx]

            data.par[:n_par] = p_vals

            if data.gap_tolerant and data.segments is None:
                detect_segments(data)

            call_model(data)

            if noise_model == 'ar1':
                noise = generate_ar1_noise(data.n_tot, sigma, rho_used, segments_for_noise, rng)
            else:
                noise = np.random.normal(0, sigma, data.n_tot)

            noisy_simulation = data.Twat_mod + noise

            ensemble_simulations.append(noisy_simulation)

        ensemble_simulations = np.array(ensemble_simulations)

        perc_5 = np.percentile(ensemble_simulations, 5, axis=0)
        perc_50 = np.percentile(ensemble_simulations, 50, axis=0)
        perc_95 = np.percentile(ensemble_simulations, 95, axis=0)

        # Replace calculated percentiles with NaN where the base model has missing data gaps
        perc_5 = np.where(data.Twat_mod == -999.0, np.nan, perc_5)
        perc_50 = np.where(data.Twat_mod == -999.0, np.nan, perc_50)
        perc_95 = np.where(data.Twat_mod == -999.0, np.nan, perc_95)

        env_df = pd.DataFrame({
            'Year': data.date[:, 0],
            'Month': data.date[:, 1],
            'Day': data.date[:, 2],
            'Twat_mod_p5': perc_5,
            'Twat_mod_p50': perc_50,
            'Twat_mod_p95': perc_95
        })

        env_filename = os.path.join(data.folder, f"Forward_Prediction_Envelopes_{data.station}_{data.series}_{data.time_res}.csv")
        env_df.to_csv(env_filename, index=False)
        print(f"Saved forward prediction uncertainty envelopes to {env_filename}")

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
    # all-zero initial parameters (see examples/validation/Switzerland/report.md).
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
            eff_index = results[k]
            if not np.isnan(eff_index):
                fitbest[k] = eff_index
            if not np.isnan(eff_index) and eff_index >= data.mineff_index:
                row = list(x[:, k]) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
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
                eff_index = eval_results[idx]
                fit[k] = eff_index
                if not np.isnan(eff_index) and eff_index >= data.mineff_index:
                    row = list(x[:, k]) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
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

def DE_MCMC_mode(data: CommonData, seed: Optional[int] = None) -> None:
    """
    Differential Evolution + L-BFGS-B followed by MCMC for uncertainty quantification.
    """
    print("Starting DE-MCMC Calibration Mode")
    print("Phase 1 & 2: Finding best parameters using DE + L-BFGS-B")

    # Run the standard DE mode first to find best parameters
    # DE_mode sets data.par_best and data.finalfit
    DE_mode(data, seed)

    print("Phase 3: MCMC Uncertainty Analysis")
    n_par = 8
    nwalkers = data.mcmc_walkers
    nsteps = data.mcmc_steps

    if seed is not None:
        np.random.seed(seed)

    best_params = data.par_best[:n_par].copy()

    # Determine which parameters are actively varying
    active_params = []
    for j in range(n_par):
        if data.flag_par[j] and data.parmin[j] != data.parmax[j]:
            active_params.append(j)

    ndim = len(active_params)
    if ndim == 0:
        print("Warning: No active parameters for MCMC. Skipping MCMC phase.")
        return

    # Define log_probability function for emcee
    def log_probability(theta):
        """
        Compute the log-probability of a parameter set for MCMC sampling.

        Calculates a formal concentrated Gaussian log-likelihood, assuming normally
        distributed observation errors.

        Parameters
        ----------
        theta : ndarray
            Array containing only the actively varying parameters (those not
            fixed by bounds).

        Returns
        -------
        float
            The log-likelihood of the parameter set, or -np.inf if the parameters
            fall outside the defined prior boundaries or cause simulation failure.
        """
        # theta contains only the active parameters
        p_vals = best_params.copy()
        for idx, j in enumerate(active_params):
            p_vals[j] = theta[idx]

            # Check bounds
            if p_vals[j] < data.parmin[j] or p_vals[j] > data.parmax[j]:
                return -np.inf

        data.par[:n_par] = p_vals
        if data.gap_tolerant and data.segments is None:
            detect_segments(data)
        call_model(data)
        eff_index = funcobj(data)

        if np.isnan(eff_index):
            return -np.inf
        # Formal Concentrated Gaussian Log-Likelihood
        valid_mask = (data.Twat_obs != -999.0)
        if data.eval_mask is not None:
            valid_mask &= data.eval_mask
        mod_valid = data.Twat_mod[valid_mask]
        obs_valid = data.Twat_obs[valid_mask]
        N = len(obs_valid)
        if N == 0:
            return -np.inf
        SSE = np.sum((mod_valid - obs_valid)**2)
        if SSE == 0:
            return np.inf
        log_L = -0.5 * N * np.log(SSE / N)
        return log_L

    # Initialize walkers around best parameters
    initial = np.array([best_params[j] for j in active_params])
    p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)

    # Ensure initial positions are within bounds
    for i in range(nwalkers):
        for idx, j in enumerate(active_params):
            p0[i, idx] = np.clip(p0[i, idx], data.parmin[j], data.parmax[j])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

    print(f"Running MCMC for {nsteps} steps with {nwalkers} walkers...")
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Discard the first 30% of steps as burn-in to ensure envelopes are drawn from fully converged distributions
    burnin = int(nsteps * 0.3)

    # Save raw MCMC chain (flattened, removing burnin)
    chain = sampler.get_chain(discard=burnin, flat=True)
    chain_df = pd.DataFrame(chain, columns=[f"par_{j+1}" for j in active_params])

    chain_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}.csv")
    chain_df.to_csv(chain_filename, index=False)
    print(f"Saved MCMC chain (discarded {burnin} burn-in steps) to {chain_filename}")

    # Step 5: Calculate properties for sidecar file from best_params
    print("Writing metadata sidecar...")
    data.par[:n_par] = best_params.copy()
    if data.gap_tolerant and data.segments is None:
        detect_segments(data)
    call_model(data)

    valid_mask = (data.Twat_obs != -999.0)
    if data.eval_mask is not None:
        valid_mask &= data.eval_mask

    mod_valid = data.Twat_mod[valid_mask]
    obs_valid = data.Twat_obs[valid_mask]

    N = len(obs_valid)
    if N > 0:
        SSE = np.sum((mod_valid - obs_valid)**2)
        best_sigma = float(np.sqrt(SSE / N))
    else:
        best_sigma = 0.0

    eval_mask_for_rho = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=bool)
    segments_for_rho = data.segments if data.gap_tolerant else [(0, data.n_tot - 1)]
    best_rho = estimate_ar1_rho(data.Twat_mod, data.Twat_obs, eval_mask_for_rho, segments_for_rho)

    noise_model = getattr(data, 'uncertainty_options', {}).get('noise_model', 'iid')

    sidecar_data = {
        "rho": best_rho,
        "sigma": best_sigma,
        "n_valid_pairs": N,  # N valid points used for variance, proxy for pairs
        "noise_model_used_for_this_run": noise_model,
        "mcmc_walkers": nwalkers,
        "mcmc_steps": nsteps,
        "mcmc_seed": seed
    }

    sidecar_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}_meta.json")
    with open(sidecar_filename, 'w') as f:
        json.dump(sidecar_data, f, indent=4)
    print(f"Saved MCMC metadata sidecar to {sidecar_filename}")

    # Step 6: Compute Predictive Uncertainty Envelopes
    print("Generating Predictive Uncertainty Envelopes...")
    # Take 1000 random samples from the flattened converged chain to make a robust envelope
    n_samples = min(1000, len(chain))
    sample_indices = np.random.choice(len(chain), size=n_samples, replace=False)
    samples = chain[sample_indices]

    # Store simulated time series
    ensemble_simulations = []

    if noise_model == 'ar1':
        rng = np.random.default_rng(seed)

    for i, theta in enumerate(samples):
        p_vals = best_params.copy()
        for idx, j in enumerate(active_params):
            p_vals[j] = theta[idx]

        data.par[:n_par] = p_vals

        if data.gap_tolerant and data.segments is None:
            detect_segments(data)

        call_model(data)

        # To build a true Prediction Interval (as opposed to just parameter confidence),
        # we must add the observation error variance back into the simulations.
        # We estimate sigma from the residuals of this specific parameter set.

        valid_mask_iter = (data.Twat_obs != -999.0)
        if data.eval_mask is not None:
            valid_mask_iter &= data.eval_mask

        mod_valid_iter = data.Twat_mod[valid_mask_iter]
        obs_valid_iter = data.Twat_obs[valid_mask_iter]

        N_iter = len(obs_valid_iter)
        if N_iter > 0:
            SSE_iter = np.sum((mod_valid_iter - obs_valid_iter)**2)
            sigma = np.sqrt(SSE_iter / N_iter)
        else:
            sigma = 0.0

        if noise_model == 'ar1':
            noise = generate_ar1_noise(data.n_tot, sigma, best_rho, segments_for_rho, rng)
        else:
            noise = np.random.normal(0, sigma, data.n_tot)

        noisy_simulation = data.Twat_mod + noise

        ensemble_simulations.append(noisy_simulation)

    ensemble_simulations = np.array(ensemble_simulations)

    # Calculate percentiles at each time step
    perc_5 = np.percentile(ensemble_simulations, 5, axis=0)
    perc_50 = np.percentile(ensemble_simulations, 50, axis=0)
    perc_95 = np.percentile(ensemble_simulations, 95, axis=0)

    # Replace calculated percentiles with NaN where the base model has missing data gaps
    perc_5 = np.where(data.Twat_mod == -999.0, np.nan, perc_5)
    perc_50 = np.where(data.Twat_mod == -999.0, np.nan, perc_50)
    perc_95 = np.where(data.Twat_mod == -999.0, np.nan, perc_95)

    # Export envelopes
    env_df = pd.DataFrame({
        'Year': data.date[:, 0],
        'Month': data.date[:, 1],
        'Day': data.date[:, 2],
        'Twat_mod_p5': perc_5,
        'Twat_mod_p50': perc_50,
        'Twat_mod_p95': perc_95
    })

    env_filename = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")
    env_df.to_csv(env_filename, index=False)
    print(f"Saved predictive uncertainty envelopes to {env_filename}")

    # Restore best parameters for forward pass
    data.par[:n_par] = best_params.copy()
    call_model(data)
