import os
import numpy as np
import pandas as pd
from typing import Optional
import concurrent.futures
from scipy.optimize import differential_evolution, minimize
import emcee

from .config import CommonData
from .model import call_model, funcobj, detect_segments

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
    """
    ei = sub_1(data)
    data.par_best = data.par.copy()
    data.finalfit = ei
    print(f'Efficiency Index in calibration {data.finalfit}')

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
                row = list(x[:, k]) + [eff_index]
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
                    row = list(x[:, k]) + [eff_index]
                    history.append(row)
                idx += 1

            for k in range(n_particles):
                # Fix: explicit check to ignore NaN efficiency values
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
                    print(f"Calcolo al {perc:.1f} %")

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
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index"])
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
            row = list(data.par[:n_par]) + [eff_index]
            history.append(row)

        if fit > foptim:
            foptim = fit
            gbest[:] = data.par[:n_par]

        if i >= 9:
            if (i + 1) % max(1, int(n_run / 10)) == 0:
                perc = float(i + 1) / float(n_run) * 100.0
                print(f"Calcolo al {perc:.1f} %")

    data.par_best = gbest.copy()
    data.finalfit = foptim
    print(f'Indice efficienza calibrazione {data.finalfit}')

    # Save to CSV
    # Fix: Pandas handles closing the file handle automatically via to_csv
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index"])
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
        # Update parameters (only the first n_par)
        data.par[:n_par] = p_vals

        # Evaluate
        eff_index = sub_1(data)

        # Record history if valid
        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(p_vals) + [eff_index]
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
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index"])
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

        # Using raw objective function as pseudo-likelihood
        # Higher is better for NSE/KGE/negated-RMS
        # We don't divide by temperature here, it's just the raw value.
        # Future enhancement: formal Gaussian log-likelihood
        return eff_index

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

    # Save raw MCMC chain
    chain = sampler.get_chain(flat=True)
    chain_df = pd.DataFrame(chain, columns=[f"par_{j+1}" for j in active_params])

    chain_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}.csv")
    chain_df.to_csv(chain_filename, index=False)
    print(f"Saved MCMC chain to {chain_filename}")

    # Step 5: Compute Predictive Uncertainty Envelopes
    print("Generating Predictive Uncertainty Envelopes...")
    # Take 100 random samples from the flattened chain
    n_samples = min(100, len(chain))
    sample_indices = np.random.choice(len(chain), size=n_samples, replace=False)
    samples = chain[sample_indices]

    # Store simulated time series
    ensemble_simulations = []

    for i, theta in enumerate(samples):
        p_vals = best_params.copy()
        for idx, j in enumerate(active_params):
            p_vals[j] = theta[idx]

        data.par[:n_par] = p_vals

        if data.gap_tolerant and data.segments is None:
            detect_segments(data)

        call_model(data)

        # We only care about valid days for plotting/analysis (skipping warmup/gaps)
        valid_mask = data.eval_mask
        ensemble_simulations.append(data.Twat_mod.copy())

    ensemble_simulations = np.array(ensemble_simulations)

    # Calculate percentiles at each time step
    perc_5 = np.percentile(ensemble_simulations, 5, axis=0)
    perc_50 = np.percentile(ensemble_simulations, 50, axis=0)
    perc_95 = np.percentile(ensemble_simulations, 95, axis=0)

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
