import sys
with open('pyair2stream/optimization.py', 'r') as f:
    code = f.read()

# We want to add DE_CV_MCMC_mode at the end of the file.
new_code = code + """

def DE_CV_MCMC_mode(data: CommonData, seed: Optional[int] = None) -> None:
    \"\"\"
    Differential Evolution + CV + MCMC for uncertainty quantification.
    Uses CV standard deviations to initialize MCMC walkers.
    \"\"\"
    from .cross_validation import run_leave_one_year_out_cv, CVConfig

    print("Starting DE-CV-MCMC Calibration Mode")
    print("Phase 1 & 2: Finding global best parameters using DE + L-BFGS-B")

    DE_mode(data, seed)

    print("Phase 3: Cross-Validation for parameter stability estimates")
    cv_config = data.cross_validation if data.cross_validation else CVConfig(unit="year", skip_first_year=True, min_valid_obs=10)

    # run CV. This will temporaily run DE for each fold and restore things.
    results = run_leave_one_year_out_cv(data, cv_config, "DE")

    # Calculate CV stds for each parameter
    # FoldResult has par_best
    fold_pars = np.array([r.par_best for r in results])
    if len(fold_pars) > 1:
        cv_stds = np.std(fold_pars, axis=0, ddof=1)
    else:
        # Fallback to tiny std if only 1 fold for some reason
        cv_stds = np.full(8, 1e-4)

    print("Phase 4: MCMC Uncertainty Analysis (Initialized from CV)")
    n_par = 8
    nwalkers = data.mcmc_walkers
    nsteps = data.mcmc_steps

    if seed is not None:
        np.random.seed(seed)

    best_params = data.par_best[:n_par].copy()

    active_params = []
    for j in range(n_par):
        if data.flag_par[j] and data.parmin[j] != data.parmax[j]:
            active_params.append(j)

    ndim = len(active_params)
    if ndim == 0:
        print("Warning: No active parameters for MCMC. Skipping MCMC phase.")
        return

    def log_probability(theta):
        p_vals = best_params.copy()
        for idx, j in enumerate(active_params):
            p_vals[j] = theta[idx]
            if p_vals[j] < data.parmin[j] or p_vals[j] > data.parmax[j]:
                return -np.inf

        data.par[:n_par] = p_vals
        if data.gap_tolerant and data.segments is None:
            detect_segments(data)
        call_model(data)
        eff_index = funcobj(data)

        if np.isnan(eff_index):
            return -np.inf

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

    initial = np.array([best_params[j] for j in active_params])
    cv_std_active = np.array([cv_stds[j] for j in active_params])
    # Protect against perfectly 0 standard deviation in CV
    cv_std_active = np.maximum(cv_std_active, 1e-6)

    p0 = initial + cv_std_active * np.random.randn(nwalkers, ndim)

    for i in range(nwalkers):
        for idx, j in enumerate(active_params):
            p0[i, idx] = np.clip(p0[i, idx], data.parmin[j], data.parmax[j])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

    print(f"Running MCMC for {nsteps} steps with {nwalkers} walkers...")
    sampler.run_mcmc(p0, nsteps, progress=True)

    burnin = int(nsteps * 0.3)
    chain = sampler.get_chain(discard=burnin, flat=True)
    chain_df = pd.DataFrame(chain, columns=[f"par_{j+1}" for j in active_params])

    chain_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}.csv")
    chain_df.to_csv(chain_filename, index=False)
    print(f"Saved MCMC chain (discarded {burnin} burn-in steps) to {chain_filename}")

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
        "n_valid_pairs": N,
        "noise_model_used_for_this_run": noise_model,
        "mcmc_walkers": nwalkers,
        "mcmc_steps": nsteps,
        "mcmc_seed": seed
    }

    sidecar_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}_meta.json")
    with open(sidecar_filename, 'w') as f:
        import json
        json.dump(sidecar_data, f, indent=4)
    print(f"Saved MCMC metadata sidecar to {sidecar_filename}")

    print("Generating Predictive Uncertainty Envelopes...")
    n_samples = min(1000, len(chain))
    sample_indices = np.random.choice(len(chain), size=n_samples, replace=False)
    samples = chain[sample_indices]

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

    perc_5 = np.percentile(ensemble_simulations, 5, axis=0)
    perc_50 = np.percentile(ensemble_simulations, 50, axis=0)
    perc_95 = np.percentile(ensemble_simulations, 95, axis=0)

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

    env_filename = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")
    env_df.to_csv(env_filename, index=False)
    print(f"Saved predictive uncertainty envelopes to {env_filename}")

    data.par[:n_par] = best_params.copy()
    call_model(data)

"""

with open('pyair2stream/optimization.py', 'w') as f:
    f.write(new_code)
