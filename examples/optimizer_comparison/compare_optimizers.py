import os
import time
import matplotlib.pyplot as plt
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis, detect_segments, call_model, funcobj
from pyair2stream.optimization import PSO_mode, DE_mode, DE_MCMC_mode
from pyair2stream.sensitivity import sensitivity_analysis

def run_calibration(config_path, mode):
    data = read_calibration(config_file=config_path)
    data.runmode = mode
    data.sensitivity_analysis = True
    data.sensitivity_perturbations = [1.0, 5.0]
    read_Tseries(data, 'c')
    aggregation(data)
    statis(data)

    import numpy as np
    np.random.seed(42)

    start_time = time.time()
    if mode == 'PSO':
        data.n_particles = 500
        data.n_run = 500
        PSO_mode(data)
    elif mode == 'DE':
        DE_mode(data)
    elif mode == 'DE-MCMC':
        DE_MCMC_mode(data)
    end_time = time.time()

    # Forward run
    data.par[:] = data.par_best[:]
    if data.gap_tolerant and data.segments is None:
        detect_segments(data)
    call_model(data)

    # Run Sensitivity Analysis
    print(f"Running Sensitivity Analysis for {mode}...")
    sensitivity_analysis(data)

    # Determine history file path
    history_file = os.path.join(data.folder, f"0_{mode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.csv")

    res = {
        'time': end_time - start_time,
        'objective': data.finalfit,
        'params': data.par_best.copy(),
        'mod_temp': data.Twat_mod.copy(),
        'obs_temp': data.Twat_obs.copy(),
        'date': data.date.copy(),
        'folder': data.folder,
        'history_file': history_file
    }

    if mode == 'DE-MCMC':
        env_file = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")
        res['env_file'] = env_file

    return res

def main():
    config_path = "examples/optimizer_comparison/config.yaml"

    print("Running PSO Calibration...")
    pso_results = run_calibration(config_path, 'PSO')

    print("\nRunning DE Calibration...")
    de_results = run_calibration(config_path, 'DE')

    print("\nRunning DE-MCMC Calibration...")
    demcmc_results = run_calibration(config_path, 'DE-MCMC')

    print("\n--- Results ---")
    print("PSO Time: {:.2f} seconds".format(pso_results['time']))
    print("PSO Objective: {:.6f}".format(pso_results['objective']))
    print("PSO Parameters:", [float(f"{p:.5f}") for p in pso_results['params']])

    print("\nDE Time: {:.2f} seconds".format(de_results['time']))
    print("DE Objective: {:.6f}".format(de_results['objective']))
    print("DE Parameters:", [float(f"{p:.5f}") for p in de_results['params']])

    print("\nDE-MCMC Time: {:.2f} seconds".format(demcmc_results['time']))
    print("DE-MCMC Best Objective: {:.6f}".format(demcmc_results['objective']))
    print("DE-MCMC Best Parameters:", [float(f"{p:.5f}") for p in demcmc_results['params']])

    # Plotting
    import pandas as pd
    import numpy as np

    # Filter out warm-up and missing observed data
    valid_mask = (pso_results['obs_temp'] != -999.0)
    valid_mask[:365] = False # drop warm-up

    valid_dates_dict = {
        'year': pso_results['date'][valid_mask, 0],
        'month': pso_results['date'][valid_mask, 1],
        'day': pso_results['date'][valid_mask, 2]
    }
    dates = pd.to_datetime(valid_dates_dict)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, pso_results['obs_temp'][valid_mask], 'k.', label='Observed T_water', alpha=0.5)

    if os.path.exists(demcmc_results['env_file']):
        env_df = pd.read_csv(demcmc_results['env_file'])
        # Extract valid envelope data matching valid_mask
        # (the length should match the model array since it is printed row-for-row)
        if len(env_df) == len(valid_mask):
            plt.fill_between(dates, env_df['Twat_mod_p5'][valid_mask], env_df['Twat_mod_p95'][valid_mask], color='green', alpha=0.2, label='DE-MCMC 90% Envelope')

    plt.plot(dates, pso_results['mod_temp'][valid_mask], label='PSO Mod (NSE={:.3f})'.format(pso_results['objective']), alpha=0.7)
    plt.plot(dates, de_results['mod_temp'][valid_mask], '--', label='DE Mod (NSE={:.3f})'.format(de_results['objective']), alpha=0.7)

    plt.title('pyair2stream Calibration: PSO vs Hybrid DE vs DE-MCMC')
    plt.ylabel('Water Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('examples/optimizer_comparison/comparison_plot.png', dpi=300)
    print("\nSaved comparison plot to examples/optimizer_comparison/comparison_plot.png")

    # Convergence Plot
    plt.figure(figsize=(10, 5))

    if os.path.exists(pso_results['history_file']):
        df_pso = pd.read_csv(pso_results['history_file'])
        pso_eff = df_pso['eff_index'].cummax()
        plt.plot(range(1, len(pso_eff) + 1), pso_eff, label='PSO Cumulative Best NSE', color='orange')

    if os.path.exists(de_results['history_file']):
        df_de = pd.read_csv(de_results['history_file'])
        # The history includes phase 1 (DE) and phase 2 (L-BFGS-B).
        de_eff = df_de['eff_index'].cummax()
        plt.plot(range(1, len(de_eff) + 1), de_eff, label='Hybrid DE Cumulative Best NSE', color='green')

    if os.path.exists(demcmc_results['history_file']):
        df_demcmc = pd.read_csv(demcmc_results['history_file'])
        demcmc_eff = df_demcmc['eff_index'].cummax()
        plt.plot(range(1, len(demcmc_eff) + 1), demcmc_eff, label='DE-MCMC Cumulative Best NSE', color='blue', linestyle=':')

    plt.title('Optimization Convergence (NSE over Evaluations)')
    plt.xlabel('Number of Objective Function Evaluations')
    plt.ylabel('Best Objective Value (NSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/optimizer_comparison/convergence_plot.png', dpi=300)
    print("Saved convergence plot to examples/optimizer_comparison/convergence_plot.png")

    # We copy them locally for the README
    import shutil
    shutil.copyfile(f"{pso_results['folder']}/sensitivity_PSO_NSE_MAH.png", "examples/optimizer_comparison/sensitivity_PSO.png")
    shutil.copyfile(f"{de_results['folder']}/sensitivity_DE_NSE_MAH.png", "examples/optimizer_comparison/sensitivity_DE.png")
    print("Sensitivity plots copied to examples/optimizer_comparison/sensitivity_PSO.png and sensitivity_DE.png")

if __name__ == "__main__":
    main()
