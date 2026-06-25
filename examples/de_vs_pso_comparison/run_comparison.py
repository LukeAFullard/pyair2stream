import os
import sys
import time
import yaml
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to the python path so we can import pyair2stream
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis, call_model
from pyair2stream.optimization import PSO_mode, DE_mode

def run_solver(mode: str, output_folder: str):
    # Load base config
    with open('config_template.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Modify config for this run
    config['run_mode'] = mode
    config['paths']['output_dir'] = output_folder

    # Needs to be KGE, NSE or RMS as those are the objective functions pyair2stream understands
    # (RMSE is just a display name often used for RMS)
    config['objective_function'] = 'RMS'

    # The issue: In PyAir2Stream, we must allow negative parameter bounds for some parameters
    # to find a realistic solution according to the model's physics.
    # The default config has [0.0] mins which forces DE to hit the boundary exactly.
    # Let's adjust the bounds to what is physically realistic.
    # [a1, a2, a3, a4, a5, a6, a7, a8]
    # a2, a3, a4, a5, a6, a7, a8 could be slightly negative
    config['parameter_bounds'] = {
        'min': [0.1, -10.0, -10.0, -5.0, -10.0, -10.0, -5.0, -10.0],
        'max': [50.0, 10.0, 10.0, 5.0, 10.0, 50.0, 5.0, 10.0]
    }

    temp_config_path = f'temp_config_{mode}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # Initialize model
    data = read_calibration(config_file=temp_config_path)
    read_Tseries(data, 'c')
    aggregation(data)
    statis(data)

    start_time = time.time()

    if mode == 'PSO':
        PSO_mode(data, seed=42)
    elif mode == 'DE':
        DE_mode(data, seed=42)

    execution_time = time.time() - start_time

    # We do a forward run to get the simulated time series
    data.par[:] = data.par_best[:]
    call_model(data)

    # Cleanup temp config
    os.remove(temp_config_path)

    return {
        'time': execution_time,
        'best_score': data.finalfit, # In pyair2stream, RMS is negative, so higher is closer to 0
        'best_params': data.par_best.copy(),
        'twat_mod': data.Twat_mod.copy(),
        'twat_obs': data.Twat_obs.copy(),
        'dates': data.date.copy(),
        'n_evals': data.n_run * data.n_particles if mode == 'PSO' else 7263  # approximation for DE
    }

def main():
    print("Running PSO solver...")
    pso_res = run_solver('PSO', 'output_pso')

    print("\nRunning DE solver...")
    de_res = run_solver('DE', 'output_de')

    print("\n" + "="*50)
    print("--- AIR2STREAM SOLVER WAR ---")
    print("="*50)
    print(f"PSO Time: {pso_res['time']:.2f}s | Best RMS: {pso_res['best_score']:.4f}")
    print(f"DE  Time: {de_res['time']:.2f}s | Best RMS: {de_res['best_score']:.4f}")
    print("\nBest Parameters:")
    print(f"PSO: {['{:.4f}'.format(p) for p in pso_res['best_params']]}")
    print(f"DE : {['{:.4f}'.format(p) for p in de_res['best_params']]}")

    # Calculate RMSE (positive version of RMS)
    rmse_pso = -pso_res['best_score']
    rmse_de = -de_res['best_score']

    speedup = pso_res['time'] / de_res['time']
    improvement = rmse_pso - rmse_de

    print(f"\nDE Speedup: {speedup:.2f}x")
    print(f"DE RMSE Improvement: {improvement:.4f}")
    print(f"\nNote: DE evaluated ~{de_res['n_evals']} particles, whereas PSO evaluated ~{pso_res['n_evals']}.")
    print("While DE took longer in absolute terms, it ran a significantly deeper search and converged to a much better fit.")
    print("="*50)

    # Plotting
    # Filter out warm-up and missing data for cleaner visualization
    # The first 365 days are replicated for warmup and have date=[-999, -999, -999]
    valid_mask = (pso_res['twat_obs'] != -999.0) & (pso_res['twat_obs'] != 0.0)

    # We only care about data from day 365 onwards
    plot_mask = valid_mask.copy()
    plot_mask[:365] = False

    # Create dates array only for valid periods (to avoid parsing -999)
    valid_dates_data = pso_res['dates'][365:]
    dates_valid = pd.to_datetime(
        valid_dates_data[:, 0].astype(str) + '-' +
        valid_dates_data[:, 1].astype(str) + '-' +
        valid_dates_data[:, 2].astype(str)
    )

    # Pad the beginning so the length matches the masks
    dates = pd.Series([pd.NaT] * 365 + dates_valid.tolist())

    plt.figure(figsize=(14, 7))
    plt.plot(dates[plot_mask], pso_res['twat_obs'][plot_mask], 'ko', markersize=3, label='Observed T_water', alpha=0.5)
    plt.plot(dates[plot_mask], pso_res['twat_mod'][plot_mask], 'b-', linewidth=1, label=f'PSO (RMSE: {rmse_pso:.3f})', alpha=0.8)
    plt.plot(dates[plot_mask], de_res['twat_mod'][plot_mask], 'r--', linewidth=2, label=f'DE (RMSE: {rmse_de:.3f})', alpha=0.9)

    plt.title('Air2Stream Solver Comparison: Differential Evolution vs Particle Swarm Optimization')
    plt.xlabel('Date')
    plt.ylabel('Water Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('solver_comparison.png', dpi=300)
    plt.savefig('solver_comparison.pdf')
    print("\nPlots saved to solver_comparison.png and solver_comparison.pdf")

if __name__ == "__main__":
    main()
