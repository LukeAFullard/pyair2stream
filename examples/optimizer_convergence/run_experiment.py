import time
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis
from pyair2stream.optimization import PSO_mode, DE_mode

def run_calibration(config_path, mode, n_runs, n_particles):
    data = read_calibration(config_file=config_path)
    data.runmode = mode
    data.n_run = n_runs
    data.n_particles = n_particles

    read_Tseries(data, 'c')
    aggregation(data)
    statis(data)

    start_time = time.time()
    if mode == 'PSO':
        PSO_mode(data)
    elif mode == 'DE':
        DE_mode(data)
    end_time = time.time()

    elapsed = end_time - start_time

    return data.par_best.copy(), elapsed, data.finalfit

def main():
    config_path = "examples/optimizer_convergence/config.yaml"

    # Range of n_runs for convergence tests
    run_values = [10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    n_particles = 20

    pso_params = []
    pso_times = []
    pso_nse = []
    de_params = []
    de_times = []
    de_nse = []

    for r in run_values:
        print(f"Running PSO with n_runs={r}")
        p, t, nse = run_calibration(config_path, 'PSO', r, n_particles)
        pso_params.append(p)
        pso_times.append(t)
        pso_nse.append(nse)

        print(f"Running DE with n_runs={r}")
        p, t, nse = run_calibration(config_path, 'DE', r, n_particles)
        de_params.append(p)
        de_times.append(t)
        de_nse.append(nse)

    pso_params = np.array(pso_params)
    de_params = np.array(de_params)

    # 8 Subplots for the 8 parameters
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i in range(8):
        axes[i].plot(run_values, pso_params[:, i], marker='o', label='PSO', color='orange')
        axes[i].plot(run_values, de_params[:, i], marker='s', label='DE', color='blue')
        axes[i].set_title(f'Parameter {i+1} Convergence')
        axes[i].set_xlabel('Number of Iterations (n_runs)')
        axes[i].set_ylabel(f'Parameter {i+1} Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/optimizer_convergence/parameter_convergence.png', dpi=300)
    print("Saved parameter convergence plot to examples/optimizer_convergence/parameter_convergence.png")

    # Separate line chart for computation time
    plt.figure(figsize=(10, 6))
    plt.plot(run_values, pso_times, marker='o', label='PSO', color='orange')
    plt.plot(run_values, de_times, marker='s', label='DE', color='blue')
    plt.title('Computation Time vs Iterations')
    plt.xlabel('Number of Iterations (n_runs)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/optimizer_convergence/computation_time.png', dpi=300)
    print("Saved computation time plot to examples/optimizer_convergence/computation_time.png")


    # Plot NSE vs Iterations
    plt.figure(figsize=(10, 6))
    plt.plot(run_values, pso_nse, marker='o', label='PSO', color='orange')
    plt.plot(run_values, de_nse, marker='s', label='DE', color='blue')
    plt.title('Objective Function (NSE) vs Iterations')
    plt.xlabel('Number of Iterations (n_runs)')
    plt.ylabel('NSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('examples/optimizer_convergence/objective_convergence.png', dpi=300)
    print("Saved objective convergence plot to examples/optimizer_convergence/objective_convergence.png")

if __name__ == "__main__":
    main()
