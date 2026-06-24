import os
import numpy as np
import pandas as pd
from typing import Optional
import concurrent.futures

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
