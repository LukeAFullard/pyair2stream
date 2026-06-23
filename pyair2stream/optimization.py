import os
import numpy as np
import pandas as pd
from typing import Optional

from .config import CommonData
from .model import call_model, funcobj

def sub_1(data: CommonData) -> np.float64:
    """
    Helper function to call model and evaluate the objective function.
    Replicates SUBROUTINE sub_1
    """
    call_model(data)
    return np.float64(funcobj(data))

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
    fitbest = np.zeros(n_particles, dtype=np.float64)

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

    for k in range(n_particles):
        data.par[:n_par] = x[:, k]
        eff_index = sub_1(data)
        fitbest[k] = eff_index
        # Fix: Included initial evaluations in history
        if eff_index >= data.mineff_index:
            row = list(x[:, k]) + [eff_index]
            history.append(row)

    # Fix: use fitbest to find initial global best instead of fit
    best_idx = int(np.argmax(fitbest))
    foptim = fitbest[best_idx]
    gbest[:] = x[:, best_idx]

    for i in range(n_run):
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
                data.par[:n_par] = x[:, k]
                eff_index = sub_1(data)
                fit[k] = eff_index
                if eff_index >= data.mineff_index:
                    row = list(x[:, k]) + [eff_index]
                    history.append(row)
            else:
                fit[k] = -1e30

            if fit[k] > fitbest[k]:
                fitbest[k] = fit[k]
                pbest[:, k] = x[:, k]

        best_idx = int(np.argmax(fitbest))
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

        if eff_index >= data.mineff_index:
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
