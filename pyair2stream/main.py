import os
import sys
import time
import numpy as np
import pandas as pd

from .io import read_calibration, read_Tseries
from .model import call_model, aggregation, statis, funcobj
from .optimization import forward_mode, PSO_mode, LH_mode
from .config import CommonData

def forward(data: CommonData) -> None:
    """
    Replicates SUBROUTINE forward in AIR2STREAM_SUBROUTINES.f90
    Executes the model with the best parameters from calibration and runs validation if available.
    """
    # 1. Forward run on calibration data
    data.par[:] = data.par_best[:]
    call_model(data)

    # Calculate objective function again to ensure consistency
    ei_check = funcobj(data)

    if abs(ei_check - data.finalfit) > 0.0001:
        print('Errore efficienza in forward')
        print(ei_check, data.finalfit)
        # Replacing Fortran PAUSE with RuntimeError
        raise RuntimeError('Efficiency mismatch in forward run.')
    else:
        print('Controllo superato')

    # Output best parameters and ei (Append to 1_ file)
    param_out_path = os.path.join(data.folder, f"1_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.out")
    with open(param_out_path, 'w') as f:
        f.write(" ".join([f"{p:.6f}" for p in data.par_best]) + "\n")
        f.write(f"{ei_check:.6f}\n")

    # Output final simulated time series (calibration) as CSV instead of raw text
    out_cal_path = os.path.join(data.folder, f"2_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}c_{data.time_res}.csv")

    # Fortran format: date(i,1:3), Tair, Twat_obs, Twat_mod, Twat_obs_agg, Twat_mod_agg, Q
    cal_df = pd.DataFrame({
        'Year': data.date[:, 0],
        'Month': data.date[:, 1],
        'Day': data.date[:, 2],
        'Tair': data.Tair,
        'Twat_obs': data.Twat_obs,
        'Twat_mod': data.Twat_mod,
        'Twat_obs_agg': data.Twat_obs_agg,
        'Twat_mod_agg': data.Twat_mod_agg,
        'Q': data.Q
    })
    cal_df.to_csv(out_cal_path, index=False)

    # 2. Validation period
    read_Tseries(data, 'v')

    if data.n_tot < 365:
        ei = -999.0
        return

    aggregation(data)
    statis(data)
    print('mean, TSS and standard deviation (validation)')
    print(f"{data.mean_obs:.5f} {data.TSS_obs:.5f} {data.std_obs:.5f}")

    call_model(data)
    ei = funcobj(data)

    with open(param_out_path, 'a') as f:
        f.write(f"{ei:.6f}\n")

    out_val_path = os.path.join(data.folder, f"3_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}v_{data.time_res}.csv")
    val_df = pd.DataFrame({
        'Year': data.date[:, 0],
        'Month': data.date[:, 1],
        'Day': data.date[:, 2],
        'Tair': data.Tair,
        'Twat_obs': data.Twat_obs,
        'Twat_mod': data.Twat_mod,
        'Twat_obs_agg': data.Twat_obs_agg,
        'Twat_mod_agg': data.Twat_mod_agg,
        'Q': data.Q
    })
    val_df.to_csv(out_val_path, index=False)


def main():
    print('       .__       ________            __                                  ')
    print('_____  |__|______\_____  \   _______/  |________   ____ _____    _____   ')
    print('\__  \ |  \_  __ \/  ____/  /  ___/\   __\_  __ \_/ __ \\__  \  /     \  ')
    print(' / __ \|  ||  | \/       \  \___ \  |  |  |  | \/\  ___/ / __ \|  Y Y  \ ')
    print('(____  /__||__|  \_______ \/____  > |__|  |__|    \___  >____  /__|_|  / ')
    print('     \/                  \/     \/                    \/     \/      \/  ')
    print('pyair2stream Version 1.0.0 (Python Port)')
    print('')

    t1 = time.time()

    data = read_calibration()
    read_Tseries(data, 'c')
    aggregation(data)
    statis(data)

    print('mean, TSS and standard deviation (calibration)')
    print(f"{data.mean_obs:.5f} {data.TSS_obs:.5f} {data.std_obs:.5f}")

    if data.runmode == 'FORWARD':
        forward_mode(data)
    elif data.runmode == 'PSO':
        PSO_mode(data)
    elif data.runmode == 'LATHYP':
        LH_mode(data)

    forward(data)

    t2 = time.time()
    print(f"Computation time was {t2 - t1:.4f} seconds.")

if __name__ == '__main__':
    main()
