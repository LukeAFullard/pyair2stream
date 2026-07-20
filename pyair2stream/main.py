"""
Command-line interface and execution dispatch for pyair2stream.

This module provides the main entry point for the CLI, parsing arguments,
loading the configuration, executing the specified operational mode (calibration,
forward simulation, cross-validation), and generating final output reports.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from .io import read_calibration, read_Tseries
from .model import call_model, aggregation, statis, funcobj
from .optimization import forward_mode, PSO_mode, LH_mode, DE_mode, DE_MCMC_mode, DE_CV_MCMC_mode
from .config import CommonData
from .post_processing import post_process
from .sensitivity import sensitivity_analysis
from . import __version__

from .model import call_model, aggregation, statis, funcobj, detect_segments

def run_optimizer(data: CommonData) -> None:
    """Dispatches to the correct optimizer based on data.runmode."""
    if data.runmode == 'FORWARD':
        forward_mode(data)
    elif data.runmode == 'PSO':
        PSO_mode(data)
    elif data.runmode == 'LATHYP':
        LH_mode(data)
    elif data.runmode == 'DE':
        DE_mode(data)
    elif data.runmode == 'DE-MCMC':
        DE_MCMC_mode(data)
    elif data.runmode == 'DE-CV-MCMC':
        DE_CV_MCMC_mode(data)


def forward(data: CommonData) -> None:
    """
    Replicates SUBROUTINE forward in AIR2STREAM_SUBROUTINES.f90
    Executes the model with the best parameters from calibration and runs validation if available.
    """
    # 1. Forward run on calibration data
    data.par[:] = data.par_best[:]

    if data.gap_tolerant and data.segments is None:
        detect_segments(data)

    call_model(data)

    # Calculate objective function again to ensure consistency
    ei_check = funcobj(data)

    if abs(ei_check - data.finalfit) > 0.0001:
        print(f'Error: efficiency mismatch in forward run ({ei_check} vs {data.finalfit})')
        print(ei_check, data.finalfit)
        # Replacing Fortran PAUSE with RuntimeError
        raise RuntimeError('Efficiency mismatch in forward run.')
    else:
        print('Consistency check passed.')

    # Output best parameters and ei (Append to 1_ file)
    param_out_path = os.path.join(data.folder, f"1_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.out")
    with open(param_out_path, 'w') as f:
        f.write(" ".join([f"{p:.6f}" for p in data.par_best]) + "\n")
        f.write(f"{ei_check:.6f}\n")

    # Construct gap columns
    tair_gap = np.where(data.Tair == -999.0, 1, 0)
    q_gap = np.where(data.Q == -999.0, 1, 0)
    segment_id = np.full(data.n_tot, -999)
    if data.gap_tolerant and data.segments:
        for idx, (start, end) in enumerate(data.segments):
            segment_id[start:end+1] = idx

    # Output final simulated time series (calibration) as CSV instead of raw text
    out_cal_path = os.path.join(data.folder, f"2_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}c_{data.time_res}.csv")

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

    if data.gap_tolerant:
        cal_df['Tair_gap'] = tair_gap
        cal_df['Q_gap'] = q_gap
        cal_df['segment_id'] = segment_id

    cal_df.to_csv(out_cal_path, index=False)

    # Generate gaps_summary.txt
    if data.gap_tolerant:
        summary_path = os.path.join(data.folder, "gaps_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=== pyair2stream Gap Summary ===\n")
            f.write(f"Qmedia source: {'User-supplied' if data.Qmedia_user is not None else 'Computed'}\n")
            f.write(f"Qmedia value: {data.Qmedia:.5f}\n")

            # Since index 0 to 364 are warmups we look from 365
            n_data_points = data.n_tot - 365
            tair_gap_count = np.sum(tair_gap[365:])
            q_gap_count = np.sum(q_gap[365:])
            f.write(f"T_air missing fraction: {tair_gap_count}/{n_data_points} ({tair_gap_count/n_data_points:.2%})\n")
            f.write(f"Q missing fraction: {q_gap_count}/{n_data_points} ({q_gap_count/n_data_points:.2%})\n")

            total_valid_days = 0
            if data.segments:
                f.write(f"Segments found: {len(data.segments)}\n")
                for i, (start, end) in enumerate(data.segments):
                    length = end - start + 1
                    total_valid_days += length
                    # Formatted dates
                    start_date = f"{data.date[start, 0]:04d}-{data.date[start, 1]:02d}-{data.date[start, 2]:02d}"
                    end_date = f"{data.date[end, 0]:04d}-{data.date[end, 1]:02d}-{data.date[end, 2]:02d}"
                    f.write(f"  Segment {i}: {start_date} to {end_date} (length: {length} days)\n")
            else:
                f.write("Segments found: 0\n")

            f.write(f"Total valid forcing days: {total_valid_days}\n")
            f.write(f"T_water observations used in calibration: {data.n_dat}\n")

    # 2. Validation period
    read_Tseries(data, 'v')

    if data.n_tot < 365:
        ei = -999.0
        return

    if data.gap_tolerant:
        try:
            detect_segments(data)
        except ValueError as e:
            print(f"Validation skipped: {e}")
            data.n_tot = 0
            return

        if not data.segments:
            print("Validation skipped: No valid segments found.")
            data.n_tot = 0
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

    val_tair_gap = np.where(data.Tair == -999.0, 1, 0)
    val_q_gap = np.where(data.Q == -999.0, 1, 0)
    val_segment_id = np.full(data.n_tot, -999)
    if data.gap_tolerant and data.segments:
        for idx, (start, end) in enumerate(data.segments):
            val_segment_id[start:end+1] = idx

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

    if data.gap_tolerant:
        val_df['Tair_gap'] = val_tair_gap
        val_df['Q_gap'] = val_q_gap
        val_df['segment_id'] = val_segment_id

    val_df.to_csv(out_val_path, index=False)


def main():
    """
    Command-line interface entry point.

    Parses the `--config` argument to locate the YAML configuration file, reads the
    settings, loads the input datasets, executes cross-validation or calibration
    (PSO, DE, MCMC, etc.), and triggers post-processing to generate outputs.
    """
    parser = argparse.ArgumentParser(description="pyair2stream - Python Port of air2stream")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    print(r'       .__       ________            __                                  ')
    print(r'_____  |__|______\_____  \   _______/  |________   ____ _____    _____   ')
    print(r'\__  \ |  \_  __ \/  ____/  /  ___/\   __\_  __ \_/ __ \__  \  /     \  ')
    print(r' / __ \|  ||  | \/       \  \___ \  |  |  |  | \/\  ___/ / __ \|  Y Y  \ ')
    print(r'(____  /__||__|  \_______ \/____  > |__|  |__|    \___  >____  /__|_|  / ')
    print(r'     \/                  \/     \/                    \/     \/      \/  ')
    print(f'pyair2stream Version {__version__} (Python Port)')
    print('')

    t1 = time.time()

    try:
        data = read_calibration(config_file=args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    read_Tseries(data, 'c')
    aggregation(data)
    statis(data)

    print('mean, TSS and standard deviation (calibration)')
    print(f"{data.mean_obs:.5f} {data.TSS_obs:.5f} {data.std_obs:.5f}")

    if getattr(data, 'cross_validation', None):
        from .cross_validation import run_leave_one_year_out_cv, summarize
        results = run_leave_one_year_out_cv(data, data.cross_validation, data.runmode)
        df = summarize(results)
        df.to_csv(os.path.join(data.folder, "cv_results.csv"), index=False)
        print("Cross-validation completed.")
        print(df)

        t2 = time.time()
        print(f"Computation time was {t2 - t1:.4f} seconds.")
        return  # skip the normal single calibration + forward() + post_process()

    run_optimizer(data)

    forward(data)

    t2 = time.time()
    print(f"Computation time was {t2 - t1:.4f} seconds.")

    # Automatically trigger post-processing visualization
    print('Starting post-processing visualizations...')
    post_process(data)
    print('Post-processing completed.')

    if data.sensitivity_analysis:
        sensitivity_analysis(data)

if __name__ == '__main__':
    main()
