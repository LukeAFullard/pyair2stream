"""
Command-line interface and execution dispatch for pyair2stream.

This module provides the main entry point for the CLI, parsing arguments,
loading the configuration, executing the specified operational mode (calibration,
forward simulation, cross-validation), and generating final output reports.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

from .io import read_calibration, read_Tseries, precheck_validation, SettingsFileNotFoundError, theta_of_days, fit_settings
from .optimization import forward_mode, PSO_mode, LH_mode, DE_mode, DE_MCMC_mode
from .config import CommonData, theta_floor_of, zero_flow_ok
from .post_processing import post_process
from .sensitivity import sensitivity_analysis
from .results import (RunResult, capture_output, collect_parameters, collect_scores, files_of_this_run,
                      messages_from, write_filled_series, write_summary)
from . import __version__

from .model import (call_model, aggregation, statis, funcobj, detect_segments, warn_on_stability,
                    check_numerical_divergence, check_segment_warmup, check_daily_plausibility)

JACKKNIFE_NOTE = (
    "The rows jackknife_{level}_lower/upper are approximate {level}% intervals for the parameters "
    "(uncertainty_options.parameter_interval; in validation, 90% intervals contained the true values "
    "83-95% of the time; validation/REPORT.md, V4). The 'std' row is only the spread between folds: "
    "it is far too small to use as an uncertainty."
)


def write_yearly_statistics_check(data: CommonData, folds) -> None:
    """Write cv_yearly_statistics.csv and cv_yearly_statistics_summary.csv: did the predicted
    ranges of yearly statistics hold in the held-out years (docs/METHODS.md §11)?"""
    from .config import DEFAULT_NOISE_MODEL
    from .cross_validation import check_yearly_statistics, check_interval_coverage
    cv = data.cross_validation
    options = data.uncertainty_options or {}
    level = float(options.get('prediction_interval', 90.0))
    coverage = check_interval_coverage(folds, extra_level=level,
                                       noise_model=options.get('noise_model', DEFAULT_NOISE_MODEL),
                                       seed=data.random_seed)
    coverage.to_csv(os.path.join(data.folder, "cv_interval_coverage.csv"), index=False)
    print("Prediction intervals in the held-out years (share of measured days / 7-day means inside):")
    for r in coverage.to_dict("records"):
        print(f"  {r['level']:g}% interval: days {r['daily inside']:.1%}, 7-day means {r['7-day inside']:.1%}")
    per_year, summary = check_yearly_statistics(
        folds, threshold=cv.threshold, season_months=cv.season_months,
        noise_model=options.get('noise_model', DEFAULT_NOISE_MODEL), level=level, seed=data.random_seed)
    if per_year.empty:
        print("Yearly statistics check: no held-out year had enough of its season measured.")
        return
    per_year.to_csv(os.path.join(data.folder, "cv_yearly_statistics.csv"), index=False)
    summary.to_csv(os.path.join(data.folder, "cv_yearly_statistics_summary.csv"), index=False)
    print(f"Yearly statistics in the held-out years (threshold {per_year.threshold.iloc[0]:.2f} degC, "
          f"season months {per_year.season_months.iloc[0]}):")
    col = f"inside_{level:g}"
    for r in summary.to_dict("records"):
        print(f"  {r['statistic']}: inside the {level:g}% range in {r['share_' + col]:.0%} of {r['n_years']} years "
              f"(expected by chance {r['expected_' + col + '_low']:.0%}-{r['expected_' + col + '_high']:.0%}); "
              f"measured minus predicted median {r['mean_deviation']:+.2f} "
              f"(95% CI {r['mean_deviation_ci95_lower']:+.2f} to {r['mean_deviation_ci95_upper']:+.2f})")
    print("A confidence interval that excludes zero means the model is biased in that statistic; "
          "scenario.correct_statistic corrects for it (docs/METHODS.md §13).")


def run_optimizer(data: CommonData) -> None:
    """
    Dispatches to the correct optimizer based on data.runmode, passing
    `data.random_seed` through for reproducibility. `None`
    (the default) reproduces the previous unseeded behaviour.
    """
    if data.runmode == 'FORWARD':
        forward_mode(data)
    elif data.runmode == 'PSO':
        PSO_mode(data, seed=data.random_seed)
    elif data.runmode == 'LATHYP':
        LH_mode(data, seed=data.random_seed)
    elif data.runmode == 'DE':
        DE_mode(data, seed=data.random_seed)
    elif data.runmode == 'DE-MCMC':
        DE_MCMC_mode(data, seed=data.random_seed)


def _write_calibration_metadata(data: CommonData) -> None:
    """Write calibration_metadata.json (Qmedia, theta range, version, integrator, parameters)."""
    # The flows the parameters were fitted on, as the model used them (zero-flow days at
    # min_theta_floor when it is set): a FORWARD run reports days outside this range.
    theta_min = theta_max = None
    if data.Qmedia > 0:
        theta_cal, _ = theta_of_days(data.Q[365:data.n_tot], data.Qmedia, theta_floor_of(data.version, data.min_theta_floor))
        if theta_cal.size:
            theta_min = float(np.min(theta_cal))
            theta_max = float(np.max(theta_cal))

    calibration_metadata = {
        "qmedia": float(data.Qmedia),
        "qmedia_source": "user" if data.Qmedia_user is not None else "computed",
        "n_q_valid": int(data.n_Q),
        "theta_min": theta_min,
        "theta_max": theta_max,
        "version": int(data.version),
        "integrator": data.mod_num,
        "par_best": [float(x) for x in data.par_best],
        **fit_settings(data),
        "pyair2stream_version": __version__,
        "random_seed": data.random_seed,
    }
    metadata_path = os.path.join(data.folder, "calibration_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(calibration_metadata, f, indent=2)


def _warm_up_column(data: CommonData) -> np.ndarray:
    """1 on the days that are not scored while the model forgets its start value, else 0:
    the first warmup_drop_days of each segment in gap-tolerant mode, or of a file shorter
    than a year (which starts from its first day's conditions, io.read_Tseries)."""
    warm_up = np.zeros(data.n_tot, dtype=int)
    starts = [start for start, _ in data.segments] if data.gap_tolerant and data.segments else [365]
    ends = [end for _, end in data.segments] if data.gap_tolerant and data.segments else [data.n_tot - 1]
    for start, end in zip(starts, ends):
        warm_up[start:min(start + data.warmup_drop_days, end + 1)] = 1
    return warm_up


def forward(data: CommonData) -> None:
    """
    Replicates SUBROUTINE forward in AIR2STREAM_SUBROUTINES.f90
    Executes the model with the best parameters from calibration and runs validation if available.
    """
    # 1. Forward run on calibration data
    data.par[:] = data.par_best[:]

    if data.gap_tolerant and data.segments is None:
        detect_segments(data)

    warn_on_stability(data, error_fraction=data.stability_error_fraction)
    check_segment_warmup(data, suggest_shorter=data.runmode != 'FORWARD')
    call_model(data)
    check_numerical_divergence(data, max_plausible_twat=data.max_plausible_twat)
    check_daily_plausibility(data)

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

    # Persist Qmedia (and the calibrated theta range) alongside the fitted
    # parameters so a later FORWARD run on different discharge (e.g. a
    # naturalised-flow or climate-projection scenario) can pin theta to the
    # value the parameters were actually fitted under instead of silently
    # rescaling it (USER_GUIDE.md §6, Qmedia). Not written by a FORWARD run: it
    # is not a calibration, and would overwrite the real calibration's file
    # (with the scenario's theta range) when both share an output folder.
    if data.runmode != 'FORWARD':
        _write_calibration_metadata(data)

    # Construct gap columns
    tair_gap = np.where(data.Tair == -999.0, 1, 0)
    # Discharge that cannot be used: missing, or (versions 4/7/8 without min_theta_floor, which
    # cannot simulate a day without flow) zero or negative; gap-tolerant mode treats both as gaps.
    q_zero = np.zeros(data.n_tot, dtype=bool)
    if not zero_flow_ok(data.version, data.min_theta_floor):
        q_zero = (data.Q != -999.0) & (data.Q <= 0.0)
    q_gap = np.where((data.Q == -999.0) | q_zero, 1, 0)
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
    if data.warmup_from_first_day or data.gap_tolerant:
        cal_df['warm_up'] = _warm_up_column(data)

    # Drop the warm-up block: it is a verbatim copy of year one with sentinel
    # dates (Year=-999), an implementation detail that broke pd.to_datetime and
    # row-count expectations for anyone reading the file directly.
    cal_df.iloc[365:].to_csv(out_cal_path, index=False)

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
            q_zero_count = int(np.sum(q_zero[365:]))
            if q_zero_count:
                f.write(f"  of which zero or negative discharge (treated as gaps; set min_theta_floor to "
                        f"simulate them): {q_zero_count}\n")

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
    # Do not recompute Qmedia here: the validation period must be scored under
    # the same normalisation the parameters were calibrated with.
    # read_Tseries already rebuilds segments/eval_mask for the validation data (and
    # turns "no valid segments" in gap-tolerant mode into the same validation-skipped
    # early return as a missing/too-short validation file).
    read_Tseries(data, 'v', recompute_qmedia=False)

    # Gate on the explicit flag, not data.n_tot: a too-short (or missing, or
    # gap-tolerant-with-no-valid-segments) validation period returns from
    # read_Tseries before data.n_tot is overwritten, so it stays at the
    # calibration value and would otherwise silently pass this guard, re-running
    # "validation" on the calibration arrays.
    if not data.validation_available:
        return

    aggregation(data)
    statis(data)
    print('mean, TSS and standard deviation (validation)')
    print(f"{data.mean_obs:.5f} {data.TSS_obs:.5f} {data.std_obs:.5f}")

    warn_on_stability(data, error_fraction=data.stability_error_fraction)
    check_segment_warmup(data, suggest_shorter=False)
    call_model(data)
    check_numerical_divergence(data, max_plausible_twat=data.max_plausible_twat)
    ei = funcobj(data)

    with open(param_out_path, 'a') as f:
        f.write(f"{ei:.6f}\n")

    out_val_path = os.path.join(data.folder, f"3_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}v_{data.time_res}.csv")

    val_tair_gap = np.where(data.Tair == -999.0, 1, 0)
    # As for the calibration: zero-flow days that versions 4/8 (without min_theta_floor) cannot
    # simulate are gaps too.
    val_q_zero = np.zeros(data.n_tot, dtype=bool)
    if not zero_flow_ok(data.version, data.min_theta_floor):
        val_q_zero = (data.Q != -999.0) & (data.Q <= 0.0)
    val_q_gap = np.where((data.Q == -999.0) | val_q_zero, 1, 0)
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
    if data.warmup_from_first_day or data.gap_tolerant:
        val_df['warm_up'] = _warm_up_column(data)

    val_df.iloc[365:].to_csv(out_val_path, index=False)  # drop the warm-up block


def run(config, verbose: bool = True) -> RunResult:
    """
    Run pyair2stream from Python, exactly as the command line does.

    `config` is the path of a settings file (YAML) or a dict with the same keys. With
    `verbose=False` nothing is printed; the warnings and notes are still returned. Errors
    raise exceptions (for example FileNotFoundError, or ValueError naming the file, column
    and line of a problem in the data).

    Returns a RunResult: the output folder, the best parameters, the scores, the warnings
    and notes, and the output files. Every run also writes summary.md and, when it
    simulates a period, filled_water_temperature_<period>.csv into its output folder.

        result = pyair2stream.run("config.yaml")
        result.scores["validation"]["RMSE"]
    """
    settings = config if isinstance(config, str) else "(a dict passed to pyair2stream.run)"
    t1 = time.time()
    with capture_output(verbose) as printed:
        data = _run(config, t1)
    messages = messages_from(printed.lines)
    scores = collect_scores(data)
    parameters = collect_parameters(data)
    summary = None
    try:   # the summary and the filled series are extras: a problem with them must not lose the run
        write_filled_series(data)
        summary = write_summary(data, scores, parameters, messages, settings=settings, seconds=time.time() - t1)
    except Exception as err:  # noqa: BLE001
        print(f"Warning: the run finished, but summary.md or the filled series could not be written: {err!r}")
    if verbose and summary:
        print(f"Summary of this run: {summary}")
    return RunResult(output_dir=data.folder, run_mode=data.runmode, version=data.version, parameters=parameters,
                     scores=scores, messages=messages, summary=summary, files=files_of_this_run(data), data=data)


def _run(config, t1: float) -> CommonData:
    """The run itself: load and check the data, calibrate or simulate, write the outputs."""
    data = read_calibration(config_file=config)   # records what the output folder already holds
    if data.folder_before:
        print(f"Warning: the output folder {data.folder} already holds {len(data.folder_before):,} file(s) from "
              "earlier runs. This run replaces those with the same names as its own outputs (for example "
              "summary.md) and keeps the others; summary.md describes only this run's files and counts the others.")

    read_Tseries(data, 'c')
    # The validation file is used only after the calibration, which can take hours:
    # check it now, so a problem in it stops the run before any calibration.
    precheck_validation(data)

    # FORWARD mode does not calibrate, so it may legitimately have no T_water
    # observations at all (a pure projection). statis() raises when there are
    # none, so it -- and the preceding aggregation() -- must not run
    # unconditionally here. forward_mode() handles both cases itself via its own
    # has_obs check.
    if data.runmode != 'FORWARD':
        aggregation(data)
        statis(data)

        print('mean, TSS and standard deviation (calibration)')
        print(f"{data.mean_obs:.5f} {data.TSS_obs:.5f} {data.std_obs:.5f}")

    if getattr(data, 'cross_validation', None):
        if data.runmode in ('PSO', 'DE', 'LATHYP'):
            from .cross_validation import cross_validate, held_out_series
            from .post_processing import write_bias_by_month
            if data.version in (4, 7, 8) and data.Qmedia_user is None:
                print("Note: Qmedia is recomputed for each fold. Set Qmedia: in the config so every "
                      "fold uses the same discharge scaling; otherwise the parameters also move with it.")
            df, folds = cross_validate(data, data.runmode, return_folds=True)
            df.to_csv(os.path.join(data.folder, "cv_results.csv"), index=False)
            # Mean error by month and season over the held-out years (out of sample).
            dates, obs, sim = held_out_series(folds)
            if np.isfinite(obs - sim).any():
                write_bias_by_month(dates, obs, sim, data.folder, "cv_bias_by_month",
                                    "Cross-validation, held-out years")
            print("Cross-validation completed.")
            print(df)
            print(JACKKNIFE_NOTE.format(level=f"{(data.uncertainty_options or {}).get('parameter_interval', 90.0):g}"))
            write_yearly_statistics_check(data, folds)

            t2 = time.time()
            print(f"Computation time was {t2 - t1:.4f} seconds.")
            return data  # skip the normal single calibration + forward() + post_process()
        else:
            print(f"Warning: cross_validation is enabled in config, but run mode '{data.runmode}' does not support it. Ignoring cross_validation block.")

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
    return data


def main():
    """
    Command-line interface entry point: `pyair2stream --config settings.yaml`.

    Prints the banner and runs `run()` on the settings file.
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

    try:
        run(args.config)
    except SettingsFileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
