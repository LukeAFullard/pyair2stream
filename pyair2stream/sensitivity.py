import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from .config import CommonData
from .model import call_model, detect_segments
from .io import read_Tseries

def sensitivity_analysis(data: CommonData):
    """
    Perform a One-At-A-Time (OAT) local sensitivity analysis.
    For each parameter, we perturb it by the given percentages of its valid range,
    and measure the mean absolute change in the predicted water temperatures.
    """
    perturbations = data.sensitivity_perturbations if data.sensitivity_perturbations else [1.0]
    print(f"Starting Local Sensitivity Analysis (perturbations = {perturbations}% of parameter range)...")

    n_par = 8
    sensitivities = []

    # Restore the calibration data
    # (Since `forward` may have loaded validation data and altered `data.n_tot`)
    read_Tseries(data, 'c')

    # Ensure baseline is run
    data.par[:] = data.par_best[:]
    if data.gap_tolerant and data.segments is None:
        detect_segments(data)
    call_model(data)

    # Determine valid indices (to skip warmup)
    valid_mask = data.Twat_mod != -999.0

    if data.gap_tolerant and data.segments is not None:
        for start, end in data.segments:
            # Drop warmup_drop_days from the beginning of each segment
            drop_end = min(start + data.warmup_drop_days, end + 1)
            valid_mask[start:drop_end] = False
    else:
        # Standard runs always prepend exactly 365 days of warmup data
        valid_mask[:365] = False

    for delta_pct in perturbations:
        for j in range(n_par):
            if not data.flag_par[j]:
                # Parameter is inactive
                sensitivities.append({
                    "Parameter": f"par_{j+1}",
                    "Perturbation_%": delta_pct,
                    "Sensitivity_Index": np.nan,
                    "Status": "Inactive"
                })
                continue

            param_range = data.parmax[j] - data.parmin[j]
            if param_range <= 0:
                sensitivities.append({
                    "Parameter": f"par_{j+1}",
                    "Perturbation_%": delta_pct,
                    "Sensitivity_Index": 0.0,
                    "Status": "Fixed"
                })
                continue

            # Use the actual parameter value to determine scale, with a fallback for exactly zero
            base_scale = abs(data.par_best[j])
            if base_scale < 1e-4:
                base_scale = 1e-4

            delta = (delta_pct / 100.0) * base_scale

            # Central difference bounded by parameter ranges
            p_plus = data.par_best[j] + delta
            p_minus = data.par_best[j] - delta

            if p_plus > data.parmax[j]:
                p_plus = data.parmax[j]
            if p_minus < data.parmin[j]:
                p_minus = data.parmin[j]

            actual_delta = p_plus - p_minus
            if actual_delta <= 0:
                sensitivities.append({
                    "Parameter": f"par_{j+1}",
                    "Perturbation_%": delta_pct,
                    "Sensitivity_Index": 0.0,
                    "Status": "Fixed"
                })
                continue

            # Run plus
            data.par[:] = data.par_best[:]
            data.par[j] = p_plus
            data.Twat_mod[:] = -999.0
            call_model(data)
            twat_plus = data.Twat_mod.copy()

            # Run minus
            data.par[:] = data.par_best[:]
            data.par[j] = p_minus
            data.Twat_mod[:] = -999.0
            call_model(data)
            twat_minus = data.Twat_mod.copy()

            # Compute mean absolute difference in the time series per unit change in normalized parameter
            diff = np.abs(twat_plus[valid_mask] - twat_minus[valid_mask])
            mean_diff = np.mean(diff)

            # Normalize sensitivity index based on the parameter's actual scale
            sens_index = mean_diff / (actual_delta / base_scale)

            sensitivities.append({
                "Parameter": f"par_{j+1}",
                "Perturbation_%": delta_pct,
                "Sensitivity_Index": sens_index,
                "Status": "Active"
            })

    # Restore best parameters
    data.par[:] = data.par_best[:]
    call_model(data)

    # Save to CSV
    df_sens = pd.DataFrame(sensitivities)
    out_csv = os.path.join(data.folder, f"sensitivity_{data.runmode}_{data.fun_obj}_{data.station}.csv")
    df_sens.to_csv(out_csv, index=False)

    # Plotting
    _plot_sensitivity(data, df_sens)

    print(f"Sensitivity analysis completed. Results saved to {out_csv}")

    return df_sens

def _plot_sensitivity(data: CommonData, df_sens: pd.DataFrame):
    plt.rcParams['font.family'] = 'serif'
    # Fallback to sans-serif if Times New Roman is not available
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 10

    # Filter active parameters
    df_active = df_sens[df_sens['Status'] == 'Active'].copy()
    if df_active.empty:
        return

    fig, ax = plt.subplots(figsize=(12/2.54, 8/2.54))

    parameters = df_active['Parameter'].unique()
    perturbations = df_active['Perturbation_%'].unique()

    x = np.arange(len(parameters))
    width = 0.8 / len(perturbations)

    colors = ['#56B4E9', '#E69F00', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

    for i, pert in enumerate(perturbations):
        subset = df_active[df_active['Perturbation_%'] == pert]
        # Ensure alignment with parameters
        subset = subset.set_index('Parameter').reindex(parameters).reset_index()

        offset = (i - len(perturbations) / 2 + 0.5) * width
        ax.bar(x + offset, subset['Sensitivity_Index'], width, label=f'{pert}%', color=colors[i % len(colors)])

    ax.set_ylabel('Sensitivity Index [\u00B0C]')
    ax.set_title('Local Parameter Sensitivity')
    ax.set_xticks(x)
    ax.set_xticklabels(parameters, rotation=45)
    if len(perturbations) > 1:
        ax.legend(title="Perturbation")

    plt.tight_layout()

    out_pdf = os.path.join(data.folder, f"sensitivity_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
    out_png = os.path.join(data.folder, f"sensitivity_{data.runmode}_{data.fun_obj}_{data.station}.png")

    plt.savefig(out_pdf, dpi=300)
    plt.savefig(out_png, dpi=300)
    plt.close()
