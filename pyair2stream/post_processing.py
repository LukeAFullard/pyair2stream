"""
Post-processing, reporting, and visualization for pyair2stream.

This module generates diagnostic plots (time series comparisons, dotty plots
of parameter convergence) and exports calibration performance metrics and
simulated series to CSV format.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .config import CommonData


def select_dotty_data(df_0: pd.DataFrame) -> tuple:
    """
    Extract parameter columns and the efficiency-index column from a `0_*.csv`
    optimizer history file, by name rather than position: the file has 12
    columns (`par_1`..`par_8`, `eff_index`, `NSE`, `R2`, `MAE` --
    `optimization.py`'s `PSO_mode`/`DE_mode`/`LH_mode` history rows), so a
    positional `[:-1]`/`[-1]` split silently treated `NSE`/`R2` as parameter
    columns and `MAE` as the plotted "efficiency" (docs/audit/06_diagnostics_and_plots.md,
    Defect A).

    Returns
    -------
    parset : ndarray, shape (n_rows, n_par)
    eff : ndarray, shape (n_rows,) -- the `eff_index` column
    n_par : int
    """
    par_cols = [c for c in df_0.columns if c.startswith('par_')]
    return df_0[par_cols].values, df_0['eff_index'].values, len(par_cols)


def gap_aware_acf(residuals: pd.Series, max_lag: int) -> tuple:
    """
    Pairwise-complete sample autocorrelation of `residuals` -- a regularly-spaced
    daily series that may contain NaN at data gaps -- for lags 1..max_lag.

    `pd.plotting.autocorrelation_plot` on a NaN-dropped series concatenates
    non-adjacent days, so its lag-k is not lag-k in time. This instead pairs day
    `t` with day `t+k` only where both are non-NaN, generalising
    `uncertainty.estimate_ar1_rho`'s lag-1 adjacency handling to arbitrary lags
    (docs/audit/06_diagnostics_and_plots.md, Defect F).

    Returns
    -------
    lags : ndarray of int, 1..max_lag (clipped to len(residuals)-1)
    acf : ndarray of float, NaN where fewer than 2 valid pairs exist at that lag
    n_pairs : ndarray of int, the number of valid pairs used at each lag
    """
    values = residuals.to_numpy(dtype=np.float64)
    n = len(values)
    valid = ~np.isnan(values)

    max_lag = min(max_lag, max(n - 1, 0))
    lags = np.arange(1, max_lag + 1)
    acf = np.full(max_lag, np.nan)
    n_pairs = np.zeros(max_lag, dtype=np.int64)

    for i, k in enumerate(lags):
        pair_mask = valid[:n - k] & valid[k:]
        n_pairs[i] = pair_mask.sum()
        if n_pairs[i] > 1:
            a = values[:n - k][pair_mask]
            b = values[k:][pair_mask]
            if np.std(a) > 0 and np.std(b) > 0:
                acf[i] = np.corrcoef(a, b)[0, 1]

    return lags, acf, n_pairs


def post_process(data: CommonData, toll: float = None):
    """
    Analyzes and plots the results of the pyair2stream simulation.
    Replicates post_processing.m
    """
    if toll is None:
        # Acceptability threshold for the dotty-plot "good parameter set" region.
        # RMS is an unbounded error metric (lower = better), so `eff <= toll` with
        # toll=2.0 is a sensible absolute cutoff. NSE/KGE are bounded above by 1.0
        # (higher = better), so a shared toll=2.0 would require eff >= 2.0, which
        # is impossible and silently produced an empty "acceptable" region for
        # every NSE/KGE calibration. 0.5 follows the common Moriasi et al.
        # "satisfactory" NSE/KGE threshold used in hydrology.
        if data.fun_obj == 'RMS':
            toll = 2.0
        elif data.fun_obj == 'NSE':
            toll = 0.5
        elif data.fun_obj == 'KGE':
            toll = 0.5
        else:
            toll = 0.5

    # Colorblind barrier-free color palette
    orange = '#E69F00'
    blue = '#0072B2'
    light_blue = '#56B4E9'
    black = '#000000'

    # Set fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10

    # Output paths
    file_0 = os.path.join(data.folder, f"0_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.csv")
    file_cal = os.path.join(data.folder, f"2_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}c_{data.time_res}.csv")
    file_val = os.path.join(data.folder, f"3_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}v_{data.time_res}.csv")

    # 1a. Convergence Plots
    if os.path.exists(file_0):
        df_opt = pd.read_csv(file_0)

        if 'NSE' in df_opt.columns and 'R2' in df_opt.columns and 'MAE' in df_opt.columns:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # Cumulative best for NSE and R2 (maximize)
            df_opt['cummax_nse'] = df_opt['NSE'].cummax()
            df_opt['cummax_r2'] = df_opt['R2'].cummax()

            # Cumulative best for MAE (minimize)
            df_opt['cummin_mae'] = df_opt['MAE'].cummin()

            x_vals = range(1, len(df_opt) + 1)

            axes[0].plot(x_vals, df_opt['NSE'], '.', color='lightgray', alpha=0.5, label='Evaluation NSE')
            axes[0].plot(x_vals, df_opt['cummax_nse'], '-', color='blue', linewidth=2, label='Cumulative Best NSE')
            axes[0].set_ylabel('NSE')
            axes[0].set_title('Optimization Convergence')
            axes[0].legend()

            axes[1].plot(x_vals, df_opt['R2'], '.', color='lightgray', alpha=0.5, label='Evaluation R2')
            axes[1].plot(x_vals, df_opt['cummax_r2'], '-', color='green', linewidth=2, label='Cumulative Best R2')
            axes[1].set_ylabel('R2')
            axes[1].legend()

            axes[2].plot(x_vals, df_opt['MAE'], '.', color='lightgray', alpha=0.5, label='Evaluation MAE')
            axes[2].plot(x_vals, df_opt['cummin_mae'], '-', color='red', linewidth=2, label='Cumulative Best MAE')
            axes[2].set_ylabel('MAE [°C]')
            axes[2].set_xlabel('Evaluation Number')
            axes[2].legend()

            plt.tight_layout()
            conv_pdf = os.path.join(data.folder, f"convergence_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
            conv_png = os.path.join(data.folder, f"convergence_{data.runmode}_{data.fun_obj}_{data.station}.png")
            plt.savefig(conv_pdf, dpi=300)
            plt.savefig(conv_png, dpi=300)
            plt.close()

    # 1b. Dotty plots
    if os.path.exists(file_0):
        # Read the parameter tracking CSV
        df_0 = pd.read_csv(file_0)

        parset, eff, n_par = select_dotty_data(df_0)

        plot_data = False
        if len(eff) > 0:
            if data.fun_obj == 'RMS':
                eff = -eff
                valid_indices = np.where(eff <= toll)[0]
                if len(valid_indices) > 0:
                    parset = parset[valid_indices]
                    eff = eff[valid_indices]
                    plot_data = True

                if plot_data or len(eff) > 0:
                    best_eff = np.min(eff)
                    i_best = np.argmin(eff)
                    if plot_data:
                        plot_limits = [best_eff * 0.9, toll]
                    else:
                        plot_limits = [best_eff * 0.9, np.max(eff)]
                    plot_data = True # enable plotting even if fallback to all points
            else:
                valid_indices = np.where(eff >= toll)[0]
                if len(valid_indices) > 0:
                    parset = parset[valid_indices]
                    eff = eff[valid_indices]
                    plot_data = True

                if plot_data or len(eff) > 0:
                    best_eff = np.max(eff)
                    i_best = np.argmax(eff)
                    if plot_data:
                        plot_limits = [toll, best_eff * 1.1]
                    else:
                        plot_limits = [np.min(eff), best_eff * 1.1]
                    plot_data = True

        if plot_data:
            fig, axes = plt.subplots(2, 4, figsize=(18/2.54, 10/2.54))
            axes = axes.flatten()

            for i in range(8):
                # Not a for...else: that clause runs once after normal loop
                # completion (i == 7) and blanked the par_8 panel on every run
                # (audit report 06, Defect B). Hide only the genuinely unused
                # panels (i >= n_par), explicitly, inside the loop.
                if i >= n_par:
                    axes[i].axis('off')
                    continue

                axes[i].plot(parset[:, i], eff, '.k', markersize=2)
                if len(parset) > 0:
                    axes[i].plot(parset[i_best, i], best_eff, '.', color=orange, markersize=10)

                axes[i].set_ylim(plot_limits)
                axes[i].set_xlabel(f'par{i+1}')

                if data.fun_obj == 'RMS':
                    axes[i].set_ylabel(f"{data.fun_obj} [\u00B0C]")
                else:
                    axes[i].set_ylabel(data.fun_obj)

            plt.tight_layout()
            dotty_pdf = os.path.join(data.folder, f"dottyplots_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
            dotty_png = os.path.join(data.folder, f"dottyplots_{data.runmode}_{data.fun_obj}_{data.station}.png")
            plt.savefig(dotty_pdf, dpi=300)
            plt.savefig(dotty_png, dpi=300)
            plt.close()

    # 1c. MCMC Parameter Significance & Correlation
    env_file_mcmc = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")
    chain_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}.csv")
    if os.path.exists(chain_filename):
        chain_df = pd.read_csv(chain_filename)

        # Calculate statistics
        stats = []
        for col in chain_df.columns:
            mean_val = chain_df[col].mean()
            std_val = chain_df[col].std()
            ci_lower = chain_df[col].quantile(0.025)
            ci_upper = chain_df[col].quantile(0.975)
            # A simple significance check: if 0 is not in the 95% CI
            significant = not (ci_lower <= 0 <= ci_upper)
            stats.append({
                'Parameter': col,
                'Mean': mean_val,
                'StdDev': std_val,
                '95%_CI_Lower': ci_lower,
                '95%_CI_Upper': ci_upper,
                'Significantly_Diff_From_Zero': significant
            })

        stats_df = pd.DataFrame(stats)
        sig_file = os.path.join(data.folder, f"parameter_significance_{data.runmode}_{data.station}.csv")
        stats_df.to_csv(sig_file, index=False)
        print(f"Saved parameter significance report to {sig_file}")

        # Parameter Correlation Matrix
        corr_matrix = chain_df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)

        # Formatting
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45)
        ax.set_yticklabels(corr_matrix.columns)

        # Add text annotations
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')

        ax.set_title("Parameter Correlation Matrix (MCMC Chain)", pad=20)
        plt.tight_layout()
        corr_pdf = os.path.join(data.folder, f"parameter_correlation_{data.runmode}_{data.station}.pdf")
        corr_png = os.path.join(data.folder, f"parameter_correlation_{data.runmode}_{data.station}.png")
        plt.savefig(corr_pdf, dpi=300)
        plt.savefig(corr_png, dpi=300)
        plt.close()

    # 2. Time-Series Plots
    def plot_series(file_path, title_prefix, output_name, filter_to_obs=True):
        """
        Helper to plot modeled vs. observed water temperatures from a CSV.

        Produces an overlaid time-series graph with a secondary axis for air
        temperature (if present), saving it as a PNG and PDF.

        Parameters
        ----------
        file_path : str
            Path to the output CSV containing the simulation data.
        title_prefix : str
            Prefix for the plot title (e.g., 'Calibration' or 'Validation').
        output_name : str
            Base name for the saved plot files (e.g., 'calibration_plot').
        filter_to_obs : bool, optional
            If True, only plot ranges where actual observations exist (defaults to True).
        """
        if not os.path.exists(file_path):
            return

        df = pd.read_csv(file_path)

        # The warm-up block is dropped when this file is written (report 05,
        # Defect C), so no positional slice is needed here any more.

        # Replace sentinel values with NaN
        df.replace(-999.0, np.nan, inplace=True)

        # Create datetime index
        dates = pd.to_datetime(df[['Year', 'Month', 'Day']])

        if filter_to_obs:
            # Filter down the plotting range to the span of available observation data
            obs_valid_indices = np.where(df['Twat_obs_agg'].notna())[0]
            if len(obs_valid_indices) > 0:
                start_idx = obs_valid_indices[0]
                end_idx = obs_valid_indices[-1]
                df = df.iloc[start_idx:end_idx+1].copy()
                dates = dates.iloc[start_idx:end_idx+1].copy()

        # Calculate goodness of fit metrics
        # df has columns: 'Year', 'Month', 'Day', 'Tair', 'Twat_obs', 'Twat_mod', 'Twat_obs_agg', 'Twat_mod_agg', 'Q'
        valid_mask = df['Twat_obs_agg'].notna() & df['Twat_mod_agg'].notna()

        # Calculate number of active parameters (k)
        k = 0
        if data.flag_par is not None and data.parmin is not None and data.parmax is not None:
            for j in range(len(data.flag_par)):
                if data.flag_par[j] and data.parmin[j] != data.parmax[j]:
                    k += 1

        aic = np.nan
        bic = np.nan

        if valid_mask.sum() > 0:
            obs = df.loc[valid_mask, 'Twat_obs_agg']
            mod = df.loc[valid_mask, 'Twat_mod_agg']
            n = len(obs)

            rmse = np.sqrt(np.mean((obs - mod)**2))
            mae = np.mean(np.abs(obs - mod))

            # R2 calculation
            ss_res = np.sum((obs - mod)**2)
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

            # AIC and BIC calculation
            if n > 0 and ss_res > 0:
                aic = n * np.log(ss_res / n) + 2 * k
                bic = n * np.log(ss_res / n) + k * np.log(n)

            # Export Metrics Summary
            metrics_df = pd.DataFrame({
                'Metric': ['R2', 'RMSE', 'MAE', 'AIC', 'BIC'],
                'Value': [r2, rmse, mae, aic, bic]
            })
            metrics_csv = os.path.join(data.folder, f"goodness_of_fit_{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.csv")
            metrics_df.to_csv(metrics_csv, index=False)

        else:
            rmse = np.nan
            mae = np.nan
            r2 = np.nan

        fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(18/2.54, 14/2.54), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        # Check if Forward Prediction or MCMC Envelopes exist
        env_file_fwd = os.path.join(data.folder, f"Forward_Prediction_Envelopes_{data.station}_{data.series}_{data.time_res}.csv")
        env_file_mcmc = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")
        env_file = env_file_mcmc if os.path.exists(env_file_mcmc) else env_file_fwd

        metrics_str = f"R²={r2:.3f}, RMSE={rmse:.2f}°C, MAE={mae:.2f}°C\nAIC={aic:.1f}, BIC={bic:.1f}"

        if os.path.exists(env_file_mcmc):
            pi_val = (data.uncertainty_options or {}).get('prediction_interval', 90.0)
            ax.set_title(f"Historical Calibration with {pi_val:g}% Prediction Interval\n{metrics_str}")
        else:
            pi_val = (data.uncertainty_options or {}).get('prediction_interval', 90.0)
            title_text = f"Forward Projection with {pi_val:g}% Prediction Interval" if os.path.exists(env_file_fwd) else title_prefix
            ax.set_title(f"{title_text}\n{metrics_str}")

        if not filter_to_obs:
            ax.set_title(f"Full Simulation Timeline (with all forcing data)\n{metrics_str}")

        # Plot temperatures on primary y-axis
        l1 = ax.plot(dates, df['Tair'], '.', color=light_blue, label='Air temperature', markersize=2)
        l2 = ax.plot(dates, df['Twat_obs_agg'], '.', color=black, label='Observed water temperature', markersize=2)

        # If Twat_mod_agg is mostly NaN (like in forward projections where obs is missing), fallback to Twat_mod
        mod_series = df['Twat_mod_agg'] if df['Twat_mod_agg'].notna().sum() > 0 else df['Twat_mod']

        # We can also backfill/overlay Twat_mod for dates where aggregation failed but raw model ran
        mod_series = mod_series.combine_first(df['Twat_mod'])



        l3 = ax.plot(dates, mod_series, '.', color=orange, label='Simulated water temperature', markersize=2)

        ax.set_ylabel('Temperature [\u00B0C]')

        # Plot discharge on secondary y-axis
        ax2 = ax.twinx()
        l4 = ax2.plot(dates, df['Q'], '-', color='grey', alpha=0.3, label='Discharge (Q)', linewidth=1)
        ax2.set_ylabel('Discharge')
        ax2.set_ylim(bottom=0) # Discharge shouldn't be negative

        # Combine legends
        l_env = []
        if os.path.exists(env_file):
            env_df = pd.read_csv(env_file)
            # The warm-up block is dropped when this file is written (report 05,
            # Defect C), so only the observation-range slice below is needed.
            if filter_to_obs and 'start_idx' in locals() and len(env_df) > end_idx:
                env_df = env_df.iloc[start_idx:end_idx+1].copy()

            if len(env_df) == len(dates):
                env_df['Twat_mod_lower'] = np.where(env_df['Twat_mod_lower'] == -999.0, np.nan, env_df['Twat_mod_lower'])
                env_df['Twat_mod_upper'] = np.where(env_df['Twat_mod_upper'] == -999.0, np.nan, env_df['Twat_mod_upper'])
                lower_col = 'Twat_mod_lower'
                upper_col = 'Twat_mod_upper'

                pi_val = (data.uncertainty_options or {}).get('prediction_interval', 90.0)
                l_env = [ax.fill_between(dates, env_df[lower_col], env_df[upper_col], color='green', alpha=0.3, label=f'{pi_val:g}% Prediction Interval')]

        lines = l1 + l2 + l3 + l4
        if l_env:
            # We add a proxy artist for the fill_between to show up nicely in the legend
            import matplotlib.patches as mpatches
            pi_val = (data.uncertainty_options or {}).get('prediction_interval', 90.0)
            proxy = mpatches.Patch(color='green', alpha=0.3, label=f'{pi_val:g}% Prediction Interval')
            lines.append(proxy)

        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower left', fontsize='small')

        # Residuals plot
        residuals = mod_series - df['Twat_obs_agg']
        ax_res.plot(dates, residuals, '.', color='purple', markersize=2, label='Residuals')
        ax_res.axhline(0, color='black', linewidth=1, linestyle='--')
        ax_res.set_ylabel('Residuals [\u00B0C]')
        ax_res.grid(True, alpha=0.3)

        ax_res.set_xlabel('Time')
        ax_res.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        fig.autofmt_xdate()

        plt.tight_layout()
        pdf_path = os.path.join(data.folder, f"{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
        png_path = os.path.join(data.folder, f"{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.png")
        plt.savefig(pdf_path, dpi=300)
        plt.savefig(png_path, dpi=300)
        plt.close()

        # Generate Residual Diagnostics Plot
        # Drop NaNs from residuals
        res_clean = residuals.dropna()
        if len(res_clean) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Histogram
            axes[0].hist(res_clean, bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_title(f'Residuals Histogram ({output_name})')
            axes[0].set_xlabel('Residual [°C]')
            axes[0].set_ylabel('Frequency')

            # Q-Q plot
            import scipy.stats as stats
            stats.probplot(res_clean, dist="norm", plot=axes[1])
            axes[1].set_title(f'Normal Q-Q Plot ({output_name})')

            # Autocorrelation (ACF), computed gap-aware on the full (NaN-retaining)
            # residual series so lag-k pairs days k calendar days apart, not the
            # k-th and (k+1)-th surviving non-missing days (audit report 06, Defect F).
            lags, acf_vals, n_pairs = gap_aware_acf(residuals, max_lag=50)
            if len(lags) > 0:
                axes[2].bar(lags, acf_vals, width=0.8, color='steelblue')
                axes[2].axhline(0, color='black', linewidth=0.8)
                valid_n = n_pairs[n_pairs > 1]
                if len(valid_n) > 0:
                    band = 1.96 / np.sqrt(float(valid_n.min()))
                    axes[2].axhline(band, color='red', linestyle='--', linewidth=0.8)
                    axes[2].axhline(-band, color='red', linestyle='--', linewidth=0.8)
                axes[2].set_xlim([0, len(lags)])
                axes[2].set_xlabel(
                    f'Lag (days) -- {int(n_pairs[0])} valid pairs at lag 1, '
                    f'{int(n_pairs.min())}-{int(n_pairs.max())} across all lags shown'
                )
            axes[2].set_title(f'Autocorrelation ({output_name})')

            plt.tight_layout()
            diag_pdf = os.path.join(data.folder, f"residual_diagnostics_{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
            diag_png = os.path.join(data.folder, f"residual_diagnostics_{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.png")
            plt.savefig(diag_pdf, dpi=300)
            plt.savefig(diag_png, dpi=300)
            plt.close()

        # Generate Predicted vs Measured (Q-Q type plot)
        obs = df['Twat_obs_agg']
        mod = mod_series
        mask = obs.notna() & mod.notna()
        if mask.sum() > 0:
            obs_clean = obs[mask]
            mod_clean = mod[mask]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(obs_clean, mod_clean, c='blue', s=5, alpha=0.5, label='Predicted vs Measured')

            # 1:1 line
            min_val = min(obs_clean.min(), mod_clean.min())
            max_val = max(obs_clean.max(), mod_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')

            ax.set_xlabel('Measured Temperature [\u00B0C]')
            ax.set_ylabel('Predicted Temperature [\u00B0C]')
            ax.set_title(f'Predicted vs Measured ({output_name})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            qq_pdf = os.path.join(data.folder, f"predicted_vs_measured_{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
            qq_png = os.path.join(data.folder, f"predicted_vs_measured_{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.png")
            plt.savefig(qq_pdf, dpi=300)
            plt.savefig(qq_png, dpi=300)
            plt.close()

    if data.runmode == 'FORWARD':
        # We plot directly from the arrays since FORWARD mode might not save a CSV by default
        fig, ax = plt.subplots(figsize=(18/2.54, 10/2.54))

        # When pure FORWARD, dates might be full of zeros from initialized array
        # Let's clip to actual data.n_tot if needed, though they should be filled by IO
        df_dates = pd.DataFrame({'year': data.date[:, 0], 'month': data.date[:, 1], 'day': data.date[:, 2]})
        # The warm-up block is marked with Year=-999 by read_Tseries, so this mask
        # already excludes it -- no separate positional [365:] slice is needed or
        # correct here. (data.date/Tair/Twat_mod/Q are the live in-memory arrays,
        # still full-length, unlike the on-disk CSVs -- report 05, Defect C.)
        valid_dates_mask = df_dates['year'] > 0
        df_dates = df_dates[valid_dates_mask]
        dates = pd.to_datetime(df_dates)

        Tair = data.Tair[valid_dates_mask]
        Twat_mod = data.Twat_mod[valid_dates_mask]
        Q = data.Q[valid_dates_mask]

        # Replace sentinel values with NaN so they break the plot line rather than dropping to -999.0
        Tair = np.where(Tair == -999.0, np.nan, Tair)
        Twat_mod = np.where(Twat_mod == -999.0, np.nan, Twat_mod)
        Q = np.where((Q == -999.0) | (Q <= 0.0), np.nan, Q)

        l1 = ax.plot(dates, Tair, '.', color=light_blue, label='Air temperature', markersize=2)

        l_env = []
        env_file_fwd = os.path.join(data.folder, f"Forward_Prediction_Envelopes_{data.station}_{data.series}_{data.time_res}.csv")
        env_file_mcmc = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")

        env_file = env_file_mcmc if os.path.exists(env_file_mcmc) else env_file_fwd

        if os.path.exists(env_file_mcmc):
            pi_val = (data.uncertainty_options or {}).get('prediction_interval', 90.0)
            ax.set_title(f"Historical Calibration with {pi_val:g}% Prediction Interval")
        else:
            pi_val = (data.uncertainty_options or {}).get('prediction_interval', 90.0)
            ax.set_title(f"Forward Projection with {pi_val:g}% Prediction Interval")

        if os.path.exists(env_file):
            env_df = pd.read_csv(env_file)
            # env_file no longer contains the warm-up block (report 05, Defect C),
            # so it already aligns 1:1 with the valid (non-warm-up) portion of
            # data.date -- no mask/slice needed, just the length safety check below.
            if len(env_df) == len(dates):
                env_df['Twat_mod_lower'] = np.where(env_df['Twat_mod_lower'] == -999.0, np.nan, env_df['Twat_mod_lower'])
                env_df['Twat_mod_upper'] = np.where(env_df['Twat_mod_upper'] == -999.0, np.nan, env_df['Twat_mod_upper'])
                lower_col = 'Twat_mod_lower'
                upper_col = 'Twat_mod_upper'

                pi_val = (data.uncertainty_options or {}).get('prediction_interval', 90.0)
                l_env = [ax.fill_between(dates, env_df[lower_col], env_df[upper_col], color='green', alpha=0.3, label=f'{pi_val:g}% Prediction Interval')]

        l3 = ax.plot(dates, Twat_mod, '-', color=orange, label='Simulated median water temp.', linewidth=1.5)

        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature [°C]')

        ax2 = ax.twinx()
        l4 = ax2.plot(dates, Q, '-', color='grey', alpha=0.3, label='Discharge (Q)', linewidth=1)
        ax2.set_ylabel('Discharge')
        ax2.set_ylim(bottom=0)

        lines = l1 + l3 + l4
        if l_env:
            import matplotlib.patches as mpatches
            pi_val = (data.uncertainty_options or {}).get('prediction_interval', 90.0)
            proxy = mpatches.Patch(color='green', alpha=0.3, label=f'{pi_val:g}% Prediction Interval')
            lines.append(proxy)

        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower left', fontsize='small')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(data.folder, "forward_projection.png"), dpi=300)
        plt.close()
    else:
        plot_series(file_cal, "Calibration", "calibration")
        plot_series(file_cal, "Full Simulation", "full_simulation", filter_to_obs=False)
        plot_series(file_val, "Validation", "validation")
