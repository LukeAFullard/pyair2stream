import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .config import CommonData

def post_process(data: CommonData, toll: float = None):
    """
    Analyzes and plots the results of the pyair2stream simulation.
    Replicates post_processing.m
    """
    if toll is None:
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

    # Set fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10

    # Output paths
    file_0 = os.path.join(data.folder, f"0_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}_{data.time_res}.csv")
    file_cal = os.path.join(data.folder, f"2_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}c_{data.time_res}.csv")
    file_val = os.path.join(data.folder, f"3_{data.runmode}_{data.fun_obj}_{data.station}_{data.series}v_{data.time_res}.csv")

    # 1. Dotty plots
    if os.path.exists(file_0):
        # Read the parameter tracking CSV
        df_0 = pd.read_csv(file_0)

        # Determine number of parameters dynamically
        n_par = len(df_0.columns) - 1

        parset = df_0.iloc[:, :-1].values
        eff = df_0.iloc[:, -1].values

        if data.fun_obj == 'RMS':
            eff = -eff
            valid_indices = np.where(eff <= toll)[0]
            if len(valid_indices) > 0:
                parset = parset[valid_indices]
                eff = eff[valid_indices]
                best_eff = np.min(eff)
                i_best = np.argmin(eff)
                plot_limits = [best_eff * 0.9, toll]
            else:
                best_eff = np.min(eff)
                i_best = np.argmin(eff)
                plot_limits = [best_eff * 0.9, np.max(eff)]
        else:
            valid_indices = np.where(eff >= toll)[0]
            if len(valid_indices) > 0:
                parset = parset[valid_indices]
                eff = eff[valid_indices]
                best_eff = np.max(eff)
                i_best = np.argmax(eff)
                plot_limits = [toll, best_eff * 1.1]
            else:
                best_eff = np.max(eff)
                i_best = np.argmax(eff)
                plot_limits = [np.min(eff), best_eff * 1.1]

        fig, axes = plt.subplots(2, 4, figsize=(18/2.54, 10/2.54))
        axes = axes.flatten()

        for i in range(8):
            if i < n_par:
                axes[i].plot(parset[:, i], eff, '.k', markersize=2)
                if len(parset) > 0:
                    axes[i].plot(parset[i_best, i], best_eff, '.', color=orange, markersize=10)

                axes[i].set_ylim(plot_limits)
                axes[i].set_xlabel(f'par{i+1}')

                if data.fun_obj == 'RMS':
                    axes[i].set_ylabel(f"{data.fun_obj} [\u00B0C]")
                else:
                    axes[i].set_ylabel(data.fun_obj)
            else:
                axes[i].axis('off')

        plt.tight_layout()
        dotty_pdf = os.path.join(data.folder, f"dottyplots_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
        dotty_png = os.path.join(data.folder, f"dottyplots_{data.runmode}_{data.fun_obj}_{data.station}.png")
        plt.savefig(dotty_pdf, dpi=300)
        plt.savefig(dotty_png, dpi=300)
        plt.close()

    # 2. Time-Series Plots
    def plot_series(file_path, title_prefix, output_name):
        if not os.path.exists(file_path):
            return

        df = pd.read_csv(file_path)

        # First 365 is a warm up year, drop it if available
        if len(df) > 365:
            df = df.iloc[365:].copy()

        # Replace sentinel values with NaN
        df.replace(-999.0, np.nan, inplace=True)

        # Create datetime index
        dates = pd.to_datetime(df[['Year', 'Month', 'Day']])

        # Calculate RMSE
        # df has columns: 'Year', 'Month', 'Day', 'Tair', 'Twat_obs', 'Twat_mod', 'Twat_obs_agg', 'Twat_mod_agg', 'Q'
        valid_mask = df['Twat_obs_agg'].notna() & df['Twat_mod_agg'].notna()
        if valid_mask.sum() > 0:
            rmse = np.sqrt(np.mean((df.loc[valid_mask, 'Twat_obs_agg'] - df.loc[valid_mask, 'Twat_mod_agg'])**2))
        else:
            rmse = np.nan

        fig, ax = plt.subplots(figsize=(18/2.54, 10/2.54))
        ax.set_title(f"{title_prefix}, RMSE={rmse:.4f}\u00B0C")

        # Plot temperatures on primary y-axis
        l1 = ax.plot(dates, df['Tair'], '.', color=light_blue, label='Air temperature', markersize=2)
        l2 = ax.plot(dates, df['Twat_obs_agg'], '.', color=blue, label='Observed water temperature', markersize=2)
        l3 = ax.plot(dates, df['Twat_mod_agg'], '.', color=orange, label='Simulated water temperature', markersize=2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature [\u00B0C]')

        # Plot discharge on secondary y-axis
        ax2 = ax.twinx()
        l4 = ax2.plot(dates, df['Q'], '-', color='grey', alpha=0.3, label='Discharge (Q)', linewidth=1)
        ax2.set_ylabel('Discharge')
        ax2.set_ylim(bottom=0) # Discharge shouldn't be negative

        # Combine legends
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower right')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        fig.autofmt_xdate()

        plt.tight_layout()
        pdf_path = os.path.join(data.folder, f"{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
        png_path = os.path.join(data.folder, f"{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.png")
        plt.savefig(pdf_path, dpi=300)
        plt.savefig(png_path, dpi=300)
        plt.close()

    plot_series(file_cal, "Calibration", "calibration")
    plot_series(file_val, "Validation", "validation")
