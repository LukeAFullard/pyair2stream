import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def analyze_timeseries(df, output_plot_path=None, output_summary_path=None, gap_tolerant=True, min_segment_days=30):
    """
    Analyzes a timeseries dataframe for modeling suitability.
    Identifies missing data, contiguous segments, and generates a plot and summary report.

    Args:
        df (pd.DataFrame): Dataframe with columns Date, T_air, T_water, Discharge
        output_plot_path (str, optional): Path to save the output plot
        output_summary_path (str, optional): Path to save the summary text
        gap_tolerant (bool): If true, looks for segments. If false, model needs one continuous block.
        min_segment_days (int): Minimum length of a valid contiguous segment.

    Returns:
        dict: A dictionary containing summary statistics.
    """
    df = df.copy()

    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Standardize missing values (-999.0 to NaN for analysis)
    for col in ['T_air', 'T_water', 'Discharge']:
        if col in df.columns:
            df.loc[df[col] == -999.0, col] = np.nan
            if col == 'Discharge':
                # Treat zero or negative discharge as missing data to prevent mathematical errors in the ODE
                df.loc[df[col] <= 0.0, col] = np.nan

    # Calculate missing percentages
    total_days = len(df)
    missing_stats = {}
    for col in ['T_air', 'T_water', 'Discharge']:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / total_days) * 100 if total_days > 0 else 0
            }

    # Identify segments based on forcing data (T_air and Discharge if present)
    # pyair2stream requires valid T_air (and Discharge if used). T_water can have internal gaps (-999.0).
    df['is_valid_forcing'] = ~df['T_air'].isna()
    if 'Discharge' in df.columns:
        df['is_valid_forcing'] = df['is_valid_forcing'] & (~df['Discharge'].isna())

    # Identify contiguous blocks
    # A change in 'is_valid_forcing' marks a new block
    df['block_id'] = (df['is_valid_forcing'] != df['is_valid_forcing'].shift(1)).cumsum()

    blocks = df.groupby('block_id').agg(
        is_valid=('is_valid_forcing', 'first'),
        start_date=('Date', 'min'),
        end_date=('Date', 'max'),
        length=('Date', 'count')
    ).reset_index()

    # Classify blocks
    valid_segments = []
    too_short_segments = []
    gap_segments = []

    for _, row in blocks.iterrows():
        if not row['is_valid']:
            gap_segments.append(row)
        else:
            if row['length'] >= min_segment_days:
                valid_segments.append(row)
            else:
                too_short_segments.append(row)

    # Calculate how many valid T_water observations are inside the valid segments
    total_valid_obs = 0
    if 'T_water' in df.columns:
        for row in valid_segments:
            mask = (df['Date'] >= row['start_date']) & (df['Date'] <= row['end_date'])
            total_valid_obs += df.loc[mask, 'T_water'].notna().sum()

    summary = {
        'total_days': total_days,
        'start_date': df['Date'].min(),
        'end_date': df['Date'].max(),
        'missing_stats': missing_stats,
        'valid_segments_count': len(valid_segments),
        'too_short_segments_count': len(too_short_segments),
        'gap_segments_count': len(gap_segments),
        'total_valid_days': sum(row['length'] for row in valid_segments),
        'total_valid_T_water_obs': total_valid_obs
    }

    # Generate Report Text
    report_lines = [
        "========================================",
        "      Timeseries Pre-Analysis Report      ",
        "========================================",
        f"Total Range: {summary['start_date'].strftime('%Y-%m-%d')} to {summary['end_date'].strftime('%Y-%m-%d')} ({total_days} days)",
        f"Gap Tolerant Mode: {'Enabled' if gap_tolerant else 'Disabled'} (Min Segment: {min_segment_days} days)",
        "",
        "--- Missing Data ---"
    ]

    for col, stat in missing_stats.items():
        report_lines.append(f"{col}: {stat['missing_count']} days missing ({stat['missing_percentage']:.1f}%)")

    report_lines.extend([
        "",
        "--- Segments Analysis (Based on T_air and Discharge) ---",
        f"Total Valid Segments: {len(valid_segments)} (Totaling {summary['total_valid_days']} days)",
        f"Segments Too Short (< {min_segment_days} days): {len(too_short_segments)}",
        f"Gap Periods: {len(gap_segments)}",
        "",
        "--- Calibration Potential ---",
        f"Total Valid T_water Observations in Valid Segments: {summary['total_valid_T_water_obs']}",
    ])

    if summary['total_valid_T_water_obs'] > 0:
        report_lines.extend([
            "Data points per parameter for different model versions:",
            f"  - Version 3 (3 parameters): {summary['total_valid_T_water_obs'] / 3:.1f} pts/param",
            f"  - Version 4 (4 parameters): {summary['total_valid_T_water_obs'] / 4:.1f} pts/param",
            f"  - Version 5 (5 parameters): {summary['total_valid_T_water_obs'] / 5:.1f} pts/param",
            f"  - Version 7 (7 parameters): {summary['total_valid_T_water_obs'] / 7:.1f} pts/param",
            f"  - Version 8 (8 parameters): {summary['total_valid_T_water_obs'] / 8:.1f} pts/param"
        ])

    if len(valid_segments) > 0:
        report_lines.append("\nValid Segments List:")
        for idx, row in enumerate(valid_segments):
            report_lines.append(f"  {idx+1}. {row['start_date'].strftime('%Y-%m-%d')} to {row['end_date'].strftime('%Y-%m-%d')} ({row['length']} days)")
    else:
        report_lines.append("\nWARNING: No valid segments found. The model cannot run on this dataset.")

    report_text = "\n".join(report_lines)

    if output_summary_path:
        with open(output_summary_path, 'w') as f:
            f.write(report_text)

    # Generate Plot
    if output_plot_path:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Timeseries Pre-Analysis', fontsize=16)

        variables = [('T_air', 'Air Temperature (°C)', 'tab:orange'),
                     ('T_water', 'Water Temperature (°C)', 'tab:blue'),
                     ('Discharge', 'Discharge', 'tab:green')]

        for ax, (col, ylabel, color) in zip(axes, variables):
            if col in df.columns:
                ax.plot(df['Date'], df[col], color=color, linewidth=1)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)

            # Add background shading for segments
            for row in valid_segments:
                ax.axvspan(row['start_date'], row['end_date'], color='green', alpha=0.1)
            for row in too_short_segments:
                ax.axvspan(row['start_date'], row['end_date'], color='yellow', alpha=0.2)
            for row in gap_segments:
                ax.axvspan(row['start_date'], row['end_date'], color='red', alpha=0.1)

        axes[-1].set_xlabel('Date')

        # Format x-axis dates
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')

        # Custom legend for shading
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.2, label=f'Valid Segment (>= {min_segment_days}d)'),
            Patch(facecolor='yellow', alpha=0.3, label=f'Too Short (< {min_segment_days}d)'),
            Patch(facecolor='red', alpha=0.2, label='Missing Forcing Data (Gap)')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.95))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    return summary, report_text
