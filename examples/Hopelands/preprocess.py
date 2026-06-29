import pandas as pd
import os
import numpy as np
from pyair2stream.preprocessing import merge_timeseries
from pyair2stream.pre_analysis import analyze_timeseries

def main():
    base_dir = 'examples/Hopelands'

    # Pre-process air temperature to convert Kelvin to Celsius
    air_temp_raw = os.path.join(base_dir, 'daily_temperature_station_id_32504.csv')
    print(f"Converting air temperature from Kelvin to Celsius in {air_temp_raw}...")
    df_air = pd.read_csv(air_temp_raw)
    df_air['temperature'] = df_air['temperature'] - 273.15
    air_temp_processed = os.path.join(base_dir, 'daily_temperature_celsius.csv')
    df_air.to_csv(air_temp_processed, index=False)

    # Pre-process water temperature to exclude outliers close to zero
    water_temp_raw = os.path.join(base_dir, 'WT_Mean_Hopelands.csv')
    print(f"Excluding water temperature outliers (< 0.1°C) in {water_temp_raw}...")
    df_water = pd.read_csv(water_temp_raw)
    wt_col = 'Water Temperature Daily Mean'
    # Set outliers to NaN
    df_water.loc[df_water[wt_col] < 0.1, wt_col] = np.nan
    water_temp_processed = os.path.join(base_dir, 'WT_Mean_Hopelands_processed.csv')
    df_water.to_csv(water_temp_processed, index=False)

    configs = [
        {
            'file_path': air_temp_processed,
            'date_col': 'time',
            'value_col': 'temperature',
            'standard_col_name': 'T_air'
        },
        {
            'file_path': water_temp_processed,
            'date_col': 'Time',
            'value_col': 'Water Temperature Daily Mean',
            'standard_col_name': 'T_water',
            'date_format': '%d/%m/%Y %H:%M'
        },
        {
            'file_path': os.path.join(base_dir, 'daily_flow_station_id_32504.csv'),
            'date_col': 'time',
            'value_col': 'flow',
            'standard_col_name': 'Discharge'
        }
    ]

    output_path = os.path.join(base_dir, 'Hopelands_merged.csv')

    print("Merging time series...")
    merged_df = merge_timeseries(configs, output_file=output_path)
    print(f"Merged data saved to {output_path}")

    # Run Pre-Analysis
    print("\nRunning pre-analysis...")
    plot_path = os.path.join(base_dir, 'pre_analysis_report.png')
    summary_path = os.path.join(base_dir, 'pre_analysis_summary.txt')

    summary, report_text = analyze_timeseries(
        merged_df,
        output_plot_path=plot_path,
        output_summary_path=summary_path,
        gap_tolerant=True,
        min_segment_days=30
    )

    print(report_text)
    print(f"\nPre-analysis plot saved to {plot_path}")
    print(f"Pre-analysis summary saved to {summary_path}")

    # Clean up temporary files
    if os.path.exists(air_temp_processed):
        os.remove(air_temp_processed)
    if os.path.exists(water_temp_processed):
        os.remove(water_temp_processed)

if __name__ == '__main__':
    main()
