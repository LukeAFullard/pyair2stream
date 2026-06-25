import os
from pyair2stream.preprocessing import merge_timeseries
from pyair2stream.pre_analysis import analyze_timeseries

def main():
    base_dir = 'examples/Pukeokahu'

    configs = [
        {
            'file_path': os.path.join(base_dir, 'Kawhatau Catchment at Upper Kawhatau.csv'),
            'date_col': 'DateTime',
            'value_col': 'Air Temperature (1.5m)',
            'standard_col_name': 'T_air'
        },
        {
            'file_path': os.path.join(base_dir, 'Water_Temperature_Pukeokahu.csv'),
            'date_col': 'Time',
            'value_col': 'Rangitikei at Pukeokahu',
            'standard_col_name': 'T_water',
            'date_format': '%d/%m/%Y %H:%M'
        },
        {
            'file_path': os.path.join(base_dir, 'Discharge_Pukeokahu.csv'),
            'date_col': 'Time',
            'value_col': 'Rangitikei at Pukeokahu',
            'standard_col_name': 'Discharge',
            'date_format': '%d/%m/%Y %H:%M'
        }
    ]

    output_path = os.path.join(base_dir, 'Pukeokahu_merged.csv')

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

if __name__ == '__main__':
    main()