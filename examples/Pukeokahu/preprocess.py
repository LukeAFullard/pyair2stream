import os
from pyair2stream.preprocessing import merge_timeseries

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

if __name__ == '__main__':
    main()