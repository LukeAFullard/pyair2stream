"""
Data preprocessing utilities for pyair2stream.

This module provides helper functions to read, resample, and align raw,
high-frequency observational data into the daily, uniformly-formatted CSVs
required by the air2stream core.
"""

import pandas as pd

def read_and_resample(file_path, date_col, value_col, standard_col_name, date_format=None):
    """
    Reads a CSV file, parses the date column, resamples to daily averages,
    and renames the target column to a standard name.

    Args:
        file_path (str): Path to the CSV file.
        date_col (str): The name of the date/time column in the CSV.
        value_col (str): The name of the value column in the CSV.
        standard_col_name (str): The standardized column name (e.g., 'T_air', 'Discharge').
        date_format (str, optional): The expected format of the date string.

    Returns:
        pd.DataFrame: A dataframe with a standardized 'Date' column and standard_col_name,
                      aggregated to daily frequency.
    """
    df = pd.read_csv(file_path)

    # Parse dates
    if date_format:
        df['Date'] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
    else:
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')

    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['Date'])

    # Rename and select columns
    df = df.rename(columns={value_col: standard_col_name})
    df = df[['Date', standard_col_name]]

    # Group by Date and calculate daily mean
    df_daily = df.groupby('Date')[standard_col_name].mean().reset_index()

    return df_daily


def merge_timeseries(file_configs, output_file=None):
    """
    Reads multiple time series from different files using read_and_resample,
    and outer joins them on the standard 'Date' column.

    Args:
        file_configs (list of dict): A list where each dict contains:
            - 'file_path': str
            - 'date_col': str
            - 'value_col': str
            - 'standard_col_name': str
            - 'date_format': str (optional)
        output_file (str, optional): If provided, saves the merged dataframe to this path.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    merged_df = None

    for config in file_configs:
        df = read_and_resample(
            file_path=config['file_path'],
            date_col=config['date_col'],
            value_col=config['value_col'],
            standard_col_name=config['standard_col_name'],
            date_format=config.get('date_format')
        )

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')

    # Sort by date
    if merged_df is not None:
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        merged_df = merged_df.sort_values('Date')

        # Reindex to a complete daily calendar to expose completely missing days as NaN rows
        min_date = merged_df['Date'].min()
        max_date = merged_df['Date'].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        merged_df = merged_df.set_index('Date').reindex(full_date_range).reset_index()
        merged_df = merged_df.rename(columns={'index': 'Date'})
        merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')

        # Ensure standard column order if all 3 are present
        standard_cols = ['Date', 'T_air', 'T_water', 'Discharge']
        existing_cols = ['Date'] + [c for c in standard_cols[1:] if c in merged_df.columns]
        other_cols = [c for c in merged_df.columns if c not in existing_cols]

        merged_df = merged_df[existing_cols + other_cols]

        if output_file:
            merged_df.to_csv(output_file, index=False)

    return merged_df
