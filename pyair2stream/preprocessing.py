"""
Data preprocessing utilities for pyair2stream.

This module provides helper functions to read, resample, and align raw,
high-frequency observational data into the daily, uniformly-formatted CSVs
required by the air2stream core.
"""

import numpy as np
import pandas as pd

MISSING_CODE = -999.0


def _line(i: int) -> int:
    """Line of the CSV file that holds table row i (line 1 is the header)."""
    return int(i) + 2


def read_and_resample(file_path, date_col, value_col, standard_col_name, date_format=None, na_values=None,
                      min_readings_per_day=None):
    """
    Reads a CSV file, parses the date column, resamples to daily averages,
    and renames the target column to a standard name.

    It reports what it cannot use, rather than dropping it silently: rows whose time
    cannot be read (left out, with a warning), values of -999 (treated as missing),
    and days with far fewer readings than usual (their mean may not represent the
    whole day). Text that is not a number stops it with a clear message.

    Args:
        file_path (str): Path to the CSV file.
        date_col (str): The name of the date/time column in the CSV.
        value_col (str): The name of the value column in the CSV.
        standard_col_name (str): The standardized column name (e.g., 'T_air', 'Discharge').
        date_format (str, optional): The expected format of the date string.
        na_values (list of str, optional): Extra codes that mean "missing" in this file
            (for example ["n.a.", "ERR"]), as in `pandas.read_csv`.
        min_readings_per_day (int, optional): Days with fewer valid readings are left blank.

    Returns:
        pd.DataFrame: A dataframe with a standardized 'Date' column and standard_col_name,
                      aggregated to daily frequency.
    """
    df = pd.read_csv(file_path, na_values=na_values)
    for col in (date_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {file_path}. Its columns are: "
                             f"{', '.join(map(str, df.columns))}.")

    # Parse dates
    times = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
    unreadable = times.isna()
    if unreadable.any():
        i = int(np.argmax(unreadable.to_numpy()))
        shown = df[date_col].iloc[i]
        shown = "blank" if pd.isna(shown) or not str(shown).strip() else repr(str(shown).strip())
        print(f"Warning: {int(unreadable.sum())} row(s) of {file_path} have a time in column {date_col!r} that "
              f"cannot be read, and were left out (first: {shown} on line {_line(i)}).")

    # Values: text that is not a number is an error; -999 means missing.
    values = pd.to_numeric(df[value_col], errors='coerce')
    text = df[value_col].notna() & values.isna()
    if text.any():
        i = int(np.argmax(text.to_numpy()))
        raise ValueError(f"Column {value_col!r} in {file_path} has {int(text.sum())} value(s) that are not numbers "
                         f"(first: {str(df[value_col].iloc[i]).strip()!r} on line {_line(i)}). Correct them, or "
                         f"name such codes as missing with na_values (for example "
                         f"{{'na_values': [{str(df[value_col].iloc[i]).strip()!r}]}} in this file's settings).")
    n_code = int((values == MISSING_CODE).sum())
    if n_code:
        print(f"Note: {n_code} value(s) of -999 in column {value_col!r} of {file_path} are treated as missing.")
    values = values.where(values != MISSING_CODE)

    daily = pd.DataFrame({'day': times.dt.normalize(), 'value': values})[~unreadable]
    grouped = daily.groupby('day')['value']
    means, counts = grouped.mean(), grouped.count()

    if min_readings_per_day is not None:
        few = (counts < int(min_readings_per_day)) & means.notna()
        if few.any():
            print(f"Note: {int(few.sum())} day(s) of {file_path} have fewer than {int(min_readings_per_day)} "
                  f"readings and are left blank (first: {few[few].index[0].date()} with "
                  f"{int(counts[few].iloc[0])}).")
        means = means.where(~few)
    elif (counts > 0).any():
        usual = float(np.median(counts[counts > 0]))
        few = (counts > 0) & (counts < usual / 2)
        if usual > 1 and few.any():
            print(f"Warning: {int(few.sum())} day(s) of {file_path} have fewer than half the usual number of "
                  f"readings (usual: {usual:g} a day; first: {few[few].index[0].date()} with "
                  f"{int(counts[few].iloc[0])}). Their daily means may not represent the whole day. Set "
                  "min_readings_per_day to leave such days blank.")

    return pd.DataFrame({'Date': means.index.strftime('%Y-%m-%d'), standard_col_name: means.to_numpy()})


def merge_timeseries(file_configs, output_file=None, min_readings_per_day=None):
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
            - 'na_values': list of str (optional): extra codes meaning "missing"
            - 'min_readings_per_day': int (optional): overrides the argument below
        output_file (str, optional): If provided, saves the merged dataframe to this path.
        min_readings_per_day (int, optional): days with fewer valid readings are left blank
            (see read_and_resample).

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
            date_format=config.get('date_format'),
            na_values=config.get('na_values'),
            min_readings_per_day=config.get('min_readings_per_day', min_readings_per_day),
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

        # Days with no value in a column: a run needs complete air temperature and discharge.
        for col in [c for c in merged_df.columns if c != 'Date']:
            empty = merged_df[col].isna()
            if empty.any():
                print(f"Note: {col} has no value on {int(empty.sum())} of the {len(merged_df)} days from "
                      f"{min_date.date()} to {max_date.date()} (first: {merged_df['Date'][empty].iloc[0].date()}).")
        merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')

        # Ensure standard column order if all 3 are present
        standard_cols = ['Date', 'T_air', 'T_water', 'Discharge']
        existing_cols = ['Date'] + [c for c in standard_cols[1:] if c in merged_df.columns]
        other_cols = [c for c in merged_df.columns if c not in existing_cols]

        merged_df = merged_df[existing_cols + other_cols]

        if output_file:
            merged_df.to_csv(output_file, index=False)

    return merged_df
