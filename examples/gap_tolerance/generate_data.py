import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 5 years of daily data (2018-2022)
dates = pd.date_range(start="2018-01-01", end="2022-12-31", freq="D")
n_days = len(dates)

# Generate synthetic sinusoidal air temperature
day_of_year = dates.dayofyear
T_air = 12.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0) + np.random.normal(0, 2.0, n_days)

# Generate synthetic discharge (base flow + random spikes + seasonal)
Discharge = 15.0 + 5.0 * np.sin(2 * np.pi * (day_of_year - 50) / 365.0) + np.random.exponential(5.0, n_days)

# Generate synthetic water temperature (smoothed, damped response to air temp)
T_water = 10.0 + 7.0 * np.sin(2 * np.pi * (day_of_year - 120) / 365.0) + np.random.normal(0, 0.5, n_days)
T_water = np.maximum(T_water, 0.0)

# Create dataframe
df = pd.DataFrame({
    'Date': dates,
    'T_air': T_air,
    'T_water': T_water,
    'Discharge': Discharge
})

# Save complete dataset
df.to_csv("examples/gap_tolerance/data_complete.csv", index=False)

# 1-Gap Dataset (Winter 2019)
df_1gap = df.copy()
mask1 = (df_1gap['Date'] >= '2019-01-01') & (df_1gap['Date'] <= '2019-03-15')
df_1gap.loc[mask1, ['T_air', 'Discharge']] = np.nan
df_1gap.to_csv("examples/gap_tolerance/data_1gap.csv", index=False)

# 2-Gaps Dataset (Winter 2019 + Spring 2020 Flood)
df_2gaps = df_1gap.copy()
mask2 = (df_2gaps['Date'] >= '2020-04-10') & (df_2gaps['Date'] <= '2020-05-20')
df_2gaps.loc[mask2, ['T_air', 'Discharge', 'T_water']] = np.nan
df_2gaps.to_csv("examples/gap_tolerance/data_2gaps.csv", index=False)

# 3-Gaps Dataset (Winter 2019 + Spring 2020 + Autumn 2021 + scattered T_water)
df_3gaps = df_2gaps.copy()
mask3 = (df_3gaps['Date'] >= '2021-10-01') & (df_3gaps['Date'] <= '2021-10-07')
df_3gaps.loc[mask3, 'Discharge'] = np.nan
random_missing = np.random.choice(df_3gaps.index, size=int(0.05 * n_days), replace=False)
df_3gaps.loc[random_missing, 'T_water'] = np.nan
df_3gaps.to_csv("examples/gap_tolerance/data_3gaps.csv", index=False)

print("Generated 4 datasets in examples/gap_tolerance/ (complete, 1gap, 2gaps, 3gaps).")
