import pandas as pd
import numpy as np
import os

out_dir = os.path.dirname(__file__)

# Load historical data
hist_path = os.path.join(out_dir, 'synthetic_data.csv')
df_hist = pd.read_csv(hist_path)

# Ensure Date is parsed correctly to find the end
df_hist['Date'] = pd.to_datetime(df_hist['Date'])
last_date = df_hist['Date'].iloc[-1]

# Generate 1 extra year of daily data for projection
np.random.seed(123) # Different seed for projection year noise
dates_proj = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')
tt = (dates_proj.dayofyear - 1) / 365.0

# Base realistic data for the projection year
base_T_air = 15.0 + 12.0 * np.sin(2.0 * np.pi * (tt - 0.25)) + np.random.normal(0, 1.0, len(dates_proj))
base_Discharge = 20.0 + 10.0 * np.sin(2.0 * np.pi * (tt - 0.1)) + np.random.lognormal(mean=0, sigma=0.5, size=len(dates_proj))

# We don't have true "observed" water temperatures for future. We fill it entirely with -999.0
missing_T_water = np.full(len(dates_proj), -999.0)

# Protect against Index mutable operations by coercing to raw arrays
base_T_air = np.array(base_T_air)
base_Discharge = np.array(base_Discharge)
summer_mask = np.array((dates_proj.dayofyear >= 150) & (dates_proj.dayofyear <= 250))

# SCENARIO 1: Hot Summer
# Increase air temperature by 5 degrees between day 150 (late May) and 250 (early Sept)
hot_T_air = base_T_air.copy()
hot_T_air[summer_mask] += 5.0

df_hot_proj = pd.DataFrame({
    'Date': dates_proj,
    'T_air': np.round(hot_T_air, 2),
    'T_water': missing_T_water,
    'Discharge': np.round(base_Discharge, 2)
})

# SCENARIO 2: Low Flow Summer
# Reduce discharge by 75% between day 150 and 250
low_Discharge = base_Discharge.copy()
low_Discharge[summer_mask] *= 0.25

df_low_flow_proj = pd.DataFrame({
    'Date': dates_proj,
    'T_air': np.round(base_T_air, 2),
    'T_water': missing_T_water,
    'Discharge': np.round(low_Discharge, 2)
})

# Concatenate historical with projections
df_hot_full = pd.concat([df_hist, df_hot_proj], ignore_index=True)
df_low_full = pd.concat([df_hist, df_low_flow_proj], ignore_index=True)

path_hot = os.path.join(out_dir, 'projection_hot_summer.csv')
path_low = os.path.join(out_dir, 'projection_low_flow.csv')

df_hot_full.to_csv(path_hot, index=False)
df_low_full.to_csv(path_low, index=False)

print(f"Successfully generated {path_hot} and {path_low} with historical context.")
