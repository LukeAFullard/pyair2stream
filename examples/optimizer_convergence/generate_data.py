import pandas as pd
import numpy as np
import os

# We generate 5 years of daily data (1 year for warmup, 4 years for actual calibration/validation)
np.random.seed(42)
dates = pd.date_range(start="2018-01-01", periods=365 * 5, freq='D')

# Time fraction for seasonal signals
tt = (dates.dayofyear - 1) / 365.0

# Generate synthetic realistic data
# T_air: Seasonal sine wave + some random noise
T_air = 15.0 + 12.0 * np.sin(2.0 * np.pi * (tt - 0.25)) + np.random.normal(0, 2.0, len(dates))

# Discharge: Base flow + some seasonal peaks
Discharge = 20.0 + 10.0 * np.sin(2.0 * np.pi * (tt - 0.1)) + np.random.lognormal(mean=0, sigma=0.5, size=len(dates))

# T_water: Dampened and delayed response to air temperature
T_water = 12.0 + 8.0 * np.sin(2.0 * np.pi * (tt - 0.3)) + np.random.normal(0, 1.0, len(dates))

# Convert to numpy array safely
T_water = np.array(T_water)
missing_indices = np.random.choice(len(dates), size=30, replace=False)
T_water[missing_indices] = -999.0

# Ensure T_water doesn't drop significantly below 0 (ice cover threshold logic)
T_water = np.where(T_water < 0.0, 0.0, T_water)
T_water[missing_indices] = -999.0 # Restore missing markers

df = pd.DataFrame({
    'Date': dates,
    'T_air': np.round(T_air, 2),
    'T_water': np.round(T_water, 2),
    'Discharge': np.round(Discharge, 2)
})

# Output to CSV
output_path = os.path.join(os.path.dirname(__file__), 'synthetic_data.csv')
df.to_csv(output_path, index=False)
print(f"Successfully generated {output_path}")
