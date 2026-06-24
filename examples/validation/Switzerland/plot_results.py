import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import glob

# Provide paths to actual generated output files.
# For example, if someone runs the program again, the outputs end up here.
cal_files = glob.glob('examples/validation/Switzerland/output_*/2_*.out')
val_files = glob.glob('examples/validation/Switzerland/output_*/3_*.out')

if not cal_files or not val_files:
    raise RuntimeError("Could not find calibration or validation outputs in examples/validation/Switzerland/output_*/")

cal_file = cal_files[0]
val_file = val_files[0]

cols = ['Year', 'Month', 'Day', 'Tair', 'Twat_obs', 'Twat_mod', 'Twat_obs_agg', 'Twat_mod_agg', 'Q']

cal_df = pd.read_csv(cal_file, sep=r'\s+', header=None, names=cols)
val_df = pd.read_csv(val_file, sep=r'\s+', header=None, names=cols)

# We just care about sequential order, replace dates with index
cal_df['index'] = range(len(cal_df))
val_df['index'] = range(len(cal_df), len(cal_df) + len(val_df))

plt.figure(figsize=(15, 6))

# Filter valid obs (not -999)
cal_valid_obs = cal_df[cal_df['Twat_obs'] != -999]
val_valid_obs = val_df[val_df['Twat_obs'] != -999]

plt.plot(cal_df['index'], cal_df['Twat_mod'], label='Model (Calibration)', color='blue', alpha=0.7)
plt.plot(cal_valid_obs['index'], cal_valid_obs['Twat_obs'], label='Observed (Calibration)', color='black', alpha=0.5, marker='.', linestyle='none')

plt.plot(val_df['index'], val_df['Twat_mod'], label='Model (Validation/Prediction)', color='red', alpha=0.7)
plt.plot(val_valid_obs['index'], val_valid_obs['Twat_obs'], label='Observed (Validation)', color='green', alpha=0.5, marker='.', linestyle='none')

plt.axvline(x=len(cal_df), color='grey', linestyle='--', label='Calibration/Validation Split')

plt.title('Air2Stream Fortran Results - Switzerland (DAV)')
plt.xlabel('Time Step (Days)')
plt.ylabel('Water Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.savefig('examples/validation/Switzerland/fortran_results.png')

print("Plot saved to examples/validation/Switzerland/fortran_results.png")
