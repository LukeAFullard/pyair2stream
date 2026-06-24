import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Python params from running pyair2stream
python_params = [0.305253, 0.409846, 0.470845, 0.706343, 7.966767, 5.250985, 0.581982, 1.780136]

# Read original parameters from Fortran report:
# 3.164, 0.417, 0.829, 0.340, 1.343, 5.192, 0.574, 0.883
fortran_params = [3.164, 0.417, 0.829, 0.340, 1.343, 5.192, 0.574, 0.883]

# Create forward config to run with python params
def create_forward_config(params):
    with open('examples/validation/Switzerland/config_forward.yaml', 'w') as f:
        f.write(f"""project_name: "validation"
station_name: "DAV"
series: "c"
time_resolution: "1d"
version: 8
Tice_cover: 0.0
objective_function: "NSE"
integrator: "RK4"
run_mode: "FORWARD"
parameters_forward: {params}

paths:
  input_data: "examples/validation/Switzerland/DAV_2327_cc.csv"
  output_dir: "examples/validation/Switzerland/output_forward"
""")

# run python forward
create_forward_config(python_params)
subprocess.run(['python', '-m', 'pyair2stream.main', '--config', 'examples/validation/Switzerland/config_forward.yaml'], check=True)
df_py = pd.read_csv('examples/validation/Switzerland/output_forward/2_FORWARD_NSE_DAV_cc_1d.csv')

# run fortran forward
create_forward_config(fortran_params)
subprocess.run(['python', '-m', 'pyair2stream.main', '--config', 'examples/validation/Switzerland/config_forward.yaml'], check=True)
df_for = pd.read_csv('examples/validation/Switzerland/output_forward/2_FORWARD_NSE_DAV_cc_1d.csv')

# plot
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

axes[0].plot(df_py['Twat_mod'].values, label='Python Integrator + Python Params')
axes[0].set_title('Python Parameters')
axes[0].legend()

axes[1].plot(df_for['Twat_mod'].values, label='Python Integrator + Fortran Params')
axes[1].set_title('Fortran Parameters')
axes[1].legend()

diff = df_py['Twat_mod'] - df_for['Twat_mod']
axes[2].plot(diff.values, label='Difference (Python - Fortran)', color='red')
axes[2].set_title('Difference')
axes[2].legend()

plt.tight_layout()
plt.savefig('examples/validation/Switzerland/parameter_comparison.png')
print("Plot saved to examples/validation/Switzerland/parameter_comparison.png")
