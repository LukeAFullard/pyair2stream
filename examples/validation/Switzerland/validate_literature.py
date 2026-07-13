import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt

def create_forward_config(station, params, in_file, val_file):
    with open('examples/validation/Switzerland/config_forward.yaml', 'w') as f:
        f.write(f"""project_name: "validation"
station_name: "{station}"
series: "c"
time_resolution: "1d"
version: 8
Tice_cover: 0.0
objective_function: "NSE"
integrator: "RK4"
run_mode: "FORWARD"
parameters_forward: {params}

paths:
  input_data: "examples/validation/Switzerland/{in_file}"
  validation_data: "examples/validation/Switzerland/{val_file}"
  output_dir: "examples/validation/Switzerland/output_forward"
""")

def create_pso_config(station, in_file, val_file):
    with open(f'examples/validation/Switzerland/config_PSO_{station}.yaml', 'w') as f:
        f.write(f"""project_name: "validation"
station_name: "{station}"
series: "c"
time_resolution: "1d"
version: 8
Tice_cover: 0.0
objective_function: "NSE"
integrator: "RK4"
run_mode: "PSO"
prc: 1.0
mineff_index: -999.0

paths:
  input_data: "examples/validation/Switzerland/{in_file}"
  validation_data: "examples/validation/Switzerland/{val_file}"
  output_dir: "examples/validation/Switzerland/output_8"

optimization:
  n_runs: 3000
  n_particles: 500
  c1: 2.0
  c2: 2.0
  wmax: 0.9
  wmin: 0.4

parameter_bounds:
  min: [-5.0, -5.0, -5.0, -1.0, 0.0, 0.0, 0.0, -1.0]
  max: [15.0, 1.5, 5.0, 1.0, 20.0, 10.0, 1.0, 5.0]
""")

def create_de_config(station, in_file, val_file):
    with open(f'examples/validation/Switzerland/config_DE_{station}.yaml', 'w') as f:
        f.write(f"""project_name: "validation"
station_name: "{station}"
series: "c"
time_resolution: "1d"
version: 8
Tice_cover: 0.0
objective_function: "NSE"
integrator: "RK4"
run_mode: "DE"
mineff_index: -999.0

paths:
  input_data: "examples/validation/Switzerland/{in_file}"
  validation_data: "examples/validation/Switzerland/{val_file}"
  output_dir: "examples/validation/Switzerland/output_8"

optimization:
  n_runs: 3000
  n_particles: 500

parameter_bounds:
  min: [-5.0, -5.0, -5.0, -1.0, 0.0, 0.0, 0.0, -1.0]
  max: [15.0, 1.5, 5.0, 1.0, 20.0, 10.0, 1.0, 5.0]
""")

# Literature parameters from Table 1 of Toffolon & Piccolroaz (2015)
lit_params = {
    "MAH": [0.889, 0.649, 0.765, 0.129, 2.318, 1.536, 0.603, 0.241],
    "SIO": [0.346, 0.219, 0.178, 0.718, 7.773, 2.217, 0.529, 1.280],
    "DAV": [4.794, 0.629, 1.410, 0.270, 0.000, 4.912, 0.582, 0.637]
}

stations = [
    ("MAH", "MAH_2369_cc.csv", "MAH_2369_cv.csv"),
    ("SIO", "SIO_2011_cc.csv", "SIO_2011_cv.csv"),
    ("DAV", "DAV_2327_cc.csv", "DAV_2327_cv.csv")
]

os.makedirs('examples/validation/Switzerland/output_forward', exist_ok=True)
os.makedirs('examples/validation/Switzerland/output_8', exist_ok=True)

# 1. Run forward models to plot literature values
for station, in_file, val_file in stations:
    params = lit_params[station]
    print(f"\n================ Running Forward Model for {station} ================")
    create_forward_config(station, params, in_file, val_file)
    subprocess.run(['python', '-m', 'pyair2stream.main', '--config', 'examples/validation/Switzerland/config_forward.yaml'], check=True)

# Plotting function
def plot_results(station):
    cal_file = f'examples/validation/Switzerland/output_forward/2_FORWARD_NSE_{station}_cc_1d.csv'
    val_file = f'examples/validation/Switzerland/output_forward/3_FORWARD_NSE_{station}_cv_1d.csv'

    cal_df = pd.read_csv(cal_file)
    val_df = pd.read_csv(val_file)

    cal_df['index'] = range(len(cal_df))
    val_df['index'] = range(len(cal_df), len(cal_df) + len(val_df))

    plt.figure(figsize=(15, 6))

    cal_valid_obs = cal_df[cal_df['Twat_obs'] != -999]
    val_valid_obs = val_df[val_df['Twat_obs'] != -999]

    plt.plot(cal_df['index'], cal_df['Twat_mod'], label='Model (Calibration)', color='blue', alpha=0.7)
    plt.plot(cal_valid_obs['index'], cal_valid_obs['Twat_obs'], label='Observed (Calibration)', color='black', alpha=0.5, marker='.', linestyle='none')

    plt.plot(val_df['index'], val_df['Twat_mod'], label='Model (Validation)', color='red', alpha=0.7)
    plt.plot(val_valid_obs['index'], val_valid_obs['Twat_obs'], label='Observed (Validation)', color='green', alpha=0.5, marker='.', linestyle='none')

    plt.axvline(x=len(cal_df), color='grey', linestyle='--', label='Calibration/Validation Split')

    plt.title(f'Air2Stream Python Forward Results - Switzerland ({station})')
    plt.xlabel('Time Step (Days)')
    plt.ylabel('Water Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'examples/validation/Switzerland/forward_{station}.png')
    print(f"Plot saved to examples/validation/Switzerland/forward_{station}.png")

for station, _, _ in stations:
    plot_results(station)

# 2. Run PSO and DE to optimize and compare parameters against literature
with open('examples/validation/Switzerland/README.md', 'a') as report:
    report.write("\n\n### Extended Evaluation: Differential Evolution (DE) vs PSO vs Literature\n")
    report.write("Following the initial evaluation, a high-intensity Differential Evolution pass (500 particles, 3000 runs) was executed to ascertain whether a stronger global search bounds the parameters closer to literature values, and to evaluate absolute convergence limits of the model equifinality.\n\n")

    for station, in_file, val_file in stations:
        # PSO
        print(f"\n================ Running PSO Model for {station} ================")
        create_pso_config(station, in_file, val_file)
        subprocess.run(['python', '-m', 'pyair2stream.main', '--config', f'examples/validation/Switzerland/config_PSO_{station}.yaml'], check=True)

        pso_out_file = f'examples/validation/Switzerland/output_8/1_PSO_NSE_{station}_c_1d.out'
        with open(pso_out_file, 'r') as f:
            lines = f.readlines()
            pso_params = [float(x) for x in lines[0].split()]
            pso_nse = float(lines[1].strip())

        # DE
        print(f"\n================ Running DE Model for {station} ================")
        create_de_config(station, in_file, val_file)
        subprocess.run(['python', '-m', 'pyair2stream.main', '--config', f'examples/validation/Switzerland/config_DE_{station}.yaml'], check=True)

        de_out_file = f'examples/validation/Switzerland/output_8/1_DE_NSE_{station}_c_1d.out'
        with open(de_out_file, 'r') as f:
            lines = f.readlines()
            de_params = [float(x) for x in lines[0].split()]
            de_nse = float(lines[1].strip())

        # FORWARD (Literature NSE)
        fwd_out_file = f'examples/validation/Switzerland/output_forward/1_FORWARD_NSE_{station}_c_1d.out'
        with open(fwd_out_file, 'r') as f:
            lines = f.readlines()
            fwd_nse = float(lines[1].strip())

        print(f"\n--- {station} Parameter Comparison ---")
        print(f"Literature (NSE: {fwd_nse:.4f}): {lit_params[station]}")
        print(f"PSO        (NSE: {pso_nse:.4f}): {pso_params}")
        print(f"DE         (NSE: {de_nse:.4f}): {de_params}")

        report.write(f"**{station} Dataset:**\n")
        report.write("| Source | NSE | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |\n")
        report.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        report.write(f"| **Literature** | {fwd_nse:.4f} | {lit_params[station][0]:.3f} | {lit_params[station][1]:.3f} | {lit_params[station][2]:.3f} | {lit_params[station][3]:.3f} | {lit_params[station][4]:.3f} | {lit_params[station][5]:.3f} | {lit_params[station][6]:.3f} | {lit_params[station][7]:.3f} |\n")
        report.write(f"| **Python PSO** | {pso_nse:.4f} | {pso_params[0]:.3f} | {pso_params[1]:.3f} | {pso_params[2]:.3f} | {pso_params[3]:.3f} | {pso_params[4]:.3f} | {pso_params[5]:.3f} | {pso_params[6]:.3f} | {pso_params[7]:.3f} |\n")
        report.write(f"| **Python DE**  | {de_nse:.4f} | {de_params[0]:.3f} | {de_params[1]:.3f} | {de_params[2]:.3f} | {de_params[3]:.3f} | {de_params[4]:.3f} | {de_params[5]:.3f} | {de_params[6]:.3f} | {de_params[7]:.3f} |\n\n")
