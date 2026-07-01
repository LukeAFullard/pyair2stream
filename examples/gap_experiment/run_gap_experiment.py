import pandas as pd
import numpy as np
import subprocess
import os
import concurrent.futures
from pyair2stream.pre_analysis import analyze_timeseries

# Configuration
BASELINE_DATA = 'examples/validation/Switzerland/DAV_2327_cc.csv'
VAL_DATA = 'examples/validation/Switzerland/DAV_2327_cv.csv'
OUT_DIR = 'examples/gap_experiment/output'
STATION = 'DAV'

os.makedirs(OUT_DIR, exist_ok=True)

def create_de_config(scenario_name, in_file, val_file):
    config_file = f'{OUT_DIR}/config_DE_{scenario_name}.yaml'
    with open(config_file, 'w') as f:
        f.write(f"""project_name: "gap_experiment"
station_name: "{scenario_name}"
series: "c"
time_resolution: "1d"
version: 8
Tice_cover: 0.0
objective_function: "NSE"
integrator: "RK4"
run_mode: "DE"
mineff_index: -999.0
gap_tolerant: true

paths:
  input_data: "{in_file}"
  validation_data: "{val_file}"
  output_dir: "{OUT_DIR}"

optimization:
  n_runs: 3000
  n_particles: 100

parameter_bounds:
  min: [-5.0, -5.0, -5.0, -1.0, 0.0, 0.0, 0.0, -1.0]
  max: [15.0, 1.5, 5.0, 1.0, 20.0, 10.0, 1.0, 5.0]
""")
    return config_file

def apply_gaps(df, scenario):
    df_new = df.copy()
    n = len(df_new)
    if scenario == 'baseline':
        pass
    elif scenario == 'few_short':
        for _ in range(5):
            start = np.random.randint(0, n - 3)
            df_new.loc[start:start+2, 'T_air'] = np.nan
    elif scenario == 'many_short':
        for _ in range(20):
            start = np.random.randint(0, n - 3)
            df_new.loc[start:start+2, 'T_air'] = np.nan
    elif scenario == 'few_long':
        for _ in range(2):
            start = np.random.randint(0, n - 30)
            df_new.loc[start:start+29, 'T_air'] = np.nan
    elif scenario == 'many_long':
        for _ in range(5):
            start = np.random.randint(0, n - 30)
            df_new.loc[start:start+29, 'T_air'] = np.nan
    elif scenario == 'random':
        drop_indices = np.random.choice(df_new.index, size=int(0.05 * n), replace=False)
        df_new.loc[drop_indices, 'T_air'] = np.nan
    return df_new

def get_results(scenario_name):
    out_file = f'{OUT_DIR}/1_DE_NSE_{scenario_name}_c_1d.out'
    with open(out_file, 'r') as f:
        lines = f.readlines()
        params = [float(x) for x in lines[0].split()]
        nse = float(lines[1].strip())

    gof_csv = f'{OUT_DIR}/goodness_of_fit_calibration_DE_NSE_{scenario_name}.csv'
    df = pd.read_csv(gof_csv)
    r2_row = df[df['Metric'] == 'R2']
    final_r2 = float(r2_row['Value'].iloc[0]) if not r2_row.empty else -999.0
    return params, nse, final_r2

def run_scenario(scenario):
    df_base = pd.read_csv(BASELINE_DATA)
    # Different seed for each to ensure random numbers are distinct if needed,
    # though we process df_new logic linearly here:
    np.random.seed(42 + hash(scenario) % 100)
    df_scenario = apply_gaps(df_base, scenario)
    missing_pct = df_scenario['T_air'].isna().sum() / len(df_scenario) * 100

    scenario_file = f'{OUT_DIR}/{STATION}_{scenario}_cc.csv'
    df_scenario.to_csv(scenario_file, index=False)

    config_file = create_de_config(scenario, scenario_file, VAL_DATA)


    # Run pre-analysis
    pre_plot_path = f'{OUT_DIR}/{scenario}_pre_analysis.png'
    analyze_timeseries(
        df_scenario,
        output_plot_path=pre_plot_path,
        gap_tolerant=True
    )

    # Run pyair2stream

    subprocess.run(['python', '-m', 'pyair2stream.main', '--config', config_file], check=True, capture_output=True)

    params, nse, r2 = get_results(scenario)
    return scenario, {
        'missing_pct': missing_pct,
        'params': params,
        'nse': nse,
        'r2': r2
    }

scenarios = ['baseline', 'few_short', 'many_short', 'few_long', 'many_long', 'random']
results = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(run_scenario, s): s for s in scenarios}
    for future in concurrent.futures.as_completed(futures):
        scenario = futures[future]
        try:
            scenario, result = future.result()
            results[scenario] = result
            print(f"Completed {scenario}")
        except Exception as exc:
            print(f'{scenario} generated an exception: {exc}')

# Write Report
report_path = 'examples/gap_experiment/report.md'
with open(report_path, 'w') as f:
    f.write("# Gap Analysis Experiment Results\n\n")
    f.write("This report details the stability of parameter values when gaps are introduced into the `T_air` forcing data.\n\n")
    f.write("## Method\n")
    f.write("A baseline Differential Evolution (DE) optimization was run (3000 iterations, 100 particles) using the complete DAV dataset from Switzerland. Various types of gaps were then systematically introduced to the `T_air` column (`NaN` injection), and the DE calibration was repeated to observe how equifinality and goodness-of-fit reacted to missing data.\n\n")

    f.write("## Results\n\n")
    f.write("| Scenario | Missing T_air (%) | NSE | R2 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

    for s in scenarios:
        r = results[s]
        p_str = " | ".join([f"{x:.3f}" for x in r['params']])
        f.write(f"| **{s}** | {r['missing_pct']:.2f}% | {r['nse']:.4f} | {r['r2']:.4f} | {p_str} |\n")


    f.write("\n## Pre-analysis Timelines\n\n")
    for s in scenarios:
        f.write(f"### {s}\n")
        f.write(f"![{s} Pre-analysis](output/{s}_pre_analysis.png)\n\n")

print(f"\nDone! Report written to {report_path}")
