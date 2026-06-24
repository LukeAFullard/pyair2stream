import os
import subprocess
import yaml
import glob
import pandas as pd
from contextlib import contextmanager

@contextmanager
def change_dir(destination):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)

def run_calibration(config_file):
    print(f"--- Calibrating with {config_file} ---")
    subprocess.run(["python", "-m", "pyair2stream.main", "--config", config_file], env={**os.environ, "PYTHONPATH": "../../"}, check=True)

def extract_best_parameters(output_dir, station_name):
    # Output file: 1_PSO_NSE_{station_name}_c_1d.out
    pattern = os.path.join(output_dir, f"1_PSO_NSE_{station_name}_c_1d.out")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Could not find parameter file {pattern}")

    with open(files[0], 'r') as f:
        # First line contains the parameters
        line = f.readline().strip()
        params = [float(x) for x in line.split()]

        # Second line contains the apparent calibration NSE
        ei = float(f.readline().strip())
        return params, ei

def create_forward_config(station_name, params):
    config = {
        "project_name": "gap_tolerance_experiment",
        "station_name": f"{station_name}_FWD",
        "series": "c",
        "time_resolution": "1d",
        "version": 8,
        "Tice_cover": 0.0,
        "objective_function": "NSE",
        "integrator": "RK4",
        "run_mode": "FORWARD",
        "gap_tolerant": False,
        "paths": {
            "input_data": "data_complete.csv",
            "output_dir": f"output_forward_{station_name.lower()}"
        },
        "parameters_forward": params
    }
    filename = f"forward_{station_name.lower()}.yaml"
    with open(filename, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return filename

def get_forward_nse(output_dir, station_name):
    pattern = os.path.join(output_dir, f"1_FORWARD_NSE_{station_name}_FWD_c_1d.out")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Could not find parameter file {pattern}")

    with open(files[0], 'r') as f:
        # Second line is the NSE
        f.readline()
        ei = float(f.readline().strip())
        return ei

def main():
    with change_dir("examples/gap_tolerance"):
        scenarios = [
            ("Complete", "calib_complete.yaml", "output_complete"),
            ("1Gap", "calib_1gap.yaml", "output_1gap"),
            ("2Gaps", "calib_2gaps.yaml", "output_2gaps"),
            ("3Gaps", "calib_3gaps.yaml", "output_3gaps")
        ]

        results = []

        for name, config, out_dir in scenarios:
            run_calibration(config)
            params, calib_nse = extract_best_parameters(out_dir, name)

            fwd_config = create_forward_config(name, params)
            print(f"--- Running Forward Mode for {name} ---")
            subprocess.run(["python", "-m", "pyair2stream.main", "--config", fwd_config], env={**os.environ, "PYTHONPATH": "../../"}, check=True)

            true_nse = get_forward_nse(f"output_forward_{name.lower()}", name)

            results.append({
                "Scenario": name,
                "Calibration_NSE (Apparent)": round(calib_nse, 4),
                "True_NSE (Complete Data)": round(true_nse, 4)
            })

        print("\n" + "="*50)
        print("EXPERIMENT RESULTS")
        print("="*50)
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))
        print("="*50)

if __name__ == "__main__":
    main()
