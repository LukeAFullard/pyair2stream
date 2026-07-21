import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyair2stream.main import main
from pyair2stream.config import CommonData
import sys

def create_config(mode, out_dir):
    config = {
        'project_name': f'mcmc_comparison_{mode}',
        'station_name': 'Validation_Station',
        'time_resolution': '1d',
        'version': 8,
        'objective_function': 'NSE',
        'integrator': 'RK4',
        'run_mode': mode,
        'paths': {
            'input_data': '../../examples/validation/Switzerland/DAV_2327_cc.csv',
            'output_dir': out_dir
        },
        'cross_validation': {
            'enabled': True if mode == 'DE-CV-MCMC' else False,
            'unit': 'year',
            'min_train_years': 0,
            'skip_first_year': True,
            'min_valid_obs': 10,
            'optimizer_overrides': {
                'n_run': 10,
                'n_particles': 10
            }
        },
        'optimization': {
            'n_run': 20,
            'n_particles': 20,
            'mcmc_walkers': 32,
            'mcmc_steps': 100
        },
        'parameter_bounds': {
            'min': [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            'max': [5.0, 2.0, 2.0, 2.0, 30.0, 15.0, 1.0, 2.0]
        },
        'uncertainty_options': {
            'noise_model': 'iid',
            'prediction_interval': 95.0
        }
    }
    config_path = f'config_{mode}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path

if __name__ == '__main__':
    # First, make sure the validation data is converted to csv
    os.system("cd ../.. && PYTHONPATH=. poetry run python examples/validation/Switzerland/convert_txt_to_csv.py")

    os.makedirs('output_DE_MCMC', exist_ok=True)
    os.makedirs('output_DE_CV_MCMC', exist_ok=True)

    config_de = create_config('DE-MCMC', 'output_DE_MCMC')
    config_cv = create_config('DE-CV-MCMC', 'output_DE_CV_MCMC')

    # Run DE-MCMC
    print("Running DE-MCMC...")
    sys.argv = ['main.py', '--config', config_de]
    main()

    # Run DE-CV-MCMC
    print("\nRunning DE-CV-MCMC...")
    sys.argv = ['main.py', '--config', config_cv]
    main()

    # Generate side-by-side plots
    print("Generating comparison plots...")
    de_chain_file = 'output_DE_MCMC/MCMC_chain_Validation_Station_series_1d.csv'
    cv_chain_file = 'output_DE_CV_MCMC/MCMC_chain_Validation_Station_series_1d.csv'

    if os.path.exists(de_chain_file) and os.path.exists(cv_chain_file):
        df_de = pd.read_csv(de_chain_file)
        df_cv = pd.read_csv(cv_chain_file)

        n_params = len(df_de.columns)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params))
        if n_params == 1:
            axes = [axes]

        for i, col in enumerate(df_de.columns):
            axes[i].hist(df_de[col], bins=30, alpha=0.5, label='DE-MCMC', density=True)
            axes[i].hist(df_cv[col], bins=30, alpha=0.5, label='DE-CV-MCMC', density=True)
            axes[i].set_title(col)
            axes[i].legend()

        plt.tight_layout()
        plt.savefig('posterior_comparison.png', dpi=300)
        print("Saved posterior_comparison.png")

    # Generate envelope plots
    print("Generating envelope plots...")
    de_env_file = 'output_DE_MCMC/MCMC_envelopes_Validation_Station_series_1d.csv'
    cv_env_file = 'output_DE_CV_MCMC/MCMC_envelopes_Validation_Station_series_1d.csv'

    if os.path.exists(de_env_file) and os.path.exists(cv_env_file):
        df_env_de = pd.read_csv(de_env_file)
        df_env_cv = pd.read_csv(cv_env_file)

        # Merge dates for plotting
                # Filter out warm-up rows where Year == -999
        valid_idx = df_env_de['Year'] != -999
        df_env_de = df_env_de[valid_idx].copy()
        df_env_cv = df_env_cv[valid_idx].copy()
        dates = pd.to_datetime(df_env_de[['Year', 'Month', 'Day']])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(dates, df_env_de['Twat_mod_lower'], df_env_de['Twat_mod_upper'], color='blue', alpha=0.3, label='DE-MCMC 95% PI')
        ax.fill_between(dates, df_env_cv['Twat_mod_lower'], df_env_cv['Twat_mod_upper'], color='red', alpha=0.3, label='DE-CV-MCMC 95% PI')
        ax.plot(dates, df_env_de['Twat_mod_p50'], color='blue', label='DE-MCMC Median')
        ax.plot(dates, df_env_cv['Twat_mod_p50'], color='red', label='DE-CV-MCMC Median')

        ax.set_title("95% Prediction Intervals: DE-MCMC vs DE-CV-MCMC")
        ax.legend()
        plt.tight_layout()
        plt.savefig('envelope_comparison.png', dpi=300)
        print("Saved envelope_comparison.png")

    print("Comparison complete.")
