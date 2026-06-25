import yaml
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import detect_segments, aggregation, statis, call_model
from pyair2stream.optimization import DE_MCMC_mode, forward_mode
from pyair2stream.post_processing import post_process
import numpy as np

def run_calibration():
    print("--- 1. Running DE-MCMC Calibration ---")
    data = read_calibration("examples/forward_prediction_intervals/config_calib.yaml")
    read_Tseries(data, 'c')
    aggregation(data)
    statis(data)

    DE_MCMC_mode(data)

    mod_valid = data.Twat_mod[365:]
    obs_valid = data.Twat_obs[365:]
    sse = np.sum((mod_valid - obs_valid)**2)
    historical_sigma = np.sqrt(sse / len(obs_valid))
    print(f"\\nCALIBRATION COMPLETE.")
    print(f"Historical Residual Sigma (Observation Error): {historical_sigma:.3f}\\n")

    return data.par_best.tolist(), historical_sigma

def run_forward(best_params, historical_sigma):
    print("--- 2. Running Probabilistic Forward Projection ---")

    with open("examples/forward_prediction_intervals/config_forward.yaml", "r") as f:
        config = yaml.safe_load(f)

    config['parameters_forward'] = [float(p) for p in best_params]
    config['forward_options']['residual_sigma'] = float(historical_sigma)

    with open("examples/forward_prediction_intervals/config_forward_injected.yaml", "w") as f:
        yaml.dump(config, f)

    data = read_calibration("examples/forward_prediction_intervals/config_forward_injected.yaml")
    read_Tseries(data, 'c')

    if data.gap_tolerant and data.segments is None:
        detect_segments(data)

    forward_mode(data)
    post_process(data)
    print("Forward run complete! Check 'examples/forward_prediction_intervals' for the plot.")

if __name__ == "__main__":
    best_params, historical_sigma = run_calibration()
    run_forward(best_params, historical_sigma)
