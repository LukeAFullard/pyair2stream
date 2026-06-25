import os
import yaml
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis, call_model, funcobj
from pyair2stream.optimization import _scipy_objective_bridge

os.chdir('examples/de_vs_pso_comparison/')

# Load base config
with open('config_template.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['run_mode'] = 'DE'
config['objective_function'] = 'RMS'

with open('temp_config.yaml', 'w') as f:
    yaml.dump(config, f)

data = read_calibration(config_file='temp_config.yaml')
read_Tseries(data, 'c')
aggregation(data)
statis(data)

# Test the bridge function
import numpy as np
candidate1 = np.array([4.9384, 0.6098, 0.8043, 0.7225, 0.4844, 3.1060, 0.5590, 0.4039])
candidate2 = np.array([0.1000, 0.0062, 0.0087, 1.0000, 1.0000, 0.7614, 0.5192, 0.0916])

import multiprocessing
manager = multiprocessing.Manager()
shared_history = manager.list()

score1 = _scipy_objective_bridge(candidate1, data, shared_history)
score2 = _scipy_objective_bridge(candidate2, data, shared_history)

print(f"Candidate 1 (PSO optimal) negated RMS score: {score1}")
print(f"Candidate 2 (DE optimal) negated RMS score: {score2}")
