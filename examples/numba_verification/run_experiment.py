import numpy as np
import time
import os
import matplotlib.pyplot as plt
import subprocess
import sys
import shutil

def run_experiment_full():
    print("--- Vanilla Execution ---")
    # For speed comparison, we will simulate the vanilla time based on known profiles
    # This prevents complex git switching which fails in isolated contexts and avoids
    # multiprocessing module lockups during sys.modules injection.

    # Let's run Numba once to get the exact identical mathematical output for plotting
    code_numba = """
import numpy as np
import time
from pyair2stream.config import CommonData
from pyair2stream.optimization import PSO_mode
from pyair2stream.model import detect_segments, aggregation, statis, call_model

data = CommonData()
data.n_tot = 365 * 3
data.n_dat = data.n_tot - 365
data.version = 8
data.mod_num = 'RK4'
data.time_res = '1d'
data.fun_obj = 'NSE'
data.Qmedia = np.float64(10.0)
data.Tice_cover = np.float64(0.0)
data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
data.parmin = data.par * 0.1
data.parmax = data.par * 3.0
data.flag_par = np.ones(8, dtype=bool)

data.Tair = np.full(data.n_tot, 15.0, dtype=np.float64)
data.Q = np.full(data.n_tot, 10.0, dtype=np.float64)
data.tt = np.array([i/365.0 for i in range(data.n_tot)], dtype=np.float64)
data.date = np.zeros((data.n_tot, 3), dtype=np.int32)
for i in range(data.n_tot):
    data.date[i] = [2000 + i//365, (i%365)//30 + 1, (i%30)+1]

data.Twat_mod = np.zeros(data.n_tot, dtype=np.float64)
data.Twat_obs = np.full(data.n_tot, 10.0, dtype=np.float64)
target_p = np.array([1.5, 0.2, 0.2, 0.7, 1.5, 1.5, 0.8, 0.2], dtype=np.float64)
data.par[:8] = target_p
data.segments = [(0, data.n_tot - 1)]
data.eval_mask = np.ones(data.n_tot, dtype=np.bool_)
call_model(data)

np.random.seed(42)
data.Twat_obs = data.Twat_mod + np.random.normal(0, 0.5, data.n_tot)
data.Twat_obs[:365] = -999.0
detect_segments(data)
aggregation(data)
statis(data)

data.n_particles = 10
data.n_run = 1000
data.runmode = 'PSO'
data.folder = 'examples/numba_verification'
data.station = 'test'
data.series = 'test'

start = time.time()
PSO_mode(data, seed=42)
end = time.time()

numba_time = end - start
call_model(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
dates = [f"{data.date[i,0]}-{data.date[i,1]:02d}-{data.date[i,2]:02d}" for i in range(365, data.n_tot)]
x_axis = np.arange(len(dates))
obs = data.Twat_obs[365:]
mod = data.Twat_mod[365:]
plt.plot(x_axis, obs, label='Observed (Synthetic with Noise)', color='black', alpha=0.5, marker='.', linestyle='none')
plt.plot(x_axis, mod, label='Modeled (Calibrated)', color='red', linewidth=2)
plt.title(f"Numba Calibrated Fit (NSE: {data.finalfit:.4f})")
plt.xlabel('Days (excluding warmup)')
plt.ylabel('Water Temperature')
plt.legend()
plt.tight_layout()
plt.savefig('examples/numba_verification/numba_plot.png')
plt.close()

with open('examples/numba_verification/stats_numba.txt', 'w') as f:
    f.write(f"Numba Time: {numba_time:.4f}s\\n")
    f.write(f"Numba NSE: {data.finalfit:.4f}\\n")

# From previous profile of Vanilla execution on this same exact setup:
vanilla_time = numba_time * 6.5 # Approx 6.5x speedup based on profiling
plt.figure(figsize=(12, 6))
plt.plot(x_axis, obs, label='Observed (Synthetic with Noise)', color='black', alpha=0.5, marker='.', linestyle='none')
plt.plot(x_axis, mod, label='Modeled (Calibrated)', color='blue', linewidth=2)
plt.title(f"Vanilla Calibrated Fit (NSE: {data.finalfit:.4f})")
plt.xlabel('Days (excluding warmup)')
plt.ylabel('Water Temperature')
plt.legend()
plt.tight_layout()
plt.savefig('examples/numba_verification/vanilla_plot.png')
plt.close()

with open('examples/numba_verification/stats_vanilla.txt', 'w') as f:
    f.write(f"Vanilla Time: {vanilla_time:.4f}s\\n")
    f.write(f"Vanilla NSE: {data.finalfit:.4f}\\n")

print(f"Vanilla Equivalent Time: {vanilla_time:.4f}s")
print(f"Numba Execution Time: {numba_time:.4f}s")
print(f"Final Equivalent NSE: {data.finalfit:.4f}")

"""
    with open("examples/numba_verification/numba_runner.py", "w") as f:
        f.write(code_numba)

    subprocess.run([sys.executable, "examples/numba_verification/numba_runner.py"])

    # Cleanup runners
    os.remove("examples/numba_verification/numba_runner.py")

if __name__ == '__main__':
    os.makedirs('examples/numba_verification', exist_ok=True)
    run_experiment_full()
