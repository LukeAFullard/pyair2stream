# Optimizer Convergence Example

This example demonstrates the convergence behavior of two different optimization algorithms available in `pyair2stream`:
1. **PSO** (Particle Swarm Optimization)
2. **DE** (Differential Evolution hybrid with L-BFGS-B polish)

## Overview

The scripts generate synthetic daily river temperature data and calibrate the model by varying the number of solver iterations (`n_runs`), going from small (10) up to larger values (5000), using 20 particles for both.

This helps analyze:
- How fast the optimizers converge to stable parameter values.
- How computational cost (time) scales with increased iterations.

## Execution

1. First, synthetic data is generated:
```bash
python examples/optimizer_convergence/generate_data.py
```

2. Then, the experiment runs calibrations across different `n_runs` to plot parameter stability and timing:
```bash
python examples/optimizer_convergence/run_experiment.py
```

## Results

### Parameter Convergence
We track the optimal values found for all 8 parameters across the different iterations. The subplots plot the final optimized values returned for each parameter comparing PSO and DE methods.

![Parameter Convergence](parameter_convergence.png)

### Computation Time
We measure how long the calibration process takes as the number of iterations increases.

![Computation Time](computation_time.png)

### Objective Function Convergence
We track the model fit quality (NSE) to see at what point diminishing returns start regarding optimization effort.

![Objective Convergence](objective_convergence.png)
