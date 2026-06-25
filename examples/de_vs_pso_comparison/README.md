# Differential Evolution vs Particle Swarm Optimization

This example demonstrates the performance and accuracy improvements gained by utilizing the optional SciPy **Differential Evolution (DE)** solver instead of the legacy **Particle Swarm Optimization (PSO)** method.

## Background
In `pyair2stream`, calculating objective parameters traditionally relies on a custom implementation of PSO. However, PSO is highly sensitive to collinear parameter combinations (e.g. $a_2$ and $a_3$).

By opting into `run_mode: "DE"`, the engine delegates optimization to `scipy.optimize.differential_evolution`. DE effectively addresses 'thermal inertia' traps using a `best1bin` strategy and polishes the solution by finalizing bounds using the L-BFGS-B minimizer. Due to SciPy's optimized `workers=-1` architecture, the parallelization handles large generations across CPU cores highly efficiently.

## How to Run

1. Generate the synthetic dataset:
   ```bash
   python generate_data.py
   ```
2. Execute the comparison script:
   ```bash
   python run_comparison.py
   ```

## Expected Output
The `run_comparison.py` script will optimize a synthetic 5-year river record using both solvers. Upon completion, it outputs an execution scorecard:

```
==================================================
--- AIR2STREAM SOLVER WAR ---
==================================================
PSO Time: ~7.00s | Best RMS: -1.1912
DE  Time: ~70.00s | Best RMS: -0.9850

Best Parameters:
PSO: ['4.9384', '0.6098', '0.8043', '0.7225', '0.4844', '3.1060', '0.5590', '0.4039']
DE : ['0.1000', '0.0062', '0.0087', '1.0000', '1.0000', '0.7614', '0.5192', '0.0916']

DE Speedup: ~0.10x
DE RMSE Improvement: ~0.2062

Note: DE evaluated ~7263 particles, whereas PSO evaluated ~2500.
While DE took longer in absolute terms, it ran a significantly deeper search and converged to a much better fit.
==================================================
```

*Note: In synthetic examples where iteration counts are bounded strictly for demonstration, execution time comparisons may vary based on your system configuration. Generally, you can expect noticeable RMSE improvements (closer to 0) with DE on complex topographies. DE executes a much larger number of evaluations per "iteration" or "run" due to the way its generation logic and popsize work compared to the fixed particle limits in PSO, which leads to DE being slower per iteration but resulting in a much more optimal solution.*

The script also generates a plot `solver_comparison.png` visualizing the simulated fits over the actual `T_water` data points.
