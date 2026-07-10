# Optimizer Convergence Report

*(Report current as of commit `dc4cba90a0f5591292c16de3407d30ad6fbaf279`)*

This report evaluates the convergence behavior of two optimization algorithms available in `pyair2stream`:
1. **PSO** (Particle Swarm Optimization)
2. **DE** (Differential Evolution hybrid with L-BFGS-B polish)

## Overview
The algorithms were executed across increasing iteration counts (from 10 up to 5000), using 20 particles, to assess:
- **Parameter Convergence**: The stability and values of the model parameters.
- **Objective Function Convergence**: The goodness-of-fit measured by Nash-Sutcliffe Efficiency (NSE).
- **Computation Time**: The runtime required for each algorithm.

## Results & Discussion

### Objective Function Convergence (Goodness of Fit)
![Objective Convergence](objective_convergence.png)

*Discussion*: The plot above shows how the objective function (NSE) improves as the number of iterations increases. DE consistently converges faster and reaches a higher NSE value with fewer iterations compared to PSO.

**Final Goodness Parameters (NSE) at 5000 iterations:**
- **PSO**: 0.97052
- **DE**: 0.97053

### Parameter Convergence (Model Fit Parameters)
![Parameter Convergence](parameter_convergence.png)

*Discussion*: This plot displays the progression of the 8 parameter values as iterations increase. We can see how quickly the parameters stabilize. DE typically shows more stability at earlier iterations, whereas PSO takes longer to find the stable parameter space.

**Final Fit Parameters at 5000 iterations:**
- **PSO**: [0.28601, 0.0, 0.00634, 0.99954, 9.99951, 6.94042, 0.54674, 0.85726]
- **DE**: [0.1, 0.0, 0.00652, 0.61331, 9.50175, 6.43965, 0.54866, 0.79872]

### Computation Time
![Computation Time](computation_time.png)

*Discussion*: The time taken by both optimizers scales roughly linearly with the number of iterations.

## Conclusion
DE reaches convergence and achieves stable parameter values in fewer iterations than PSO for this configuration, at a similar computational cost. Both optimizers reach near-identical NSE values (~0.9705), but the final parameter sets diverge substantially (e.g., `p4`: 0.6133 vs 0.9995). This indicates equifinality, where multiple, differing parameter sets yield similarly high objective function efficiencies.
