# Optimizer Comparison: PSO vs Hybrid DE

This directory contains an experiment (`compare_optimizers.py`) comparing the native Particle Swarm Optimizer (PSO) algorithm built into `pyair2stream` against the modern, hybrid Differential Evolution + L-BFGS-B (DE) method introduced as an alternative.

The experiment was run against the dataset generated in `examples/synthetic_example/`.

## Why a Hybrid Optimizer?

Particle Swarm Optimization works well for roughly narrowing down bounds globally but tends to converge extremely slowly on exact local optimums (or fails to entirely, getting caught in local minima).

For an 8-parameter problem, **Differential Evolution (DE)** explores the parameter space significantly better and is less prone to premature convergence. Once DE finds the correct underlying basin of attraction, switching to **L-BFGS-B** (which uses numerical gradients) allows us to mathematically "snap" to the true local optimum instantly, skipping thousands of fine-tuning evaluations.

## Results

Executing the identical extended configuration (`50 particles/popsize`, `100 runs/maxiter`) to ensure full saturation yields:

### Particle Swarm Optimization (PSO)
* **Execution Time:** ~17.36 seconds
* **Accuracy (NSE):** 0.961474
* **Parameters Found:** `[4.61483, 0.37989, 0.61807, 0.28403, 0.08293, 3.1011, 0.54667, 0.26237]`

### Differential Evolution + L-BFGS-B (DE)
* **Execution Time:** ~258.27 seconds
* **Accuracy (NSE):** **0.969964** (Substantial improvement)
* **Parameters Found:** `[0.10000, 0.02667, 0.04163, 0.00269, 0.99981, 0.71131, 0.54188, 0.08516]`

### Convergence Tracking
![Convergence Plot](convergence_plot.png)
By tracking the cumulative best `NSE` over every evaluation iteration, we see how DE's initial global search rapidly finds a superior basin to PSO, and then the vertical "snap" (the perfectly straight line at the end of the green curve) represents the L-BFGS-B gradient polishing finding the exact local minimum instantly.

### Visual Fit Comparison

![Calibration Fit Comparison](comparison_plot.png)

As seen in the plot above, the orange line (PSO) captures the generic sinusoidal curve of the water temperature but misses the extreme peak amplitudes. The green line (Hybrid DE) traces the underlying signal significantly closer to the true observations, validating the leap in the NSE score.

## Sensitivity Analysis Plausibility

Environmental parameters should ideally map physically to the system. While PSO finds a mathematically plausible fit, it is extremely prone to the "equifinality" problem (multiple different parameter sets giving similar decent results).

Let's look at the One-At-A-Time local sensitivity of both parameter sets:

### PSO Sensitivity
![PSO Sensitivity](sensitivity_PSO.png)

### Hybrid DE Sensitivity
![DE Sensitivity](sensitivity_DE.png)

The DE solution exhibits sharper bounds in sensitivity. Parameter 1 ($a_1$) is extremely sensitive in the DE fit, reflecting its fundamental role in the physics mass-balance equations compared to the overly-distributed sensitivity in the PSO local minimum.

## Conclusion

While PSO is rapid for prototyping, the **Hybrid Differential Evolution + L-BFGS-B** method provides objectively superior mathematical convergence to the true global optimum. In scientific research scenarios where finding the exact system coefficients is more important than pure computational speed, the DE method is highly recommended.
