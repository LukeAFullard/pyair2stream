# DE-MCMC vs DE-CV-MCMC

This directory contains an end-to-end example comparing the standard `DE-MCMC` run mode with the new `DE-CV-MCMC` mode.

## Hydrological and Statistical Reasoning

The standard `DE-MCMC` mode finds a global optimum using Differential Evolution and then initializes MCMC walkers in a very tight, uniform "ball" (standard deviation of 1e-4) around that single optimum. While mathematically valid in the limit of infinite time, this initialization strategy suffers from severe burn-in inefficiencies given limited computational budgets. The tight prior prevents walkers from quickly discovering the true equifinality and variance profile of the parameter posterior distribution.

The `DE-CV-MCMC` mode solves this by leveraging cross-validation to inform the initial spread of the MCMC walkers. It first performs a leave-one-year-out cross-validation using the DE optimizer to sample realistic parameter sets across different temporal folds. The standard deviations of the parameters obtained from these folds provide a mathematically informed variance. This variance is then used to initialize the MCMC walkers, allowing them to start with a spread that reflects the parameter equifinality observed in the temporal subsets of the data.

As a result, `DE-CV-MCMC` can discover a wider variance profile and reach convergence much faster, producing more realistic uncertainty bounds for hydrological predictions without requiring excessive burn-in periods.

## Running the Example

Run the comparison script:
```bash
python compare_mcmc.py
```
This script will configure both modes (using a configurable 95% `prediction_interval` setting in the YAML config), run them on the Switzerland validation dataset, and produce a `posterior_comparison.png` plot showing the side-by-side posterior histograms, as well as an `envelope_comparison.png` plot displaying the differing prediction interval widths.
