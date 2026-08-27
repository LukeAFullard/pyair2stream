# DE-MCMC vs DE-CV-MCMC

This directory contains an end-to-end example comparing the standard `DE-MCMC` run mode with the new `DE-CV-MCMC` mode.

## Hydrological and Statistical Reasoning

The standard `DE-MCMC` mode finds a global optimum using Differential Evolution and then initializes MCMC walkers in a tight ball (scaled to each active parameter's own bound width, reflected off bounds rather than clipped) around that single optimum. While mathematically valid in the limit of infinite time, this initialization strategy suffers from severe burn-in inefficiencies given limited computational budgets. The tight prior prevents walkers from quickly *discovering* the true equifinality and variance profile of the parameter posterior distribution -- it does not change what that distribution actually is.

The `DE-CV-MCMC` mode addresses the discovery-speed problem by leveraging cross-validation to inform the initial spread of the MCMC walkers. It first performs a leave-one-year-out cross-validation using the DE optimizer to sample realistic parameter sets across different temporal folds. The standard deviations of the parameters obtained from these folds provide a mathematically informed variance, used to initialize the MCMC walkers so they start with a spread that already reflects the parameter equifinality observed across temporal subsets of the data, rather than one they have to discover through burn-in alone.

**`posterior_comparison.png` and `envelope_comparison.png` are a non-convergence diagnostic, not evidence that `DE-CV-MCMC` finds a "better" or wider posterior.** Both modes sample the same target distribution; if either chain has genuinely converged, the two must agree on posterior width as well as on the point estimate -- that is what convergence means. Any visible disagreement between the two panels within the step budget configured here is evidence that the tightly-initialized `DE-MCMC` chain has not yet mixed out to the true posterior width at that step count, which is exactly the failure mode `DE-CV-MCMC`'s informed initialization is meant to avoid. Check the convergence diagnostics reported in each run's `MCMC_chain_*_meta.json` sidecar (`mean_autocorr_time`, `max_split_rhat`) before drawing conclusions from either panel; do not use this comparison as a substitute for actually checking convergence.

## Running the Example

Run the comparison script:
```bash
python compare_mcmc.py
```
This script will configure both modes (using a configurable 95% `prediction_interval` setting in the YAML config), run them on the Switzerland validation dataset, and produce a `posterior_comparison.png` plot showing the side-by-side posterior histograms, as well as an `envelope_comparison.png` plot displaying the differing prediction interval widths -- read both as the non-convergence diagnostic described above, not as evidence for either mode's uncertainty being more "correct".
