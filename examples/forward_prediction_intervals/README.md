# Probabilistic Forward Prediction Intervals

This example demonstrates how to project future water temperatures probabilistically, using the parameter distributions and residual error ($\sigma$) derived during a `DE-MCMC` historical calibration.

## The Problem
By default, the `FORWARD` run mode in `pyair2stream` is deterministic. It accepts a single `parameters_forward` array and outputs exactly one predicted line. While useful, this ignores the parameter uncertainty (equifinality) and intrinsic data noise.

## The Solution
By providing the `MCMC_chain.csv` and the historical residual error ($\sigma$), `FORWARD` mode can generate a robust 90% Prediction Interval encompassing parameter uncertainty and observation noise.

This example showcases the two noise generation methods available:
* **IID (Independent and Identically Distributed):** Standard white noise. It assumes residuals have no memory.
* **AR(1) (Autoregressive lag-1):** Time-correlated noise. Environmental data like water temperature often has strong serial correlation, meaning if today is warmer than predicted, tomorrow is likely to be as well. AR(1) preserves this structure, typically resulting in wider and more realistic prediction bounds. `pyair2stream` automatically estimates the correlation coefficient ($\rho$) from the historical residuals and passes it to the projection run via a sidecar JSON file (`_meta.json`).

## How to Run

1. Generate synthetic historical (with AR(1) structured noise injected) and future climate data:
```bash
poetry run python examples/forward_prediction_intervals/generate_data.py
```

2. Run the full calibration and projection pipeline:
```bash
poetry run python examples/forward_prediction_intervals/run_example.py
```

This script will:
1. Run `DE-MCMC` to calibrate against `historical_data.csv`. This automatically calculates the $\rho$ autocorrelation and saves it to a sidecar file.
2. Extract the historical standard deviation of residuals ($\sigma$).
3. Dynamically inject these properties and run `FORWARD` mode twice: once forcing `noise_model: "iid"`, and once allowing `noise_model: "ar1"` (which automatically reads the sidecar file).
4. Draw 500 parameter samples, simulate, add the corresponding noise model realizations, and calculate percentiles.
5. Generate a side-by-side comparative plot of the IID vs AR(1) bounds.

## Example Output

Check `examples/forward_prediction_intervals/comparison_iid_vs_ar1.png` to see how the temporally-correlated structure of the AR(1) noise generates more representative bounds compared to standard white noise.
