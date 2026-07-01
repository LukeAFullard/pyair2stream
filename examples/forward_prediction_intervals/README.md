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

The generated plot features four panels that illustrate the practical difference between IID and AR(1) noise models:
1. **Forward Projection - IID Noise**: Shows the standard 90% Prediction Interval using white noise.
2. **Forward Projection - AR(1) Noise**: Shows the 90% Prediction Interval using autoregressive noise. Note that at a daily scale, the overall width of the interval is mathematically identical to the IID interval, because both models are calibrated to the exact same marginal variance ($\sigma^2$).
3. **Sample Individual Trajectories**: Overlays individual simulation traces. Here the structural difference becomes visible: the IID trace (green) oscillates rapidly day-to-day around the median, while the AR(1) trace (blue) exhibits realistic "memory", wandering away from the median for several days at a time (e.g., simulating a multi-day heatwave or cold snap).
4. **7-Day Rolling Average Prediction Interval**: This is the most crucial panel for understanding the real-world impact. When we calculate a 7-day rolling average of the envelopes, the rapid daily oscillations of the IID white noise cancel each other out, causing the green uncertainty bounds to shrink drastically. In contrast, because the AR(1) errors are temporally correlated, they do not perfectly cancel out over a week. The blue AR(1) bounds remain much wider, correctly preserving the uncertainty for time-averaged metrics (e.g., weekly compliance thresholds) which IID models dangerously underestimate.
