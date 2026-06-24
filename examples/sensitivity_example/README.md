# pyair2stream Sensitivity Analysis Example

This example demonstrates how to use the built-in One-At-A-Time (OAT) local sensitivity analysis feature in `pyair2stream`.

## Setup and Execution

1.  **Generate Data:**
    Run the generation script to create synthetic time-series data.
    ```bash
    python examples/sensitivity_example/generate_data.py
    ```
    This creates `synthetic_data.csv`.

2.  **Run the Model:**
    Execute the model with the local configuration. The `config.yaml` is configured to run a calibration (PSO) followed by the sensitivity analysis.
    ```bash
    python -m pyair2stream.main --config examples/sensitivity_example/config.yaml
    ```

## Understanding the Configuration

The `config.yaml` includes two key flags:
*   `sensitivity_analysis: true`: Instructs the model to perform the OAT sensitivity analysis immediately after calibration.
*   `sensitivity_perturbations: [1.0, 2.0, 5.0]`: Defines the exact perturbation percentages to test. For each parameter, it will perturb the calibrated optimal value by +/- 1%, 2%, and 5% of its total valid range.

## Outputs

After running, check the generated outputs:
*   **`sensitivity_PSO_NSE_River_Beta.csv`**: A detailed report of the calculated sensitivity index for each parameter and perturbation step.
*   **`sensitivity_PSO_NSE_River_Beta.png`**: A grouped bar chart visualizing the sensitivity index. Parameters with taller bars are those where a small change in their value causes a large change in the predicted water temperatures.

This analysis is vital for understanding model identifiability—parameters with near-zero sensitivity are poorly constrained by the calibration data.
