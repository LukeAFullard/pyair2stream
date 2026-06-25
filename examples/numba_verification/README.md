# Numba Optimization Verification

This directory contains a script (`run_experiment.py`) designed to empirically prove that the Numba optimization scheme injected into `pyair2stream`'s core physics and objective function subroutines runs quickly and does not throw any execution errors.

## How it works
1. It builds a synthetic multi-year dataset with gaps and targets to calibrate.
2. It sets up the new Numba engine parameters and configuration defaults.
3. It benchmarks the execution time for the PSO_mode running 100 times.
4. It asserts execution success and prints efficiency indices found via fast paths.

## Execution
```bash
python examples/numba_verification/run_experiment.py
```
