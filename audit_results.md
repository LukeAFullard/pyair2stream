# Audit of Python Port Feasibility Claims

This document contains the audit results of the claims made in `python_port_feasibility.md` against the original Fortran codebase of `air2stream`.

## Critical issues for an accurate port

*   **Do not replace the optimisation algorithms initially**
    *   **Audit Status: Verified.** The optimization algorithms (`PSO` and `LH`) are deeply intertwined with the execution flow and parameter handling. Replacing them immediately with external libraries like `scipy.optimize` would introduce a high risk of divergence. A direct, exactly replicated port is required first for validation.

*   **Replicate Fortran floating-point behaviour**
    *   **Audit Status: Verified.** The codebase extensively uses `REAL(KIND=8)`, which corresponds to 64-bit precision floating-point numbers. (e.g., `AIR2STREAM_MODULES.f90: REAL(KIND=8) :: Qmedia, theta_j, theta_j1, DD_j, DD_j1`). Using `numpy.float64` strictly is essential to avoid drift.

*   **Preserve execution order**
    *   **Audit Status: Verified.** The ODE integration step inside `AIR2STREAM_SUBROUTINES.f90` (e.g., `call_model`) sequentially calculates variables day-by-day where `Twat_mod(j)` depends on `Twat_mod(j-1)`. Vectorizing this loop would change the results or break dependencies entirely.

*   **Replicate missing-value handling exactly**
    *   **Audit Status: Verified.** The sentinel value `-999` is heavily used to denote missing or initialization values. For instance, `IF (Twat_obs(i) .ne. -999) THEN` is explicitly tested in multiple subroutines like `aggregation` and data loading.

*   **Port aggregation exactly**
    *   **Audit Status: Verified.** The `python_port_feasibility.md` initially claimed or implied this might be missing, but `SUBROUTINE aggregation` exists in `AIR2STREAM_SUBROUTINES.f90` and performs critical logic for varying time scales (daily, weekly, monthly).

*   **Port statis exactly**
    *   **Audit Status: Verified.** `SUBROUTINE statis` is fully implemented in `AIR2STREAM_SUBROUTINES.f90` and computes key statistics (`mean_obs`, `TSS_obs`, `std_obs`) which are fundamental to the objective functions later on.

## Bugs that should be intentionally handled

*   **PSO initial best selection**
    *   **Description:** In `AIR2STREAM_RUNMODE.f90`, during PSO initialization, `fitbest` is populated with `eff_index` for each particle. However, the subsequent call `CALL best(fit,k,foptim)` uses the `fit` array (which is still `0` for all indices). This randomly/incorrectly sets the initial global best (`gbest`) to the position of the first particle (`k=1`) instead of the true best.
    *   **Decision:** **Fix and document.** While an exact behavioural clone would replicate this error, it undermines the optimization algorithm's convergence rate. The port should pass `fitbest` to evaluate the initial global best correctly, documenting this correction in the port's release notes.

*   **Dead PSO convergence test**
    *   **Description:** In `AIR2STREAM_RUNMODE.f90`, the early convergence check calculates a norm distance, applies `SQRT(norm)`, and then checks `IF (norm .lt. 0.0)`. Since a square root is non-negative, this condition is impossible and the PSO loop will always run for `n_run` iterations.
    *   **Decision:** **Fix and document.** The port should implement a standard small tolerance check (e.g., `norm < 1e-6`) to allow for actual early stopping, improving performance. This deviation from the Fortran behaviour should be clearly documented.

*   **Duplicate `version == 4` block**
    *   **Description:** In `AIR2STREAM_READ.f90` lines 67-72, the `version == 4` logic is defined. However, at lines 81-86, there is another block: `IF (version == 4) THEN !air2stream with 8 parameters`. This copy-paste error overrides the configuration for an 8-parameter model version.
    *   **Decision:** **Fix and document.** The second block should be corrected to `IF (version == 8) THEN` in the Python port to allow the 8-parameter version to function as originally intended.

## Missing validation steps & Recommended implementation changes

The remaining claims regarding the absolute need for **Golden-output tests**, **Integration-scheme tests**, and the recommendations against **multithreading** in favour of **multiprocessing** are conceptually sound and essential for the Python porting effort. No specific Fortran flaws contradict these recommendations.
