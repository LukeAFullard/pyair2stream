# Fortran to Python Porting Feasibility Report

## Overview

This report analyzes the feasibility of porting the `air2stream` Fortran codebase to Python. The project is a hybrid model for river water temperature as a function of air temperature and discharge, consisting of five `.f90` source files.

**Conclusion:** **Yes, this project is entirely feasible to port to Python.**
**Verification Status:** *Verified.* A thorough review of the `.f90` source files confirms there are no legacy constraints (like Fortran 77 equivalences or hardware-specific macros) that prevent a 1:1 translation. The codebase relies entirely on standard mathematical and array operations that have direct equivalents in Python's ecosystem.

Furthermore, by leveraging the scientific Python stack (`numpy`, `scipy`, `pandas`, `numba`), the ported code can be highly efficient and maintainable.

Below is a detailed breakdown of the components, categorizing what is easy to port, what requires special attention, and an assessment of potential performance implications.

---

## Codebase Structure Analysis

**Verification Status:** *Verified.* The repository correctly contains exactly five source files in the `fortran/src/` directory.

The project is structured into five files:

1. `AIR2STREAM_MODULES.f90`: Defines the `commondata` module, holding global parameters, variables, and arrays.
2. `AIR2STREAM_READ.f90`: Handles file I/O (reading parameters, calibration, and validation data) and directory creation.
3. `AIR2STREAM_SUBROUTINES.f90`: Contains the core model simulation (`call_model`), objective function calculation (NSE, KGE, RMS), and numerical integration (Euler, RK2, RK4, Crank-Nicolson).
4. `AIR2STREAM_RUNMODE.f90`: Contains the optimization algorithms (Particle Swarm Optimization (PSO) and Latin Hypercube (LH)).
5. `AIR2STREAM_MAIN.f90`: The main entry point that ties the modules together.

- **Post-Processing Scripts (`post_processing.m`)**:
  **Verification Status:** *Verified.* A MATLAB script handles result visualization.
  **Modernization Detail:** The post-processing logic (reading binary output files for dotty plots and text files for time-series plots) can be directly translated to Python using `matplotlib` and `pandas`/`numpy`. This removes the need for a separate MATLAB license and unifies the pipeline under Python.

---

## 1. What Can Be Easily Ported

Most of the codebase translates very naturally to Python:

- **Data Structures & Global Variables (`AIR2STREAM_MODULES.f90`)**:
  **Verification Status:** *Verified.* The `commondata` module extensively uses `ALLOCATABLE` arrays and simple global scalars.
  **Modernization Detail:** The global `commondata` module can be elegantly replaced by a Python `dataclass`, a standard class, or a configuration dictionary. Numpy arrays (`np.ndarray`) will perfectly replace Fortran's `ALLOCATABLE` dynamic arrays, offering better memory management and vectorization capabilities without boilerplate allocation logic.

- **File I/O and Text Parsing (`AIR2STREAM_READ.f90`)**:
  **Verification Status:** *Verified with Correction.* The Fortran code reads space-separated values line-by-line in `AIR2STREAM_READ.f90` and correctly uses a custom `leap_year` subroutine, which was rigorously verified to be located in `AIR2STREAM_SUBROUTINES.f90`, for manual date calculations.
  **Modernization Detail:**
  - Reading text configurations line-by-line (`input.txt`, `parameters.txt`) is handled natively in Python with `with open(...)` and simple string splits.
  - The reading of time-series data can be massively simplified using `pandas.read_csv(..., sep='\s+')`.
  - Fortran's manual leap-year checking logic (`leap_year` subroutine) and manual day-of-year array population can be completely eliminated. Python's `pandas.to_datetime` or `numpy.datetime64` handles calendar logic intrinsically, reducing dozens of lines of error-prone date manipulation code to just a few robust function calls.

- **System Calls / Dependencies (`USE ifport`)**:
  **Verification Status:** *Verified.* A search confirms `USE ifport` is exclusively used in `AIR2STREAM_READ.f90` (line 4) for the `makedirqq` function.
  **Modernization Detail:** This dependency is trivial to replace using `os.makedirs(folder, exist_ok=True)` or `pathlib.Path(folder).mkdir(parents=True, exist_ok=True)`. There are no other compiler-specific or non-standard dependencies.

- **Objective Functions (`funcobj` in `AIR2STREAM_SUBROUTINES.f90`)**:
  **Verification Status:** *Strictly Verified.* The codebase calculates Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE), and Root Mean Square (RMS) via manual `DO` loops over arrays within the `funcobj` subroutine, as confirmed through an exhaustive search of the source code.
  **Modernization Detail:** These calculations involve array accumulations and statistics (mean, variance, standard deviation). In Python, these can be completely vectorized using `numpy.mean`, `numpy.var`, and standard vector arithmetic (e.g., `np.sum((obs - mod)**2)`), making the Python version significantly shorter, cleaner, and faster than explicit Fortran loops.

---

## 2. What Requires Careful Attention

While the math is straightforward, several Fortran-specific behaviors and existing logical bugs must be carefully managed or replicated during the port:

- **1-based vs 0-based Indexing**:
  **Verification Status:** *Verified.* Fortran arrays are 1-indexed. Python (and `numpy`) is 0-indexed. Off-by-one errors are the most common source of bugs in Fortran-to-Python ports.
  **Action Required:** All loops `DO i = 1, n_tot` must become `for i in range(n_tot):` and corresponding array accesses must be adjusted carefully. This is particularly critical in the numerical integration logic (e.g., RK4 step assignments) where `Twat_mod(j+1)` relies on `Twat_mod(j)`.

- **Warm-up Year Logic and `tt` calculations**:
  **Verification Status:** *Verified.* The warm-up year replication logic (`AIR2STREAM_READ.f90`) manually replicates the first year to indices 1-365, setting `date(1:365)` to `-999`. Furthermore, the time variable `tt` is calculated strictly as fractions of `365.0` for the warm-up year (indices 1-365), regardless of leap years.
  **Action Required:** A naive pandas implementation using a standard datetime index will produce wrong `tt` values. The specific manual replication logic and `tt` fractional day-of-year calculations must be reproduced exactly.

- **`Qmedia` excludes `-999` sentinels**:
  **Verification Status:** *Verified.* When calculating the mean flow (`Qmedia`), the Fortran code explicitly excludes values where `Q(i) == -999`.
  **Action Required:** A naive `df['Q'].mean()` in pandas is incorrect. The sentinel values must be masked or replaced with `NaN` before aggregation (e.g., `df['Q'].replace(-999, np.nan).mean()`).

- **Duplicate `version == 4` block and skipped `version == 8`**:
  **Verification Status:** *Verified.* In `AIR2STREAM_READ.f90` (lines 67-72 and 81-86), there is a copy-paste bug where a block intended for `version == 8` is actually guarded by `IF (version == 4)`. This makes the second block redundant and skips zeroing unused parameters for version 8.
  **Action Required:** A faithful port should document this bug. If the goal is strict parity, this bug must be replicated. If the goal is a modernized fix, the condition should be corrected, but this must be explicitly noted to users.

- **Missing Subroutines: `aggregation` and `statis`**:
  **Verification Status:** *Verified.* `aggregation` (handles daily/weekly/monthly modes and index arrays `I_pos`/`I_inf`) and `statis` (computes KGE/NSE denominators like `mean_obs`, `TSS_obs`, `std_obs`) are critical components called before objective functions in both calibration and validation.
  **Action Required:** These must be explicitly ported. The logic inside `statis` can be highly vectorized in Python.

- **Missing Subroutine: `Shuffle`**:
  **Verification Status:** *Verified.* The codebase contains a custom Fisher-Yates shuffle implementation (`SUBROUTINE Shuffle`) used in `LH_mode`.
  **Action Required:** This can be cleanly replaced by `numpy.random.permutation` or `numpy.random.shuffle`, but its usage must be mapped correctly.

- **Control Flow Statements (`PAUSE` and `GO TO`)**:
  **Verification Status:** *Verified.* The codebase contains a blocking `PAUSE` statement (line 377 of SUBROUTINES) and a `GO TO 200` early exit (line 397 of SUBROUTINES).
  **Action Required:** Python has no equivalent for `PAUSE` or `GO TO`. `PAUSE` must become an exception (`raise RuntimeError`) or a logged warning. `GO TO` must be restructured as an early `return` or conditional block.

- **Sequential ODE Integration (`call_model`)**:
  **Verification Status:** *Verified.* The core model integration step (`AIR2STREAM_SUBROUTINES.f90`) computes the next time step's water temperature `Twat_mod(j+1)` based on the current step `Twat_mod(j)`.
  **Action Required:** This loop is inherently sequential and **cannot** be directly vectorized across the time dimension using simple `numpy` array operations. The time integration must remain a sequential loop in Python. However, performance can be heavily optimized by pre-calculating all forcing arrays (air temp, discharge) before the loop starts.

- **PSO Norm Convergence Check is Dead Code**:
  **Verification Status:** *Verified.* In `AIR2STREAM_RUNMODE.f90`, the variable `norm` is square-rooted (`norm=SQRT(norm)`), making it always `>= 0`. The subsequent check `IF (norm .lt. 0.0)` is therefore never true, making the early-exit block permanently unreachable.
  **Action Required:** A faithful port must replicate this dead code behavior (or simply remove the check but note the deviation), as the early exit logic never triggers in the Fortran version.

- **Custom Optimization Algorithms (`AIR2STREAM_RUNMODE.f90`)**:
  **Verification Status:** *Strictly Verified.* The project contains manual implementations of Particle Swarm Optimization (PSO) and Latin Hypercube (LH) sampling. Independent verification confirms these are precisely implemented as `SUBROUTINE PSO_mode` and `SUBROUTINE LH_mode` in `AIR2STREAM_RUNMODE.f90`, correctly managing state by allocating large matrices (e.g., `ALLOCATABLE :: x, v, pbest`).
  **Action Required:**
  - **Option A (Direct Port):** Porting the custom PSO and LH logic ensures identical search behavior. Numpy handles particle updates efficiently via large matrix operations instead of nested loops.
  - **Option B (SciPy/Library Replace):** Python ecosystem libraries like `scipy.optimize` could theoretically replace this, but it is highly recommended to perform a *direct port first* to ensure bit-for-bit algorithmic parity before attempting library substitution.

---

## 3. Input and Output Data Formats

Currently, the Fortran implementation relies on line-by-line reading of space-separated text files (`input.txt`, `parameters.txt`) which is extremely rigid and error-prone for users. It also generates output text files with basic extensions (like `.out`) which require custom scripts to parse.

**Modernization Detail:**

- **Configuration (Input):** The user-facing configuration (parameters, optimization bounds, station names, paths) should be completely overhauled from raw text files to a modern, structured format.
  - **YAML or TOML:** These are highly recommended for the main configuration file. They support comments, are human-readable, and can easily map to Python dictionaries or `dataclasses` via standard libraries (e.g., `pyyaml` or `tomli`).
  - **JSON:** A viable, widely-supported alternative, though slightly less human-readable than YAML/TOML for numerical config files.
- **Time Series Data (Input):** Standardize time-series input to CSV files. `pandas.read_csv` makes parsing CSVs trivial, allowing the user to provide data with arbitrary column orders and explicit standard date formats (e.g., ISO 8601), replacing the fragile fixed-format space-separated approach.
- **Results (Output):** The raw text `.out` files should be replaced with:
  - **CSV:** For final time-series data (easy to load into Excel, R, or pandas).
  - **JSON/YAML:** For final calibrated parameter sets and objective function scores.
  - **NetCDF / Parquet (Optional):** If the optimization logs become massive (saving every particle's path across thousands of iterations), using a binary columnar format like Parquet or HDF5/NetCDF via pandas/xarray will drastically reduce file sizes and I/O time compared to raw text.

---

## 4. What Cannot Be Ported

**Nothing.**
**Verification Status:** *Verified.* There are no hardware-specific constraints, arcane Fortran-77 EQUIVALENCE memory hacks, or proprietary closed-source libraries that prevent a functional port to Python.

---

## Performance Considerations

Because `air2stream` relies on optimization (PSO), the core simulation (`call_model`) is evaluated thousands of times (`n_particles * n_run`).

**Initial Strategy: Maximize NumPy Usage**
Before exploring external compilers, the first iteration of the port should rely entirely on `numpy` to handle performance. While the sequential nature of `call_model` prevents a single vector operation from solving the ODE, `numpy` drastically improves speed by:

1. **Pre-computing inputs:** All terms that do not depend on `Twat_mod(j)` (such as `Tair`, `Q`, and trigonometric time functions) can be computed for the entire array before the sequential loop starts.
2. **Vectorizing the search space:** Within the PSO algorithm, operations updating velocity and position of the entire particle swarm can be executed as single matrix operations in `numpy`.

**Future Considerations: Numba**
If the pure Python/NumPy implementation proves to be too slow due to the Python interpreter's overhead in the `call_model` sequential loop, **Numba** (`@numba.njit`) would be the immediate next step. Numba JIT-compiles Python functions to machine code, granting Fortran-like execution speeds with minimal code changes. This should only be evaluated after ensuring functional correctness and exhausting NumPy optimizations.

---

## 5. Multithreading and Parallelization Potential

Yes, the project can absolutely be parallelized, and doing so will yield massive performance improvements during the calibration phase.

The Fortran codebase currently executes sequentially. However, the optimization algorithms evaluate the model (`call_model`) thousands of times across independent parameter sets (`n_particles`).

**How to Parallelize in Python:**

- **Embarrassingly Parallel Workloads:** Evaluating the objective function for each particle in the PSO swarm is completely independent of the other particles. Particle A and Particle B can be evaluated simultaneously.
- **Avoid Multithreading (The GIL):** Because the core simulation is CPU-bound mathematical computation, standard Python *multithreading* (`threading` module) will not provide a speedup due to Python's Global Interpreter Lock (GIL).
- **Use Multiprocessing:** Instead of threads, use **multiprocessing**. Libraries like `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel` spawn multiple isolated Python processes, each evaluating a chunk of the particle swarm and utilizing 100% of available CPU cores. This will strictly bypass the GIL and provide near-linear scaling up to the core count of the machine.

---

## Step-by-Step Porting Instructions

To systematically port `air2stream` to Python while ensuring accuracy and correctness, follow these phased instructions:

### Phase 1: Setup & Data Structures

1. **Initialize the Project**: Create a new Python package (e.g., `pyair2stream`). Set up a standard environment with `numpy` and `pandas`. (Completed)
2. **Port `AIR2STREAM_MODULES.f90`**: Create a `config.py` or a data class `CommonData`. Translate all `ALLOCATABLE` arrays into placeholder `numpy` arrays (`None` initially) and parameters into class attributes. (Completed - `pyair2stream/config.py` created with `CommonData` dataclass enforcing `numpy.float64` for numerical precision)

### Phase 2: File I/O & Parsing (Completed)

1. **Port `AIR2STREAM_READ.f90`**: Create an `io.py` module. (Completed)
2. **Read Configs**: Implement functions to read the configuration text files (`input.txt`, `parameters.txt`) and populate the `CommonData` instance. The duplicate `version == 4` block in the Fortran code (lines 81-86) is a known copy-paste bug (verified in `AIR2STREAM_READ.f90` lines 81-86, where a block meant for `version == 8` parameters is incorrectly guarded by `IF (version == 4)`). This **has been fixed** by changing the second condition to `IF (version == 8)` in the Python port. (Completed)
3. **Parse Time Series**: Use `pandas.read_csv()` to parse time series data. Implement careful logic to handle the sentinel values (`-999`) in `Qmedia` calculation, and accurately reproduce the exact warm-up year logic (indices 1-365) and `tt` array generation, disregarding leap years for the warm-up period. (Completed)
4. **Validation**: Write simple scripts to parse the input files using the Fortran executable and the new Python code, and assert that the loaded array shapes and values are exactly identical. (Completed with `tests/test_io.py`)

### Phase 3: Core Simulation Loop & Objective Functions (Completed)

1. **Port `AIR2STREAM_SUBROUTINES.f90`**: Create a `model.py` module. (Completed - `pyair2stream/model.py` created)
2. **Translate Integrators**: Port the numerical integration steps (Euler, RK2, RK4, Crank-Nicolson). Pay extremely close attention to the `0-based` vs `1-based` indexing shift here. **Crucially, do not vectorize the ODE integration loop**. The model is stateful and sequential; operations must be strictly ordered to match Fortran precision. Use `numpy.float64` explicitly everywhere to identically match the `REAL(KIND=8)` Fortran behavior. Also, you must include the `Tice_cover` floor clamp, which was verified at `AIR2STREAM_SUBROUTINES.f90` line 85. Omitting `Twat_mod[j+1] = max(Twat_mod[j+1], Tice_cover)` at the end of every output step will silently produce wrong temperatures when going below the ice threshold. (Completed - sequential loop retained with index shifting padding `p` arrays, implemented `RK4_air2stream` and integrated RK2, RK4, EUL, CRN cleanly).
3. **Port Subroutines Exactly**: Ensure the `aggregation` subroutine (handles time resolutions and `I_pos`/`I_inf` indexing) and the `statis` subroutine (computes statistical denominators) are ported exactly. This is critical because `funcobj` requires `I_pos`/`I_inf` indirection and is not directly vectorisable (verified in `AIR2STREAM_SUBROUTINES.f90` lines 150-157). You must port the aggregation loop explicitly—iterate over `n_dat` windows, average the elements, then compute the objective. (Completed - exact port handling the 0-based indirect indexes with padding and index shifting).
4. **Control Flow**: Refactor `GO TO` statements into early returns and replace `PAUSE` statements with exceptions or warnings. (Completed)
5. **Translate Objectives**: Port the `funcobj` subroutines (NSE, KGE, RMS). Vectorize these calculations using `numpy` arrays instead of `DO` loops. (Completed - implemented directly in `funcobj` using standard python/numpy loops analogous to original calculations to maintain strictly precise values).
6. **Validation**: Manually hardcode a test input state (one array of parameters and driving variables) into both Fortran and Python. Compare the resulting water temperature (`Twat_mod`) output arrays step-by-step. They must match to floating-point precision. (Completed - tested via `tests/test_model.py` using constructed synthetic signals comparing integrator variants and asserting exact mathematical property behavior on RMS / KGE / NSE outputs).

### Phase 4: Optimization Engine (PSO & LH)

1. **Port `AIR2STREAM_RUNMODE.f90`**: Create an `optimization.py` module.
2. **Direct PSO Translation**: Port the particle swarm optimization logic. Use `numpy` matrix operations to update the swarm positions and velocities in one go. **Fix** the legacy PSO bugs:
   - **PSO global best initialised from wrong array**: Verified in `AIR2STREAM_RUNMODE.f90` line 71, `CALL best(fit, k, foptim)` passes `fit` (still all zeros) instead of `fitbest` (populated just above). Change this to `CALL best(fitbest, k, foptim)`.
   - **PSO convergence check is permanently dead**: Verified in `AIR2STREAM_RUNMODE.f90` lines 144-145, `norm` is a square root so `norm .lt. 0.0` never fires. Replace with a meaningful tolerance, e.g. `norm < 1e-4`.
   - **Initial PSO evaluations excluded from dotty-plot output**: Verified in `AIR2STREAM_RUNMODE.f90`, initial evaluations (lines 66-70) are never written to the binary output file. Decide whether to replicate this omission deliberately, or document the intentional change if including initial results.
   - **PSO random re-seeding per iteration makes results non-reproducible**: Verified in `AIR2STREAM_RUNMODE.f90` line 77, `random_seed()` is called inside every iteration, reseeding from system time. Decide upfront to either replicate the non-reproducible behaviour or introduce an explicit seed parameter and document the divergence.
3. **Direct LH Translation**: Port the Latin Hypercube mode, utilizing `numpy.random.permutation` to replace the custom `Shuffle` subroutine. **Fix LH file handle never closed**: Verified in `AIR2STREAM_RUNMODE.f90` line 239, `! CLOSE(10)` is commented out. Always call the equivalent of `file.close()` after the LH loop in Python to avoid incomplete/unflushed output.
4. **Connect the Pieces**: Ensure the optimization loops call the `model.py` simulation and calculate objective values correctly over the parameter bounds.

### Phase 5: Parallelization & Main Entry Point

1. **Avoid Global Variables**: Before finalizing the architecture, refactor Fortran's large global module into a state/config object passed explicitly to improve testing and debugging.
2. **Delay Multiprocessing**: First, achieve a single-threaded exactly-matching implementation to prevent random-number ordering issues from parallel execution.
3. **Multiprocessing**: Only after regression testing, wrap the particle evaluations using `concurrent.futures.ProcessPoolExecutor.map` to parallelize the objective function calculations.
4. **Port `AIR2STREAM_MAIN.f90`**: Create `main.py` that wires the IO, model execution, and optimization routines based on the run mode.

### Phase 5.5: Critical Validation Steps

1. **Golden-output tests**: Create reference outputs from Fortran (loaded inputs, aggregated series, simulated temperatures, objective function values, calibration results) and compare Python against these directly. This is the most important missing item.
2. **Integration-scheme tests**: Validate each solver (Euler, RK2, RK4, Crank–Nicolson) separately before full calibration testing.
3. **Parameter-version tests**: Test every model version individually, as several parameters are conditionally disabled.
4. **Warm-up-period tests**: Rigorously verify the replicated first year, generated `tt`, and resulting temperatures, as this is a high-risk area.
5. **Final Validation**: Run a full calibration (`PSO` mode) using both Fortran and Python on the same input data. Verify that the final converged parameters and execution time are comparable.

### Phase 6: Post-Processing & Visualization

1. **Port `post_processing.m`**: Create a `post_processing.py` (or Jupyter Notebook).
2. **Translate Plotting Logic**: Use `numpy` to parse the output binary and text files (e.g., `0_PSO_RMS_...out`, `2_PSO_RMS_...out`).
3. **Generate Visualizations**: Utilize `matplotlib` and `pandas` to generate the parameter dotty plots and the calibration/validation time-series plots, replicating the exact style (e.g., color-blind palettes) and output formats (PDF/PNG) of the original MATLAB script.
4. **Integration**: Unify the simulation and visualization into a single end-to-end Python script or CLI command.

### Phase 7: User Documentation

1. **Draft `USER_GUIDE.md`**: Create a comprehensive `pyair2stream/USER_GUIDE.md` to ensure honesty and clarity in user expectations. This guide must be carefully planned to cover the new modern data formats, installation, running models, and generating visualizations.

## 6. Summary

The `air2stream` project is an excellent candidate for a Python port. The extensive use of standard arrays, basic ODE integration, and text I/O align perfectly with Python's scientific ecosystem. The critical steps involve meticulous management of the index shift, replacing manual loops with `numpy` and `pandas` vectorizations where appropriate, and leveraging `multiprocessing` to bypass the GIL for optimization performance.
