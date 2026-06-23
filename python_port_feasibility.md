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

---

## 1. What Can Be Easily Ported

Most of the codebase translates very naturally to Python:

- **Data Structures & Global Variables (`AIR2STREAM_MODULES.f90`)**:
  **Verification Status:** *Verified.* The `commondata` module extensively uses `ALLOCATABLE` arrays and simple global scalars.
  **Modernization Detail:** The global `commondata` module can be elegantly replaced by a Python `dataclass`, a standard class, or a configuration dictionary. Numpy arrays (`np.ndarray`) will perfectly replace Fortran's `ALLOCATABLE` dynamic arrays, offering better memory management and vectorization capabilities without boilerplate allocation logic.

- **File I/O and Text Parsing (`AIR2STREAM_READ.f90`)**:
  **Verification Status:** *Verified.* The Fortran code reads space-separated values line-by-line and implements a custom `leap_year` subroutine for date calculations.
  **Modernization Detail:**
  - Reading text configurations line-by-line (`input.txt`, `parameters.txt`) is handled natively in Python with `with open(...)` and simple string splits.
  - The reading of time-series data can be massively simplified using `pandas.read_csv(..., sep='\s+')`.
  - Fortran's manual leap-year checking logic (`leap_year` subroutine) and manual day-of-year array population can be completely eliminated. Python's `pandas.to_datetime` or `numpy.datetime64` handles calendar logic intrinsically, reducing dozens of lines of error-prone date manipulation code to just a few robust function calls.

- **System Calls / Dependencies (`USE ifport`)**:
  **Verification Status:** *Verified.* A search confirms `USE ifport` is exclusively used in `AIR2STREAM_READ.f90` for the `makedirqq` function.
  **Modernization Detail:** This dependency is trivial to replace using `os.makedirs(folder, exist_ok=True)` or `pathlib.Path(folder).mkdir(parents=True, exist_ok=True)`. There are no other compiler-specific or non-standard dependencies.

- **Objective Functions (`funcobj` in `AIR2STREAM_SUBROUTINES.f90`)**:
  **Verification Status:** *Verified.* The codebase calculates Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE), and Root Mean Square (RMS) via manual `DO` loops over arrays.
  **Modernization Detail:** These calculations involve array accumulations and statistics (mean, variance, standard deviation). In Python, these can be completely vectorized using `numpy.mean`, `numpy.var`, and standard vector arithmetic (e.g., `np.sum((obs - mod)**2)`), making the Python version significantly shorter, cleaner, and faster than explicit Fortran loops.

---

## 2. What Is More Difficult / Requires Care

While entirely possible, the following aspects will require careful translation to avoid bugs or performance regressions:

- **0-Based vs. 1-Based Indexing**:
  **Verification Status:** *Critical Note.* Fortran arrays are 1-based by default, and slice operations are inclusive on both ends. Python arrays are 0-based, and slices are exclusive on the upper bound.
  **Action Required:** The mathematical logic uses indices heavily (e.g., `Twat_mod(j+1) = Twat_mod(j) + ...`). The porting effort must meticulously map index bounds. If a loop runs `DO j=1, n_tot-1`, the Python equivalent `range(0, n_tot - 1)` must be checked so the indices align correctly with array shapes.

- **Sequential State Updates (The Time Loop)**:
  **Verification Status:** *Verified.* The simulation loop (`call_model`) sequentially calculates water temperature at `j+1` using RK4, RK2, Euler, or Crank-Nicolson, relying on the state at `j`.
  **Action Required:** Because the state at step `t+1` depends strictly on `t`, the final temporal integration loop cannot be fully vectorized. However, inputs (like air temperature and discharge evaluations) and non-recursive terms can be pre-vectorized using `numpy`. The remaining sequential ODE step must be written clearly, prioritizing algorithmic correctness over premature vectorization attempts that might violate causality.

- **Custom Optimization Algorithms (`AIR2STREAM_RUNMODE.f90`)**:
  **Verification Status:** *Verified.* The project contains manual implementations of Particle Swarm Optimization (PSO) and Latin Hypercube (LH) sampling, allocating large matrices (e.g., `ALLOCATABLE :: x, v, pbest`).
  **Action Required:**
  - **Option A (Direct Port):** Porting the custom PSO and LH logic ensures identical search behavior. Numpy handles particle updates efficiently via large matrix operations instead of nested loops.
  - **Option B (SciPy/Library Replace):** Python ecosystem libraries like `scipy.optimize` could theoretically replace this, but it is highly recommended to perform a *direct port first* to ensure bit-for-bit algorithmic parity before attempting library substitution.

---

## 3. What Cannot Be Ported

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

## 4. Multithreading and Parallelization Potential

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
1. **Initialize the Project**: Create a new Python package (e.g., `pyair2stream`). Set up a standard environment with `numpy` and `pandas`.
2. **Port `AIR2STREAM_MODULES.f90`**: Create a `config.py` or a data class `CommonData`. Translate all `ALLOCATABLE` arrays into placeholder `numpy` arrays (`None` initially) and parameters into class attributes.

### Phase 2: File I/O & Parsing
1. **Port `AIR2STREAM_READ.f90`**: Create an `io.py` module.
2. **Read Configs**: Implement functions to read the configuration text files (`input.txt`, `parameters.txt`) and populate the `CommonData` instance.
3. **Parse Time Series**: Replace manual loop-based data reading and the `leap_year` subroutine with `pandas.read_csv()` to parse time series data, ensuring the dates map cleanly to a standard `datetime` index.
4. **Validation**: Write simple scripts to parse the input files using the Fortran executable and the new Python code, and assert that the loaded array shapes and values are exactly identical.

### Phase 3: Core Simulation Loop & Objective Functions
1. **Port `AIR2STREAM_SUBROUTINES.f90`**: Create a `model.py` module.
2. **Translate Integrators**: Port the numerical integration steps (Euler, RK2, RK4, Crank-Nicolson). Pay extremely close attention to the `0-based` vs `1-based` indexing shift here.
3. **Translate Objectives**: Port the `funcobj` subroutines (NSE, KGE, RMS). Vectorize these calculations using `numpy` arrays instead of `DO` loops.
4. **Validation**: Manually hardcode a test input state (one array of parameters and driving variables) into both Fortran and Python. Compare the resulting water temperature (`Twat_mod`) output arrays step-by-step. They must match to floating-point precision.

### Phase 4: Optimization Engine (PSO)
1. **Port `AIR2STREAM_RUNMODE.f90`**: Create an `optimization.py` module.
2. **Direct PSO Translation**: Port the particle swarm optimization logic. Use `numpy` matrix operations to update the swarm positions and velocities in one go.
3. **Connect the Pieces**: Ensure the PSO loop calls the `model.py` simulation and calculates objective values correctly over the parameter bounds.

### Phase 5: Parallelization & Main Entry Point
1. **Multiprocessing**: In the PSO implementation, wrap the particle evaluations using `concurrent.futures.ProcessPoolExecutor.map` to parallelize the objective function calculations.
2. **Port `AIR2STREAM_MAIN.f90`**: Create `main.py` that wires the IO, model execution, and optimization routines based on the run mode.
3. **Final Validation**: Run a full calibration (`PSO` mode) using both Fortran and Python on the same input data. Verify that the final converged parameters and execution time are comparable.

## Summary
The `air2stream` project is an excellent candidate for a Python port. The extensive use of standard arrays, basic ODE integration, and text I/O align perfectly with Python's scientific ecosystem. The critical steps involve meticulous management of the index shift, replacing manual loops with `numpy` and `pandas` vectorizations where appropriate, and leveraging `multiprocessing` to bypass the GIL for optimization performance.
