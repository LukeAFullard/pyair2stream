# Differential Evolution + L-BFGS-B Calibration Mode in pyair2stream

## 1. Overview

`pyair2stream` is a Python reimplementation of the Fortran `air2stream` model (Toffolon and Piccolroaz, 2015), which simulates daily river water temperature from air temperature and, optionally, discharge. The original Fortran implementation offers two calibration algorithms: Particle Swarm Optimization (PSO) and Latin Hypercube sampling (LATHYP). `pyair2stream` retains both for backward compatibility but introduces a third calibration mode, `DE`, which is set as the recommended default.

`DE` mode performs a two-phase hybrid optimization:

1. **Global search** using Differential Evolution (DE; Storn and Price, 1997), which explores the full parameter space without requiring gradient information or a good initial guess.
2. **Local polish** using the L-BFGS-B quasi-Newton method, which refines the DE solution to high precision within the same parameter bounds.

This document describes the algorithm, its implementation in `pyair2stream`, its configuration, its outputs, and its empirical performance relative to PSO. It is intended to serve as a citable technical description of this feature for use in derivative scientific work.

## 2. Motivation

The original `air2stream` model is calibrated with PSO with inertia weight, an algorithm now several decades old. Piotrowski and Napiorkowski (2018) evaluated twelve calibration methods for `air2stream` and found that the choice of optimizer materially affects model performance, and that PSO with inertia weight was not competitive with more modern global optimization strategies. DE variants performed favourably in that comparison.

`pyair2stream` addresses this by replacing PSO as the default optimizer with a DE-based hybrid scheme. The motivation is threefold:

- **Robustness**: DE is a population-based, derivative-free global optimizer that is less prone than PSO to premature convergence on multimodal objective surfaces, which are expected given the equifinality known to affect `air2stream`'s 7- and 8-parameter formulations.
- **Precision**: DE alone converges to the neighbourhood of the global optimum but is comparatively slow to refine a solution to high numerical precision. Appending a local, gradient-based polish step (L-BFGS-B) after DE addresses this without sacrificing the global-search robustness of the first phase.
- **Efficiency**: in internal benchmarking (Section 8), the hybrid DE approach reaches an equivalent or better objective value than PSO in a small fraction of the wall-clock time and function evaluations.

## 3. Algorithm Description

### 3.1 Phase 1 — Differential Evolution (global search)

DE is implemented via SciPy's `scipy.optimize.differential_evolution`. DE maintains a population of candidate parameter vectors and iteratively improves it over a fixed number of generations. For each candidate ("target vector") in the population, a "trial vector" is constructed by:

1. **Mutation**: selecting other members of the population and combining them (in the default `best1bin` strategy used here, by perturbing the current best-known vector with the scaled difference of two randomly chosen population members).
2. **Crossover (recombination)**: mixing components of the mutant vector with the target vector according to a crossover probability, producing the trial vector.
3. **Selection**: evaluating the objective function for the trial vector and replacing the target vector in the population if the trial is better.

This process repeats for a specified number of generations, or until a convergence tolerance is met. Because DE only requires objective function evaluations (no gradients) and works with a full population rather than a single point, it is well suited to non-smooth, multimodal, or noisy objective surfaces, which is the expected character of the `air2stream` calibration problem given its stiff ODE integration and known parameter equifinality.

`pyair2stream` calls `differential_evolution` with the following configuration:

| Argument | Value | Meaning |
|---|---|---|
| `bounds` | Per-parameter `(min, max)` pairs from the config file | Defines the 8-dimensional search box |
| `maxiter` | `data.n_run` (config: `optimization.n_run`) | Maximum number of generations |
| `popsize` | `data.n_particles` (config: `optimization.n_particles`) | Population size multiplier |
| `workers` | `1` | Single-process evaluation (see Section 7) |
| `polish` | `False` | Disables SciPy's own internal polishing step, since `pyair2stream` performs its own explicit L-BFGS-B phase afterward |
| `seed` | User-supplied or `None` | Controls reproducibility of the stochastic search |

All other DE hyperparameters (mutation constant range, crossover/recombination probability, initialization scheme, and convergence tolerance) are left at SciPy's defaults (`best1bin` strategy, dithered mutation in `(0.5, 1.0)`, recombination probability `0.7`, Latin hypercube initialization, relative tolerance `0.01`).

### 3.2 Phase 2 — L-BFGS-B (local polish)

The best parameter vector returned by DE, `result_de.x`, is passed as the initial guess to `scipy.optimize.minimize` with `method="L-BFGS-B"`, using the same parameter bounds. L-BFGS-B is a limited-memory, quasi-Newton, box-constrained optimizer. It approximates the objective function's curvature from a short history of gradient evaluations (gradients are obtained by SciPy via finite differences, since the `air2stream` objective is not differentiated analytically) and takes a sequence of bounded Newton-like steps to converge on a local optimum.

Because DE Phase 1 has already located the basin of the global optimum, L-BFGS-B in Phase 2 typically converges to a highly precise solution within that basin in a small number of iterations. This division of labour — DE for global exploration, L-BFGS-B for local exploitation — is a standard hybrid metaheuristic design used to combine the robustness of stochastic global search with the convergence speed of gradient-based local search.

### 3.3 Objective Function Handling

`air2stream` supports three objective (goodness-of-fit) functions, selected via `objective_function` in the config file: Nash–Sutcliffe Efficiency (NSE), Kling–Gupta Efficiency (KGE), and root-mean-square error (RMS). All three are defined such that **larger values indicate a better fit** in the model's internal convention — RMS is stored internally as its negation, `-RMSE`, so that all three objectives are consistently subject to maximization.

Because SciPy's optimizers are formulated as minimizers, `DE_mode` wraps the model evaluation function in an `objective_wrapper` that negates the returned efficiency index before passing it to `differential_evolution` and `minimize`:

```
objective_wrapper(p) = -efficiency_index(p)          if efficiency_index(p) is defined
objective_wrapper(p) = 1e30                            if efficiency_index(p) is NaN
```

The large finite penalty (`1e30`) is used in place of `NaN` or `inf` because SciPy's optimizers require finite, comparable objective values; returning `NaN` would otherwise propagate through comparisons in the DE population update and corrupt the search. Parameter sets that trigger a NaN objective (for example, due to numerical instability in the ODE integrator for extreme parameter combinations) are therefore penalized as strongly suboptimal rather than causing a failure. This mirrors the same NaN-handling logic used in `pyair2stream`'s PSO implementation.

### 3.4 Fixed and Degenerate Parameters

Some `air2stream` model versions (3, 4, 5, and 7) do not use the full 8-parameter set; unused parameters are fixed by setting their lower and upper bounds equal (`parmin[j] == parmax[j]`), and the corresponding entry in the boolean array `flag_par` is set to `False`. `differential_evolution` requires strictly non-degenerate bounds (`lower < upper`) for every dimension. `DE_mode` therefore detects any parameter that is either flagged inactive (`flag_par[j] is False`) or has `parmin[j] == parmax[j]`, and substitutes a bound of `(parmin[j], parmin[j] + 1e-12)` — an interval narrow enough to be numerically fixed, but wide enough to avoid a zero-width bound that would cause `differential_evolution` to raise an error. After optimization, these parameters are snapped back exactly to `parmin[j]`, removing the epsilon before the final objective evaluation and before parameters are recorded.

## 4. Implementation Summary

The core routine is `DE_mode(data, seed=None)` in `pyair2stream/optimization.py`, where `data` is the `CommonData` configuration/state object shared across all calibration modes. Its control flow is:

1. Seed the NumPy random generator if a `seed` is supplied.
2. Construct per-parameter bounds from `data.parmin`, `data.parmax`, and `data.flag_par`, applying the fixed-parameter epsilon adjustment described above.
3. Run `differential_evolution` (Phase 1) with `maxiter = data.n_run` and `popsize = data.n_particles`, `polish=False`, `workers=1`.
4. Run `minimize(..., method="L-BFGS-B")` (Phase 2), initialized at the Phase 1 solution, using identical bounds.
5. Snap any fixed parameters exactly back to their bound values.
6. Re-evaluate the model once more at the final parameter vector to populate `data.par_best` and `data.finalfit` and to guarantee internal consistency between the reported objective value and the reported parameters.
7. Write every parameter set evaluated during the search — for which the objective exceeded the user-configured acceptability threshold `mineff_index` — to a history CSV file (`0_DE_<objective>_<station>_<series>_<time_res>.csv`), together with the corresponding NSE, R², and MAE diagnostic statistics for that evaluation.

Every candidate parameter set evaluated by either phase passes through the same underlying model call (`sub_1`, which wraps `call_model` and `funcobj`), so the physical model and objective function are identical to those used by PSO and LATHYP; only the search strategy differs.

## 5. Configuration

`DE` is selected via `run_mode` in the YAML configuration file:

```yaml
run_mode: "DE"

optimization:
  n_run: 100          # maximum number of DE generations
  n_particles: 50     # DE population size multiplier

mineff_index: 0.0      # minimum objective value for a parameter set to be
                        # retained in the "0_*.csv" history output
```

`n_run` and `n_particles` map directly onto DE's `maxiter` and `popsize` arguments. Note that the total number of model evaluations during Phase 1 is approximately `popsize × n_dimensions × n_run` (the exact count also depends on early convergence against SciPy's internal tolerance), where `n_dimensions` is 8 for the full parameter set; the L-BFGS-B polish in Phase 2 adds a comparatively small number of additional evaluations.

`DE` mode can also be invoked programmatically:

```python
from pyair2stream.optimization import DE_mode

DE_mode(data, seed=42)   # populates data.par_best and data.finalfit
```

## 6. Reproducibility

`DE_mode` accepts an optional `seed` argument, which is passed both to `numpy.random.seed()` and directly to `differential_evolution`'s own `seed` parameter. Supplying an explicit seed makes the stochastic global-search phase, and therefore the full two-phase calibration, deterministic and reproducible across runs, which is the configuration used for the benchmarking reported in Section 8 and recommended for any calibration intended to support a scientific publication.

## 7. Related Modes Built on DE

Two additional `run_mode` options extend `DE_mode`, reusing its result rather than re-implementing the search:

- **`DE-MCMC`**: runs `DE_mode` to obtain a best-fit parameter vector, then uses this as the starting point for an ensemble MCMC sampler (via `emcee`) to characterize posterior parameter uncertainty and to generate predictive uncertainty envelopes.
- **`DE-CV-MCMC`**: runs `DE_mode`, then performs leave-one-year-out block cross-validation, using the cross-validation results to inform the subsequent MCMC initialization.

Both inherit the global-search robustness and computational efficiency of the DE + L-BFGS-B hybrid described in this document.

## 8. Empirical Performance

### 8.1 Optimizer Comparison (Switzerland/Mentue dataset)

Using a fixed random seed for a like-for-like comparison, and an intentionally over-specified PSO configuration (500 particles, 500 iterations = 250,000 evaluations) to give PSO its best chance of converging:

| Optimizer | Wall-clock time (s) | Best objective (NSE) |
|---|---|---|
| PSO (500 particles, 500 iterations) | 170.51 | 0.987811 |
| DE + L-BFGS-B | 2.75 | 0.987811 |
| DE-MCMC | 44.52 | 0.987811 |

DE + L-BFGS-B reached an NSE statistically indistinguishable from a heavily over-provisioned PSO run, in roughly 1.6% of the wall-clock time. The additional time for `DE-MCMC` reflects the subsequent 2000-step, 32-walker MCMC sampling phase used for uncertainty quantification, not the DE search itself.

### 8.2 Convergence Behaviour with Iteration Count

A separate experiment varied the iteration budget (10 to 5000 iterations, 20 particles) for both PSO and DE:

- At 5000 iterations, DE reached NSE = 0.95270 versus PSO's 0.95255.
- DE's parameter estimates stabilized earlier in the iteration sequence than PSO's, indicating faster practical convergence at a given computational budget.

### 8.3 Validation Against Literature Parameter Sets

Independent re-calibration with DE against three Swiss river datasets and literature-derived parameter sets (Piccolroaz et al., 2016) gave:

| River (station) | Flow regime | Literature NSE | pyair2stream NSE (DE) |
|---|---|---|---|
| Mentue (MAH-2369) | Natural | 0.989 | 0.9886 |
| Rhône (SIO-2011) | Regulated | 0.923 | 0.9242 |
| Dischmabach (DAV-2327) | Snow-fed | 0.950 | 0.9558 |

In the same study, PSO was observed to converge to different parameter combinations than DE for the 7- and 8-parameter model versions, consistent with known equifinality in `air2stream`'s parameter space; DE and DE-MCMC matched the literature parameter sets more closely than PSO across all three catchments.

## 9. Practical Considerations and Limitations

- **Single-process evaluation**: `DE_mode` runs with `workers=1`. The objective function passed to SciPy is a closure (`objective_wrapper`) that captures local state (the shared `data` object and the evaluation `history` list), which is not picklable and therefore cannot be distributed across multiple worker processes using SciPy's built-in `workers` parallelization. In practice this has not been a binding constraint: the benchmarking in Section 8 shows the hybrid DE approach completing in a small fraction of the time required by PSO even without parallel workers.
- **Equifinality**: as with any calibration of the 7- or 8-parameter `air2stream` formulations, multiple parameter combinations can yield similar objective values. DE's population-based global search reduces (but does not eliminate) the risk of settling on a poor local optimum relative to single-point methods; users working toward publication-quality results should still inspect dotty plots and consider `DE-MCMC` for a fuller picture of parameter identifiability.
- **Recommended use**: for scientific or publication use, `DE` or `DE-MCMC` is preferred over `PSO` unless PSO convergence has been independently verified for the dataset in question.

## 10. References

- Storn, R. and Price, K. (1997). Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11(4), 341–359.
- Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *Environmental Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011
- Piotrowski, A. P. and Napiorkowski, J. J. (2018). Performance of the air2stream model that relates air and stream water temperatures depends on the calibration method. *Journal of Hydrology*, 561, 395–412.
- Piccolroaz, S., Calamita, E., Majone, B., Gallice, A., Siviglia, A., and Toffolon, M. (2016). Prediction of river water temperature: a comparison between a new family of hybrid models and statistical approaches. *Hydrological Processes*, 30(21), 3901–3917.
- Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261–272. (`scipy.optimize.differential_evolution`, `scipy.optimize.minimize` with `method="L-BFGS-B"`)
- Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). A Limited Memory Algorithm for Bound Constrained Optimization. *SIAM Journal on Scientific Computing*, 16(5), 1190–1208.

## Appendix: Source Reference

Implementation: `pyair2stream/optimization.py`, function `DE_mode` (with supporting dispatch in `pyair2stream/main.py::run_optimizer` and configuration in `pyair2stream/config.py::CommonData`). Repository: https://github.com/LukeAFullard/pyair2stream.
