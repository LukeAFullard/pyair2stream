# One-at-a-Time Local Sensitivity Analysis in pyair2stream

## 1. Overview

`pyair2stream` is a Python reimplementation of the Fortran `air2stream` model (Toffolon and Piccolroaz, 2015), which simulates daily river water temperature from air temperature and, optionally, discharge, using an 8-parameter (`a1`–`a8`) ordinary differential equation (ODE) heat-budget formulation. The original Fortran implementation returns only a calibrated parameter vector and its goodness-of-fit; it provides no built-in mechanism for characterizing how strongly the simulated output responds to each individual parameter.

`pyair2stream` adds an optional one-at-a-time (OAT) local sensitivity analysis, enabled with `sensitivity_analysis: true`, that does not exist in the Fortran reference implementation. Around the calibrated best-fit parameter vector, each active parameter is perturbed independently by a small percentage of its own value, the model is re-simulated, and the resulting mean absolute change in simulated water temperature is reported as that parameter's sensitivity index. This document describes the algorithm, its statistical interpretation and known limitations, its implementation, its configuration, its outputs, and empirical results from its application to a real calibrated dataset. It is intended to serve as a citable technical description of this feature for use in derivative scientific work.

## 2. Motivation

`air2stream`'s 7- and 8-parameter formulations are known to exhibit equifinality, where several different parameter combinations produce similar goodness-of-fit. A local sensitivity analysis around the calibrated optimum provides a complementary, computationally cheap diagnostic to this: it identifies which of the calibrated parameters the simulated water temperature actually depends on strongly at that point in parameter space, versus which parameters have comparatively little influence on the output there. This is useful for two practical purposes: prioritizing which `parameter_bounds` are worth tightening (a parameter with low local sensitivity is unlikely to gain much from a narrower bound, whereas a highly sensitive parameter's bounds warrant particular scrutiny), and building physical intuition about which of the governing equation's terms (constant offset, air-temperature coupling, heat-loss damping, discharge-dependent terms, seasonal cycle) are actually driving the fitted response for a given catchment.

## 3. Algorithm

### 3.1 Design: Local, One-at-a-Time Perturbation

The method implemented is a local OAT sensitivity analysis: sensitivities are computed only in the immediate neighbourhood of the single calibrated parameter vector `par_best`, and only one parameter is varied at a time while all others are held at their calibrated value. This is distinct from global sensitivity analysis methods (e.g., Sobol indices, the Morris method), which explore the full parameter space and can attribute output variance to parameter interactions; OAT sensitivity is cheaper (requiring only two additional model evaluations per parameter per perturbation magnitude) but characterizes local behaviour around one point only, and cannot detect sensitivity that arises from interaction between two or more parameters (Section 7).

### 3.2 Perturbation Construction

For each active parameter `a_j` (indexed 0–7 internally, `par_1`–`par_8` in output), the perturbation magnitude is scaled by the parameter's own calibrated value, not by the width of its configured bounds:

```
base_scale = max(|par_best[j]|, 1e-4)
delta = (perturbation_pct / 100) * base_scale
p_plus  = clip(par_best[j] + delta, parmin[j], parmax[j])
p_minus = clip(par_best[j] - delta, parmin[j], parmax[j])
```

A floor of `1e-4` is applied to `base_scale` to avoid a degenerate, exactly-zero perturbation for a parameter whose calibrated value happens to be exactly (or very near) zero. If either `p_plus` or `p_minus` would fall outside the parameter's configured bounds, it is clipped to that bound instead — this occurs, for example, whenever the calibrated value sits close to `parmin[j]` or `parmax[j]` and a full-size perturbation in that direction would exceed the feasible range. The resulting `actual_delta = p_plus - p_minus` is the true, bound-respecting span between the two evaluated points, which may therefore be asymmetric or narrower than `2 x delta` for a parameter calibrated near one of its bounds.

Multiple perturbation magnitudes may be requested in a single run via `sensitivity_perturbations` (e.g., `[1.0, 2.0, 5.0]`, meaning 1%, 2%, and 5% of each parameter's own calibrated value); comparing the resulting sensitivity indices across magnitudes is a simple, practical check of local linearity around the optimum (Section 6).

### 3.3 Simulating the Perturbed Response

For each parameter and each requested perturbation percentage, the model is simulated twice — once with that single parameter set to `p_plus` and once set to `p_minus`, with every other parameter held at `par_best` — using the same integrator, model version, and (if applicable) gap-tolerant segmentation as the original calibration run. The resulting simulated water temperature time series (`Twat_mod` for the `p_plus` and `p_minus` runs) are compared day-by-day.

### 3.4 Scoring Window

The comparison is restricted to days that are both physically meaningful and part of the genuinely evaluated record:

- The standard one-year warm-up prefix is always excluded (the first 365 internal rows in non-gap-tolerant mode; in gap-tolerant mode, the first `warmup_drop_days` of every retained segment, matching exactly the `eval_mask` construction used elsewhere in the package — see the gap-tolerant mode documentation).
- Only days with a genuine `T_water` observation are included, so the sensitivity index reflects the model's behaviour specifically over the period being used to judge its fit, not over days where no ground truth exists to contextualize the comparison.
- A numerical-stability filter excludes any day where either the `p_plus` or `p_minus` simulation produced a value above 50°C. This guards against a known artefact in which an aggressively perturbed parameter — particularly the discharge-exponent term `a4`, which scales a rating-curve-like exponent — can push the chosen numerical integrator into an unstable regime, producing physically impossible temperature spikes that would otherwise dominate and distort the mean-difference calculation. If every scoring day is excluded by this filter for a given parameter and perturbation magnitude, the sensitivity index for that combination is reported as undefined (`NaN`) rather than a misleading value derived from an unstable simulation.

### 3.5 Sensitivity Index

Over the remaining (stable, in-window, observed) days, the sensitivity index is the mean absolute difference between the two perturbed simulations, rescaled to correct for any bound-induced shrinkage of the actual perturbation span:

```
sensitivity_index = mean(|Twat_mod(p_plus) - Twat_mod(p_minus)|) / (actual_delta / base_scale)
```

The denominator, `actual_delta / base_scale`, equals `2 x (perturbation_pct / 100)` whenever neither `p_plus` nor `p_minus` was clipped by a bound, so in the unclipped case the index is simply the mean absolute response divided by the nominal (symmetric) perturbation fraction requested. When one side has been clipped (calibrated value near a bound), this rescaling compensates for the resulting narrower `actual_delta`, so that sensitivity indices remain comparable across parameters regardless of how close each one's calibrated value sits to its bound — though see Section 7 for a caution on interpreting sensitivities computed this way very close to a bound. The reported unit is degrees Celsius of simulated water temperature change corresponding to the nominal requested perturbation fraction of the parameter's own calibrated value.

### 3.6 Inactive and Fixed Parameters

Parameters that are not used by the selected model version (`flag_par[j] is False`) are reported with `Sensitivity_Index = NaN` and `Status = "Inactive"`, without running any additional model evaluations for them. Parameters that are active but have a degenerate range (`parmax[j] - parmin[j] <= 0`), or whose perturbation could not produce any actual separation between `p_plus` and `p_minus` after bound-clipping, are reported with `Sensitivity_Index = 0.0` and `Status = "Fixed"`. All other parameters are reported with `Status = "Active"` and a numeric (or `NaN`, if fully filtered by the stability check) sensitivity index.

## 4. Implementation Summary

The routine is `sensitivity_analysis(data)` in `pyair2stream/sensitivity.py`, invoked automatically by the CLI entry point (`main.py`) after the main calibration run completes, whenever `sensitivity_analysis: true` is set. Its control flow is:

1. Re-read the calibration-period time series (`read_Tseries(data, 'c')`), since a preceding validation or forward run may have altered `data`'s internal arrays; sensitivity analysis is always performed against the calibration record.
2. Set `data.par` to `data.par_best` and run the baseline simulation once (re-detecting gap-tolerant segments if not already available).
3. Construct the scoring window (`valid_mask`) as described in Section 3.4.
4. For every combination of requested perturbation percentage and the eight parameter slots, either record the parameter as `Inactive`/`Fixed` (Section 3.6), or run the `p_plus`/`p_minus` simulations, apply the stability filter, and compute the sensitivity index (Sections 3.2–3.5).
5. Restore `data.par` to `data.par_best` and re-run the model once more, so the calibration state is left consistent for any subsequent step in the pipeline.
6. Write the full results table to a CSV file and generate a grouped bar-chart visualization (Section 5).

## 5. Configuration and Outputs

```yaml
sensitivity_analysis: true
sensitivity_perturbations: [1.0, 2.0, 5.0]   # percent of each parameter's own calibrated value
```

`sensitivity_analysis` defaults to `false`, and `sensitivity_perturbations` defaults to `[1.0]` (a single 1% perturbation) if the key is omitted while `sensitivity_analysis` is enabled. Both fields sit at the top level of the YAML configuration, alongside `gap_tolerant` and the other top-level run-mode settings.

Two output files are produced in the run's output folder:

| Output | Contents |
|---|---|
| `sensitivity_<run_mode>_<objective>_<station>.csv` | One row per (parameter, perturbation percentage) combination: `Parameter` (`par_1`–`par_8`), `Perturbation_%`, `Sensitivity_Index`, `Status` (`Active`, `Fixed`, or `Inactive`) |
| `sensitivity_<run_mode>_<objective>_<station>.png` / `.pdf` | A grouped bar chart of the sensitivity index for every active parameter, with one bar group per requested perturbation percentage |

## 6. Empirical Application

The sensitivity analysis was applied to a Version-8 (full 8-parameter) PSO calibration of the Dischmabach dataset (station DAV-2327, Switzerland), with `sensitivity_perturbations: [1.0, 2.0, 5.0]`.

| Parameter | Physical role | Sensitivity @ 1% | Sensitivity @ 2% | Sensitivity @ 5% |
|---|---|---|---|---|
| `a1` (par_1) | Constant offset | 0.140 | 0.140 | 0.140 |
| `a2` (par_2) | Air-temperature coupling | 0.989 | 0.989 | 0.989 |
| `a3` (par_3) | Linear heat-loss (damping toward equilibrium) | 1.393 | 1.401 | 1.467 |
| `a4` (par_4) | Discharge-exponent (effective thermal capacity) | 2.491 | 3.087 | 6.329 |
| `a5` (par_5) | Discharge-scaled constant offset | 2.635 | 2.636 | 2.638 |
| `a6` (par_6) | Seasonal-cycle amplitude | 1.030 | 1.030 | 1.030 |
| `a7` (par_7) | Seasonal-cycle phase | 4.298 | 4.297 | 4.283 |
| `a8` (par_8) | Discharge-scaled heat-loss | 4.676 | 5.171 | 6.156 |

(Sensitivity index units: degrees Celsius of mean absolute change in simulated water temperature, per the nominal requested perturbation fraction of that parameter's own calibrated value, as defined in Section 3.5.)

### 6.1 Ranking and Physical Interpretation

At the smallest (1%) perturbation, the seasonal-phase term `a7` and the discharge-scaled heat-loss term `a8` produce the largest simulated response, followed by the discharge-scaled constant offset `a5` and the discharge-exponent term `a4`; the constant offset `a1` is, by a wide margin, the least influential parameter for this dataset. This is consistent with a catchment where the river's thermal regime is strongly shaped by its seasonal cycle and by discharge-mediated heat exchange, rather than by a simple additive offset.

### 6.2 Local Linearity Across Perturbation Magnitudes

Parameters `a1`, `a2`, `a5`, `a6`, and `a7` show sensitivity indices that change by less than 1% across the 1%, 2%, and 5% perturbation magnitudes, indicating the model's response to these parameters is closely linear in the immediate neighbourhood of the calibrated optimum. Parameters `a3`, `a4`, and `a8` — all of which enter the governing equation through discharge-dependent terms — show a visibly increasing sensitivity index as the perturbation magnitude grows, most sharply for `a4` (2.49 at 1%, rising to 6.33 at 5%, a 2.5-fold increase). This is consistent with genuine local nonlinearity in the discharge-coupled terms of the governing equation, compounded by the stability-filtering behaviour described in Section 3.4: larger perturbations of the discharge-exponent term in particular are more likely to push a subset of days into the numerically unstable regime that the >50°C filter screens out, so the reported index reflects the response averaged only over the days that remained numerically stable at each magnitude. Comparing sensitivity indices across more than one perturbation magnitude, as done here, is therefore a useful and recommended check before concluding that a single-magnitude sensitivity ranking is representative.

## 7. Practical Considerations and Limitations

- **Local, not global.** All results characterize only the immediate neighbourhood of the single calibrated parameter vector; a different (potentially equally well-fitting, given known equifinality) parameter vector could exhibit a materially different sensitivity ranking. This method does not explore the full parameter space and should not be used to draw conclusions about global parameter identifiability — the cross-validation and MCMC-based uncertainty features are better suited to that question (see the accompanying documentation for those features).
- **No parameter interactions.** Because only one parameter is varied at a time, this method cannot detect sensitivity that emerges from the interaction of two or more parameters; a parameter that appears locally insensitive when varied alone could still matter jointly with another. Global variance-based methods (e.g., Sobol indices) or elementary-effects screening (e.g., the Morris method) are the appropriate tools if interaction effects are of interest.
- **Perturbations near a parameter's own value, not its bound range.** Because the perturbation is scaled by the parameter's own calibrated value rather than by the width of `parameter_bounds`, a parameter calibrated to a very small value (near the `1e-4` floor) receives a correspondingly tiny absolute perturbation even under a large percentage setting; conversely, two parameters with very different calibrated magnitudes are perturbed by very different absolute amounts even at the same requested percentage. This makes sensitivity indices most directly comparable within a single parameter's own perturbation-magnitude sweep (Section 6.2), and only loosely comparable across parameters of very different magnitude.
- **Bound-clipping and near-bound calibrations.** A parameter calibrated very close to `parmin[j]` or `parmax[j]` has its perturbation clipped on that side, and the reported sensitivity index is rescaled to compensate for the resulting narrower `actual_delta` (Section 3.5). This rescaling keeps the index numerically comparable to an unclipped case, but the underlying finite-difference estimate is then based on an asymmetric, one-sided step; results for such parameters should be interpreted with some caution, and a calibrated value sitting hard against a bound is, independently, a signal (documented elsewhere in the package) that the bound itself may be too tight and worth revisiting.
- **Numerical-instability filtering can silently reduce the effective sample.** The >50°C stability filter (Section 3.4) is a deliberate safeguard against a known numerical artefact, but it means the number of days contributing to the mean-absolute-difference calculation can shrink — sometimes substantially — for aggressively perturbed, discharge-coupled parameters, and can in the extreme case leave zero valid days (reported as `NaN`). Section 6.2's cross-magnitude comparison is one practical way to notice when this is happening.

## 8. References

- Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *Environmental Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011
- Hamby, D. M. (1994). A review of techniques for parameter sensitivity analysis of environmental models. *Environmental Monitoring and Assessment*, 32(2), 135–154. (General reference on one-at-a-time and other local sensitivity analysis methods.)
- Morris, M. D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics*, 33(2), 161–174. (Elementary-effects screening method as a global alternative to OAT.)
- Saltelli, A., Ratto, M., Andres, T., et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley. (Variance-based global sensitivity methods as an alternative or complement to local OAT analysis.)
- Piotrowski, A. P. and Napiorkowski, J. J. (2018). Performance of the air2stream model that relates air and stream water temperatures depends on the calibration method. *Journal of Hydrology*, 561, 395–412.

## Appendix: Source Reference

Implementation: `pyair2stream/sensitivity.py` (`sensitivity_analysis`, `_plot_sensitivity`), invoked from `pyair2stream/main.py` after calibration completes, with configuration parsing in `pyair2stream/io.py`. Worked example with full results: `examples/validation/Switzerland/config_sensitivity.yaml` and `examples/validation/Switzerland/output_8/sensitivity_PSO_NSE_DAV.csv`. Repository: https://github.com/LukeAFullard/pyair2stream.