# MCMC Uncertainty Quantification in pyair2stream

## 1. Overview

`pyair2stream` is a Python reimplementation of the Fortran `air2stream` model (Toffolon and Piccolroaz, 2015), which simulates daily river water temperature from air temperature and, optionally, discharge. The original Fortran implementation is deterministic: calibration returns a single best-fit parameter vector, and a forward run returns a single predicted time series, with no quantification of parameter uncertainty or predictive uncertainty.

`pyair2stream` adds a Markov Chain Monte Carlo (MCMC) uncertainty quantification layer, built on the affine-invariant ensemble sampler implemented in `emcee` (Foreman-Mackey et al., 2013), that does not exist in the Fortran reference implementation. This layer provides:

- **Posterior parameter distributions** around the Differential Evolution (DE) best fit, quantifying parameter identifiability and equifinality (`run_mode: DE-MCMC` and `run_mode: DE-CV-MCMC`).
- **Predictive uncertainty envelopes** for the calibration period itself, combining parameter uncertainty with an estimate of residual observation error.
- **Probabilistic forward projections**, in which a previously generated posterior parameter chain is reused to propagate both parameter and residual uncertainty into an out-of-sample forward simulation (e.g., a future climate scenario).
- **An optional autoregressive AR(1) noise model** for residual error, in place of the default independent-and-identically-distributed (i.i.d.) assumption, to better reflect the serial correlation typically present in river water temperature residuals.

This document describes the statistical design, the algorithm, its implementation, its configuration, its outputs, and empirical results from its application. It is intended to serve as a citable technical description of this feature for use in derivative scientific work.

## 2. Motivation

A single deterministic calibration cannot distinguish a well-identified parameter from one that happens to sit at a locally optimal but poorly constrained value — a distinction that matters both scientifically (interpreting what the model has and has not learned about a catchment) and practically (communicating a defensible range, rather than a single number, for any downstream use of the simulation, such as a thermal-stress threshold exceedance projection). `air2stream`'s 7- and 8-parameter formulations are known to exhibit equifinality: multiple parameter combinations can yield similar goodness-of-fit. `pyair2stream` addresses this by sampling the posterior distribution of the parameters — rather than reporting only their point estimate — and by propagating that distribution, together with an estimate of residual observation error, into predictive uncertainty envelopes for both the calibration period and future forward projections.

## 3. Statistical Design

### 3.1 Likelihood Function

MCMC sampling requires a probabilistic (not merely goodness-of-fit) formulation of the calibration problem. `pyair2stream` uses a formal concentrated Gaussian log-likelihood, under the assumption that residuals `T_water_obs - T_water_sim` are normally distributed with a fixed but unknown variance that is analytically profiled out of the likelihood:

```
log L(theta) = -0.5 * N * log(SSE(theta) / N)
```

where `SSE(theta)` is the sum of squared residuals between simulated and observed water temperature, computed only over valid (non-missing, and — if `eval_mask` is set — evaluation-flagged) observations, and `N` is the number of such valid observations. This is the standard "concentrated" or "profile" Gaussian likelihood used when the residual variance is not itself a parameter being sampled, but is instead estimated afterward from the best-fit residuals (Section 3.4). Parameter sets that fall outside the configured `parameter_bounds`, or that cause the ODE integration to return a NaN objective, are assigned `-inf` log-probability, effectively implementing a uniform (bounded-box) prior over the active parameters and excluding numerically invalid regions of parameter space from the posterior.

Only parameters that are both flagged active (`flag_par[j] is True`) and non-degenerate (`parmin[j] != parmax[j]`) are sampled; parameters fixed by the model version (e.g., unused parameters in Versions 3–7) are held at their calibrated value and excluded from the MCMC dimensionality.

### 3.2 Sampler

Posterior sampling uses `emcee.EnsembleSampler`, an implementation of the affine-invariant ensemble MCMC method of Goodman and Weare (2010). This method evolves a population ("ensemble") of walkers simultaneously, using the spread of the ensemble itself to propose new positions, which makes it well suited to bounded, correlated parameter spaces like `air2stream`'s without requiring the user to hand-tune a proposal covariance matrix, as a single-chain Metropolis-Hastings sampler would typically require.

`pyair2stream` enforces `mcmc_walkers >= 2 x ndim` (where `ndim` is the number of active parameters), raising a configuration error otherwise; this is both an `emcee` requirement and good sampling practice, since the ensemble method needs enough walkers relative to dimensionality for its stretch-move proposals to explore the space effectively.

### 3.3 Walker Initialization

The two run modes differ in how they initialize the ensemble's starting spread around the DE best fit:

- **`DE-MCMC`**: walkers are initialized in a tight ball, `theta_best + 1e-4 * N(0, 1)`, around the single DE-optimized parameter vector, clipped to the parameter bounds. This is a standard emcee initialization pattern, but it encodes essentially no prior information about the true posterior width — the ensemble must discover the spread of the posterior purely through burn-in.
- **`DE-CV-MCMC`**: walkers are instead initialized with a per-parameter standard deviation derived from an internal leave-one-year-out cross-validation (Section 4), so that the initial ensemble spread already reflects the parameter variability observed empirically across independent temporal folds of the same dataset, rather than starting from an arbitrarily tight point estimate.

### 3.4 Convergence Diagnostics

After sampling, `pyair2stream` computes and reports:

- **Integrated autocorrelation time** (`sampler.get_autocorr_time()`), per parameter, used to judge whether the chain is long enough to be considered well-mixed. A warning is issued if the configured `mcmc_steps` is less than 50 times the largest estimated autocorrelation time, following the standard `emcee` convergence heuristic; if the autocorrelation time itself cannot be reliably estimated (e.g., because the chain is too short), a warning is issued instead and the mean autocorrelation time is recorded as unavailable.
- **Mean acceptance fraction** across all walkers (`sampler.acceptance_fraction`), a standard MCMC health check — acceptance fractions far from a reasonable range (roughly 0.2–0.5 for typical ensemble samplers) can indicate an ensemble that is not mixing well.
- **Burn-in discard**: the first 30% of steps are discarded before the chain is saved or used for any downstream envelope calculation, on the assumption that this is sufficient for the ensemble to move away from its initial (potentially unrepresentative) starting spread and reach its stationary distribution.

### 3.5 Residual Error and the AR(1) Noise Model

The Gaussian likelihood in Section 3.1 concentrates out the residual variance, so it is not directly returned by the sampler; `pyair2stream` estimates it after the fact, at the best-fit parameter vector, as the empirical residual standard deviation, `sigma = sqrt(SSE / N)`, and estimates the lag-1 residual autocorrelation coefficient `rho` from the same best-fit residuals (`estimate_ar1_rho()` in `uncertainty.py`), using only residual pairs that fall on consecutive days within the same valid segment (respecting gap-tolerant mode's segmentation and any `eval_mask`). If fewer than 30 valid consecutive-day residual pairs are available, `rho` falls back to `0.0` with a warning, since the sample is judged too small for a reliable lag-1 estimate. The estimated `rho` is clipped to `[0.0, 0.99]` — restricting it to non-negative persistence and stopping just short of a unit root, which would make the AR(1) process non-stationary.

When generating predictive ensembles (Section 4), noise is added to each sampled trajectory in one of two ways, selected via `uncertainty_options.noise_model`:

- **`iid` (default)**: independent Gaussian noise, `N(0, sigma^2)`, drawn independently for every day.
- **`ar1`**: a stationary AR(1) noise process (`generate_ar1_noise()` in `uncertainty.py`) with the estimated `sigma` and `rho`, generated independently within each valid segment (so that gap boundaries do not induce spurious autocorrelation across a data gap) via `scipy.signal.lfilter` applied to appropriately scaled Gaussian innovations, giving each segment the exact stationary AR(1) marginal variance from its very first time step.

At the daily scale the two noise models are calibrated to the same marginal variance `sigma^2`, so their instantaneous prediction interval widths are similar; the practical difference emerges under temporal aggregation (e.g., a multi-day rolling average), where independent daily noise partially cancels while AR(1) noise, having day-to-day memory, does not cancel to the same degree and yields a wider, more representative interval for multi-day or threshold-exceedance quantities (see Section 6.2).

## 4. DE-CV-MCMC: Cross-Validation-Informed Initialization

`DE-CV-MCMC` extends `DE-MCMC` with an intermediate phase: after the initial DE + L-BFGS-B fit, it runs the leave-one-year-out (or leave-N-years-out) block cross-validation procedure described separately (see the cross-validation documentation) using DE as the per-fold optimizer, then computes the sample standard deviation of each calibrated parameter across the resulting folds. These per-parameter standard deviations become the initial per-dimension spread of the MCMC walker ensemble (with a small floor value, `1e-4`, applied to any parameter whose cross-validation standard deviation is zero, undefined, or based on fewer than two folds, to avoid initializing walkers with zero spread in a dimension).

The rationale is that a single DE optimum gives no information about how sensitive that optimum is to which subset of the record was used to find it; independently recalibrating on several different temporal subsets does. Using this empirically observed spread — rather than an arbitrary tight ball — as the MCMC starting configuration is intended to let the ensemble discover the true posterior width and reach a representative sample more efficiently, without requiring an extended burn-in to first grow out of an artificially narrow starting distribution.

If `data.cross_validation` has not been separately configured (i.e., no `cross_validation:` block was supplied in the YAML config), `DE-CV-MCMC` falls back to `CVConfig()` defaults for this internal step.

## 5. Implementation Summary

The functions implementing this feature are in `pyair2stream/optimization.py` (`DE_MCMC_mode`, `DE_CV_MCMC_mode`) and `pyair2stream/uncertainty.py` (`estimate_ar1_rho`, `generate_ar1_noise`), with the forward-projection consumer in `forward_mode()`.

### 5.1 `DE_MCMC_mode(data, seed=None)`

1. Validate `mcmc_walkers >= 2 x ndim`.
2. Run `DE_mode(data, seed)` to obtain the point-estimate best fit (`data.par_best`).
3. If no parameters are active (a fully fixed model configuration), skip the MCMC phase entirely with a warning.
4. Initialize `nwalkers` in a tight ball around the DE optimum (Section 3.3) and run `emcee.EnsembleSampler` for `mcmc_steps` steps.
5. Compute and report convergence diagnostics (Section 3.4); discard a 30% burn-in and flatten the remaining chain.
6. Save the flattened, burn-in-discarded chain to `MCMC_chain_<station>_<series>_<time_res>.csv` (one column per active parameter, named `par_<j+1>`).
7. Re-evaluate the model at the DE best fit to compute `sigma` (residual standard deviation) and `rho` (AR(1) coefficient), and write both, together with sampler diagnostics (`mcmc_walkers`, `mcmc_steps`, `mcmc_seed`, `mean_acceptance_fraction`, `mean_autocorr_time`, and which `noise_model` was configured for this run) to a JSON sidecar file, `MCMC_chain_<station>_<series>_<time_res>_meta.json`.
8. Draw up to 1000 random posterior samples from the saved chain; for each, simulate the full time series, estimate a per-sample residual `sigma` from that sample's own residuals, generate noise (i.i.d. or AR(1) per the configured `noise_model`), and add it to the simulated trajectory to build an ensemble of noisy realizations.
9. Compute the requested percentile envelope (default: 5th/50th/95th, from `uncertainty_options.prediction_interval`, default 90%) across the ensemble at every time step, masking out days where the underlying deterministic simulation itself has no value (e.g., inside an undetected gap-tolerant segment), and write the result to `MCMC_envelopes_<station>_<series>_<time_res>.csv`.
10. Restore the DE best-fit parameters as `data.par`/`data.par_best` and recompute `data.finalfit`, so that downstream reporting (e.g., the standard `forward()`/post-processing pipeline) reflects the deterministic best fit rather than the last MCMC-sampled parameter set evaluated.

### 5.2 `DE_CV_MCMC_mode(data, seed=None)`

Identical to `DE_MCMC_mode`, except that between steps 3 and 4 above, it runs `run_leave_one_year_out_cv()` using DE as the fold optimizer, computes per-parameter cross-fold standard deviations, and uses those (rather than a fixed `1e-4`) as the initial per-dimension walker spread (Section 4). The remainder of the procedure — sampling, diagnostics, chain/envelope output, sidecar metadata — is identical to `DE_MCMC_mode`.

### 5.3 `forward_mode()` — Forward Prediction Intervals

When `run_mode: FORWARD` is used together with `forward_options.enable_prediction_intervals: true` and a path to a previously generated `MCMC_chain_*.csv`, `forward_mode()`:

1. Runs the deterministic forward simulation using `parameters_forward` as usual, and — if genuine `T_water` observations are present in the forward dataset — computes the corresponding efficiency index for reporting.
2. Reads the saved MCMC chain and draws `n_samples` (default 1000, capped at the chain length) random parameter sets from it.
3. Resolves the residual standard deviation `sigma` to use from `forward_options.residual_sigma` (an explicit, user-supplied value — since a pure future projection typically has no observations of its own from which to estimate residual error).
4. Resolves the AR(1) coefficient `rho` (only if `noise_model: ar1`) using a strict priority order:
   1. **Explicit override** — `uncertainty_options.ar1_rho`, if supplied.
   2. **Own residuals** — if the forward dataset itself contains genuine `T_water` observations, `rho` is estimated directly from this run's own residuals.
   3. **Sidecar carry-forward** — if a `_meta.json` sidecar exists alongside the supplied `mcmc_chain_path` (as automatically written by `DE-MCMC`/`DE-CV-MCMC`), `rho` is read from it.
   4. **Fallback** — `rho = 0.0` (equivalent to `iid`), with a warning.
5. For each of the `n_samples` parameter draws, simulates the full forward series, generates noise (i.i.d. or AR(1), per the resolved `rho`), and adds it to the deterministic trajectory.
6. Computes the requested percentile envelope across the resulting ensemble and writes it to `Forward_Prediction_Envelopes_<station>_<series>_<time_res>.csv`, masking days with no underlying deterministic value.
7. Restores the deterministic `parameters_forward` and re-runs the model once more, so `data.Twat_mod` reflects the single deterministic projection rather than the last noisy ensemble member evaluated.

## 6. Configuration

### 6.1 Enabling `DE-MCMC` / `DE-CV-MCMC`

```yaml
run_mode: "DE-MCMC"          # or "DE-CV-MCMC"

optimization:
  n_run: 100                  # DE phase: max generations
  n_particles: 50              # DE phase: population size
  mcmc_walkers: 32              # must be >= 2x the number of active parameters
  mcmc_steps: 1000

uncertainty_options:
  noise_model: "iid"           # "iid" (default) or "ar1"
  prediction_interval: 90.0    # width of the reported percentile envelope
```

`DE-CV-MCMC` additionally reads any `cross_validation:` block present in the config (Section 4); if absent, internal cross-validation defaults are used for the CV-informed initialization step only, independent of whether the user wants a full CV report.

### 6.2 Enabling Forward Prediction Intervals

```yaml
run_mode: "FORWARD"
parameters_forward: [1.2, 0.3, 0.2, 0.5, 0.1, 1.5, 0.4, 0.1]

forward_options:
  enable_prediction_intervals: true
  mcmc_chain_path: "output/MCMC_chain_<station>_<series>_<time_res>.csv"
  residual_sigma: 1.0          # observation error standard deviation
  n_samples: 1000
  random_seed: 42

uncertainty_options:
  noise_model: "ar1"           # or "iid"
  ar1_rho: null                 # optional explicit override, in (-1, 1)
```

### 6.3 Caveats Documented for This Feature

- Residual error (`sigma`, `rho`) is estimated from daily residuals even if the calibration objective function was computed on aggregated (e.g., weekly) data — the noise model operates at daily resolution regardless of the calibration's temporal aggregation.
- Both noise models add noise after the physical ODE integration, so a lower prediction bound can, in principle, dip below the model's physical ice-cover floor (`Tice_cover`); this is a known, accepted limitation rather than a bug.
- The AR(1) coefficient `rho` used for an interval is a fixed, plug-in estimate — it is not jointly calibrated with the physical model parameters inside the MCMC sampler itself.

## 7. Outputs

| File | Produced by | Contents |
|---|---|---|
| `MCMC_chain_<station>_<series>_<time_res>.csv` | `DE-MCMC`, `DE-CV-MCMC` | Flattened, burn-in-discarded posterior samples, one column per active parameter (`par_<j+1>`) |
| `MCMC_chain_<station>_<series>_<time_res>_meta.json` | `DE-MCMC`, `DE-CV-MCMC` | Sidecar metadata: estimated `sigma`, `rho`, number of valid residual pairs, configured noise model, walker/step counts, seed, mean acceptance fraction, mean autocorrelation time |
| `MCMC_envelopes_<station>_<series>_<time_res>.csv` | `DE-MCMC`, `DE-CV-MCMC` | Percentile prediction envelope (lower/median/upper, per `prediction_interval`) over the calibration-period simulation |
| `Forward_Prediction_Envelopes_<station>_<series>_<time_res>.csv` | `FORWARD` with `enable_prediction_intervals: true` | Percentile prediction envelope over a forward/projection simulation, built from a previously saved MCMC chain |

## 8. Empirical Results

### 8.1 DE-MCMC vs. DE-CV-MCMC Initialization (Dischmabach dataset)

An internal comparison ran both `DE-MCMC` and `DE-CV-MCMC` on the same Swiss validation dataset (`DAV_2327`, Version 8, 32 walkers, 100 MCMC steps, 90th-percentile-configured interval reported at 95%), holding the DE phase and MCMC step count fixed so that only the walker-initialization strategy differed.

| Quantity | `DE-MCMC` | `DE-CV-MCMC` |
|---|---|---|
| Estimated residual `sigma` (deg C) | 0.6064 | 0.6026 |
| Estimated AR(1) `rho` | 0.7122 | 0.7107 |
| Valid residual pairs used for `rho` | 2202 | 1837 |

The close agreement in `sigma` and `rho` between the two modes confirms that both converge to essentially the same best-fit residual structure; the intended difference between the two modes is in how quickly and representatively the MCMC ensemble explores the posterior parameter spread, not in the point estimate itself. `DE-CV-MCMC`'s cross-validation-derived initial spread is designed to let the ensemble discover the true parameter-space equifinality (visible, for example, as wider or narrower posterior histograms and correspondingly wider or narrower predictive envelopes) more efficiently than a tightly-initialized `DE-MCMC` ensemble would within the same, limited step budget — this is illustrated directly in the package's `posterior_comparison.png` and `envelope_comparison.png` diagnostic plots for this comparison.

### 8.2 IID vs. AR(1) Forward Prediction Intervals

A separate worked example calibrated `DE-MCMC` (Version 8) against a synthetic historical dataset with injected AR(1)-structured noise, then used the resulting chain to generate forward prediction intervals under both noise models.

**Calibration results:** NSE = 0.9728, R² = 0.9728, MAE = 0.6463 deg C. Estimated residual standard deviation `sigma` = 0.8185 deg C, closely matching the injected synthetic noise level; estimated AR(1) coefficient `rho` = 0.5910, confirming clear day-to-day memory in the residuals.

**Forward interval comparison:** at the daily scale, the IID and AR(1) 90% prediction intervals have essentially the same width, since both are calibrated to the same marginal residual variance. The practical difference appears under temporal aggregation: computing a 7-day rolling average of the ensemble, the IID interval visibly narrows (independent daily noise partially cancels when averaged), while the AR(1) interval remains close to its original width (temporally correlated noise does not cancel to the same degree). This makes the AR(1) noise model the more conservative and structurally realistic choice for any downstream use involving multi-day averages or sustained-threshold-exceedance questions (e.g., multi-day thermal stress events for aquatic organisms).

## 9. Practical Considerations and Limitations

- **Cost.** MCMC sampling is the most computationally expensive calibration mode in `pyair2stream`, scaling with `mcmc_walkers x mcmc_steps` on top of the initial DE + L-BFGS-B fit (and, for `DE-CV-MCMC`, on top of a full N-fold cross-validation as well). Reduce `mcmc_steps`/`mcmc_walkers` or use `optimizer_overrides` within the CV phase of `DE-CV-MCMC` to control runtime, but check the reported acceptance fraction and autocorrelation-time warnings before trusting a shortened chain.
- **Likelihood assumptions.** The Gaussian, homoscedastic, temporally-independent-within-the-likelihood assumption used to derive the sampling log-likelihood is a simplification — the AR(1) noise model in Section 3.5 is applied only to the downstream predictive envelope construction, not to the likelihood the sampler itself explores; the two are decoupled in the current implementation.
- **`rho` is a plug-in estimate, not a sampled parameter.** As noted in Section 6.3, the AR(1) coefficient used for prediction intervals is estimated once from best-fit residuals and held fixed; it is not integrated over as part of the posterior uncertainty.
- **Forward projections require a prior MCMC run.** Forward prediction intervals are only available if a `DE-MCMC` or `DE-CV-MCMC` calibration has already been run and its chain (and, for convenient `rho` carry-forward, its sidecar file) is available on disk; `FORWARD` mode cannot generate probabilistic intervals from a single deterministic `parameters_forward` vector alone.
- **Small-sample AR(1) fallback.** If a dataset (or a forward run's own observations) provides fewer than 30 valid consecutive-day residual pairs, `rho` is set to 0.0 with a warning rather than estimated from an unreliable small sample — users relying on the AR(1) model with sparse observational records should check for this warning.

## 10. References

- Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *Environmental Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011
- Foreman-Mackey, D., Hogg, D. W., Lang, D., and Goodman, J. (2013). emcee: The MCMC Hammer. *Publications of the Astronomical Society of the Pacific*, 125(925), 306–312.
- Goodman, J. and Weare, J. (2010). Ensemble samplers with affine invariance. *Communications in Applied Mathematics and Computational Science*, 5(1), 65–80.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley. (AR(1) process definition and stationarity conditions.)
- Piotrowski, A. P. and Napiorkowski, J. J. (2018). Performance of the air2stream model that relates air and stream water temperatures depends on the calibration method. *Journal of Hydrology*, 561, 395–412.

## Appendix: Source Reference

Implementation: `pyair2stream/optimization.py` (`DE_MCMC_mode`, `DE_CV_MCMC_mode`, and the prediction-interval branch of `forward_mode`), `pyair2stream/uncertainty.py` (`estimate_ar1_rho`, `generate_ar1_noise`), with configuration parsing in `pyair2stream/io.py` and dispatch in `pyair2stream/main.py::run_optimizer`. Worked examples: `examples/mcmc_comparison/README.md` (DE-MCMC vs. DE-CV-MCMC) and `examples/forward_prediction_intervals/README.md` (IID vs. AR(1) forward projection). Repository: https://github.com/LukeAFullard/pyair2stream.
