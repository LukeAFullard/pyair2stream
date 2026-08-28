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

MCMC sampling requires a probabilistic (not merely goodness-of-fit) formulation of the calibration problem. `pyair2stream` uses a formal concentrated Gaussian log-likelihood, under the assumption that residuals `T_water_obs - T_water_sim` are normally distributed with a fixed but unknown variance that is analytically profiled out of the likelihood. Two forms are available, selected via `uncertainty_options.noise_model` (the same key that selects the noise model for predictive envelopes, Section 3.5):

- **`iid` (default; unchanged from earlier versions)**:

  ```
  log L(theta) = -0.5 * N * log(SSE(theta) / N)
  ```

  where `SSE(theta)` is the sum of squared residuals between simulated and observed water temperature, computed only over valid (non-missing, and — if `eval_mask` is set — evaluation-flagged) observations, and `N` is the number of such valid observations.
- **`ar1`**: daily stream-temperature residuals typically show lag-1 autocorrelation of 0.8–0.95, which the `iid` form ignores entirely — it scores a chain of highly redundant, non-independent residuals as if each one were fresh information, which sharpens (over-narrows) the posterior by roughly a factor of `N / N_eff` in the log-likelihood, where `N_eff = N * (1-rho) / (1+rho)` (for `rho = 0.9`, `N_eff ~ N/19`). The `ar1` likelihood instead whitens the residuals within each temporally contiguous run of valid, in-segment days (the same adjacency logic used by `estimate_ar1_rho()`, Section 3.5): `u[0] = e[0] * sqrt(1 - rho**2)`, `u[t] = e[t] - rho * e[t-1]` for `t >= 1`. These are iid under the AR(1) model, giving

  ```
  log L(theta) = -0.5 * N * log(SSE_u(theta) / N) + 0.5 * n_runs * log(1 - rho**2)
  ```

  where `SSE_u` is the sum of squared whitened residuals and `n_runs` is the number of independent runs (each contributing its own `0.5*log(1-rho**2)` term). `rho` is treated as **fixed**: estimated once, at the DE optimum, by `estimate_ar1_rho()` (not jointly sampled) — see Section 9 for why this is an accepted simplification.

Parameter sets that fall outside the configured `parameter_bounds`, or that cause the ODE integration to return a NaN objective, are assigned `-inf` log-probability under either likelihood, effectively implementing a uniform (bounded-box) prior over the active parameters and excluding numerically invalid regions of parameter space from the posterior.

Only parameters that are both flagged active (`flag_par[j] is True`) and non-degenerate (`parmin[j] != parmax[j]`) are sampled; parameters fixed by the model version (e.g., unused parameters in Versions 3–7) are held at their calibrated value and excluded from the MCMC dimensionality.

### 3.2 Sampler

Posterior sampling uses `emcee.EnsembleSampler`, an implementation of the affine-invariant ensemble MCMC method of Goodman and Weare (2010). This method evolves a population ("ensemble") of walkers simultaneously, using the spread of the ensemble itself to propose new positions, which makes it well suited to bounded, correlated parameter spaces like `air2stream`'s without requiring the user to hand-tune a proposal covariance matrix, as a single-chain Metropolis-Hastings sampler would typically require.

`pyair2stream` enforces `mcmc_walkers >= 2 x ndim` (where `ndim` is the number of active parameters), raising a configuration error otherwise; this is both an `emcee` requirement and good sampling practice, since the ensemble method needs enough walkers relative to dimensionality for its stretch-move proposals to explore the space effectively.

### 3.3 Walker Initialization

The two run modes differ in how they initialize the ensemble's starting spread around the DE best fit:

- **`DE-MCMC`**: walkers are initialized in a ball, `theta_best + 1e-3*(parmax[j]-parmin[j]) * N(0, 1)` per active parameter `j`, around the single DE-optimized parameter vector. The spread scales with each parameter's own bound width rather than a fixed constant, so it does not become negligible for a wide parameter or oversized for a narrow one.
- **`DE-CV-MCMC`**: walkers are instead initialized with a per-parameter standard deviation derived from an internal leave-one-year-out cross-validation (Section 4), so that the initial ensemble spread already reflects the parameter variability observed empirically across independent temporal folds of the same dataset, rather than starting from an arbitrarily tight point estimate.

In both modes, draws that fall outside `[parmin[j], parmax[j]]` are **reflected** back inside the bound rather than clipped to it. Clipping is the more common textbook pattern, but it collapses the ensemble's spread in any dimension where the DE optimum sits exactly on a bound (which happens in practice — e.g. `a5`/`a6` pinned at a bound in some validation datasets, see `examples/validation/Switzerland/README.md`): every walker gets clipped to the same value, and `emcee`'s stretch move cannot generate spread from a degenerate ensemble. After construction, `pyair2stream` asserts every active dimension has non-zero variance across walkers and raises `ValueError` if not, rather than silently proceeding with a collapsed ensemble.

### 3.4 Convergence Diagnostics

After sampling, `pyair2stream` computes and reports:

- **Burn-in discard**: a rough autocorrelation-time estimate is first computed on the *full* chain (`discard=0`); the burn-in is then `max(0.3*mcmc_steps, 5*max(tau))` when that estimate is available, or the historical flat 30% when it is not. An explicit `uncertainty_options.burnin_fraction` (in `(0, 1)`) overrides this entirely. The chosen burn-in (in steps) is recorded in the sidecar as `burnin`.
- **Integrated autocorrelation time** (`sampler.get_autocorr_time(discard=burnin)`), per parameter, recomputed on the *post-burn-in* chain (not the full chain used to size the burn-in above) so the reported value reflects the chain actually used downstream. Used to judge whether the chain is long enough to be considered well-mixed: a warning is issued if `mcmc_steps` is less than 50 times the largest such estimate, following the standard `emcee` convergence heuristic. If `uncertainty_options.strict_convergence: true` is set, this is instead a hard `RuntimeError` rather than a warning. If the autocorrelation time itself cannot be reliably estimated (e.g., because the chain is too short), a warning is issued instead and the mean autocorrelation time is recorded as unavailable.
- **Mean acceptance fraction** across all walkers (`sampler.acceptance_fraction`), a standard MCMC health check — acceptance fractions far from a reasonable range (roughly 0.2–0.5 for typical ensemble samplers) can indicate an ensemble that is not mixing well.
- **Split-Rhat** (Gelman-Rubin potential scale reduction, computed per parameter by splitting each walker's post-burn-in chain in half): flags both between-walker disagreement and within-walker non-stationarity that acceptance fraction and autocorrelation time alone can miss. A warning is printed if the largest per-parameter value is `>= 1.01`; the value is recorded in the sidecar as `max_split_rhat` (`null` if it could not be computed, e.g. too few post-burn-in steps).

### 3.5 Residual Error and the AR(1) Noise Model

The Gaussian likelihood in Section 3.1 concentrates out the residual variance, so it is not directly returned by the sampler; `pyair2stream` estimates it after the fact, at the best-fit parameter vector, as the empirical residual standard deviation, `sigma = sqrt(SSE / N)`, and estimates the lag-1 residual autocorrelation coefficient `rho` from the same best-fit residuals (`estimate_ar1_rho()` in `uncertainty.py`), using only residual pairs that fall on consecutive days within the same valid segment (respecting gap-tolerant mode's segmentation and any `eval_mask`). If fewer than 30 valid consecutive-day residual pairs are available, `rho` falls back to `0.0` with a warning, since the sample is judged too small for a reliable lag-1 estimate. The estimated `rho` is clipped to `[0.0, 0.99]` — restricting it to non-negative persistence and stopping just short of a unit root, which would make the AR(1) process non-stationary.

When generating predictive ensembles (Section 4), noise is added to each sampled trajectory in one of two ways, selected via `uncertainty_options.noise_model`:

- **`iid` (default)**: independent Gaussian noise, `N(0, sigma^2)`, drawn independently for every day.
- **`ar1`**: a stationary AR(1) noise process (`generate_ar1_noise()` in `uncertainty.py`) with the estimated `sigma` and `rho`, generated independently within each valid segment (so that gap boundaries do not induce spurious autocorrelation across a data gap) via `scipy.signal.lfilter` applied to appropriately scaled Gaussian innovations, giving each segment the exact stationary AR(1) marginal variance from its very first time step.

At the daily scale the two noise models are calibrated to the same marginal variance `sigma^2`, so their instantaneous prediction interval widths are similar; the practical difference emerges under temporal aggregation (e.g., a multi-day rolling average), where independent daily noise partially cancels while AR(1) noise, having day-to-day memory, does not cancel to the same degree and yields a wider, more representative interval for multi-day or threshold-exceedance quantities (see Section 6.2).

Percentile bands alone cannot produce such aggregate statistics correctly — the 5th percentile of a 7-day rolling mean is *not* the 7-day rolling mean of the 5th-percentile series. When `uncertainty_options.save_ensemble: true` is set, `DE-MCMC`/`DE-CV-MCMC`/`FORWARD` additionally write the raw `(n_samples, n_days)` matrix of noisy simulated trajectories (post-warm-up) as a compressed `.npz`, and `pyair2stream/scenario.py` provides `load_ensemble()`, `aggregate()`, `exceedance()`, and `paired_difference()` to work with it directly — see Section 7.

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
4. Re-evaluate the model at the DE best fit to compute `sigma` (residual standard deviation) and `rho` (AR(1) coefficient, treated as fixed for the likelihood — Section 3.1).
5. Initialize `nwalkers` around the DE optimum with reflected, bound-scaled spread (Section 3.3) and run `emcee.EnsembleSampler` for `mcmc_steps` steps, using either the `iid` or `ar1` log-likelihood per `uncertainty_options.noise_model` (Section 3.1).
6. Compute and report convergence diagnostics (Section 3.4): burn-in, post-burn-in autocorrelation time, acceptance fraction, split-Rhat.
7. Save the flattened, burn-in-discarded chain to `MCMC_chain_<station>_<series>_<time_res>.csv` (one column per active parameter, named `par_<j+1>`).
8. Write `sigma`, `rho`, and sampler/convergence diagnostics (`mcmc_walkers`, `mcmc_steps`, `mcmc_seed`, `burnin`, `mean_acceptance_fraction`, `mean_autocorr_time`, `max_split_rhat`, `strict_convergence`, and which `noise_model` was configured for this run) to a JSON sidecar file, `MCMC_chain_<station>_<series>_<time_res>_meta.json`.
9. Draw up to 1000 random posterior samples from the saved chain; for each, simulate the full time series, estimate a per-sample residual `sigma` from that sample's own residuals, generate noise (i.i.d. or AR(1) per the configured `noise_model`), and add it to the simulated trajectory to build an ensemble of noisy realizations.
10. Compute the requested percentile envelope (default: 5th/50th/95th, from `uncertainty_options.prediction_interval`, default 90%) across the ensemble at every time step, masking out days where the underlying deterministic simulation itself has no value (e.g., inside an undetected gap-tolerant segment), and write the result to `MCMC_envelopes_<station>_<series>_<time_res>.csv`. If `uncertainty_options.save_ensemble: true`, also write the raw ensemble matrix to `MCMC_ensemble_<station>_<series>_<time_res>.npz`.
11. Restore the DE best-fit parameters as `data.par`/`data.par_best` and recompute `data.finalfit`, so that downstream reporting (e.g., the standard `forward()`/post-processing pipeline) reflects the deterministic best fit rather than the last MCMC-sampled parameter set evaluated.

### 5.2 `DE_CV_MCMC_mode(data, seed=None)`

Identical to `DE_MCMC_mode`, except that between steps 3 and 4 above, it runs `run_leave_one_year_out_cv()` using DE as the fold optimizer, computes per-parameter cross-fold standard deviations, and uses those (rather than the bound-width-scaled default) as the initial per-dimension walker spread (still reflected off bounds, Section 3.3/Section 4). The remainder of the procedure — likelihood, sampling, diagnostics, chain/envelope output, sidecar metadata — is identical to `DE_MCMC_mode` (both call the same shared implementation).

### 5.3 `forward_mode()` — Forward Prediction Intervals

When `run_mode: FORWARD` is used together with `forward_options.enable_prediction_intervals: true` and a path to a previously generated `MCMC_chain_*.csv`, `forward_mode()`:

1. Runs the deterministic forward simulation using `parameters_forward` as usual, and — if genuine `T_water` observations are present in the forward dataset — computes the corresponding efficiency index for reporting.
2. Reads the saved MCMC chain and either draws `n_samples` (default 1000, capped at the chain length) random parameter sets from it, or — if `forward_options.reuse_sample_indices_from` is set — reuses the exact indices a prior `forward_mode()` run saved to its own sidecar, ignoring `n_samples`/`random_seed`/global random state entirely for this draw (see Section 5.4).
3. Resolves the residual standard deviation `sigma` to use, in priority order: (a) an explicit `forward_options.residual_sigma` override; (b) the `sigma` field of the `_meta.json` sidecar alongside `mcmc_chain_path` (written automatically by `DE-MCMC`/`DE-CV-MCMC`), mirroring the `rho` carry-forward in step 4 below. If neither yields a usable (`> 0.0`) value, `forward_mode()` raises `ValueError` rather than silently building a prediction interval with no residual term — a pure future projection typically has no observations of its own from which to estimate residual error, so silently defaulting to `sigma = 0.0` would produce an interval that reflects parameter uncertainty only, with no warning beyond a `print`.
4. Resolves the AR(1) coefficient `rho` (only if `noise_model: ar1`) using a strict priority order:
   1. **Explicit override** — `uncertainty_options.ar1_rho`, if supplied.
   2. **Own residuals** — if the forward dataset itself contains genuine `T_water` observations, `rho` is estimated directly from this run's own residuals.
   3. **Sidecar carry-forward** — if a `_meta.json` sidecar exists alongside the supplied `mcmc_chain_path` (as automatically written by `DE-MCMC`/`DE-CV-MCMC`), `rho` is read from it.
   4. **Fallback** — `rho = 0.0` (equivalent to `iid`), with a warning.
5. For each parameter draw, simulates the full forward series and checks it for numerical divergence (non-finite, or exceeding `max_plausible_twat`) before adding noise (Section 5.5); a divergent draw is excluded (default) or raised on, per `uncertainty_options.on_divergent_draw`. For a draw that passes, noise (i.i.d. or AR(1), per the resolved `rho`) is generated and added to the deterministic trajectory.
6. Computes the requested percentile envelope across the resulting (post-exclusion) ensemble and writes it to `Forward_Prediction_Envelopes_<station>_<series>_<time_res>.csv`, masking days with no underlying deterministic value. If `uncertainty_options.save_ensemble: true`, also writes the raw ensemble matrix to `Forward_Prediction_Ensemble_<station>_<series>_<time_res>.npz`, and always writes a provenance/divergence sidecar to `Forward_Prediction_Ensemble_<station>_<series>_<time_res>_meta.json` (Section 7).
7. Restores the deterministic `parameters_forward` and re-runs the model once more, so `data.Twat_mod` reflects the single deterministic projection rather than the last noisy ensemble member evaluated.

### 5.4 Pairing two scenario runs: ensemble provenance and `reuse_sample_indices_from`

`scenario.paired_difference()` is only statistically meaningful if both ensembles were built from the SAME posterior parameter draws in the SAME order — the exact requirement of the water-abstraction study (observed vs. naturalised flow) and the climate-projection study (historical vs. projected flow). Relying on a shared `forward_options.random_seed` across two separate CLI invocations/config files to guarantee this is fragile: the seed is easy to omit, or to set to two different values by mistake, and nothing previously checked for it after the fact — two ensembles built from unrelated draws would pass `paired_difference()`'s shape-only check silently.

Every `forward_mode()` run with `enable_prediction_intervals: true` now writes its sample provenance into the ensemble's sidecar JSON (`Forward_Prediction_Ensemble_<...>_meta.json`, Section 7): a SHA-256 content hash and row count identifying the source chain file, the resolved `random_seed`, the requested `sample_indices` (drawn or reused), and `valid_draw_indices` — the subset of those indices that actually survived the per-draw divergence check in step 5 above and therefore ended up as rows in the saved ensemble.

Setting `forward_options.reuse_sample_indices_from: "<path to a prior run's sidecar>"` makes the current run skip the random draw entirely and reuse that prior run's exact `sample_indices` — byte-identical regardless of global random state — after cross-checking that both runs' `mcmc_chain_path` resolves to the same chain (by content hash and row count, not just the path string), raising `ValueError` on a mismatch. `_run_mcmc_uncertainty()` (`DE-MCMC`/`DE-CV-MCMC`) writes the same provenance fields into `MCMC_chain_*_meta.json` for consistency, but does not itself support `reuse_sample_indices_from` — the paired-scenario workflow pairs two `FORWARD` runs, not two calibration runs.

`scenario.paired_difference_from_files(path_a, path_b)` (Section 7.1) is the recommended way to actually difference two such ensembles: it loads both sidecars and verifies the source chain, requested sample count, requested `sample_indices`, and `valid_draw_indices` all match exactly before differencing — `valid_draw_indices` is the authoritative check, since two runs can request identical `sample_indices` and still end up with differently-excluded (and therefore misaligned) rows if one scenario's discharge diverges on draws the other's does not.

### 5.5 Per-draw divergence handling (`forward_mode()` and `_run_mcmc_uncertainty()`)

Prior to this feature, `check_numerical_divergence` (the guard that catches a non-finite or implausibly large simulated water temperature) only ever ran on the single deterministic best-fit simulation — never inside either ensemble-generation loop, each of which calls `call_model()` once per posterior/parameter draw (hundreds to ~1000 times). A single bad draw (e.g. a scenario discharge the chain was never fitted under, or a divergence induced by a zero-discharge day — see the "Known deviations" section of the README) either crashed the whole batch with no context, or — if it happened to stay finite — was silently written into the percentile envelope CSV and the raw `.npz` ensemble that `scenario.paired_difference`/`paired_difference_from_files` consume.

Both loops now call `model.is_numerically_divergent()` on each draw's simulated series before adding noise. The behaviour is controlled by `uncertainty_options.on_divergent_draw`:

- **`"drop"` (default)**: the draw is excluded from the ensemble/percentile calculation. The number of draws excluded (and, for each, its draw index, its position in the source chain, and its parameter values) is printed to the console and recorded in the sidecar metadata (`n_divergent_draws_excluded`, `divergent_draw_fraction`, `excluded_draws`).
- **`"raise"`**: the run raises `NumericalDivergenceError` immediately on the first divergent draw, naming it, instead of dropping it.

Regardless of `on_divergent_draw`, if every draw diverges (an empty ensemble is never silently returned as a successful result), or if the excluded fraction exceeds `uncertainty_options.max_divergent_fraction` (default `0.10`, mirroring `stability_error_fraction`'s pattern for the pre-flight stability check), the run raises rather than proceeding with a depleted ensemble. The saved `.npz` ensemble (when `save_ensemble: true`) already contains only the surviving rows, so it is never out of sync with the reported percentile envelope or with `n_divergent_draws_excluded`.

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
  noise_model: "iid"           # "iid" (default) or "ar1" -- also selects the likelihood (Section 3.1)
  prediction_interval: 90.0    # width of the reported percentile envelope
  save_ensemble: false         # if true, also write the raw (n_samples, n_days) ensemble as .npz
  strict_convergence: false    # if true, insufficient chain length raises instead of warning
  burnin_fraction: null        # optional override, in (0, 1); default is adaptive (Section 3.4)
  on_divergent_draw: "drop"    # "drop" (default) or "raise" -- per-draw divergence handling (Section 5.5)
  max_divergent_fraction: 0.10 # raise if more than this fraction of draws are excluded as divergent
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
  # Optional: reuse a prior run's exact sample_indices instead of drawing new ones
  # (ignores n_samples/random_seed/global random state above for this draw) --
  # required for a statistically valid paired scenario comparison (Section 5.4):
  # reuse_sample_indices_from: "output/scenario_a/Forward_Prediction_Ensemble_<...>_meta.json"

uncertainty_options:
  noise_model: "ar1"           # or "iid"
  ar1_rho: null                 # optional explicit override, in (-1, 1)
  save_ensemble: false         # if true, also write the raw ensemble as .npz
  on_divergent_draw: "drop"    # "drop" (default) or "raise" -- per-draw divergence handling (Section 5.5)
  max_divergent_fraction: 0.10 # raise if more than this fraction of draws are excluded as divergent
```

### 6.3 Caveats Documented for This Feature

- Residual error (`sigma`, `rho`) is estimated from daily residuals even if the calibration objective function was computed on aggregated (e.g., weekly) data — the noise model operates at daily resolution regardless of the calibration's temporal aggregation.
- Both noise models add noise after the physical ODE integration, so a lower prediction bound can, in principle, dip below the model's physical ice-cover floor (`Tice_cover`); this is a known, accepted limitation rather than a bug.
- The AR(1) coefficient `rho` used for an interval is a fixed, plug-in estimate — it is not jointly calibrated with the physical model parameters inside the MCMC sampler itself.

## 7. Outputs

| File | Produced by | Contents |
|---|---|---|
| `MCMC_chain_<station>_<series>_<time_res>.csv` | `DE-MCMC`, `DE-CV-MCMC` | Flattened, burn-in-discarded posterior samples, one column per active parameter (`par_<j+1>`) |
| `MCMC_chain_<station>_<series>_<time_res>_meta.json` | `DE-MCMC`, `DE-CV-MCMC` | Sidecar metadata: estimated `sigma`, `rho`, number of valid residual pairs, configured noise model, walker/step counts, seed, burn-in, mean acceptance fraction, post-burn-in mean autocorrelation time, max split-Rhat, `strict_convergence`; plus per-draw divergence handling (`on_divergent_draw`, `max_divergent_fraction`, `n_draws_requested`, `n_divergent_draws_excluded`, `divergent_draw_fraction`, `excluded_draws`) and ensemble provenance (`chain_path`, `chain_content_sha256`, `chain_n_rows`, `envelope_sample_seed`, `sample_indices`, `valid_draw_indices`) -- see Section 5.4/5.5 |
| `MCMC_envelopes_<station>_<series>_<time_res>.csv` | `DE-MCMC`, `DE-CV-MCMC` | Percentile prediction envelope (lower/median/upper, per `prediction_interval`) over the calibration-period simulation |
| `MCMC_ensemble_<station>_<series>_<time_res>.npz` | `DE-MCMC`, `DE-CV-MCMC` with `save_ensemble: true` | Raw `(n_samples, n_days)` matrix of noisy simulated trajectories, post-warm-up (any divergent draws already excluded, per Section 5.5), plus `year`/`month`/`day` column dates |
| `Forward_Prediction_Envelopes_<station>_<series>_<time_res>.csv` | `FORWARD` with `enable_prediction_intervals: true` | Percentile prediction envelope over a forward/projection simulation, built from a previously saved MCMC chain |
| `Forward_Prediction_Ensemble_<station>_<series>_<time_res>.npz` | `FORWARD` with `enable_prediction_intervals: true` and `save_ensemble: true` | Raw ensemble matrix for the forward/projection simulation, same layout as `MCMC_ensemble_*.npz` (divergent draws already excluded) |
| `Forward_Prediction_Ensemble_<station>_<series>_<time_res>_meta.json` | `FORWARD` with `enable_prediction_intervals: true` (always written, regardless of `save_ensemble`) | Sidecar metadata: same divergence-handling fields as `MCMC_chain_*_meta.json` above, plus provenance (`chain_path`, `chain_content_sha256`, `chain_n_rows`, `requested_seed`, `sample_indices`, `valid_draw_indices`, `reused_sample_indices_from`) -- this is what `scenario.paired_difference_from_files()` reads (Section 5.4/7.1) |

### 7.1 Working with the raw ensemble (`pyair2stream/scenario.py`)

Percentile bands cannot produce aggregate statistics: the 5th percentile of a 7-day rolling mean is not the 7-day rolling mean of the 5th-percentile series. Degree-days, days-above-threshold, sustained-exceedance duration, and summer-mean warming — the outputs the water-abstraction and climate-projection studies actually need (see `docs/study_design` guidance) — require the raw ensemble instead:

```python
from pyair2stream import scenario

ensemble, dates = scenario.load_ensemble("MCMC_ensemble_Alpha_historical_1d.npz")

weekly_mean = scenario.aggregate(ensemble, dates, how="mean", freq="7D")
days_above_20C = scenario.exceedance(ensemble, threshold=20.0, consecutive_days=1)

# RECOMMENDED: two Forward_Prediction_Ensemble_*.npz files from the two-run workflow
# in Section 5.4 (the second built with forward_options.reuse_sample_indices_from
# pointing at the first's sidecar). This verifies both runs' saved provenance
# (source chain, requested and surviving sample indices) match exactly before
# differencing, raising ValueError naming what disagreed if they don't:
delta = scenario.paired_difference_from_files(
    "output/scenario_a/Forward_Prediction_Ensemble_Alpha_historical_1d.npz",
    "output/scenario_b/Forward_Prediction_Ensemble_Alpha_historical_1d.npz",
)

# Advanced/same-process alternative: only checks that the two arrays have the same
# shape, NOT that they came from the same draws in the same order -- use only when
# you built both arrays yourself in this same script/session:
delta = scenario.paired_difference(ensemble_scenario_a, ensemble_scenario_b)
```

`paired_difference_from_files()` is the recommended way to pair two saved ensembles for anything other than same-process use: it loads each `.npz`'s provenance sidecar (Section 5.4/7) and requires the source chain, requested sample count, requested `sample_indices`, and the `valid_draw_indices` that actually survived per-draw divergence filtering (Section 5.5) to match exactly, raising `ValueError` naming what disagreed otherwise. The plain `paired_difference()` only checks `.shape` and raises `ValueError` on a mismatch there, but cannot detect two same-shaped ensembles built from different or misaligned draws -- see the water-abstraction/climate-projection workflow in USER_GUIDE.md Section 12.

## 8. Empirical Results

### 8.1 DE-MCMC vs. DE-CV-MCMC Initialization (Dischmabach dataset)

An internal comparison ran both `DE-MCMC` and `DE-CV-MCMC` on the same Swiss validation dataset (`DAV_2327`, Version 8, 32 walkers, 100 MCMC steps, 90th-percentile-configured interval reported at 95%), holding the DE phase and MCMC step count fixed so that only the walker-initialization strategy differed.

| Quantity | `DE-MCMC` | `DE-CV-MCMC` |
|---|---|---|
| Estimated residual `sigma` (deg C) | 0.6064 | 0.6026 |
| Estimated AR(1) `rho` | 0.7122 | 0.7107 |
| Valid residual pairs used for `rho` | 2202 | 1837 |

The close agreement in `sigma` and `rho` between the two modes confirms that both converge to essentially the same best-fit residual structure. **This comparison is a non-convergence diagnostic, not evidence that `DE-CV-MCMC` finds a "better" or "wider" posterior.** If both chains have genuinely converged to the stationary posterior, they *must* agree on the parameter spread as well as the point estimate — MCMC convergence means the chain's distribution no longer depends on where it started. Any visible difference between the two modes' posterior histograms or predictive envelopes within a shared, limited step budget (as shown in the package's `posterior_comparison.png` and `envelope_comparison.png` diagnostic plots) is therefore evidence that the tightly-initialized `DE-MCMC` ensemble has *not yet* mixed out to the true posterior width at that step count — a useful thing to show, but the opposite of a methodological endorsement of `DE-CV-MCMC`'s spread as more "correct". Use the convergence diagnostics in Section 3.4 (split-Rhat, autocorrelation time relative to `mcmc_steps`) to check whether either chain has actually converged before trusting its reported width; `DE-CV-MCMC`'s CV-informed initialization is a **speed** device — it can reach a representative sample with fewer steps — not a change to the target distribution.

### 8.2 IID vs. AR(1) Forward Prediction Intervals

A separate worked example calibrated `DE-MCMC` (Version 8) against a synthetic historical dataset with injected AR(1)-structured noise, then used the resulting chain to generate forward prediction intervals under both noise models.

**Calibration results:** NSE = 0.9728, R² = 0.9728, MAE = 0.6463 deg C. Estimated residual standard deviation `sigma` = 0.8185 deg C, closely matching the injected synthetic noise level; estimated AR(1) coefficient `rho` = 0.5910, confirming clear day-to-day memory in the residuals.

**Forward interval comparison:** at the daily scale, the IID and AR(1) 90% prediction intervals have essentially the same width, since both are calibrated to the same marginal residual variance. The practical difference appears under temporal aggregation: computing a 7-day rolling average of the ensemble, the IID interval visibly narrows (independent daily noise partially cancels when averaged), while the AR(1) interval remains close to its original width (temporally correlated noise does not cancel to the same degree). This makes the AR(1) noise model the more conservative and structurally realistic choice for any downstream use involving multi-day averages or sustained-threshold-exceedance questions (e.g., multi-day thermal stress events for aquatic organisms).

## 9. Practical Considerations and Limitations

- **Cost.** MCMC sampling is the most computationally expensive calibration mode in `pyair2stream`, scaling with `mcmc_walkers x mcmc_steps` on top of the initial DE + L-BFGS-B fit (and, for `DE-CV-MCMC`, on top of a full N-fold cross-validation as well). Reduce `mcmc_steps`/`mcmc_walkers` or use `optimizer_overrides` within the CV phase of `DE-CV-MCMC` to control runtime, but check the reported acceptance fraction and autocorrelation-time warnings before trusting a shortened chain.
- **Likelihood assumptions.** The Gaussian, homoscedastic likelihood is still a simplification (no heteroscedasticity, no hierarchical error model — see the "Out of scope" note in `docs/audit/04_uncertainty_and_mcmc.md`). Temporal independence, however, is now optional rather than baked in: setting `uncertainty_options.noise_model: ar1` couples the same estimated `rho` into the sampler's own likelihood (Section 3.1), not just into the downstream predictive envelope (Section 3.5) as in earlier versions. The `iid` likelihood (still the default, for backward compatibility) remains fully temporally-independent-within-the-likelihood, and understates posterior width by roughly `sqrt((1+rho)/(1-rho))` whenever the true residuals are autocorrelated.
- **`rho` is a plug-in estimate, not a sampled parameter — under either likelihood.** The AR(1) coefficient used for prediction intervals, and for the `ar1` likelihood itself, is estimated once from best-fit residuals and held fixed; it is not integrated over as part of the posterior uncertainty. Treating `rho` as an additional sampled dimension would remove this simplification but was judged unnecessary complexity for the accuracy gain (`docs/audit/04_uncertainty_and_mcmc.md`, 4.1).
- **Forward projections require a prior MCMC run.** Forward prediction intervals are only available if a `DE-MCMC` or `DE-CV-MCMC` calibration has already been run and its chain (and, for convenient `rho` carry-forward, its sidecar file) is available on disk; `FORWARD` mode cannot generate probabilistic intervals from a single deterministic `parameters_forward` vector alone.
- **Small-sample AR(1) fallback.** If a dataset (or a forward run's own observations) provides fewer than 30 valid consecutive-day residual pairs, `rho` is set to 0.0 with a warning rather than estimated from an unreliable small sample — users relying on the AR(1) model with sparse observational records should check for this warning.

## 10. References

- Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *Environmental Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011
- Foreman-Mackey, D., Hogg, D. W., Lang, D., and Goodman, J. (2013). emcee: The MCMC Hammer. *Publications of the Astronomical Society of the Pacific*, 125(925), 306–312.
- Goodman, J. and Weare, J. (2010). Ensemble samplers with affine invariance. *Communications in Applied Mathematics and Computational Science*, 5(1), 65–80.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley. (AR(1) process definition and stationarity conditions.)
- Piotrowski, A. P. and Napiorkowski, J. J. (2018). Performance of the air2stream model that relates air and stream water temperatures depends on the calibration method. *Journal of Hydrology*, 561, 395–412.

## Appendix: Source Reference

Implementation: `pyair2stream/optimization.py` (`DE_MCMC_mode`, `DE_CV_MCMC_mode`, their shared `_run_mcmc_uncertainty` phase, the prediction-interval branch of `forward_mode`, and the shared `_check_ensemble_divergence`/`_hash_file` helpers), `pyair2stream/model.py` (`is_numerically_divergent`, the per-draw divergence check shared by both ensemble loops), `pyair2stream/uncertainty.py` (`estimate_ar1_rho`, `generate_ar1_noise`, `build_ar1_runs`, `ar1_whitened_stats`), `pyair2stream/scenario.py` (`load_ensemble`, `aggregate`, `exceedance`, `paired_difference`, `paired_difference_from_files`), with configuration parsing in `pyair2stream/io.py` and dispatch in `pyair2stream/main.py::run_optimizer`. Worked examples: `examples/mcmc_comparison/README.md` (DE-MCMC vs. DE-CV-MCMC) and `examples/forward_prediction_intervals/README.md` (IID vs. AR(1) forward projection). Repository: https://github.com/LukeAFullard/pyair2stream.
