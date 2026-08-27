# 04 — Uncertainty quantification

**Priority: P1.** Posterior intervals are too narrow by a large and unquantified factor,
and the output format cannot supply the aggregate statistics both studies need.

**Depends on report 03.** Land the `eval_mask` fix first.

## What is already correct

Stated up front so it is not disturbed:

- `uncertainty.generate_ar1_noise` (`uncertainty.py:56`) produces a genuinely stationary
  AR(1) process — marginal SD equal to `sigma`, correct lag-1 structure, correctly reset
  per segment. Verified numerically. Do not touch.
- `uncertainty.estimate_ar1_rho` correctly pairs only temporally adjacent, in-segment,
  observed days. Do not replace with a naive `Series.autocorr()`.
- `docs/MCMC_uncertainty.md` §3.2 and §6.2 already state correctly that iid and AR(1) give
  essentially identical *pointwise* interval widths and that the difference appears under
  temporal aggregation. That analysis is right.
- `docs/MCMC_uncertainty.md` §4 and §5.2 correctly frame `DE-CV-MCMC` as an
  **initialisation** device that changes convergence speed, not the target distribution.

## Defect A — the likelihood assumes independent residuals

```python
# optimization.py:586 and 873
log_L = -0.5 * N * np.log(SSE / N)
```

This is the concentrated Gaussian log-likelihood with `sigma` profiled out, and it is the
correct form — *for iid residuals*. Daily stream-temperature residuals typically have
lag-1 autocorrelation of 0.8-0.95. The effective sample size is roughly

```
N_eff ≈ N * (1 - rho) / (1 + rho)
```

which for `rho = 0.9` is `N/19`. Since `log_L` scales linearly with `N`, the posterior is
sharpened by roughly that factor; parameter standard deviations are understated by
something on the order of `sqrt(19) ≈ 4.4`.

This is internally inconsistent: the code estimates `rho` and uses it to generate AR(1)
noise for the envelopes, then discards it when inferring the posterior.

For both studies the deliverable is a *credible interval on a temperature change*, so this
is the dominant source of error in the reported uncertainty.

## Defect B — the ensemble is discarded

`optimization.py:686` (and `972`, `126`) accumulates `ensemble_simulations`, takes three
percentiles, and lets the array go out of scope. Only `Twat_mod_lower`, `Twat_mod_p50`,
`Twat_mod_upper` reach disk.

Percentile bands cannot produce aggregate statistics. The p5 of a 7-day rolling mean is not
the 7-day rolling mean of the p5 series. So degree-days, days above a threshold, sustained
exceedance duration, and summer-mean warming — the actual outputs of a water-quality
projection — are unobtainable from the current output format.

`docs/MCMC_uncertainty.md` §6.2 explicitly advertises the aggregation benefit of AR(1),
and the repo's own example
(`examples/forward_prediction_intervals/run_example.py:102-140`) has to work around this by
re-implementing the AR(1) generator inline and substituting the ensemble median for the
deterministic base — which drops parameter uncertainty from the demonstration entirely.

## Defect C — `sigma` is not carried forward the way `rho` is

`DE-MCMC` writes both to `MCMC_chain_*_meta.json` (`optimization.py:657-668`).
`forward_mode` reads `rho` from that sidecar automatically (`optimization.py:107-119`) but
takes `sigma` only from config, defaulting to `0.0`:

```python
# optimization.py:91-93
sigma = float(data.forward_options.get('residual_sigma', 0.0))
if sigma <= 0.0:
    print("Warning: residual_sigma is 0.0. ...")
```

Forget to hand-copy it and you silently get parameter-only intervals behind a one-line
`print`. Asymmetric and easy to get wrong.

## Defect D — walker initialisation can collapse on a bound

```python
# optimization.py:591
p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
# then clipped to [parmin, parmax]
```

If the DE optimum sits on a bound — which happens, see the `a6 = 10.000` and `a5 = 0.000`
cases in `examples/validation/Switzerland/README.md` — clipping collapses the ensemble in
that dimension. emcee's stretch move cannot generate spread from a degenerate ensemble.

## Required changes

### 4.1 Account for residual autocorrelation in the likelihood

Preferred: implement an explicit AR(1) Gaussian likelihood. For residuals `e`, with the
first element of each contiguous run scaled by `sqrt(1-rho^2)`, the whitened residuals

```
u[0] = e[0] * sqrt(1 - rho**2)
u[t] = e[t] - rho * e[t-1]
```

are iid, and the concentrated log-likelihood becomes
`-0.5*N*log(SSE_u/N) + 0.5*log(1-rho**2)`. Reuse the adjacency logic already in
`estimate_ar1_rho` to build the runs. Treat `rho` either as fixed (estimated once at the
DE optimum) or as an extra sampled dimension; fixed is simpler and adequate.

Minimum acceptable alternative if the above is deferred: keep the iid form but replace `N`
with `N_eff` computed from the estimated `rho`, and record `rho`, `N` and `N_eff` in the
sidecar. This is crude but removes most of the bias in interval width.

Gate the choice on the existing `uncertainty_options.noise_model` key so `iid` reproduces
today's behaviour exactly.

### 4.2 Emit the ensemble

Add `uncertainty_options.save_ensemble: bool` (default `false`). When true, write the
`(n_samples, n_tot)` matrix — Parquet or compressed npz, not CSV; at 1000 samples x 20
years this is ~60 MB as CSV.

Also add a small helper module so users are not forced to reconstruct aggregates by hand:

```python
# new: pyair2stream/scenario.py — signatures only
def load_ensemble(path) -> np.ndarray: ...
def aggregate(ensemble, dates, how='mean', freq='7D') -> np.ndarray: ...
def exceedance(ensemble, threshold, consecutive_days=1) -> np.ndarray: ...
def paired_difference(ens_a, ens_b) -> np.ndarray:
    """Row-aligned difference. Requires both ensembles generated from the SAME
    parameter draws in the SAME order — see report 09."""
```

`paired_difference` is the function both studies actually need and is currently absent.

### 4.3 Read `sigma` from the sidecar

In `forward_mode`, load `sigma` from `<chain_path>_meta.json` when
`forward_options.residual_sigma` is absent, mirroring the existing `rho` logic. Keep the
config key as an override. Escalate the `sigma <= 0.0` case from `print` to a raised error
when `enable_prediction_intervals` is true — a prediction interval with no residual term
is not a prediction interval.

### 4.4 Robust walker initialisation

Scale the initial ball to each parameter's bound width rather than a fixed `1e-4`, and
reflect off bounds rather than clipping. After construction, assert every dimension has
non-zero variance across walkers and raise if not.

### 4.5 Report the `DE-CV-MCMC` comparison honestly

`docs/MCMC_uncertainty.md` §5.2 points at `envelope_comparison.png` as evidence the
CV-informed initialisation works. If both chains have converged they must agree; if they
disagree, at least one has not converged. Reframe that figure as a *non-convergence
diagnostic* — "how far from stationary a tightly-initialised ensemble still is after N
steps" — which is a genuinely useful thing to show. Do not present envelope divergence as
a methodological improvement.

### 4.6 Convergence diagnostics

- Burn-in is hardcoded at 30% (`optimization.py:625`, `896`). Make it configurable and
  default to `max(0.3*nsteps, 5*max(tau))`.
- `get_autocorr_time` is computed on the full chain including burn-in. Compute it on the
  post-burn-in chain.
- Add split-Rhat across walker halves and report it in the sidecar. Acceptance threshold
  `< 1.01`.
- Promote the existing chain-length warning to a hard failure when
  `nsteps < 50 * max(tau)` and a new `uncertainty_options.strict_convergence` is set.

## Acceptance criteria

- A synthetic test: generate data with known AR(1) residuals of known `rho`, fit, and
  assert the AR(1) likelihood recovers posterior widths within a factor of 1.5 of the truth
  while the iid likelihood understates them by more than 2x.
- A test asserting `save_ensemble` produces a matrix whose row-wise percentiles reproduce
  the envelope CSV to 1e-9.
- A test asserting `paired_difference` on two ensembles from identical draws with identical
  forcing returns exactly zero.
- A test asserting `forward_mode` errors when prediction intervals are enabled and no
  `sigma` is available from config or sidecar.
- A test asserting walker initialisation retains non-zero spread when the optimum is
  exactly on a bound.

## Out of scope

Do not replace emcee. Do not add a hierarchical or heteroscedastic error model — the AR(1)
correction is the change that matters, and further elaboration is unwarranted given the
equifinality already present in the parameters (report 09).
