# 09 — Study design notes

Not a code-change report. These are properties of the model and the calibration problem
that will affect the two studies regardless of how many bugs are fixed. An implementing
agent should read this for context; the actionable items are cross-referenced to the other
reports.

## Equifinality is severe, and it lands directly on the discharge parameters

Two `DE` runs with byte-identical configuration:

```
run 1  NSE 0.99214   a4=-0.317  a5=11.16  a6=8.09  a8=1.15
run 2  NSE 0.99268   a4=-0.750  a5= 3.49  a6=2.44  a8=0.35
```

The fits are statistically indistinguishable. But `a5`, `a6` and `a8` are the coefficients
of the `theta` group — **the entire discharge dependence of the model** — and they differ
by a factor of more than three. `a4`, the thermal-capacity exponent, differs in magnitude
by a factor of two and is negative in both.

This is not a bug and it is not news; `examples/validation/Switzerland/README.md` already
discusses equifinality, and the literature does too. But it has a specific consequence for
an abstraction study: **the flow sensitivity of a single best-fit parameter vector is close
to arbitrary.** A point estimate of "abstraction warms the river by X °C" derived from one
`DE` run is not defensible.

The honest response is to propagate it. Sample parameter sets from the posterior, run the
paired scenario per draw, and report the distribution of ΔT. Given that the posterior is
itself too narrow until report 04 lands, treat the resulting spread as a lower bound.

## Scenario differences must be paired

The quantity of interest is a difference between two simulations, not a simulation. That
requires, per parameter draw `theta_i`:

```
dT_i = simulate(theta_i, Q_naturalised) - simulate(theta_i, Q_observed)
```

then percentiles over `i`. Two rules follow.

**Use the same draw for both arms.** Differencing independently-drawn ensembles inflates
the variance of `dT` by roughly a factor of two and destroys the correlation structure that
makes the difference well-determined even when the individual levels are not.

**Do not add residual noise to the difference.** The envelope machinery adds
`sigma`-scaled iid or AR(1) noise to each simulation (`optimization.py:718-727`). That
noise represents observation and structural error in the *level*. It is not signal in the
difference and does not cancel automatically in the current code, because the two arms draw
independent noise realisations. Compute `dT` from the noise-free simulations.

The package has no facility for any of this. Report 04 change 4.2 proposes
`scenario.paired_difference`; until it exists, this must be scripted against the MCMC chain
CSV directly.

## Watch `a4`

Both DE runs above returned `a4 < 0`, and the bundled Switzerland results contain converged
CRN fits with `a4 = -0.530` alongside `a6` pinned at its upper bound of 10.0.

`a4` is the exponent in `theta**a4`, representing how thermal inertia scales with
discharge. Physically it should lie in `[0, 1]`: more water, more thermal mass, slower
response. A negative `a4` inverts that, so reduced flow makes the river respond *more
slowly* — the opposite of the mechanism an abstraction study is trying to quantify. It will
extrapolate in the wrong direction under scenario flows.

Negative `a4` is also the specific condition that makes higher flow destabilise the
explicit integrators (report 02): `B = (a3 + a8*theta)/theta**a4` increases with `theta`
when `a4 < 0`.

Recommendations:

- Constrain `parameter_bounds` for `a4` to `[0, 1]` for scenario work.
- Treat any parameter that converges onto a bound as a failed fit, not a result. `a6 = 10.000`
  and `a5 = 0.000` in the bundled results are both bound-limited.
- Report the fitted `a4` alongside any ΔT estimate.

## Scenario flows move you outside the calibrated θ range

The model is fitted over the `theta` range present in the observed record. A naturalised
series has systematically higher `theta`; a climate scenario may have both tails extended.
Nothing in the package tracks this. Report 01 change 1.4 and report 02 change 2.2 add the
diagnostics; use them, and report the fraction of scenario days outside the calibrated
range as a caveat on the result.

Two specific hazards at the low-flow end:

- `Q = 0` raises a bare `ZeroDivisionError` from inside the Numba kernel in
  non-gap-tolerant mode. Only the gap-tolerant path screens `Q <= 0` (`model.py:38`).
- As `theta -> 0` with `a4 > 0`, `theta**a4 -> 0` and the derivative diverges. Severe
  abstraction scenarios approach this.

I tested 80% and 95% abstraction on the quickstart data with a well-behaved parameter set
and got stable, plausible results (+1.9 °C and +2.5 °C on the summer maximum). So this is
not automatically fatal — but it needs checking per case rather than assuming.

## Confounding in the abstraction fit

The parameters are calibrated on a record in which abstraction was already occurring. If
abstraction correlates with season — which it usually does, being highest in summer — then
`a2` and `a3` (the air-temperature and relaxation terms) will absorb part of the flow
effect, and the fitted `theta` group will understate the true flow sensitivity.

This is a structural identification problem, not something the code can fix. Mitigations
worth considering: calibrate on the sub-period with the least abstraction if one exists;
check whether abstraction volume and air temperature are correlated in the record and say
so; and test whether including a period of naturally low flow with no abstraction changes
the fitted `theta` group.

## What the climate study needs that the outputs cannot currently supply

The deliverables of a water-quality projection are usually aggregate: degree-days above a
threshold, number of days above a threshold, longest sustained exceedance, seasonal mean
warming. None of these can be computed from the three percentile columns the package
writes — the p5 of a rolling mean is not the rolling mean of the p5. Report 04 change 4.2
(emit the ensemble, plus an aggregation helper) is a prerequisite for study 2, not a
nice-to-have.

`docs/MCMC_uncertainty.md` §6.2 correctly identifies that AR(1) versus iid only matters
under aggregation. That is precisely the regime study 2 operates in, so the AR(1) noise
model should be used — and the AR(1) *likelihood* correction (report 04 change 4.1) matters
correspondingly more.

## Suggested minimum protocol for study 1

1. Calibrate with `run_mode: DE-MCMC`, `integrator: CRN`, `a4` bounded to `[0, 1]`,
   an explicit `random_seed`, and cross-validation enabled.
2. Confirm no parameter sits on a bound; confirm split-Rhat and acceptance fraction are
   sane; confirm cross-fold parameter spread is not larger than the posterior spread (if it
   is, the posterior is too narrow — expected until report 04 lands).
3. Record the calibration `Qmedia` and the calibrated `theta` range.
4. For each posterior draw, run both arms in `FORWARD` mode with `Qmedia` **pinned** to the
   calibration value, noise disabled.
5. Difference paired, then take percentiles of the differences.
6. Report the fraction of scenario days outside the calibrated `theta` range and the
   fraction flagged by the stability screen.
7. Sanity-check that `max(Twat_mod)` is physically plausible in every run.

Steps 3, 4 and 7 are the ones that currently have no support in the package and are
covered by reports 01, 02 and 04.
