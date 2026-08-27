# 02 — Explicit integrators diverge silently under scenario flows

**Priority: P0.** Produces either astronomically wrong or plausible-but-wrong output with
no error, no NaN and no warning.

## Background

For all five model versions the ODE is **linear in the state variable**:

```
dTw/dt = A(t) - B(t) * Tw

version 8:   A = (a1 + a2*Ta + theta*(a5 + a6*cos(2*pi*(t-a7)))) / theta**a4
             B = (a3 + a8*theta) / theta**a4
```

`B` is the reciprocal thermal response time in 1/day. The integration step is fixed at
`dt = 1 day` (`model_numba.py:13`, `TTT = 1.0/365.0` is the *time-of-year* increment, not
the step size). Stability of an explicit scheme therefore depends entirely on the
dimensionless product `B·dt = B`.

Because `B` contains `theta`, **B changes when the discharge scenario changes**. A
parameter set that is stable on the calibration record can be unstable on a scenario
record. With `a4 < 0` — which the optimiser does select, see below — higher flow
*increases* B.

## Evidence

### One-step amplification factors, computed analytically

```
 B (=lambda*dt)     exact       CRN         RK4         RK2
          0.500    0.6065    0.6000      0.6068      0.6250
          1.000    0.3679    0.3333      0.3750      0.5000
          2.000    0.1353    0.0000      0.3333      1.0000
          2.785    0.0617   -0.1641      0.9996      2.0931
          3.000    0.0498   -0.2000      1.3750      2.5000
          6.000    0.0025   -0.5000     31.0000     13.0000
         10.000    0.0000   -0.6667    291.0000     41.0000

  RK4                  unstable for B >= 2.785
  RK2                  unstable for B >= 2.000
  EUL (as implemented) unstable for B >= 2.000
  CRN                  unconditionally stable
  exponential          unconditionally stable and exact for constant coefficients
```

`EUL` in `model_numba.py` evaluates the forcing at `j+1` but the state at `j`, so it is
explicit in `Tw` with `R = 1 - B`; the forward forcing does not buy stability.

### Real failure on a real fit

Two DE runs with byte-identical config produced:

```
run 1  NSE 0.99214   a4=-0.317  a5=11.16  a6=8.09  a8=1.15
run 2  NSE 0.99268   a4=-0.750  a5= 3.49  a6=2.44  a8=0.35
```

Statistically indistinguishable fits whose discharge terms differ threefold (this is a
separate equifinality problem, see report 09). Run 1 pushed through a naturalised-flow
scenario, Qmedia correctly pinned:

```
DE run 1 params (NSE=0.9921 on calibration), max simulated Twat:
   RK4  flow x1.0   max Twat =         17.7
   RK4  flow x1.1   max Twat =         17.7
   RK4  flow x1.2   max Twat =     1.26e+05   <-- DIVERGED
   RK4  flow x1.5   max Twat =     1.32e+62   <-- DIVERGED
   RK4  flow x2.0   max Twat =           13   <-- plausible-looking, still wrong
   CRN  flow x1.0   max Twat =         17.6
   CRN  flow x1.5   max Twat =         17.4
   CRN  flow x2.0   max Twat =         17.3
```

Three things matter here. Stability at the calibration flow says nothing about stability
at scenario flow. `Tice_cover` clamps only from below, so nothing catches divergence. And
the `x2.0` case is the dangerous one: divergence followed by re-stabilisation leaves a
plausible number (12.96) where the truth is 17.19.

### Exponential integrator

Because the ODE is linear in `Tw`, an integrating-factor step is exact for
piecewise-constant coefficients:

```
Tw[j+1] = Tw[j] * exp(-B) + (A/B) * (1 - exp(-B))        # B > 0
Tw[j+1] = Tw[j] + A                                       # B -> 0 limit
```

with `A`, `B` taken as the trapezoidal average of their endpoint values. Tested against
the same divergent parameter set:

```
 flow x |      RK4 max   CRN max   EXP max |  CRN-EXP mean|d|
    1.0 |        17.72    17.606    17.596 |           0.0161
    1.2 |    1.256e+05    17.502    17.474 |           0.0192
    1.5 |    1.323e+62    17.397    17.338 |           0.0233
    2.0 |        12.96    17.273    17.189 |           0.0284
```

### Note on CRN in the stiff regime

CRN is A-stable but not L-stable: its amplification factor goes negative for `B > 2`,
which in principle causes spurious day-to-day oscillation. **This was tested and does not
materialise on realistic forcing.** A deliberately stiff set (`a3=3.0, a8=3.0, a4=0`,
giving `B` from 3.85 to 8.45) gave:

```
  RK4 max Twat = nan                        <-- diverged
  CRN vs EXP: mean|diff| = 0.036 degC, max|diff| = 0.152 degC
  day-to-day sign-flipping jumps >0.5C:  CRN = 0,  EXP = 0
```

The homogeneous mode decays within days and the solution is forcing-dominated. **CRN is an
adequate fix on its own.** The exponential integrator is a refinement, not a prerequisite.

## Required changes

### 2.1 Add a divergence guard (do this first — it is the cheapest and highest value)

After every `call_model` in a user-facing path, check the simulated range and fail loudly.

```python
# model.py, at the end of call_model / call_model_segmented — shape only
finite = np.isfinite(Twat_mod) & (Twat_mod != -999.0)
if not finite.all() or (Twat_mod[finite] > TWAT_SANITY_MAX).any():
    raise NumericalDivergenceError(...)   # include first offending index, date, theta, B
```

Suggested `TWAT_SANITY_MAX = 60.0`. Make it a config key `max_plausible_twat`.

This must **not** raise inside the optimiser hot loop, where diverged parameter sets are a
normal occurrence and are already handled by returning a penalty. Gate it on a flag, or
apply it only in `forward_mode`, `main.forward()` and `sensitivity_analysis`. In the
optimiser, convert a diverged simulation into the existing NaN/penalty path instead.

### 2.2 Pre-flight stability check

Before any simulation, compute `B` over the whole forcing series and compare against the
scheme's limit.

```python
# new helper, e.g. model.stability_report(data) -> dict
theta = Q / Qmedia
B     = (a3 + a8*theta) / theta**a4          # versions 4/7/8; a8=0 for v4, a4=0 for v7
limit = {'EUL': 2.0, 'RK2': 2.0, 'RK4': 2.785, 'CRN': np.inf, 'EXP': np.inf}[mod_num]
```

Report `max(B)`, the fraction of days exceeding `limit`, and the dates of the worst days.
Warn on any exceedance; error if more than a small fraction of days exceed it.

This criterion is **conservative, not exact**. Run 1 above is flagged at `flow x1.0`
(`max B = 3.007`) yet simulates fine, because isolated high-θ days let the transient decay
before it compounds; fatal divergence requires consecutive high-θ days. Report it as a
screening warning, not a hard verdict, and pair it with 2.1 which catches the real failure.

### 2.3 Change the default integrator to CRN

`io.py:35` currently defaults `integrator` to `RK4`. Change the default to `CRN`.

This changes results for any user who relied on the default. It is the right trade: CRN is
unconditionally stable, second-order, and matches the exponential integrator to under
0.04 °C on the cases tested. Document under "Known deviations" and in `CHANGELOG.md`.

Keep RK4/RK2/EUL available — they are needed for the Fortran golden tests and are more
accurate than CRN when `B < 1`.

### 2.4 Add an `EXP` integrator

Add `mod_num_idx == 4` to `fast_run_integration` in `model_numba.py`, implementing the
integrating-factor step above. Requirements:

- Derive `A` and `B` per version. For versions 3 and 5, `theta` does not appear:
  `B = a3`, `A = a1 + a2*Ta + a6*cos(...)`. For version 4, `a5 = a6 = a8 = 0`. For
  version 7, `a4 = 0` so the divisor is 1.
- Guard the `B -> 0` limit with the linear fallback; use a relative threshold, not an
  absolute one.
- Apply the same `Tice_cover` clamp as the other schemes.
- Wire `'EXP'` into the `mod_num` mapping in `model._run_integration` (lines 108-116).

Do **not** make `EXP` the default in this change. Land it, validate it against CRN across
the example datasets, then consider promoting it.

### 2.5 Document the stability constraint

Add a section to `USER_GUIDE.md`. The existing troubleshooting row ("Results look
implausible... Switch to RK4 or CRN") is actively misleading — switching *to* RK4 is the
wrong advice. Replace it. State the `B` criterion, the per-scheme limits, and the rule
that scenario work should use CRN or EXP.

## Acceptance criteria

- A test using the run-1 parameter vector and a x1.5 flow scenario that asserts RK4 raises
  `NumericalDivergenceError` and CRN does not.
- A test asserting CRN and EXP agree to within 0.1 °C on the quickstart dataset and on the
  stiff parameter set above.
- A test asserting `stability_report` flags `max(B) > 2.785` for RK4 on the run-1 set.
- `tests/test_golden.py` and `tests/test_physics_golden.py` unchanged and passing — the
  guard must not fire on the reference cases.

## Out of scope

Adaptive step-size control and sub-daily stepping. Both would break the Numba path and the
Fortran equivalence, and the exponential integrator makes them unnecessary for this ODE.
