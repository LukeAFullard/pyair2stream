# 04 Scenario: what difference would a change make?

**Question:** if 30% of the Mentue's flow had been taken out (abstracted) in
2010–2012, how much would its water temperature have changed? And how sure can
we be?

The same approach works for any change in the inputs: naturalised flows, a dam
release, or air temperatures from a climate projection.

## The key idea: pair the simulations

Run the model twice: once on the measured discharge, and once on the reduced
discharge. Then subtract. Each simulation in the second run must use **the same
parameter set** as its partner in the first run. The uncertainty the two runs
share then cancels. What remains is the uncertainty of the *difference*.

pyair2stream also gives each pair the same day-to-day model error, which
cancels too. This assumes the model's error on a given day would be the same in
both scenarios.

## Steps

Run everything with one command, from the repository's top folder:

```bash
python examples/04_scenario/run.py
```

It takes two to three minutes. First it makes the scenario's input file: the
measured file with the discharge multiplied by 0.7. It removes the water
temperature, because that was not measured under the scenario. Then it runs:

```bash
pyair2stream --config examples/04_scenario/calibrate.yaml     # calibrate with uncertainty, as in example 02
pyair2stream --config examples/04_scenario/baseline.yaml      # 1,000 simulations with the measured discharge
pyair2stream --config examples/04_scenario/abstraction.yaml   # the same, with 70% of the discharge
```

Two settings in [`abstraction.yaml`](abstraction.yaml) matter:

- `reuse_sample_indices_from` points at the baseline run's record of which
  parameter sets it used. So this run uses exactly the same ones.
- `calibration_metadata` keeps the calibration's mean discharge (`Qmedia`). The
  model sees discharge only relative to `Qmedia`. If `Qmedia` were recomputed
  from the reduced flows, the reduction would cancel out. The scenario would
  then show no effect at all.

Then the difference, simulation by simulation:

```python
from pyair2stream import scenario

out = "examples/04_scenario/output"
diff = scenario.paired_difference_from_files(
    f"{out}/abstraction/Forward_Prediction_Ensemble_Mentue_c_1d.npz",
    f"{out}/baseline/Forward_Prediction_Ensemble_Mentue_c_1d.npz")
# abstraction minus baseline: one row per simulation, one column per day.
# It refuses runs that did not use the same parameter sets.
```

## Results

| | Median | 90% range |
|---|---|---|
| Average summer (Jun–Aug) change | +0.04 °C | +0.02 to +0.08 °C |
| Average winter (Dec–Feb) change | −0.07 °C | −0.12 to −0.03 °C |
| Largest warming on a single day | +0.33 °C | +0.27 to +0.39 °C |
| Extra days per year above 18 °C | 1.0 | 0.0 to 2.0 |
| *Summer change if the runs were not paired* | *+0.05 °C* | *−0.32 to +0.43 °C* |

![Daily effect of the abstraction and the summer average](figures/abstraction_effect.png)

*Left: the change on each day, the median of the 1,000 paired simulations and
their 90% range, drawn with `plots.change(diff, dates, by="day")`. Right: each
simulation's average summer change, paired (orange) and not paired (grey).*

**Reading it.**

- According to the fitted model, this abstraction changes the Mentue's
  temperature only slightly on average.
- Summers are a little warmer and winters a little colder. With less water,
  the river follows the air more closely.
- On single low-flow days, the warming reaches about 0.3 °C.
- Without pairing (the last row, grey in the figure), the same simulations
  cannot even tell whether the river gets warmer or colder.

## Limits of this approach

- **Parameter uncertainty only.** The narrow ranges show the uncertainty of the
  parameters *within this model*. They do not cover the chance that the
  model's response to discharge is wrong. The model learns that response from
  the natural changes in flow during the calibration years.
- **Stay within the calibrated flows.** If the scenario's discharge goes
  outside the range seen in calibration, the run prints a warning. The model
  is then extrapolating.
- The validation suite checks that paired differences are exact where the true
  answer is known ([V8](../../validation/REPORT.md#v8)).

## Next

Example [05](../05_gaps/README.md) deals with gaps in the data.
