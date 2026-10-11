# 04 Scenario: what difference would a change make?

**Question:** if 30% of the Mentue's flow had been taken out (abstracted) in
2010–2012, how much would its water temperature have changed? And how sure can
we be?

The same approach works for any change in the inputs: naturalised flows, a dam
release, or air temperatures from a climate projection.

## Paired and not paired

The model is run twice: once with the measured discharge (the **baseline**),
and once with the reduced discharge (the **abstraction**). The effect of the
abstraction is the difference between the two. But how the two runs are
compared matters a great deal.

**Why there are 1,000 simulations.** As example
[02](../02_uncertainty/README.md) explains, the model's parameters are not
known exactly: many sets of parameter values fit the calibration years almost
equally well. Each run here draws 1,000 of these parameter sets. Each one gives
one **simulation** of 2010–2012, with its own parameter values and its own
day-to-day model error.

The simulations disagree about how warm the river is. Their average summer
water temperature ranges from 16.3 to 16.8 °C (90% range). The abstraction
changes it by only about 0.05 °C, ten times less. To see such a small effect,
the uncertainty about the river's temperature must be taken out.

**Paired** means that simulation 1 of the abstraction run uses the same
parameter set, and the same model error, as simulation 1 of the baseline run.
Simulation 2 is paired with simulation 2, and so on. Each pair is then
subtracted. Whatever a parameter set gets wrong, it gets wrong in both runs, so
it cancels. What is left is the effect of the abstraction:

| | Baseline: average summer temperature | Abstraction | Difference |
|---|---|---|---|
| Simulation 1 | 16.57 °C | 16.63 °C | +0.07 °C |
| Simulation 2 | 16.43 °C | 16.50 °C | +0.07 °C |
| Simulation 3 | 16.36 °C | 16.39 °C | +0.03 °C |

Simulations 1 and 3 disagree by 0.2 °C about the river. But they agree that the
abstraction warms it a little in summer.

**Not paired** means comparing simulations that used different parameter sets.
The abstraction of simulation 3 minus the baseline of simulation 2 gives
+0.13 °C. The other way round, it gives −0.04 °C, a cooling. These numbers mostly show the
difference between two parameter sets, not the effect of the abstraction. This
is what you would get by running the two scenarios separately, each with its
own random parameter sets, and comparing their results.

An everyday comparison: to find out what a diet does, weigh the same people
before and after it (paired). Weighing one group before and a different group
after (not paired) mixes the effect with the differences between people.

The last row of the results table shows the cost. There, `run.py` matches the
simulations of the two runs at random, on purpose. The 90% range of the summer
change becomes −0.30 to +0.39 °C, so you could not even tell whether the
abstraction warms or cools the river. It is shown only to make the point:
always pair.

Pairing assumes that the model's error on a given day would be the same in both
scenarios. On a winter day when a simulation is held at the ice floor
(`Tice_cover`, 0 °C) in one run and not in the other, the error does not
cancel exactly.

**The uncertainty method, and why.** This example uses a DE-MCMC chain (as in
example 02) and two `FORWARD` runs that draw the same 1,000 parameter sets from
it, with the same daily errors, and are then subtracted series by series (a
paired difference). It is used because the question is about a *change*: what
is uncertain about the river in both runs cancels, and what is left is the
uncertainty of the change itself, from the parameters. No cross-validated
correction is applied: it corrects a bias in a yearly statistic, which affects
both runs alike and cancels in their difference.
[docs/UNCERTAINTY.md §10](../../docs/UNCERTAINTY.md#10-comparing-two-scenarios)
explains pairing in plain words.

## Steps

Run everything with one command, from the repository's top folder:

```bash
python examples/04_scenario/run.py
```

It takes two to three minutes. First it makes the scenario's input: the
measured data with the discharge multiplied by 0.7. It removes the water
temperature, because that was not measured under the scenario. Then it runs
three steps with one `pyair2stream.Model`, made from
[`settings.yaml`](settings.yaml) (model version 8, as in example
[02](../02_uncertainty/README.md)):

```python
import pandas as pd
import pyair2stream

measured = pd.read_csv("data/switzerland/MAH_2369_validation.csv")
abstraction = measured.assign(Discharge=measured.Discharge * 0.7, T_water=float("nan"))

m = pyair2stream.Model("examples/04_scenario/settings.yaml")
m.calibrate()                                                        # with uncertainty, as in example 02
m.predict("data/switzerland/MAH_2369_validation.csv", name="baseline")      # 1,000 simulations, measured discharge
m.predict(abstraction, name="abstraction", paired_with="baseline")  # the same, with 70% of the discharge
```

A DataFrame is first written to `output/inputs/abstraction.csv`, so the run is
recorded like any other. Two settings of the abstraction run
(`output/abstraction.yaml`) matter, and the Model sets both:

- `reuse_sample_indices_from` (from `paired_with`) points at the baseline run's
  record of which parameter sets it used. So this run uses exactly the same
  ones, in the same order: this is what pairs the simulations.
- `calibration_metadata` keeps the calibration's mean discharge (`Qmedia`). The
  model sees discharge only relative to `Qmedia`. If `Qmedia` were recomputed
  from the reduced flows, the reduction would cancel out. The scenario would
  then show no effect at all.

Then the difference, simulation by simulation:

```python
diff = m.difference("abstraction", "baseline")
# abstraction minus baseline: one row per simulation, one column per day.
# It refuses runs that did not use the same parameter sets.
```

## Results

| | Median | 90% range |
|---|---|---|
| Average summer (Jun–Aug) change | +0.05 °C | +0.02 to +0.08 °C |
| Average winter (Dec–Feb) change | −0.06 °C | −0.11 to −0.02 °C |
| Largest warming on a single day | +0.33 °C | +0.27 to +0.40 °C |
| Extra days per year above 18 °C | 0.7 | 0.0 to 2.0 |
| *Summer change if the runs were not paired* | *+0.05 °C* | *−0.30 to +0.39 °C* |

![Daily effect of the abstraction and the summer average](figures/abstraction_effect.png)

*Left: the change on each day, the median of the 1,000 paired simulations and
their 90% range, drawn with `plots.change(diff, dates, by="day")`. Right: each
simulation's average summer change, paired (orange) and not paired (grey).*

**Reading it.**

- According to the fitted model, this abstraction changes the Mentue's
  temperature only slightly on average.
- Summers are a little warmer and winters a little colder. With less water,
  the river is pulled less towards the temperature linked to its flow, and more
  towards the air (see [Does the model version matter?](#does-the-model-version-matter)).
- On single days, the warming reaches about 0.3 °C, and the cooling about
  0.5 °C. These days are in winter and spring at above-median flows, not on
  the days of lowest flow (see [Other flow changes](#other-flow-changes)).
- Without pairing (the last row, grey in the figure), the same simulations
  cannot even tell whether the river gets warmer or colder (see
  [Paired and not paired](#paired-and-not-paired)).

## Does the model version matter?

Versions 4, 7 and 8 use discharge (versions 3 and 5 do not, so for them the
abstraction changes nothing). `python examples/04_scenario/compare_versions.py`
(about 10 minutes) repeats this example with each of them: the same settings,
data and seed, with only `version` changed.

| Version | RMSE, 2010–2012 | Days inside the 90% interval | a4 | Summer change | Winter change | Largest daily warming | Extra days per year above 18 °C |
|:-:|---|---|---|---|---|---|---|
| 4 | 0.94 °C | 86.9% | 0.200 | 0.00 (0.00 to 0.00) °C | +0.01 (0.00 to +0.02) °C | +0.18 (+0.05 to +0.34) °C | 0.0 (−0.3 to +1.0) |
| 7 | 0.78 °C | 89.2% | (none) | +0.05 (+0.02 to +0.08) °C | −0.07 (−0.11 to −0.02) °C | +0.31 (+0.25 to +0.37) °C | +0.7 (0.0 to +2.0) |
| 8 | 0.78 °C | 89.1% | 0.066 | +0.05 (+0.02 to +0.08) °C | −0.06 (−0.11 to −0.02) °C | +0.33 (+0.27 to +0.40) °C | +0.7 (0.0 to +2.0) |

*Median and, in brackets, 90% range over the 1,000 paired simulations
(`output/versions_summary.csv`). Version 8 is the run above.*

**How discharge enters each version.** The water is pulled towards an
equilibrium temperature, A/B in the notation of
[USER_GUIDE §9.1](../../USER_GUIDE.md#91-numerical-stability-and-the-choice-of-integrator).
In version 8 it is a weighted average of two temperatures:

    equilibrium = (1 − w) · (a1 + a2·Ta) / a3   +   w · (a5 + a6·cos(2π(t − a7))) / a8,
    with w = a8·θ / (a3 + a8·θ)

The first is set by the air. The second is linked to the flow: it varies less
through the year than the air, as water from groundwater, snowmelt or upstream
would. Its share w grows with discharge θ. The factor 1/θ^a4 cancels in the
equilibrium, so a4 changes only how fast the water gets there (its thermal
inertia), not where it goes.

- **Version 4 has only a4.** Discharge changes how fast the water follows the
  air (with less flow, faster), but not the equilibrium. So the averages cannot
  change. Single days can: a quicker response shows more of each warm spell.
- **Version 7 has only the flow-linked term** (version 8 with a4 = 0).
- **Version 8 has both, but here a4 is about zero.** Its best value is 0.066:
  at 70% of the flow, the speed changes by 2% (0.7^0.066 = 0.977). Its 90%
  credible interval, −0.20 to +0.22, includes zero: the data cannot tell a
  change of speed with flow from none. So version 8 gives version 7's answer.

**Why less flow warms the summer and cools the winter (version 8).** Monthly
means over 2010–2012, from the fitted parameters
(`output/version8_equilibrium_by_month.csv`):

| | Air-driven | Flow-linked | Share w, measured flow | Share w, 70% of the flow | Change in equilibrium |
|---|---|---|---|---|---|
| June–August | 15.9–17.1 °C | 13.5–15.8 °C (cooler) | 0.08–0.12 | 0.06–0.09 | +0.02 to +0.06 °C |
| December–February | 1.1–2.6 °C | 3.4–5.7 °C (warmer) | 0.22–0.36 | 0.17–0.29 | −0.08 to −0.20 °C |

The abstraction takes 30% of the flow in every season. In summer the
flow-linked temperature is cooler than the air-driven one, so losing some of
it warms the river; in winter it is warmer, so losing some of it cools the
river. Winter flows are higher, so the flow-linked share is larger then, and
the same abstraction moves the equilibrium further. The warming is largest in
spring (+0.10 °C in April) and the cooling in autumn and early winter (−0.11 °C
in November, −0.20 °C in December), which the summer and winter averages
above do not show.

**What this means.**

- **The model version is part of the answer.** Version 4's summer range and
  version 7 and 8's do not overlap. Each range covers the uncertainty of the
  parameters of one version only, not the choice of version.
- **Here the evidence favours versions 7 and 8**: they predict 2010–2012
  better (RMSE 0.78 against 0.94 °C), and version 4's 90% interval held on
  only 86.9% of days. Version 4's mechanism is not wrong in principle (less
  water does heat and cool faster); this record cannot detect it.
- **The flow-linked term is fitted, not measured.** "Groundwater" or
  "snowmelt" is an interpretation of its parameters; the model knows only
  that, in 2002–2009, the river was closer to this temperature when the flow
  was high.
- **For a decision, report the result for each version that fits** and say
  why the one relied on was chosen (its error and interval coverage in years
  not used for calibration).

## Other flow changes

What if flow is added instead, or changed only on the days of lowest or
highest flow? `python examples/04_scenario/flow_regimes.py` (about 10 minutes,
after `compare_versions.py`) runs six scenarios, each paired with the same
baseline. "Lowest flows" are the 10% of days of 2010–2012 with the least flow
(below 0.236 m³/s; July to December, mostly September and October), "highest
flows" the 10% with the most (above 2.25 m³/s; mostly November to February).
Every scenario stays within the flows of the calibration years.

Version 8, change from the baseline, median and 90% range
(`output/flow_regimes_summary.csv`; version 7 is within 0.03 °C of these):

| Scenario | Hottest 10% of days | Yearly highest 7-day mean | Days above 18 °C per year | Largest daily warming | Largest daily cooling |
|---|---|---|---|---|---|
| 30% taken, every day | +0.07 (+0.03 to +0.11) °C | +0.08 (+0.04 to +0.11) °C | +0.7 (0.0 to +2.0) | +0.33 °C | −0.54 °C |
| 30% added, every day | −0.07 (−0.10 to −0.03) °C | −0.07 (−0.11 to −0.04) °C | −0.7 (−2.0 to 0.0) | +0.47 °C | −0.27 °C |
| 30% taken, lowest flows | +0.01 (0.00 to +0.02) °C | +0.03 (+0.02 to +0.05) °C | 0.0 (0.0 to +0.3) | +0.10 °C | −0.10 °C |
| 30% added, lowest flows | −0.01 (−0.02 to 0.00) °C | −0.03 (−0.05 to −0.02) °C | 0.0 (−0.3 to 0.0) | +0.09 °C | −0.09 °C |
| 30% taken, highest flows | 0.00 °C | 0.00 °C | 0.0 | +0.29 °C | −0.51 °C |
| 30% added, highest flows | 0.00 °C | 0.00 °C | 0.0 | +0.38 °C | −0.21 °C |

*The hottest 10% of days are those with the warmest baseline (median of its
simulations). Largest daily warming and cooling: medians over the
simulations.*

**Reading it.**

- **Adding mirrors taking.** 30% more flow cools the hottest days and the
  yearly peaks by about as much as 30% less flow warms them.
- **Changing only the lowest flows barely matters, according to the model**:
  at most 0.03 °C on the yearly highest 7-day mean. There are two reasons.
  The lowest flows come mostly in September and October, after the hottest
  weeks: only 21% of the hottest days are among them. And on those days the
  model gives the flow little weight. A change in flow moves the equilibrium
  by w·(1 − w)·(flow-linked − air-driven) per unit change of ln θ, and w, the
  flow-linked share, is smallest when flow is lowest (0.08 in August on average,
against 0.36 in December; see the table above).
- **Changing only the highest flows changes nothing in summer.** The highest
  flows are mostly winter floods, and the largest single-day changes (±0.2 to
  0.5 °C) are on those days.
- **The hot extremes follow the flow of the whole warm season**, not the
  lowest-flow days alone: only the every-day scenarios move them.
- **Version 4** changes the hottest days and the yearly peaks by at most
  0.03 °C in every scenario.

**A caution about low flows.** That low flows barely matter is a property of
the model's structure, learned from 2002–2009, not a measured fact. In
versions 7 and 8 the flow-linked share shrinks with the flow, so the model
cannot make a given relative change of flow matter more at low flow. The
mechanism that could (shallow, slow water heating and cooling faster) is a4,
and this record did not detect it (above). The model also has no direct
sunlight or water depth: it sees them only through air temperature and
discharge. So for a question about the lowest flows, such as a drought
abstraction limit, treat these small numbers as what this model can say, and
look for measurements at such flows, or a model that represents depth and
radiation, before relying on them.

## Limits of this approach

- **Parameter uncertainty only.** The narrow ranges show the uncertainty of the
  parameters *within this model*. They do not cover the chance that the
  model's response to discharge is wrong. The model learns that response from
  the natural changes in flow during the calibration years. Another model
  version can give a different answer, outside these ranges (above).
- **Stay within the calibrated flows.** If the scenario's discharge goes
  outside the range seen in calibration, the run reports every such day, with
  their number and the first date. The model is then extrapolating. Here the
  reduced flows stay inside the calibrated range.
- **What it does not mean.** "+0.05 °C in summer" is the effect of this
  abstraction according to this model, on the weather of 2010–2012. It is not
  a measured effect, and it does not include the chance that the model's
  response to discharge is wrong (above).
- The validation suite checks that paired differences are exact where the true
  answer is known ([V8](../../validation/REPORT.md#v8)).

## Next

Example [05](../05_gaps/README.md) deals with gaps in the data.
