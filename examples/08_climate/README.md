# 08 Climate: what would a warmer climate do?

**Question:** if the air over the Mentue had been 2 °C warmer in 2010–2012, how
much warmer would the river have been? How often would a 20 °C limit on the
7-day mean have been exceeded? And what if summer flows had also been 20%
lower?

This example uses the simplest kind of climate scenario, a "delta change": the
measured air temperature plus 2 °C on every day. The same steps work for air
temperature and discharge from a climate model (see [Climate-model
data](#climate-model-data) below).

## Run it

```bash
python examples/08_climate/run.py
```

It takes about five minutes. It makes the two scenario files from the measured
2010–2012 file, then runs:

```bash
pyair2stream --config examples/08_climate/calibrate.yaml      # calibrate with uncertainty (as example 02)
pyair2stream --config examples/08_climate/baseline.yaml       # 1,000 simulations, as measured
pyair2stream --config examples/08_climate/warmer.yaml         # the same, with the air 2 °C warmer
pyair2stream --config examples/08_climate/warmer_drier.yaml   # 2 °C warmer, and 20% less flow in June-September
pyair2stream --config examples/08_climate/check.yaml          # cross-validation, to check the yearly peaks (as example 03)
```

Two settings in [`warmer.yaml`](warmer.yaml) and
[`warmer_drier.yaml`](warmer_drier.yaml) matter:

- `reuse_sample_indices_from` makes each scenario use the same 1,000
  parameter sets as the baseline. The change can then be computed simulation
  by simulation, as in example [04](../04_scenario/README.md).
- `calibration_metadata` keeps the calibration's mean discharge (`Qmedia`), so
  a reduced flow is seen as reduced.

The scenario files have no water temperature: it was not measured under these
conditions.

**The uncertainty methods, and why.** Two questions, two treatments:

- **How much warmer?** This is a change, so it is computed as a paired
  difference, as in example [04](../04_scenario/README.md): the DE-MCMC chain
  gives 1,000 parameter sets, every run uses the same ones with the same daily
  errors, and each scenario simulation is subtracted from its baseline twin.
  What the runs share cancels, and the range is the parameter uncertainty of
  the change.
- **Would the limit be exceeded?** This is about a yearly statistic, so it is
  computed in each simulated series and corrected for the model's bias in the
  yearly peaks, measured by cross-validation of 2002–2009, as in example
  [03](../03_compliance/README.md). The same correction is applied to every
  run, so it does not change the difference between them.

[docs/UNCERTAINTY.md](../../docs/UNCERTAINTY.md) explains both (§7 to §10).

## Step 1: is the scenario inside what the model has seen?

A scenario warmer than any calibration year asks the model to extrapolate. The
script compares them:

| | 2010–2012, measured | 2010–2012, +2 °C | Warmest of 2002–2009 |
|---|---|---|---|
| Mean summer (Jun–Aug) air temperature | 18.3 °C | 20.3 °C | 21.3 °C (2003) |

Only 1.0% of the scenario's days are warmer than the warmest day of
2002–2009 (25.5 °C). The 2003 heatwave summer lies in the calibration years,
so the scenario stays mostly inside the conditions the model was fitted to.
Check this for your own scenario. On the Swiss rivers, the model was
calibrated on the coolest third of the years. It predicted the warmest third
almost as well: at most 0.07 °C worse (validation
[V10](../../validation/REPORT.md#v10)).

## Step 2: how much warmer is the river?

The change, simulation by simulation (scenario minus baseline):

| Scenario | Change in | Median | 90% range |
|---|---|---|---|
| air +2 °C | summer (Jun–Aug) mean | +1.52 °C | +1.49 to +1.56 °C |
| air +2 °C | winter (Dec–Feb) mean | +1.05 °C | +0.99 to +1.10 °C |
| air +2 °C | whole-year mean | +1.36 °C | +1.32 to +1.40 °C |
| air +2 °C, 20% less summer flow | summer (Jun–Aug) mean | +1.58 °C | +1.54 to +1.63 °C |
| air +2 °C, 20% less summer flow | winter (Dec–Feb) mean | +1.05 °C | +0.99 to +1.10 °C |

![Monthly mean change in water temperature for both scenarios](figures/monthly_change.png)

*The monthly mean change: the median of the 1,000 paired simulations (dot) and
their 90% range (bar). The dashed line is the change in air temperature. Drawn
with `plots.change(diffs, dates, by="month", reference=2)`.*

**Reading it.**

- The river warms by less than the air: about 1.5 °C in summer and 1.0 °C in
  winter, for 2 °C warmer air.
- **Why less, and why less in winter?** In the model's equation, 1 °C warmer air
  raises the water's balance temperature by a2 / (a3 + a8·θ). Here θ is the
  flow relative to its mean. With this calibration, that is 0.78 °C at the
  median summer flow, and 0.63 °C at the mean flow. The Mentue's flow is higher
  in winter, so the response is smaller then. In cold spells the water also
  cannot fall below 0 °C in either run, so the change there is small.
- **Less summer flow adds little:** 0.06 °C in summer. With less water, the
  river follows the air a little more closely (as in example 04).
- The ranges are narrow because they show only the uncertainty of the
  parameters within this model (see [Limits](#limits-of-this-approach)).

## Step 3: the yearly peaks and the limit

The year's highest 7-day mean is checked and corrected by cross-validation, as
in example [03](../03_compliance/README.md). The same correction is applied to
all three runs.

| Year | P(7-day mean > 20 °C): as measured | air +2 °C | air +2 °C, less summer flow | Days above 18 °C: as measured | air +2 °C | air +2 °C, less summer flow |
|---|---|---|---|---|---|---|
| 2010 | 0.57 | 1.00 | 1.00 | 30 (22 to 38) | 50 (42 to 59) | 51 (43 to 60) |
| 2011 | 0.33 | 0.99 | 0.99 | 18 (11 to 27) | 41 (33 to 49) | 41 (33 to 50) |
| 2012 | 0.14 | 0.95 | 0.97 | 24 (15 to 33) | 56 (43 to 70) | 59 (45 to 72) |

*Days above 18 °C: median, with the 90% range in brackets.*

![The year's highest 7-day mean in 1,000 simulations, for each year and scenario](figures/yearly_peaks.png)

*Each year's highest 7-day mean: the median of the 1,000 simulations (dot) and
their 90% range (bar), corrected as in example 03. Grey: as measured; blue and
orange: the two scenarios. The number above each bar is the share of the
simulations above the 20 °C limit (dashed). Drawn with
`plots.yearly_statistic`.*

**Reading it.** The year's highest 7-day mean rises by 1.6 °C (90% range 1.56 to
1.65 °C). In 2010–2012, a 20 °C limit was exceeded in one year of three (P =
0.14 to 0.57). With 2 °C warmer air, it would be exceeded almost every year (P =
0.95 to 1.00). The number of days above 18 °C would roughly double.

## Climate-model data

To use air temperature and discharge from a climate model instead of a delta
change:

- run `FORWARD` directly on the climate model's daily series, with
  `calibration_metadata` from your calibration, as here;
- if the model uses a 365-day calendar, declare it with `calendar: "noleap"`.
  The file then needs real dates without 29 February: a file with 29 February
  rows is refused, unless `drop_29_february: true` removes them (with a
  warning). Convert a 360-day calendar to the standard one first (USER_GUIDE
  [§5](../../USER_GUIDE.md#5-preparing-your-own-data));
- climate models are biased. Adjust their air temperature and discharge to the
  measured ones first (bias correction). Or use them as a change applied to the
  measurements, as here;
- a scenario file needs at least 30 days, and a year or more is recommended
  (USER_GUIDE [§5](../../USER_GUIDE.md#5-preparing-your-own-data)).

## Limits of this approach

- **Parameter uncertainty only, for the change.** The narrow ranges show the
  uncertainty of the parameters *within this model*. They do not cover the
  chance that the river responds differently in a warmer climate: changes in
  shading, groundwater, snowmelt or the timing of flows.
- **A delta change keeps the measured weather.** Adding 2 °C to every day keeps
  the day-to-day pattern of 2010–2012. A warmer climate may also bring more or
  longer heatwaves. Climate-model series can include that.
- **The yearly peaks rest on the cross-validation check.** The correction
  assumes the model's error in the peaks is the same in a warmer climate.
- **Stay near the calibrated conditions.** Check how far the scenario goes
  beyond the calibration years (step 1). Discharge outside the calibrated range
  gives a warning.
- **What it does not mean.** "P = 0.95 to 1.00" is the probability under this
  delta change and this model, with the error the model had in 2002–2009. It
  is not a forecast of the future climate.

## Next

That completes the examples. For results that support a decision, work through
the checklist in USER_GUIDE [§14](../../USER_GUIDE.md#14-checklist-for-results-that-support-a-decision).
