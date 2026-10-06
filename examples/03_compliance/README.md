# 03 Compliance: was a temperature limit exceeded?

**Question:** suppose water temperature had not been measured in 2010–2012.
From air temperature and discharge alone, how likely is it that the 7-day mean
water temperature of the Mentue exceeded 20 °C in each year? And on how many
days did the daily mean exceed 18 °C?

The limits are illustrative. The measurements that do exist for 2010–2012 are
used only at the end, to check the answer.

## Why every simulation is needed

The daily 90% interval (example 02) says nothing directly about a 7-day mean, a
yearly peak, or a count of days. Those have to be computed within each
simulated series first, and the probability then taken across series. The
upper edge of the daily band is not the upper edge of the weekly peak.

## Steps

```bash
pyair2stream --config examples/03_compliance/calibrate.yaml    # as example 02, step 1
pyair2stream --config examples/03_compliance/predict.yaml      # 1000 simulations of 2010-2012
pyair2stream --config examples/03_compliance/check.yaml        # cross-validation of 2002-2009
python examples/03_compliance/run.py                           # runs all three, then the analysis
```

[`predict.yaml`](predict.yaml) sets `save_ensemble: true`, which keeps all 1,000
simulated series (`output/prediction/Forward_Prediction_Ensemble_Mentue_c_1d.npz`).
Each series has its own parameters and its own day-to-day model error.
`noise_model: "ar1"` makes those errors persist from day to day, as real ones
do; with `"iid"` the 7-day means would be far too certain
([V5](../../validation/REPORT.md#v5)).

[`check.yaml`](check.yaml) tests the same statistics on years the model was not
calibrated on: it hides each of 2003–2009 in turn, calibrates on the others and
records, for each hidden year, where the measured statistic fell among 1,000
simulations (`output/check/cv_yearly_statistics.csv`). It uses the same 18 °C
threshold as the question.

The analysis in [`run.py`](run.py) is a few lines:

```python
from pyair2stream import scenario
ens, dates = scenario.load_ensemble("output/prediction/Forward_Prediction_Ensemble_Mentue_c_1d.npz")
stats = scenario.year_statistics(ens, dates, threshold=18)      # each year's statistics, per simulation
peak = stats[2010]["highest 7-day mean"]                         # one value per simulation
check = pd.read_csv("output/check/cv_yearly_statistics.csv")
dev = check[check.statistic == "highest 7-day mean"].deviation   # measured minus predicted median
peak_c = scenario.correct_statistic(peak, dev)                   # corrected for the model's bias
p_exceeded = (peak_c > 20).mean()                                # share of simulations above the limit
```

## Results

**The check.** In the seven hidden years of 2003–2009, the measured yearly peaks
were lower than predicted: the highest 7-day mean by 0.65 °C on average (95%
interval 0.20 to 1.10 °C), the highest daily mean by 0.78 °C. The uncorrected
90% ranges held in only 5 (7-day mean) and 4 (daily mean) of the 7 years. The
number of days above 18 °C was not biased (+0.3 days, −5.7 to +6.3). On this
river the model, calibrated on the whole year, puts the summer peaks too high.

**The answer.**

| Year | P(7-day mean > 20 °C), corrected | uncorrected | Highest 7-day mean, 90% range, corrected | Measured | Days above 18 °C, corrected: median (90% range) | Measured |
|---|---|---|---|---|---|---|
| 2010 | 0.59 | 0.91 | 19.1 to 21.1 °C | 21.0 °C | 31 (23 to 39) | 30 |
| 2011 | 0.33 | 0.75 | 18.7 to 20.8 °C | 19.8 °C | 19 (12 to 27) | 21 |
| 2012 | 0.12 | 0.51 | 18.5 to 20.3 °C | 19.6 °C | 25 (16 to 34) | 22 |

![Highest 7-day mean in each simulation, corrected and not](figures/peak_7day_mean.png)

**Reading it.** The limit was exceeded in 2010 and not in 2011 or 2012. The
corrected probabilities (0.59, 0.33, 0.12) say so better than the uncorrected
ones (0.91, 0.75, 0.51), which would have called 2011 a likely exceedance: the
Brier score, the mean squared difference between probability and outcome, is
0.10 corrected against 0.28 uncorrected. All measured values lie inside the
corrected 90% ranges. 2010's measured peak, 21.0 °C, is near the top of its
range: the bias in 2010–2012 was smaller than in 2003–2009. Report such results
as probabilities with ranges, together with the check, not as a yes or no.

## Limits of this approach

- **Daily means only.** The model simulates daily mean temperature. It cannot
  assess limits on daily maximum temperature.
- **The correction rests on the check.** It assumes the model's average error
  in the statistic is the same in the years predicted as in the years held out.
  With 7 years its uncertainty is large, and the corrected ranges include it.
  Over 48 held-out years on three Swiss rivers, corrected 90% ranges held in
  85–94% of years, uncorrected ones in 73–92%
  ([V11](../../validation/REPORT.md#v11)).
- **The method matters.** With the alternative `likelihood: "exact"` the
  simulations run slightly cooler. The default keeps them centred on the best
  fit ([METHODS §12](../../docs/METHODS.md#12-parameter-and-prediction-uncertainty-de-mcmc)).
- **So does the time scale of the error.** The model's errors have a part that
  changes from day to day and a part that lasts for weeks. The default
  (`rho_timescale: "weekly"`) includes both (ρ = 0.86 here); `"daily"` matches
  only the day-to-day part (ρ = 0.70) and gives narrower, over-confident ranges
  for 7-day means and yearly peaks
  ([METHODS §12](../../docs/METHODS.md#12-parameter-and-prediction-uncertainty-de-mcmc)).
- **The answer depends on the definition.** Here a 7-day mean is a moving
  average over the current and previous six days, within the calendar year. Use
  your standard's own definition (moving or fixed weeks, calendar year or
  season), and the same one in the check.

## Next

Example [04](../04_scenario/README.md) compares two situations, for example
with and without a water abstraction.
