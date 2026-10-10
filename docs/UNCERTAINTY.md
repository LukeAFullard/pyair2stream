# Uncertainty in pyair2stream: a guide for water quality scientists

pyair2stream can tell you more than "the water was 19.8 °C". It can tell you how
far that prediction could be from the truth, how likely it is that a temperature
limit was exceeded, and whether those statements held up when the model was
tested on years it had never seen. This guide explains each of these, without
assuming a statistics background: what it means, when to use it, how to run it,
how to read it, what it rests on, and how to defend it.

It complements [METHODS.md](METHODS.md) (the exact steps and formulas, §11–§13)
and the [User Guide](../USER_GUIDE.md) (settings, §11–§14). Every number quoted
here comes from the package's own examples and its
[validation report](../validation/REPORT.md). The figures are drawn by
[`figures/make_uncertainty_figures.py`](figures/make_uncertainty_figures.py) from
examples 03 and 04, so anyone can redraw them.

**Contents**

1. [Start here: which tool answers your question](#1-start-here-which-tool-answers-your-question)
2. [Where uncertainty comes from](#2-where-uncertainty-comes-from)
3. [Words you will meet](#3-words-you-will-meet)
4. [How pyair2stream builds its uncertainty, in seven steps](#4-how-pyair2stream-builds-its-uncertainty-in-seven-steps)
5. [The model's errors: their size and how long they last](#5-the-models-errors-their-size-and-how-long-they-last)
6. [Daily prediction intervals](#6-daily-prediction-intervals)
7. [Weekly means, yearly peaks, days above a limit](#7-weekly-means-yearly-peaks-days-above-a-limit)
8. [Testing on years the model has not seen: cross-validation](#8-testing-on-years-the-model-has-not-seen-cross-validation)
9. [Correcting yearly statistics for the model's bias](#9-correcting-yearly-statistics-for-the-models-bias)
10. [Comparing two scenarios](#10-comparing-two-scenarios)
11. [How well are the parameters known?](#11-how-well-are-the-parameters-known)
12. [Mean error by month and season](#12-mean-error-by-month-and-season)
13. [Choosing the level: 90%, 95% or 99%?](#13-choosing-the-level-90-95-or-99)
14. [Warnings and what to do about them](#14-warnings-and-what-to-do-about-them)
15. [Is it defensible?](#15-is-it-defensible)
16. [A worked example, from data to a statement](#16-a-worked-example-from-data-to-a-statement)
17. [Common mistakes](#17-common-mistakes)
18. [References](#18-references)

---

## 1. Start here: which tool answers your question

pyair2stream has two kinds of uncertainty tool, and they work together.

- **Tools that make ranges.** A `DE-MCMC` run calibrates the model, then
  measures how uncertain its parameters are and how large and how persistent
  its daily errors are. From these, 1,000 simulated series are made: by the
  `DE-MCMC` run itself for the calibration years, and by a `FORWARD` run for
  any other period or scenario. Every range and probability is read from these
  series.
- **Tools that check ranges.** Cross-validation is a separate run
  (`run_mode: "DE"` with `cross_validation: enabled: true`). It hides each
  year in turn, predicts it, and records whether ranges made the same way held
  in that year. It also gives the correction for yearly statistics
  (section [9](#9-correcting-yearly-statistics-for-the-models-bias)) and a
  second, independent set of parameter intervals (the jackknife).

A range you report comes from the first kind. The evidence that it can be
trusted at your site comes from the second.

| Your question | Method | Run mode or setting | Output file | How well it held in the validation | Section |
|---|---|---|---|---|---|
| How far could a daily prediction be from the truth? | daily prediction interval | calibration years: `DE-MCMC`; any other period: `FORWARD` with `enable_prediction_intervals: true` | `MCMC_envelopes_*.csv`; `Forward_Prediction_Envelopes_*.csv` | 90% bands: 90% of days in held-out years (V11); 85–90% in later years (V5) | [6](#6-daily-prediction-intervals) |
| What range for a weekly or monthly mean, or a run of warm days? | the statistic computed in each simulated series | `save_ensemble: true`, then `scenario.aggregate`, `scenario.exceedance` | `MCMC_ensemble_*.npz`; `Forward_Prediction_Ensemble_*.npz` | 90% ranges for 7-day means: 87–93% (V5) | [7](#7-weekly-means-yearly-peaks-days-above-a-limit) |
| How likely is it that a yearly limit was exceeded (highest daily mean, highest 7-day mean, days above a threshold)? | share of simulated series above the limit, **corrected** by cross-validation | `scenario.year_statistics`, then `scenario.correct_statistic` | the `.npz` above, and `cv_yearly_statistics.csv` | corrected 90% ranges: 85–94% of held-out years; uncorrected 73–92% (V11) | [7](#7-weekly-means-yearly-peaks-days-above-a-limit), [9](#9-correcting-yearly-statistics-for-the-models-bias) |
| What difference would a change in flow or climate make? | paired difference of two scenario runs | two `FORWARD` runs, the second with `reuse_sample_indices_from`; `scenario.paired_difference_from_files` | the two `Forward_Prediction_Ensemble_*.npz` | equal to the exact effect (V8) | [10](#10-comparing-two-scenarios) |
| My site's bands held too rarely. Can they be widened? | conformal margin, from the misses of held-out years | cross-validation, then `FORWARD` with `forward_options.conformal_margins`; `scenario.conformal_range` | `cv_conformal_margins.csv`; columns `*_conformal` of `Forward_Prediction_Envelopes_*.csv` | V19_TABLE | [6](#widening-a-band-that-held-too-rarely-optional) |
| Did the ranges hold at my site? | cross-validation check | `cross_validation: enabled: true`, `run_mode: "DE"` | `cv_interval_coverage.csv`, `cv_yearly_statistics_summary.csv` | this is the test | [8](#8-testing-on-years-the-model-has-not-seen-cross-validation) |
| How precisely do the data fix the model's parameters? | MCMC parameter intervals; jackknife intervals | `DE-MCMC`; cross-validation | `parameter_significance_*.csv`; rows `jackknife_*` of `cv_results.csv` | 90% intervals contained the true values 92–97% (MCMC) and 81–95% (jackknife) of the time (V4) | [11](#11-how-well-are-the-parameters-known) |
| Is the model too warm or too cool in some season? | mean error by month and season | every run with measurements; cross-validation | `bias_by_month_*.csv`; `cv_bias_by_month.csv` | | [12](#12-mean-error-by-month-and-season) |
| Should I report 90%, 95% or 99%? | the level of every range | `uncertainty_options.prediction_interval`, `parameter_interval` | | 99% daily bands held on only 98.0–98.2% of days (V11) | [13](#13-choosing-the-level-90-95-or-99) |
| Which parameter changes the simulation most? | sensitivity analysis (**not** an uncertainty range) | `sensitivity_analysis: true` | `sensitivity_*.csv`, `.png` | | [11](#11-how-well-are-the-parameters-known) |

The "V" numbers refer to the checks of the
[validation report](../validation/REPORT.md).

> **The three rules that matter most**
>
> 1. Compute every multi-day or yearly quantity **in each simulated series**,
>    then take the range across series. Never compute it from the daily band.
> 2. **Check** what you report on years the model was not calibrated on.
> 3. For a **yearly** statistic, use the **corrected** probability.

---

## 2. Where uncertainty comes from

A model is a simplification. Even with the best parameters, its daily water
temperature differs from the measured one: on the Mentue (a small Swiss river)
by 0.63 °C on a typical day. That difference has three sources:

1. **Parameters.** The 3–8 coefficients are fitted to a limited record. Other
   values would fit almost as well (section [11](#11-how-well-are-the-parameters-known)).
2. **The model's structure.** One equation driven by air temperature and flow
   cannot represent everything that sets a river's temperature: shading,
   groundwater, snowmelt, a reservoir release, a heatwave unlike any in the
   record.
3. **The data.** Measurements have errors, and daily means hide what happens
   within a day.

pyair2stream represents source 1 by many parameter sets, each consistent with the
data, and sources 2 and 3 together by a statistical description of the model's
past errors: the **error model** (section [5](#5-the-models-errors-their-size-and-how-long-they-last)).
Figure 1 shows that, for daily values, the error model is by far the larger part.

![A summer predicted from earlier years: the dark band (parameters only) is narrow; the light band (parameters plus daily error) is about five times wider and contains the measurements.](figures/U02_sources.png)

*Figure 1. Mentue, summer 2011, predicted from a calibration on 2002–2009. The
90% band from parameter uncertainty alone is 0.45 °C wide on average; with the
model's daily error it is 2.1 °C wide, and it contains the measurements (dots),
which the prediction did not use.*

**What the uncertainty does not include.** The inputs you give the model are
taken as exact: a projected air temperature or a naturalised flow series is not
made uncertain by pyair2stream. And the river is assumed to behave in the years
predicted as it did in the years calibrated on (no new dam, abstraction pattern
or loss of shading in between). If either matters, treat it separately, for
example by running several input scenarios.

---

## 3. Words you will meet

| Word | Meaning |
|---|---|
| **Error** (residual) | simulated minus measured water temperature on a day. Positive: the model is too warm. |
| **Bias** | the average error. A model can have a small error overall and still be biased in summer. |
| **σ** (sigma) | the typical size of the daily error (the root-mean-square error, in °C). |
| **Persistence**, **ρ** (rho) | how much an error carries over from one day to the next: 0 means each day's error is new, near 1 means errors last for weeks. |
| **Chain** | the thousands of parameter sets that `DE-MCMC` collects (`MCMC_chain_*.csv`), each consistent with the data. |
| **Likelihood** | the rule that scores how well a parameter set explains the measurements. It sets how wide the parameter ranges are (section [11](#11-how-well-are-the-parameters-known)). |
| **Ensemble** | the 1,000 simulated series a run produces, each with its own parameter set and its own daily errors. |
| **Interval**, **range**, **level** | a band that should contain the truth with a stated chance: a 90% interval should miss about 1 value in 10. |
| **Coverage** | the share of measured values that actually fell inside the interval. For a trustworthy 90% interval it is close to 90%. |
| **Probability of exceedance** | the share of simulated series in which a statistic (say the year's highest 7-day mean) is above the limit. |
| **Held-out year** | a year whose measurements were hidden from the calibration, so the model can be tested on it. |
| **Uncorrected**, **corrected** | for a yearly statistic (a peak, a count of warm days): *uncorrected* is the range or probability read straight from the simulated series; *corrected* is the same after every simulated value has been shifted by the model's average error in that statistic in the years cross-validation held out (measured minus predicted), with an allowance for the uncertainty of that average (section [9](#9-correcting-yearly-statistics-for-the-models-bias)). |
| **PIT** | where a measured value falls among the simulations: 0.3 means 30% of the simulations were below it (section [8](#8-testing-on-years-the-model-has-not-seen-cross-validation)). |
| **Jackknife** | a way to turn the spread of the parameters fitted in the cross-validation folds into an interval, allowing for the data the folds share (section [11](#11-how-well-are-the-parameters-known)). |

---

## 4. How pyair2stream builds its uncertainty, in seven steps

![Seven boxes: calibrate, parameter uncertainty, error model, 1000 simulated series, your statistic, check, correct and report.](figures/U01_pipeline.png)

*Figure 2. The seven steps from calibration to a reported probability.*

1. **Calibrate.** Differential Evolution finds the parameters that fit the
   measured water temperature best (`run_mode: "DE-MCMC"` starts with this).
   Use `objective_function` NSE or RMS: DE-MCMC refuses KGE, because its
   ranges are measured around the least-squares fit, which a KGE fit is not.
2. **Parameter uncertainty.** MCMC (Markov chain Monte Carlo) then collects
   thousands of parameter sets, each in proportion to how well it fits
   (section [11](#11-how-well-are-the-parameters-known)).
3. **Error model.** The size σ and persistence ρ of the best fit's daily errors
   are measured (section [5](#5-the-models-errors-their-size-and-how-long-they-last)).
4. **1,000 simulated series.** A `FORWARD` run draws 1,000 parameter sets,
   simulates each, and adds a random error series of size σ and persistence ρ to
   each one (the `DE-MCMC` run does the same for the calibration years). Like
   the simulation itself, each series is kept at or above `Tice_cover`. Each
   series is one plausible version of what the water temperature was (or will
   be).
5. **Your statistic.** Whatever your question is about, a daily value, a 7-day
   mean, a yearly peak or a count of days, is computed in each of the 1,000
   series. The spread across series is its uncertainty; the share above a limit
   is the probability that the limit was exceeded (section [7](#7-weekly-means-yearly-peaks-days-above-a-limit)).
6. **Check.** Cross-validation hides each year in turn and repeats steps 1, 3,
   4 and 5 without it (step 2 is left out: each fold has one parameter set, so
   its ranges are slightly narrower). It records whether the stated ranges
   contained what was measured
   (section [8](#8-testing-on-years-the-model-has-not-seen-cross-validation)).
7. **Correct and report.** For yearly statistics, the bias found in step 6 is
   removed (section [9](#9-correcting-yearly-statistics-for-the-models-bias)).
   Report the probability, its range and the check.

---

## 5. The model's errors: their size and how long they last

This section explains the **error model**, and in particular the setting
`rho_timescale: "weekly"`. The setting affects every range and probability
that covers more than a single day. It is also the one part of the method that
is pyair2stream's own. So it is explained here in full, with the evidence and
answers to the questions reviewers ask. The formulas are in
[METHODS §12](METHODS.md#error-persistence-how-it-is-estimated-and-why-the-weekly-scale).

### In short

- The model's daily errors last for a day or two, and they also drift for
  weeks.
- pyair2stream describes them with two numbers: their size σ and their
  persistence ρ.
- ρ can be measured from one day to the next (`"daily"`), or from one week to
  the next (`"weekly"`, the default).
- Both give the same range for a single day.
- The weekly ρ keeps the slow drift. Ranges for 7-day means, monthly means,
  yearly peaks and counts of warm days are then realistic. With the daily ρ
  they are too narrow.
- Where the errors have no slow drift, the two give the same ρ. Then the
  weekly choice costs nothing.
- Keep the default.

### What it is

The **error model** describes the best fit's daily errors (simulated minus
measured temperature) with two numbers:

- **σ** (sigma): their typical size, in °C (the root-mean-square error of the
  calibration);
- **ρ** (rho): their persistence, between 0 and 1. It says how much of today's
  error carries over to tomorrow.

pyair2stream uses the simplest model of persistence, called AR(1)
("autoregressive of order 1"): each day's error is ρ times yesterday's error,
plus something new. It also assumes that the errors are normally distributed,
with the same size all year.

### Why persistence matters

Real errors come in runs: when the model is too warm today, it is likely to be
too warm tomorrow (Figure 3). A random error that is new every day would look
very different.

![Three panels: the real errors of one year form runs; independent random errors of the same size do not; AR(1) errors do.](figures/U03_errors_persist.png)

*Figure 3. Top: the model's real errors on the Mentue in 2006. Middle: random
errors of the same size that are new every day. Bottom: AR(1) errors of the same
size with ρ = 0.86, as pyair2stream generates them.*

This matters because a compliance question is usually about several days: a
7-day mean, a run of warm days, a summer peak. Errors that are new every day
cancel out in an average. Errors that persist do not (Figure 5). If the error
model ignored persistence, the range for a 7-day mean would be far too narrow.
On the Swiss rivers, 90% ranges for 7-day means made that way contained only
39–59% of the measured values ([V5](../validation/REPORT.md#v5)).

### Real errors have two memories

Figure 4 shows how long the real errors last. They have two parts:

- **a fast part**, which fades within a day or two. This is the weather of a
  single day that the model misses.
- **a slow part**, which lasts weeks. This is, for example, a summer when the
  model runs slightly warm throughout, a dry spell, or a change in the river
  that the model does not represent.

An AR(1) process has only one memory: its correlation falls by the same factor,
ρ, every day. So one number cannot follow both parts. A choice must be made:
which part should ρ follow?

![The correlation of the real errors falls quickly over the first days, then slowly over weeks; the daily-rho curve follows the first days only; the weekly-rho curve lies above the real one for ten days and below it after.](figures/U04_autocorrelation.png)

*Figure 4. How long an error lasts on the Mentue: its correlation with the error
k days later. The "daily" ρ (orange) follows only the first few days, and says
that nothing lasts beyond a week or two. The default "weekly" ρ (blue) lies
between the two parts: above the real correlation over the first ten days,
below it after. No single ρ can follow both.*

What matters in practice is how errors behave when they are averaged over
several days (Figure 5).

![Typical error of an average over 1 to 90 days: real errors decline slowly; independent errors decline fast; the weekly-rho curve is slightly above the real one up to a month and close to it at one to two months; the daily-rho curve falls below it beyond a week.](figures/U05_averaging.png)

*Figure 5. The typical error of an average over 1 to 90 days, on the Mentue.
Errors that are new every day (dashed) average out quickly. Real errors
(black) do not. The default "weekly" ρ (blue) is slightly wider than the real
errors up to about a month (cautious), and close to them at one to two months.
The "daily" ρ (orange) is close at a week, but increasingly too narrow beyond
it.*

### Two ways to measure ρ

Figure 6 shows the same errors read in two ways.

![Top: eight weeks of daily errors, with the mean error of each 7-day window. Bottom: the daily estimate correlates consecutive days (0.70); the weekly estimate correlates consecutive 7-day means (0.52) and converts that to rho = 0.86; the default is the larger, 0.86, which sets the parameter ranges, the simulated errors and the cross-validation check.](figures/U15_rho_how.png)

*Figure 6. How the two estimates of ρ are made, on the Mentue (version 8).*

- **`"daily"`** correlates each day's error with the next day's. On the Mentue
  this gives ρ = 0.70. It sees mostly the fast part.
- **`"weekly"`** (the default) works in five steps:
  1. Take every 7-day window in which all days were measured, starting on any
     day of the week.
  2. Average the error over the window, and over the 7 days that follow it.
  3. Correlate the two averages, over all such pairs of windows. On the Mentue:
     0.52.
  4. Find the AR(1) whose 7-day averages would correlate that much (Figure 7,
     left). On the Mentue: ρ = 0.86.
  5. Use the larger of this value and the daily value.

The weekly ρ is still the persistence from one day to the next of an AR(1).
Only the way it is measured differs. It is chosen so that the simulated errors
carry over from one week to the next as much as the real errors did.

![Left: the curve that converts the correlation of consecutive 7-day means into rho, with the daily rho of 0.70 implying 0.25 and the measured 0.52 giving 0.86. Right: for six river and version cases, the measured correlation of consecutive 7-day mean errors (0.52 to 0.81) against what the daily rho implies (0.25 to 0.51).](figures/U16_rho_conversion.png)

*Figure 7. Left: the conversion. If the errors were AR(1) with the daily
ρ = 0.70, consecutive weeks would correlate only 0.25. They correlate 0.52.
Right: the same on all three Swiss rivers, versions 5 and 8. Consecutive
weeks' errors correlate 0.52–0.81, about twice what the daily ρ implies
(0.25–0.51). That gap is the slow part.*

### What changes, and what does not

**A single day: almost nothing.** σ is the same, so the range for one day is
almost the same. On the Swiss rivers, the test was years not used for
calibration. There, 90% daily ranges held on 84.8–90.0% of days with the weekly
ρ, and on 84.8–89.2% with the daily ρ ([V5](../validation/REPORT.md#v5)).

**Averages over a week or more: a lot.** Figure 8 shows simulated errors with
each ρ, made from the same random numbers. Day by day they look alike. Over a
month they do not: the daily ρ averages the errors away.

![Three panels over two years: real errors with their 30-day mean; AR(1) errors with the daily rho, whose 30-day mean stays close to zero; AR(1) errors with the weekly rho, whose 30-day mean drifts like the real one.](figures/U17_rho_series.png)

*Figure 8. Same daily size, different persistence (Mentue). The typical error
of a 30-day mean is 0.34 °C for the real errors, 0.26 °C with the daily ρ and
0.37 °C with the weekly ρ.*

Figure 9 and the table below show the same on all three rivers, for averages
from 1 to 90 days. They also show two other ways to choose a single ρ, for
comparison.

![For averages over 1 to 90 days, the typical error each single-rho choice gives, divided by the real one, on six river and version cases: month-to-month persistence too wide; weekly closest to 1; matching the size of a 7-day mean and the daily rho too narrow beyond two weeks.](figures/U19_rho_scales.png)

*Figure 9. The typical error of an average, as each choice of ρ gives it,
divided by the real one. 1 is right. Above 1 the ranges are wider than needed;
below 1 they are too narrow. Bands: the six river and version cases. Lines:
their median.*

| Averaging period | daily ρ | ρ matched to the size of a 7-day mean | weekly ρ (default) | ρ matched to month-to-month persistence |
|---|---|---|---|---|
| ρ itself | 0.70–0.86 | 0.73–0.88 | 0.86–0.95 | 0.94–0.96 |
| 1 day | 1.00 | 1.00 | 1.00 | 1.00 |
| 7 days | 0.94–0.98 | 1.00 | 1.09–1.17 | 1.09–1.26 |
| 14 days | 0.84–0.90 | 0.90–0.94 | 1.08–1.17 | 1.09–1.38 |
| 30 days | 0.70–0.77 | 0.75–0.81 | 1.02–1.11 | 1.07–1.46 |
| 60 days | 0.61–0.68 | 0.65–0.73 | 0.94–1.07 | 1.07–1.46 |
| 90 days | 0.58–0.70 | 0.62–0.76 | 0.91–1.15 | 1.16–1.42 |

*The typical error of an average, as each choice gives it, divided by the real
one: ranges over the three Swiss rivers, versions 5 and 8.*

Reading it:

- For a single 7-day mean, the daily ρ is about right in the calibration
  years. The weekly ρ makes it 9–17% too wide, which is cautious.
- Beyond two weeks, the daily ρ is too narrow, by about a third at 60–90 days.
  The weekly ρ stays within 17% of the real errors up to three months (0.91–1.17).
- Yearly peaks, counts of warm days and probabilities that a limit was exceeded
  depend on how errors hang together over a season. So they need the slow part
  too.

**The parameter ranges: wider.** ρ also sets how much information the
calibration contains. With persistent errors, n consecutive measured days count
as only n(1 − ρ)/(1 + ρ) independent ones (Figure 17, section 11). With the weekly ρ,
the Mentue's 2,907 days count as about 220 instead of about 520. So the
parameter ranges are about 1.5 times wider than with the daily ρ (1.5–1.8 times
on the three rivers).

**Where ρ is used:**

| Where | What ρ does there |
|---|---|
| DE-MCMC: the parameter ranges | sets the effective number of independent days in the default likelihood |
| DE-MCMC: the band for the calibration years | sets how long the random error added to each simulation lasts |
| `FORWARD` runs with intervals | the same, for each of the 1,000 series, using the σ and ρ stored with the chain |
| Cross-validation check (section 8) | the same, with each fold's own σ and ρ |
| `likelihood: "exact"` (not the default) | always uses the daily ρ, because that likelihood removes the correlation of consecutive days |

### Does it work? The evidence

![For seven tests, the share inside the 90% range with the weekly and the daily rho: synthetic AR(1) errors, the same; synthetic fast plus slow errors, the weekly rho holds and the daily rho does not; real rivers, daily values the same, 7-day means and yearly statistics better with the weekly rho.](figures/U18_rho_evidence.png)

*Figure 10. The validation suite's tests of the two choices: the share of
cases inside the stated 90% range. For parameter ranges, above 90% is cautious
and below is over-confident.*

- **Errors that really are AR(1)** (made-up data,
  [V4](../validation/REPORT.md#v4)). The weekly and daily estimates agree
  (ρ = 0.70 and 0.69), and every result is the same. The weekly choice costs
  nothing when there is no slow part.
- **Errors with a fast and a slow part, built like the real ones** (made-up
  data, V4 and [V9](../validation/REPORT.md#v9)). How often the 90% ranges
  contained the truth:
  - all parameter ranges: 99% with the weekly ρ, 89% with the daily ρ;
  - the range of the seasonal timing `a7`: 93% against 67%;
  - the ranges of yearly peaks and counts: 89–94% against 83–90%.
- **Real rivers, years not used for calibration** (V5, V9). Daily values: the
  same with both. 90% ranges for 7-day means held 87.2–92.9% of the time with
  the weekly ρ, against 80.8–88.5% with the daily ρ. Ranges for yearly peaks
  and counts held in 80–93% of river-years, against 60–87%.

### When to use which

| Situation | Setting |
|---|---|
| Any range or probability: intervals, limits, scenarios, cross-validation | `rho_timescale: "weekly"` (the default) |
| Reproducing results made with version 0.4.1 or earlier | `rho_timescale: "daily"` |
| Showing how much a conclusion depends on the choice | run both, and report both |
| You know ρ from another source (expert use) | `uncertainty_options.ar1_rho` in a `FORWARD` run |

The weekly estimate needs at least 140 pairs of complete 7-day windows a week
apart: about 21 complete weeks of daily measurements. With fewer, the daily
value is used, with a warning. A few years of daily measurements give
thousands of pairs.

The daily value in turn needs at least 30 pairs of consecutive measured days.
With fewer, for example with measurements only every other day or once a week,
ρ is set to 0 (no persistence), with the warning `Falling back to rho=0.0`;
`summary.md` then marks ρ as **not measured** in its uncertainty table, and
the `_meta.json` files record `rho_measured: false`.
Then the parameter ranges are too narrow, and so are the ranges for anything
longer than a day. Such data cannot show how long the errors last. If you
must use them, say so, and set `uncertainty_options.ar1_rho` in the `FORWARD`
run from a site with daily measurements (expert use; the parameter ranges of
the `DE-MCMC` run are not changed by it).

### How to set it and check it

```yaml
uncertainty_options:
  noise_model: "ar1"        # the default
  rho_timescale: "weekly"   # the default; or "daily"
```

- DE-MCMC records `rho`, `rho_timescale` and `rho_likelihood` in
  `MCMC_chain_*_meta.json`.
- Every `FORWARD` run uses the chain's error model and ρ, and prints them, for
  example `Using rho=0.8585 carried from calibration run ...`. A different
  value set in the FORWARD settings is used, with a note in `summary.md`.
- To check it at your site, run a cross-validation (section 8). In
  `cv_interval_coverage.csv`, the 7-day mean column should be close to the
  stated levels. In `cv_yearly_statistics_summary.csv`, the yearly statistics'
  ranges should hold about as often as stated.

### Questions and criticisms

**Is this a standard method?** The parts are standard. AR(1) error models are
widely used in hydrology (Sorooshian and Dracup, 1980; Schoups and Vrugt, 2010;
Evin et al., 2014). The effect of persistence on averages is a textbook result
(Bayley and Hammersley, 1946). Choosing the AR(1) to match how averages carry
over from one week to the next is pyair2stream's own approximation. It is not
published under this name. That is why it is explained in full here, tested
in the validation suite (V4, V5, V9), and why the older choice is still
available.

**Why not use a model with two memories, a fast one and a slow one?** That is
the better model, in principle. It is not implemented, for three reasons:

- it adds two more numbers to estimate, alongside up to 8 model parameters;
- a few years of data contain only a few dozen independent slow episodes, so
  the size and length of the slow part would be poorly known;
- the likelihood and the effective number of days would need new formulas,
  which would need their own validation.

The single weekly ρ meets the coverage targets in the tests. A model with two
memories remains a possible future improvement.

**Why a week, and not a day or a month?** Figure 9 answers this. Matching a
day misses the slow part. Matching a month gives ρ = 0.94–0.96, and ranges up
to 1.46 times too wide. A week is the shortest period over which the slow part
dominates. The week-to-week match stays within about 0.9 to 1.2 of the real
errors from 1 to 90 days. A few years of data also give thousands of
pairs of weeks, so the estimate is stable. And many temperature standards are
set on 7-day means.

**Why match how the errors carry over between weeks, not the size of a weekly
mean?** The size of one 7-day mean says nothing about whether the next week is
likely to be off in the same direction. Yearly peaks, counts of warm days and
monthly means depend on that. Matching the size of a 7-day mean (ρ =
0.73–0.88) gets a single 7-day mean right, but it is too narrow for longer
periods: 0.75–0.81 of the real error at 30 days (Figure 9).

**Doesn't it get the first few days wrong?** A little. It overstates how
strongly an error carries over to the next few days (Figure 4). This makes
ranges for averages over 3 to 14 days 5–17% wider than needed. That errs on
the side of caution. Single-day ranges are not affected.

**Doesn't it make the parameter ranges too wide?** For some parameters, yes.
One ρ sets the width of every parameter's range.

- Parameters whose effect changes slowly over the year (the constant `a1`, the
  seasonal size `a6` and timing `a7`) need the full allowance for slow errors.
  With the daily ρ, the range of `a7` was too narrow: in the test, it contained
  the truth 67% of the time instead of 90% (V4). With the weekly ρ it held
  (93%).
- Parameters whose effect changes from day to day (`a2`, `a3`) then get ranges
  two to three times wider than they need.

A range that is too wide errs on the side of caution. One that is too narrow
claims more than the data show. Rely on the predictions, which held their
stated levels, rather than on single parameter values (section 11).

**Why take the larger of the daily and weekly values?** Two reasons. Errors are
never less persistent than consecutive days show. And when the errors are only
weakly correlated, the weekly estimate alone is imprecise; the daily value then
sets a floor. For errors that really are AR(1), the two agree anyway (0.70 and
0.69 in V4).

**What if ρ comes out very high (0.99)?** Then the errors last for months. That
usually means a systematic error, such as a bias in one season. pyair2stream
warns when this happens. Look at the error by month (section 12), and try
another model version.

**Why not resample the real errors instead (a block bootstrap)?** Resampling
blocks of the real errors would keep both memories without a model (Künsch,
1989). But it needs a block length, which is the same kind of choice. Long
blocks leave few independent blocks in a few years of data. And it cannot
produce errors larger than those already seen. It is not implemented. It would
be a reasonable cross-check.

**Is ρ tuned to make the tests pass?** No. ρ is measured on the calibration
years only, with a rule fixed in advance. Every `FORWARD` run, and every
cross-validation fold, uses the σ and ρ measured on its own calibration data.
The years the tests check were never used to set ρ.

**Does it overlap with the correction of yearly statistics (section 9)?** No.
ρ sets the *spread* of the simulations. The correction moves their *centre* by
the model's average error in that statistic. They fix different things, and
both are needed (V11 used both).

**How much does my conclusion depend on it?** Run once more with
`rho_timescale: "daily"` and compare. For single days, the answer hardly
changes. For weekly or longer quantities, the daily setting gives narrower
ranges and more extreme probabilities. If your conclusion changes between the
two, say so.

### Limits

- **One ρ cannot follow two memories.** The weekly ρ is slightly too wide for
  averages over 3 to 14 days. Real errors that last for months (Figure 8, top)
  are only partly reproduced.
- **Heavier tails.** Real errors are occasionally larger than a normal
  distribution allows. This hardly matters up to 95% intervals, but matters at
  99% (section [13](#13-choosing-the-level-90-95-or-99)).
- **The same size all year.** Real errors differ by season: on the Swiss
  rivers, the monthly sizes were 0.7–1.5 times the yearly one, in no common
  pattern. The error model uses one size for the whole year. A size that
  changes with the month was tested by cross-validation and not adopted: for
  version 8 its 90% intervals held less often on the hottest days and for the
  yearly peak, with no better result overall
  ([METHODS §12](METHODS.md#12-parameter-and-prediction-uncertainty-de-mcmc)).
  Check the coverage in the season of your limit: validation V5 reports summer
  coverage separately, and V14 the hottest days.
- **The past errors are assumed to describe the future ones.** In years unlike
  the calibration years, errors are often somewhat larger (section
  [6](#6-daily-prediction-intervals)).

### What it rests on

Describing model errors as autocorrelated, and fitting them alongside the
model, is long established in hydrology (Sorooshian and Dracup, 1980; Schoups
and Vrugt, 2010; Evin et al., 2014). Errors at several time scales are known to
matter for aggregated quantities (McInerney et al., 2020). The variance of an
average of correlated values (Figures 5 and 9) is a textbook result (Bayley and
Hammersley, 1946). So is the effective number of independent values (Zwiers
and von Storch, 1995). Choosing ρ at the weekly scale is pyair2stream's own,
documented approximation.
[METHODS §12](METHODS.md#error-persistence-how-it-is-estimated-and-why-the-weekly-scale) gives
the formulas and the reasoning, and validation V4, V5 and V9 test it.

---

## 6. Daily prediction intervals

### What it is

A band around the simulated water temperature that should contain the measured
value on a stated share of days, for example 90%.

### How it is made

1. 1,000 parameter sets are drawn from the MCMC chain (`forward_options.n_samples`
   in a `FORWARD` run).
2. The model is run with each, and an AR(1) error series (σ, ρ) is added. Each
   series is then kept at or above `Tice_cover`, as the simulation itself is,
   so in winter the lower edge does not fall below the ice floor.
3. On each day, the band runs from the 5th to the 95th percentile of the 1,000
   values (for a 90% band; the level is `uncertainty_options.prediction_interval`).

Two runs make this band, for different periods:

| | `DE-MCMC` run | `FORWARD` run with intervals |
|---|---|---|
| Period | the calibration file only | any file: validation years, a scenario, a projection |
| σ | each parameter set's own daily root-mean-square error | the best fit's, from the chain's `_meta.json` |
| ρ, error model | measured on the calibration (`rho_timescale`) | taken from the chain's `_meta.json` |
| Band | `MCMC_envelopes_*.csv` | `Forward_Prediction_Envelopes_*.csv` |
| Every series (`save_ensemble: true`) | `MCMC_ensemble_*.npz` | `Forward_Prediction_Ensemble_*.npz` |
| Coverage and settings | `MCMC_chain_*_meta.json` | `Forward_Prediction_Ensemble_*_meta.json` |

A validation file in a `DE-MCMC` run gets no band: run `FORWARD` on it.

### When to use it

To show how uncertain the daily simulation is, to fill gaps in a record with an
honest range, or as a first look at a scenario. **Not** for anything spanning
several days (section [7](#7-weekly-means-yearly-peaks-days-above-a-limit)).

### How to run it

```yaml
# calibration
run_mode: "DE-MCMC"
uncertainty_options:
  prediction_interval: 90        # any level above 0 and below 100

# prediction, on the period of interest
run_mode: "FORWARD"
paths:
  calibration_metadata: "output/calibration/calibration_metadata.json"
forward_options:
  enable_prediction_intervals: true
  mcmc_chain_path: "output/calibration/MCMC_chain_<station>_<series>_1d.csv"
uncertainty_options:
  prediction_interval: 90        # the level of this run's band
  save_ensemble: true            # keep the 1,000 series (needed for section 7)
```

Outputs: `Forward_Prediction_Envelopes_*.csv` (columns `Twat_mod_lower`,
`Twat_mod_p50` and `Twat_mod_upper` for each day) and, with `save_ensemble`,
`Forward_Prediction_Ensemble_*.npz` (every series). The `FORWARD` run stops if
the chain was fitted with another model version, integrator, `Qmedia`,
`Tice_cover` or `min_theta_floor`: hence `paths.calibration_metadata`.

### How to read it

The run reports its **coverage**: the share of measured days inside the band
(in the console, `summary.md`, and `interval_coverage` in the `_meta.json`). On the calibration years it is close to
the stated level almost automatically, because σ and ρ were measured there.
What counts is the coverage on years the model
was **not** calibrated on: a separate validation file, or cross-validation
(`cv_interval_coverage.csv`, section [8](#8-testing-on-years-the-model-has-not-seen-cross-validation)).

### What the validation shows

| Share of days inside the band | 50% | 80% | 90% | 95% | 99% |
|---|---|---|---|---|---|
| Synthetic data, model exactly right (V4)* | 47–49% | 77–78% | 87–88% | 91.9–92.5% | 96% |
| Swiss rivers, 48 held-out years per version (V11) | 52–54% | 81–82% | 90% | 94.4–94.5% | 98.0–98.2% |
| Swiss rivers, later validation years (V5) | 42–52% | 75–80% | 85–90% | 91–95% | 95–99.5% |
| Swiss rivers, the hottest 10% of days by prediction, version 8 (V14) | | 80% | 91% | 95% | |
| The same, version 5 (no discharge term) (V14) | | 68% | 84% | 92% | |

\* About 3% of the synthetic values of V4 are below 0 °C. A band is kept at or
above the ice floor (step 2 above), so those values can never be inside it, and
every level comes out up to 3.2 points low. Real water does not go below freezing
(none of the 18,225 measured days of the Swiss rivers), and the real rivers show
no such shortfall.

Up to 95% the bands hold when averaged over many years. In years unlike those
calibrated on (the validation periods of V5 came after the calibration periods,
and the model's errors there were somewhat larger), they are somewhat narrower
than stated. At 99% they miss about twice as many days as stated. On the
hottest days, where limits are breached, version 8's bands held as stated;
version 5's did not, because it predicted its hottest days about 0.5 °C too
warm ([V14](../validation/REPORT.md#v14)).

### Widening a band that held too rarely (optional)

**What it is.** If the cross-validation shows that your site's bands held less
often than stated (for example, 90% bands held on 85% of hidden days), both
edges can be moved out by a **margin**: the amount that would have made them
hold in the hidden years. On 23 rivers in British Columbia, 90% bands held on
about 87% of days and 85% of 30-day means in later years
([V18](../validation/REPORT.md#v18)), because the model's errors in new years
were larger than in the calibration years.

**How it is made.** In the cross-validation (section
[8](#8-testing-on-years-the-model-has-not-seen-cross-validation)):

1. For every measured day of a hidden year, record how far the measurement fell
   outside its band (negative if inside).
2. The margin of a 90% band is the distance that 90% of these are below, each
   hidden year counting equally, however many days it has.
3. The widened band runs from the lower edge minus the margin to the upper edge
   plus the margin, never below `Tice_cover`. A negative margin (the band held
   too often) narrows it.

7-day and 30-day means get their own margins, made the same way from their own
misses.

**How to run it.** The cross-validation writes `cv_conformal_margins.csv`. In
the `FORWARD` run, set `forward_options.conformal_margins` to that file: the
band file then also has the widened band (`Twat_mod_lower_conformal`,
`Twat_mod_upper_conformal`). For 7-day or 30-day means, use
`scenario.conformal_range` with `scenario.conformal_margin(file, level,
window_days)` ([USER_GUIDE §12](../USER_GUIDE.md#12-scenario-runs-and-prediction-intervals)).
The run stops if the file was made with another model version or error model.

**How to read it.** The column `inside_after` gives how often the widened band
held in each hidden year when its margin came from the other years only: the
test of the margin at your site. Quote it with the result.

**What the validation shows** ([V19](../validation/REPORT.md#v19); 90% bands,
share inside without and with the margin):

V19_UNC_TABLE

**Limits.**

- It assumes the model's misses in the years predicted are like those in the
  hidden years. If the river or its climate changes, they may not be.
- The margin is the same in every season. Check the widened band in the season
  of your limit.
- It needs years: from fewer than about 5 hidden years the margin is uncertain
  (the file needs at least 3).
- It errs slightly wide: the margin comes from simulations without parameter
  uncertainty and is added to a band that has it.
- It is not for yearly statistics (use the correction of section
  [9](#9-correcting-yearly-statistics-for-the-models-bias)) or for the
  difference between two scenarios (section [10](#10-comparing-two-scenarios)).

**What it rests on.** This is split conformal prediction (Vovk et al., 2005; Lei
et al., 2018), in the form that adjusts an existing interval (conformalized
quantile regression; Romano et al., 2019). It makes no assumption about the
shape of the errors. Its guarantee assumes the hidden cases and the new one are
alike in distribution; daily values of one year are not, so here the units are
whole years. For errors that depend on each other but behave alike over time,
its coverage approaches the stated level as the record grows (Oliveira et al.,
2024); for a short record it is approximate, which is why its check at your site
(`inside_after`) and V19 matter.

### What it rests on

The band is a posterior predictive interval, the standard Bayesian way of
combining parameter uncertainty and random error into a prediction (Gelman et
al., 2013). Checking intervals by their coverage on independent data is standard
practice in hydrology (Laio and Tamea, 2007; Renard et al., 2010).

---

## 7. Weekly means, yearly peaks, days above a limit

### The rule: compute it in each series

The daily band says how uncertain each day is on its own. It does not say how
uncertain a 7-day mean, a yearly peak or a count of days is, because those depend
on how the errors of successive days hang together. The only correct way is to
compute the statistic in each of the 1,000 series, then take the range across
series (Figure 11).

![Left: three simulated series wander inside the daily band. Right: the number of days above 18 °C in 2011, counted in each series, has a 90% range of 13 to 26 days; counting the days on which the band's lower or upper edge is above 18 °C would give 9 and 33.](figures/U06_band_vs_statistic.png)

*Figure 11. Mentue, 2011. Counted series by series, the number of days above 18 °C
has a 90% range of 13 to 26 days, and 21 were measured. Counting the days on
which the band's lower or upper edge is above 18 °C gives 9 and 33 days: a range
almost twice as wide, and with no stated chance at all.*

### How to do it

```python
from pyair2stream import scenario
ens, dates = scenario.load_ensemble("output/prediction/Forward_Prediction_Ensemble_<...>.npz")

# the three yearly statistics, in every series
stats = scenario.year_statistics(ens, dates, threshold=18.0)
peak = stats[2011]["highest 7-day mean"]          # 1,000 values, one per series
low, high = scenario.central_range(peak, 90)      # its 90% range
p_exceeded = (peak > 20.0).mean()                 # probability that a 20 °C limit was exceeded

# other quantities
weekly = scenario.aggregate(ens, dates, how="mean", freq="7D")      # fixed 7-day blocks; a partial one gets NaN
runs = scenario.exceedance(ens, 18.0, consecutive_days=3)           # days in runs of 3+ days above 18 °C
```

`year_statistics` gives, for each year: the highest daily mean, the highest 7-day
moving mean (the day and the six before it) and the number of days above a
threshold. A year the dates cover only in part (at either end of the file)
is left out, with a warning (`partial_years="keep"` includes it). If your
standard defines a statistic differently (fixed weeks, a season, a 30-day
mean), compute it the same way: once per series.

`aggregate` gives no value (NaN) for a week or month that the dates cover only
in part, or that has days without a simulated value, with a warning: a 3-day
mean is not a 7-day mean. `min_days=N` accepts a period with at least N days;
`return_periods=True` also returns the first day of each period. `exceedance`
counts a day without a value as not above the threshold, with a warning.

In a file shorter than a year, and in each segment in gap-tolerant mode, the
first `warmup_drop_days` days (default 15) are the model settling from its
starting point. They are marked `warm_up = 1` in the run's output series, but
not in the `.npz`. Leave them out of your statistic.

### The probability that a limit was exceeded

It is the share of series in which the statistic is above the limit. Three
points to know:

- **It is a probability, not a verdict.** 0.6 means the model cannot tell
  whether the limit was exceeded, and says so honestly. Report it as such.
- **It has a small sampling error.** With 1,000 series it is precise to about
  ±0.03 (95%), because it is a share counted from a finite number of series.
  If the answer is close to a decision threshold, run more series
  (`forward_options.n_samples`; at most the number of parameter sets in the
  chain).
- **It does not depend on the interval level you choose** (section [13](#13-choosing-the-level-90-95-or-99)):
  only the reported ranges do.

For a **yearly** statistic, use the corrected probability (section [9](#9-correcting-yearly-statistics-for-the-models-bias)).

### What it rests on

Estimating probabilities and ranges from an ensemble of simulations is standard
in forecasting (Wilks, 2011). That the percentile of a sum is not the sum of
percentiles, and therefore that a statistic must be computed before taking
percentiles, is a basic property of probability.

---

## 8. Testing on years the model has not seen: cross-validation

### What it is

The model is calibrated with one year hidden, then asked to predict that year.
This is repeated for every whole year in turn, the first one included (the
defaults since version 0.5.1). A first or last year that the record covers
only in part is used for calibration but not held out: a few months are not a
test of a year. Each prediction is honest, because the year's measurements
played no part in it.

Each held-out year gets one calibration (one best-fit parameter set, found by
DE, PSO or LATHYP), not a full MCMC. Its error size σ and persistence ρ are
measured on the years used for that calibration only, exactly as `DE-MCMC`
measures them, and 1,000 simulated series of the held-out year are made from
them, as a `FORWARD` run would make them.

### Why

A model always fits the years it was calibrated on. The question that matters
for a decision is how well it predicts other years, and whether its stated
ranges hold there. Cross-validation answers that with your own data, at your own
site.

### What is hidden, and what is not

Only the year's measured **water temperatures** are hidden. Its air temperature
and discharge are kept, and the year is predicted from them, just as any real
prediction is made from the inputs of the period predicted. That is how
cross-validation is defined: hide what is predicted, keep what it is predicted
from. The model never uses measured water temperature, so hiding it removes
everything the year could tell the model about the answer.

Validation [V12](../validation/REPORT.md#v12) checked this on 96 held-out years
of three Swiss rivers. Hiding the year's air temperature and discharge from the
calibration as well, or leaving two months unused on each side of the year,
changed no year's error by more than 0.01 °C, and did not change how often the
ranges held.

### How to run it

```yaml
run_mode: "DE"
Qmedia: 1.4915                 # your calibration's mean discharge (here the Mentue's): every fold scales flow alike
cross_validation:
  enabled: true
  threshold: 18                # °C: the threshold of your question, for "days above threshold"
                               #   (default: the 90th percentile of the held-out measurements)
  season_months: [6, 7, 8, 9]  # a year counts if 80% of these months was measured (default: the 4 warmest)
  # unit: "n_years"            # hold out blocks of n_years_per_fold years instead of single years
  # water_year_start_month: 10 # years from October to September (default 1: calendar years)
  # skip_first_year: false     # the default; true never holds out the first year
  # min_train_years: 0         # the default; N never holds out N further first years
  # min_valid_obs: 10          # the default; a year with fewer measured days is not held out
uncertainty_options:
  prediction_interval: 90      # the level you intend to report
  # noise_model and rho_timescale: as for DE-MCMC (keep the defaults)
```

Cross-validation runs only with `run_mode` `DE`, `PSO` or `LATHYP`; with
`DE-MCMC` or `FORWARD` the block is ignored, with a warning. A
cross-validation run writes only the files below, not a calibration: run the
`DE-MCMC` calibration separately. Excluding first years with
`skip_first_year` or `min_train_years` removes test years only; every fold is
still calibrated on all the years it does not hold out.

### What it writes, and how to read each

| File | What it tells you |
|---|---|
| `cv_results.csv` | one row per held-out year: its error (RMSE, NSE, KGE) and the parameters fitted without it. Then rows `mean`, `std` (spread between folds; **not** an uncertainty), `pooled` (the scores over all held-out days together) and `jackknife_se`, `jackknife_90_lower`, `jackknife_90_upper` (parameter intervals; the number follows `parameter_interval`; section [11](#11-how-well-are-the-parameters-known)) |
| `cv_bias_by_month.csv` / `.png` | the mean error by month and season in the held-out years (section [12](#12-mean-error-by-month-and-season)) |
| `cv_interval_coverage.csv` | the share of held-out days (`daily inside`), and of 7-day means (`7-day inside`), inside the interval at 50%, 80%, 90%, 95% and your level. Close to the level: the daily ranges hold. Days are not independent, so no range expected by chance is given |
| `cv_yearly_statistics.csv` | for each held-out year and yearly statistic: the measured value, the predicted `median` and range (`lower`, `upper`), the `pit`, whether it was `inside`, and the `deviation` (measured minus predicted median), which the correction of section [9](#9-correcting-yearly-statistics-for-the-models-bias) uses |
| `cv_yearly_statistics_summary.csv` | for each statistic: how often the ranges held at each level (`share_inside_90` and so on), the range expected by chance (`expected_inside_90_low`, `_high`), and the `mean_deviation` with its 95% interval |

### The PIT: where the measured value falls

For each held-out year, the check makes 1,000 simulations of the year's
statistic and records the share below the measured value: the **PIT**
(probability integral transform). If the predictions are right, the measured
value is equally likely to fall anywhere among the simulations, so over many
years the PITs spread evenly between 0 and 1. Their pattern shows what is wrong
when they do not (Figure 12). The share of years with a PIT between 0.05 and 0.95
is the coverage of the 90% range.

![Left: one year's simulations with the measured value at the 31st percentile. Right: four PIT histograms: flat (right), U-shaped (too narrow), humped (too wide), sloping (biased).](figures/U08_pit.png)

*Figure 12. Left: one held-out year; 31% of the simulations lie below the measured
value, so its PIT is 0.31. Right: what many years' PITs look like when the
ranges are right, too narrow, too wide, or biased (illustration).*

### "Expected by chance": why 8 out of 10 is not a failure

Even a perfect 90% range does not contain exactly 90% of a handful of years,
just as ten coin tosses rarely give exactly five heads. The summary therefore
gives the range of shares that chance alone produces: with 10 years, anything
from 7 to 10 inside a 90% range is consistent with a correct range (Figure 13).
A share outside that range is evidence that the ranges are wrong; a share inside
it is no proof that they are right, only that the years available cannot show
otherwise. With fewer than about 5 held-out years the check says little.

![Two bar charts of the chance of k years inside a correct 90% range: for 10 years, 7 to 10 are expected; for 48 years, 39 to 47.](figures/U09_chance.png)

*Figure 13. How many years a correct 90% range contains by chance: 7–10 of 10
years, 39–47 of 48 (central 95% of outcomes).*

### The mean deviation: is the model biased in that statistic?

The deviation is the measured value minus the predicted median. Its mean over
the held-out years is the model's average error in the statistic, given with a
95% confidence interval. An interval that excludes zero means the model is
consistently too warm (negative) or too cool (positive) in that statistic. That
is what section [9](#9-correcting-yearly-statistics-for-the-models-bias)
corrects.

### Limits

- The check's simulations use each fold's best-fit parameters, not a full MCMC,
  so their ranges are slightly narrower than a `FORWARD` run's (parameters add
  little to the width; Figure 1). The check therefore errs towards finding
  ranges too narrow, not too wide.
- Years are treated as independent. A long drift (a river that slowly changes)
  breaks that.
- The inputs are measured ones. Cross-validation shows how well the model does
  with correct air temperature and discharge; a prediction from projected or
  borrowed inputs (a climate model, another station, a flow scenario) carries
  their errors on top.
- Each year is predicted from a calibration on the other years, later ones
  included. For predicting the future, a test that uses only earlier years is
  stricter: in V12 it was as accurate, but its 90% intervals held on 87% of days
  instead of 90%. For predictions of future years, quote the later-years tests
  ([V5](../validation/REPORT.md#v5), [V10](../validation/REPORT.md#v10)) as well.

### What it rests on

Cross-validation is a standard way to estimate how well a model predicts new
data (Stone, 1974), and testing a hydrological model on periods it was not
calibrated on is a long-standing requirement in hydrology (Klemeš, 1986).
Hiding what is predicted and keeping what it is predicted from is how both are
done (Hastie et al., 2009; Coron et al., 2012); using the inputs of the period
predicted is not "leakage", which is information a real prediction would not
have (Kaufman et al., 2012). The
PIT and its histogram are the standard check of probabilistic predictions in
weather forecasting and hydrology (Dawid, 1984; Hamill, 2001; Gneiting et al.,
2007; Laio and Tamea, 2007).

---

## 9. Correcting yearly statistics for the model's bias

### Why it is needed

The model's error on the hottest days of the year is not always its typical
error. Calibrated on the whole year, it can put the summer peak systematically
too high at one river and too low at another (Figure 14). Random error of the
typical size cannot fix that: the predicted range is centred in the wrong place,
so it misses the measured peak more often than it states. On the Swiss rivers,
uncorrected 90% ranges for yearly statistics held in only 73–92% of 48 held-out
years per version ([V11](../validation/REPORT.md#v11)).

![Mean deviation of the yearly peak, with 95% intervals, for three rivers and two model versions; several intervals exclude zero, in both directions.](figures/U10_bias_at_peaks.png)

*Figure 14. The model's average error in the yearly peak, in years held out by
cross-validation. Orange: a bias whose 95% interval excludes zero. Version 8
puts the Mentue's peak 0.6–0.8 °C too high; version 5 puts the Rhône's too high
and the Dischmabach's too low.*

### How it works

1. **The check** (section [8](#8-testing-on-years-the-model-has-not-seen-cross-validation))
   gives the deviation in each held-out year: how far the measured statistic
   was from the predicted median.
2. **The correction** shifts each simulated value of the statistic by the mean
   of those deviations. Because that mean is itself uncertain (it comes from a
   few years), each simulation is shifted by a slightly different amount: the
   mean plus a random draw of its uncertainty (its standard error times a
   Student t variable, the standard allowance for a mean estimated from few
   values). With few years the corrected range is therefore wider.

![Left: deviations of eight held-out years, mean -0.65 °C with its 95% interval. Right: the 2011 highest 7-day mean before and after correction; the probability of exceeding 20 °C falls from 0.77 to 0.35; the measured value was 19.8 °C.](figures/U11_correction.png)

*Figure 15. Example 03. Left: in the eight held-out years of 2002–2009 the
measured highest 7-day mean was on average 0.65 °C below the prediction (95%
interval 0.26–1.05 °C). Right: corrected, the probability that 2011 exceeded a
20 °C limit falls from 0.77 to 0.35; the measured value was 19.8 °C, below the
limit.*

### When to use it

For every **yearly** statistic (a peak, a count of days above a threshold) when
cross-validation is available with at least 5 held-out years (`correct_statistic`
refuses fewer than 3), and **always**, not
only when the check finds a bias. Deciding afterwards would let the result steer
the method. Report the corrected probability and range, and give the
uncorrected ones and the check alongside for transparency.

### How to run it

```python
import pandas as pd
from pyair2stream import scenario

check = pd.read_csv("output/check/cv_yearly_statistics.csv")
dev = check[check.statistic == "highest 7-day mean"].deviation         # the same statistic as the question
peak = scenario.year_statistics(ens, dates, threshold=18.0)[2011]["highest 7-day mean"]
peak_corrected = scenario.correct_statistic(peak, dev, seed=1)
p_exceeded = (peak_corrected > 20.0).mean()
```

The statistic, threshold and season must be defined the same way in the check
and in the question. The check keeps a year only if 80% of its season was
measured; set `season_months` to the season of your limit.

### What the validation shows

| Held-out years inside the range (V11, 48 per version) | 80% | 90% | 95% | 99% |
|---|---|---|---|---|
| uncorrected | 56–85% | 73–92% | 85–94% | 92–98% |
| corrected | 77–85% | 85–94% | 92–100% | 100% |

After the correction every statistic was within the range expected by chance at
80%, 90% and 95%, and the measured values sat on average at the middle of the
simulations (mean PIT 0.48–0.50, against 0.31–0.52 before). Applied to genuinely
later years (V9 part C), the corrected probabilities were closer to what
happened for 5 of 6 combinations of model version and statistic. The highest
**daily** mean remains the hardest statistic: its corrected 90% ranges held in
85% of held-out years, and in 11 of 15 later years for version 8.

### Limits

- It assumes the model's average bias in the statistic is the same in the years
  predicted as in the years held out.
- It corrects the centre of the range, not its shape.
- It applies to yearly statistics. Ranges for daily values and 7-day means are
  much less affected (sections [6](#6-daily-prediction-intervals) and [13](#13-choosing-the-level-90-95-or-99)).

### What it rests on

Correcting a model's systematic error with statistics of its past errors is
standard practice in weather forecasting, where it is called model output
statistics (Glahn and Lowry, 1972) or, for ensembles, ensemble model output
statistics (Gneiting et al., 2005). pyair2stream's version is the simplest form,
a shift, estimated only from years other than the one predicted, with its
uncertainty carried through.

---

## 10. Comparing two scenarios

### What it is

The change a different flow or climate would make, for example "how much warmer
would the river have been with 30% of its flow abstracted?", with an uncertainty
band for the **change itself**.

### Why pairing matters

Run alone, each scenario's range is wide, and the two overlap almost entirely
(Figure 16, left). But most of that width is shared: a parameter set that makes
one scenario warm makes the other warm too. Pairing the two runs, so that series
number k uses the same parameter set and the same daily error in both, cancels
everything they share. What remains is the uncertainty of the difference.

![Left: the June-August mean under both scenarios, two overlapping wide histograms. Right: the paired difference is +0.00 to +0.05 °C; an unpaired difference would be -0.52 to +0.59 °C.](figures/U12_paired.png)

*Figure 16. Example 04. Left: each scenario's June–August 2011 mean, with a
90% range 0.8 °C wide. Right: the change, paired, is 0.00 to +0.05 °C (90% range);
subtracting unpaired series would give −0.52 to +0.59 °C and suggest that the
change could go either way.*

### How to run it

1. Run scenario A with `save_ensemble: true`.
2. Run scenario B from the same chain with `save_ensemble: true` and
   `forward_options.reuse_sample_indices_from:` set to scenario A's
   `Forward_Prediction_Ensemble_*_meta.json`.
3. `diff = scenario.paired_difference_from_files(path_b, path_a)` gives B minus
   A, one row per series. It refuses two runs that did not use the same chain
   and parameter sets (including those left out as divergent), the same error
   model, σ and ρ, the same dates, or that simulated different days.

Both runs must use the calibration's `Qmedia` (`paths.calibration_metadata`):
recomputing it from the scenario's flows would cancel the flow change.

### Limits

The band shows **parameter uncertainty only**. Pairing assumes the model's error
on a given day would have been the same under both scenarios. On a day when a
series is held at `Tice_cover` in one scenario and not in the other (water near
freezing), the error does not cancel exactly. Days whose flow lies outside the
calibrated flows are an extrapolation: the `FORWARD` run reports every such
day, with their number, the first date and the number of zero-flow days.

### What it rests on

Using the same random numbers for the runs being compared ("common random
numbers") is a standard technique in simulation for estimating differences
precisely (Law, 2015). The validation checks that the paired difference equals
the exact effect where it is known ([V8](../validation/REPORT.md#v8)).

---

## 11. How well are the parameters known?

### MCMC in plain words

MCMC is a guided random walk through the possible parameter values. It spends
time in each region in proportion to how plausible the parameters there are,
given the data and the bounds, so the places it visits most are the most
plausible parameters, and the spread of the places it visits is their
uncertainty. pyair2stream
runs 32 walkers at once, each proposing moves along the difference between two
others (the differential-evolution move; ter Braak, 2006), which handles
parameters that trade off against each other.

What goes into it:

- **The prior**: before seeing the data, every value inside `parameter_bounds`
  is equally plausible, and values outside are impossible. Results can depend
  on the bounds; a parameter piled up against a bound means the bound is too
  narrow.
- **The likelihood**: how well a parameter set explains the data. pyair2stream's
  default is the least-squares fit (as the original authors calibrated), with
  the number of days replaced by the **effective number of independent days**.
  Persistent errors carry less information than independent ones (Figure 17):
  on the Mentue, 2,907 measured days count as 221 independent ones. Without
  that allowance the parameter intervals would be far too narrow. Measurements
  further apart are less related, and the count allows for the actual spacing
  of the scored days (errors d days apart are taken to correlate ρ^d): with one
  measurement a week, at the same ρ = 0.86, about half of them would count as
  independent. (But ρ itself can only be measured from consecutive days:
  section [5](#5-the-models-errors-their-size-and-how-long-they-last).) Every
  scored day, week or month enters the likelihood. How ρ is chosen, and what
  the choice does to these intervals, is in section
  [5](#5-the-models-errors-their-size-and-how-long-they-last). The other
  likelihoods are described below.

![Independent days per day measured fall from 1 at rho 0 towards 0 as rho approaches 1; at rho 0.86, 2907 days count as 221.](figures/U14_effective_n.png)

*Figure 17. The effective sample size: with errors that persist (ρ = 0.86), 2,907
days carry about as much information as 221 independent ones.*

### Three ways to score a parameter set

Two settings choose the likelihood. Keep the defaults.

| Setting | What it assumes | Evidence | Use it |
|---|---|---|---|
| `noise_model: "ar1"`, `likelihood: "least_squares"` (the defaults) | errors persist (ρ); least squares with the effective number of independent days | 90% parameter intervals held the truth 92–97% of the time (V4); the band stays centred on the best fit (V5) | always |
| `noise_model: "ar1"`, `likelihood: "exact"` | errors persist; the exact AR(1) likelihood, which weighs day-to-day changes of the error more than its level, and always uses the daily ρ | version 8's 90% parameter intervals held the truth only 75% of the time (V4); on real rivers it moved the band by up to 0.11 °C, cooler in that case (V5). With weekly or monthly scoring it treats the scores as independent, with a warning | only to compare |
| `noise_model: "iid"` | every day's error is new | parameter intervals far too narrow with persistent errors (67%, V4); 90% ranges for 7-day means held only 39–59% of the time (V5) | not for decisions |

The error model chosen here is recorded in the chain's `_meta.json`, and
`FORWARD` runs use it too (section [6](#6-daily-prediction-intervals)).

With weekly or monthly scoring (`time_resolution` `"1w"` or `"1m"`), the
default likelihood counts the block means and allows for how they persist from
block to block. Keep `a2` and `a3` at or above 0 in `parameter_bounds` then: a
weekly mean can hide a simulation that zigzags from day to day, and the run
warns when it finds one (V4, cases J and K).

### Has it converged?

The walk must run long enough to forget its starting point and to visit the
plausible region thoroughly. pyair2stream checks two standard diagnostics: the
run must be at least 50 times longer than the **autocorrelation time** (how many
steps it takes the walk to produce a new, independent sample), and **split-R̂**
(whether different walkers and different halves of the run agree) must be below
1.01. The walk runs in blocks of 1,000 steps (at least 2,000) until both hold,
or until `optimization.mcmc_steps` (default 20,000) is reached. The first part
of the walk (the larger of 30% of the steps and 5 autocorrelation times) is
discarded as burn-in. The diagnostics are in `MCMC_chain_*_meta.json`
(`converged`, `max_autocorr_time`, `max_split_rhat`). If the walk has not
converged, the run stops with an error by default (`strict_convergence: true`)
and gives no prediction interval, because unconverged results are not valid.
Try more steps (`mcmc_steps`), or a simpler model version: a posterior that
will not settle usually means the data cannot pin down all the parameters.

### Parameter intervals

`parameter_significance_*.csv` gives each parameter's mean, standard deviation
and central interval at `uncertainty_options.parameter_interval` (default 90%),
and whether zero lies outside its central 95% (`Significantly_Diff_From_Zero`:
zero is outside the central 95% credible interval, the Bayesian counterpart of
a 5%-level test, whatever the interval's level; meaningful only
for parameters where zero means "no effect": `a2`, `a4`, `a5`, `a6`, `a8`).
`parameter_correlation_*.png` shows how the parameters move together, and
`MCMC_chain_*.csv` holds every parameter set collected.

![Left: a2 and a3 from the chain lie along a narrow diagonal (correlation 0.99). Right: the histogram of a7 with its 90% interval 0.590 to 0.613.](figures/U13_parameters.png)

*Figure 18. Example 03. Left: `a2` and `a3` trade off; many combinations along
the diagonal fit almost equally well. Right: the seasonal timing `a7` is fixed
precisely (90% interval 0.590–0.613).*

Read parameter intervals with care:

- **Parameters trade off.** Each interval is for one parameter on its own. Do not
  combine the ends of several intervals (a high `a2` with a low `a3`): such
  combinations do not fit the data. Rely on predictions, which keep the
  trade-offs, rather than on individual parameter values. That many parameter
  sets can fit equally well is a general property of environmental models
  ("equifinality"; Beven, 2006).
- **One allowance for persistence widens every parameter equally.** It is about
  right for the slowly acting parameters (`a1`, `a6`, `a7`) and makes the
  intervals of the fast-acting ones (`a2`, `a3`) two to three times wider than
  necessary: cautious, not optimistic ([METHODS §12](METHODS.md#12-parameter-and-prediction-uncertainty-de-mcmc)).

### Jackknife intervals from cross-validation

Cross-validation fits the parameters once per held-out year. `cv_results.csv`
turns their spread into approximate intervals (the delete-one-year jackknife):
the rows `jackknife_se` (standard error) and `jackknife_90_lower`,
`jackknife_90_upper` (the number follows `parameter_interval`). The interval is
the mean of the folds ± a Student t value times the standard error. The
standard error is about √(n − 1) times the spread between folds, for n years,
because each fold shares all but one year with the others. Only whole years count as blocks.

Do **not** use the `std` row as an uncertainty. It is only the spread between
folds, which share most of their data, so it is far too small: in V4 the
fold mean ± 1.645 × `std` contained the true values only 39–60% of the time.

The jackknife intervals depend on the optimizer too. Where parameters trade
off, one fold can end on a distant parameter set with almost the same fit, and
widen the interval. In [V12](../validation/REPORT.md#v12), another optimizer
seed alone changed the jackknife standard errors by a factor of 0.65 to 1.41
(median over the parameters, for each river and version). For poorly
determined parameters, treat them as indicative and repeat the
cross-validation with a second `random_seed`.

**MCMC or jackknife?** They answer the same question from different
calculations: the MCMC intervals from the likelihood of one calibration, the
jackknife from how much the best fit moves when a year is left out. Report the
MCMC intervals (from the converged chain) and use the jackknife as a
cross-check. Where they disagree strongly, the parameter is poorly determined;
rely on predictions instead.

### Sensitivity analysis is not an uncertainty

`sensitivity_analysis: true` moves each parameter up and down, one at a time,
around the best fit, and reports the mean change in simulated water
temperature (`sensitivity_*.csv` and `.png`). It shows which parameters the
simulation responds to most, near the best fit. It says nothing about how well
the data fix the parameters, and gives no range or probability.

### What the validation shows

On synthetic data with known parameters, the default MCMC intervals (90%)
contained the true values 92–97% of the time, and the jackknife intervals 81–95%
([V4](../validation/REPORT.md#v4)). With `likelihood: "exact"`, version 8's
MCMC intervals contained them only 75% of the time: do not quote those as
confidence statements. The parameters published for the Swiss
rivers (Piccolroaz et al., 2016) lay inside pyair2stream's MCMC intervals for 76
of 81 values ([V2](../validation/REPORT.md#v2)).

### What it rests on

Bayesian calibration by MCMC is standard in hydrology (Kuczera and Parent, 1998;
Vrugt et al., 2009). The sampler is emcee (Foreman-Mackey et al., 2013) with
the differential-evolution move (ter Braak, 2006); the convergence rules follow
Gelman et al. (2013). Reducing the number of observations to an effective number
for correlated errors is classical (Bayley and Hammersley, 1946; Zwiers and von
Storch, 1995); applying it inside a likelihood is an adjusted likelihood in the
sense of Pauli et al. (2011) and Ribatet et al. (2012), and is pyair2stream's
documented approximation. The jackknife is described by Efron and Tibshirani
(1993), and for blocks of dependent data by Künsch (1989).

---

## 12. Mean error by month and season

### What it is

`bias_by_month_*.csv` and `.png` (calibration and validation periods) and
`cv_bias_by_month.*` (held-out years) give the mean error in each calendar month,
season and the whole year, with a 95% interval.

### Why

A model can score well over the year and still be too warm in summer and too
cool in spring. For a summer limit, a summer bias matters more than the yearly
score: a model that is too warm overstates the chance a warm-water limit was
exceeded.

### How it is computed

The error is simulated minus measured: positive means the model is too warm.
The errors are first averaged within each year. A month counts in a year if
the record covers all its days and at least 10 of them were measured (a
season, 30; the whole year, 120); the unscored warm-up days are left out. Then
the bias is the mean of those yearly values, with interval
mean ± t × standard deviation / √(number of years); it needs at least two
years. Days
are not used as independent values, because errors persist for days to weeks
and an interval from days would be far too narrow.

### How to read it

An interval that excludes zero means the model is consistently biased in that
month. Use the held-out version (`cv_bias_by_month`) for the honest picture.
Where a bias remains in the season of your limit, compare model versions (on the
Rhône, version 5, which ignores discharge, was too warm every summer), and
report it.

---

## 13. Choosing the level: 90%, 95% or 99%?

### What a level means

A central L% range should miss the truth (100 − L)% of the time: a 90% range
about 1 time in 10, a 95% range 1 in 20, a 99% range 1 in 100. For normally
distributed errors, the half-width grows from 1.64 σ (90%) to 1.96 σ (95%) and
2.58 σ (99%) (Figure 19). The higher the level, the more it depends on the rare,
large errors, which are the hardest to describe.

![A normal curve with the 50, 80, 90, 95 and 99% central ranges and their half-widths in multiples of sigma.](figures/U07_levels.png)

*Figure 19. Central ranges of a normal distribution.*

### Setting it

| Setting | Controls | Default |
|---|---|---|
| `uncertainty_options.prediction_interval` | every prediction range: daily bands, the cross-validation check | 90 |
| `uncertainty_options.parameter_interval` | every parameter range: MCMC summary, jackknife | 90 |
| fixed at 95% | tests for a difference from zero: mean bias by month, the check's mean deviation, "differs from zero" | |

**The probability that a limit was exceeded does not depend on the level.** It is
the share of all 1,000 series above the limit; the level only decides which
range you quote beside it.

### The record on the real rivers

| Share inside the range | 50% | 80% | 90% | 95% | 99% |
|---|---|---|---|---|---|
| days, synthetic data (V4)* | 47–49% | 77–78% | 87–88% | 91.9–92.5% | 96% |
| days, 48 held-out years per version (V11) | 52–54% | 81–82% | 90% | 94.4–94.5% | 98.0–98.2% |
| 7-day means, held-out years (V11) | 53–59% | 84–87% | 93–94% | 96% | 98.5–99.1% |
| days, later validation years (V5) | 42–52% | 75–80% | 85–90% | 91–95% | 95–99.5% |
| days, British Columbia, later years 2021–2022 (V18) | V18_ROW |
| the same, with conformal margins (V19) | V19_ROW |
| yearly statistics, corrected (V11) | 50–69% | 77–85% | 85–94% | 92–100% | 100% |

\* Below the level only because about 3% of V4's synthetic values are below
0 °C, where no range can reach (section [6](#6-daily-prediction-intervals)).

![Stated against achieved coverage on the real rivers: days, summer days and 7-day means.](../validation/figures/V5_levels.png)

*Figure 20. Real rivers, later validation years (V5): stated level against the
share of measurements inside. On the dashed diagonal the ranges hold; the shaded
area is the accepted range (a miss rate between half and 1.5 times the stated
one).*

### Advice

- **90% (default)** holds well for days, 7-day means and corrected yearly
  statistics.
- **95%** holds for days and 7-day means over many years, and for corrected
  yearly statistics. In years unlike the calibration years it can be somewhat
  narrow (91–95% in V5; V18_95 on the British Columbia rivers, V18). Check it
  at your site (`cv_interval_coverage.csv`); if it held too rarely, widen it
  (section [6](#widening-a-band-that-held-too-rarely-optional)).
- **99%** daily intervals missed about twice as many days as stated (98.0–98.2%
  held), because real errors have heavier tails than the normal distribution.
  Do not quote a 99% daily interval without showing its coverage at your site.
  For yearly statistics, 99% cannot be verified with the years usually
  available.
- Whatever level you choose, choose it **before** looking at the results, and
  report its coverage on years not used for calibration.

---

## 14. Warnings and what to do about them

| Message | Meaning | What to do |
|---|---|---|
| `run_mode DE-MCMC cannot be used with objective_function KGE` | the ranges are measured around the least-squares fit, which a KGE fit is not | `objective_function: "NSE"` or `"RMS"` |
| `MCMC did not converge within ... steps` | the parameter sets are not yet a reliable sample | more `mcmc_steps`, or a simpler model version; never use unconverged results for a decision |
| `Only ... pairs of complete 7-day windows ...` | too few complete weeks of measurements for the weekly ρ; the daily ρ is used | expect ranges for multi-week quantities to be too narrow; check them by cross-validation |
| `Only ... valid residual pairs ... Falling back to rho=0.0` | too few consecutive measured days to measure persistence at all (section [5](#5-the-models-errors-their-size-and-how-long-they-last)) | the parameter ranges and every multi-day range are too narrow; do not use them for a decision without saying so |
| `... draws were excluded as numerically divergent` | some parameter sets make the simulation blow up (usually with an explicit integrator) | use `CRN` (the default); check `parameter_bounds` |
| `rho reached its upper limit of 0.99` | errors persist for months: a systematic error | look at the mean error by month; consider another model version |
| `the exact AR(1) likelihood treats the scored errors as independent` | with weekly or monthly scoring that likelihood ignores persistence | keep the default `likelihood: "least_squares"` |
| `The MCMC chain ... was fitted with ...` | the chain belongs to a calibration with another model version, integrator, `Qmedia`, `Tice_cover` or `min_theta_floor` | use the matching chain and `paths.calibration_metadata` |
| `the relaxation rate B is negative` / `zigzags from one day to the next` | physically impossible parameters, usually after weekly or monthly scoring | set the minimum of `a2` and `a3` to 0 and calibrate again |
| `... days ... have theta = Q/Qmedia outside the calibrated range` | the scenario's flows go beyond those calibrated on | treat the results as extrapolation |
| `Note: ... does not record the model version, integrator and Qmedia` | a chain from version 0.4.2 or earlier | make sure it matches, or rerun DE-MCMC |
| `Note: ... does not record the Tice_cover and min_theta_floor` | a chain or calibration from version 0.5.0 or earlier | make sure they match, or rerun the calibration |

---

## 15. Is it defensible?

When a result is used in a permit, a compliance assessment or a hearing, experts
are commonly asked whether the method has been tested, whether its error rate
is known, whether it rests on published and accepted methods, and whether the
result can be reproduced. In US courts these are the Daubert factors (Daubert v.
Merrell Dow Pharmaceuticals, 1993); guidance on environmental models asks the
same of model results (Jakeman et al., 2006; Refsgaard et al., 2007; US EPA,
2009). This is how pyair2stream answers them.

| Question | pyair2stream's answer |
|---|---|
| Is the method tested? | The [validation suite](../validation/REPORT.md) runs 19 checks with stated pass criteria, through the same code a user runs: identical results to the original Fortran, reproduction of the published results, recovery of known truths, coverage of intervals at several levels, and performance on real rivers in Switzerland and British Columbia. Failures are reported, not hidden: the report marks each check PASS or FAIL, and says which criterion was missed. |
| Is its error rate known? | Yes, as coverage at each level (section [13](#13-choosing-the-level-90-95-or-99)), on synthetic data and real rivers, and at your own site through cross-validation. |
| Does it rest on published methods? | Each component does (table below). The combination, and three approximations, are pyair2stream's own and documented as such. |
| Can the result be reproduced? | With `random_seed`, every run gives identical results. Each run records its settings (`calibration_metadata.json`, `_meta.json` with σ, ρ, the chain's content hash and the parameter sets used), and a FORWARD run refuses a chain fitted with another model version, integrator, `Qmedia`, `Tice_cover` or `min_theta_floor`. |

### What each component rests on

| Component | Published basis | What is pyair2stream's own | Evidence |
|---|---|---|---|
| The model | Toffolon and Piccolroaz (2015); Piccolroaz et al. (2016); Callahan and Moore (2025) | a Python version | V1, V2, V13, V15 |
| MCMC calibration | Kuczera and Parent (1998); ter Braak (2006); Foreman-Mackey et al. (2013); Gelman et al. (2013) | | V4 |
| Autocorrelated error model | Sorooshian and Dracup (1980); Schoups and Vrugt (2010); Evin et al. (2014) | ρ chosen at the weekly scale | V4, V5, V9 |
| Effective sample size in the likelihood | Bayley and Hammersley (1946); Pauli et al. (2011); Ribatet et al. (2012) | its use with the least-squares likelihood | V4 |
| Probabilities from an ensemble | Wilks (2011) | | V9 |
| Cross-validation and the PIT | Stone (1974); Klemeš (1986); Hastie et al. (2009); Coron et al. (2012); Dawid (1984); Gneiting et al. (2007) | | V11, V12 |
| Correction of yearly statistics | Glahn and Lowry (1972); Gneiting et al. (2005) | the shift with its uncertainty, from held-out years | V9 C, V11 |
| Paired scenarios | common random numbers (Law, 2015) | | V8 |
| Conformal margins (optional) | Vovk et al. (2005); Lei et al. (2018); Romano et al. (2019); Oliveira et al. (2024) | whole years as the units | V19 |

### What to state as limitations

- The model gives **daily means**; a limit on daily maxima needs a separate,
  justified step.
- The error model assumes normally distributed errors of constant size whose
  persistence is described by one number. Up to 95% this was adequate on the
  Swiss rivers; at 99% daily intervals were too narrow.
- Uncertainty is estimated from the calibration years and assumes the river
  behaves alike in the years predicted. In years unlike them, intervals were
  somewhat narrower than stated: on 23 British Columbia rivers, 90% intervals
  held on about 87% of days in later years (V18). The conformal margins correct
  this from the misses of held-out years (V19), on the same assumption.
- Yearly statistics need the cross-validation correction; the highest daily
  mean remains the least reliable.
- The weekly ρ, the effective-sample-size likelihood and the correction of
  yearly statistics are validated approximations, not methods published under
  those names.
- The evidence comes from 3 Swiss and 23 British Columbia rivers. Your river
  may differ: that is what the check at your site is for.

### How to word a result

> The model, calibrated on 2002–2009 and checked by leave-one-year-out
> cross-validation (8 held-out years), gives a probability of 0.35 that the
> highest 7-day mean water temperature in 2011 exceeded 20 °C (90% range for
> that statistic 18.8–20.7 °C), after correcting for the model's average error
> in that statistic in the held-out years (−0.65 °C, 95% interval −1.05 to
> −0.26 °C). Uncorrected, the probability would be 0.77. pyair2stream 0.5.1,
> commit `<sha>`, random seed 42.

### Checklist before you report

1. The model validates well on years not used for calibration, and no parameter
   sits on a bound.
2. The MCMC run converged.
3. Multi-day and yearly quantities were computed in each simulated series.
4. Yearly statistics were checked by cross-validation and corrected, with the
   same definition, threshold and season as the question.
5. The coverage at the level you report was checked at your site, and, if it
   held too rarely, the band was widened with the conformal margins and their
   check (`inside_after`) reported.
6. The mean error by month shows no unexplained bias in the season of the limit.
7. The config, input files, commit, `pip freeze` and output folder are kept.

---

## 16. A worked example, from data to a statement

[Example 03](../examples/03_compliance/README.md) does all of this on the
Mentue. `python examples/03_compliance/run.py` runs these three steps, then
computes the statistics, the correction and the table:

```bash
pyair2stream --config examples/03_compliance/calibrate.yaml   # DE-MCMC on 2002-2009: parameters, σ, ρ
pyair2stream --config examples/03_compliance/predict.yaml     # FORWARD on 2010-2012: 1,000 series
pyair2stream --config examples/03_compliance/check.yaml       # cross-validation of 2002-2009
```

| Year | P(7-day mean > 20 °C), corrected | uncorrected | Measured highest 7-day mean |
|---|---|---|---|
| 2010 | 0.54 | 0.92 | 21.0 °C (exceeded) |
| 2011 | 0.35 | 0.77 | 19.8 °C (not exceeded) |
| 2012 | 0.15 | 0.53 | 19.6 °C (not exceeded) |

The check found that the model put the yearly peak 0.65 °C too high in the
held-out years. Corrected, the probabilities matched what happened better (Brier
score, the mean squared difference between probability and outcome; Brier,
1950: 0.12 against 0.29), and all three measured values lay inside the corrected 90%
ranges.

---

## 17. Common mistakes

| Mistake | Why it is wrong | Instead |
|---|---|---|
| Reading a weekly mean, a yearly peak or a count of days off the daily band | the band's edges are not the edges of those statistics (Figure 11) | compute the statistic in each series (section [7](#7-weekly-means-yearly-peaks-days-above-a-limit)) |
| `noise_model: "iid"` for multi-day quantities | errors that are new every day average out, so ranges are far too narrow (Figure 5) | keep `"ar1"` |
| Quoting the coverage of the `DE-MCMC` band as evidence | on the calibration years it is close to the level almost by construction | the coverage on held-out or later years (sections [6](#6-daily-prediction-intervals), [8](#8-testing-on-years-the-model-has-not-seen-cross-validation)) |
| Reading the sensitivity analysis as an uncertainty | it shows how the simulation responds to a parameter, not how well the data fix it | MCMC or jackknife intervals (section [11](#11-how-well-are-the-parameters-known)) |
| Reporting an uncorrected probability for a yearly statistic | the model can be biased at the peaks (Figure 14) | check and correct (sections [8](#8-testing-on-years-the-model-has-not-seen-cross-validation), [9](#9-correcting-yearly-statistics-for-the-models-bias)) |
| Correcting only when the check shows a bias | the result then steers the method | always correct yearly statistics |
| Calling 8 of 10 years inside a 90% range a failure | chance alone gives 7–10 (Figure 13) | compare with the range expected by chance |
| Subtracting two scenario runs that used different parameter sets | the shared uncertainty does not cancel (Figure 16) | `reuse_sample_indices_from` and `paired_difference_from_files` |
| Recomputing `Qmedia` from scenario flows | it cancels the flow change | `paths.calibration_metadata` |
| Using the `std` row of `cv_results.csv` as a parameter uncertainty | the folds share most of their data | the jackknife rows |
| Combining the ends of several parameter intervals | parameters trade off (Figure 18) | rely on predictions |
| Quoting a 99% daily interval unchecked | real errors have heavier tails | check its coverage at your site (section [13](#13-choosing-the-level-90-95-or-99)) |
| Checking a band only on the days the limit was exceeded | those days were chosen partly because their error was positive: even a correct 90% band is exceeded there far more often than 10% of the time (V14) | check on the days predicted to be hottest, or with the hottest air |
| Using an unconverged chain | its parameter sets are not a valid sample | more steps or a simpler version |

---

## 18. References

- Bayley, G. V. and Hammersley, J. M. (1946). The "effective" number of
  independent observations in an autocorrelated time series. *Supplement to the
  Journal of the Royal Statistical Society*, 8, 184–197.
- Beven, K. (2006). A manifesto for the equifinality thesis. *Journal of
  Hydrology*, 320, 18–36.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of
  probability. *Monthly Weather Review*, 78, 1–3.
- Callahan, L. and Moore, R. D. (2025). Evaluation of the hybrid air2stream
  model for simulating daily stream temperature during extreme summer heat wave
  and autumn drought conditions. *Hydrological Processes*, 39(1), e70033.
  Data: Moore, R. D. and Callahan, L. (2024), Zenodo,
  https://doi.org/10.5281/zenodo.14502248.
- Coron, L., Andréassian, V., Perrin, C., Lerat, J., Vaze, J., Bourqui, M. and
  Hendrickx, F. (2012). Crash testing hydrological models in contrasted climate
  conditions: an experiment on 216 Australian catchments. *Water Resources
  Research*, 48, W05552.
- Daubert v. Merrell Dow Pharmaceuticals, Inc., 509 U.S. 579 (1993).
- Dawid, A. P. (1984). Statistical theory: the prequential approach. *Journal of
  the Royal Statistical Society A*, 147, 278–292.
- Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall.
- Evin, G., Thyer, M., Kavetski, D., McInerney, D. and Kuczera, G. (2014).
  Comparison of joint versus postprocessor approaches for hydrological
  uncertainty estimation accounting for error autocorrelation and
  heteroscedasticity. *Water Resources Research*, 50, 2350–2375.
- Foreman-Mackey, D., Hogg, D. W., Lang, D. and Goodman, J. (2013). emcee: the
  MCMC hammer. *Publications of the Astronomical Society of the Pacific*, 125,
  306–312.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and Rubin,
  D. B. (2013). *Bayesian Data Analysis*, 3rd edn. CRC Press.
- Glahn, H. R. and Lowry, D. A. (1972). The use of model output statistics (MOS)
  in objective weather forecasting. *Journal of Applied Meteorology*, 11,
  1203–1211.
- Gneiting, T., Raftery, A. E., Westveld, A. H. and Goldman, T. (2005).
  Calibrated probabilistic forecasting using ensemble model output statistics
  and minimum CRPS estimation. *Monthly Weather Review*, 133, 1098–1118.
- Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007). Probabilistic
  forecasts, calibration and sharpness. *Journal of the Royal Statistical
  Society B*, 69, 243–268.
- Hamill, T. M. (2001). Interpretation of rank histograms for verifying ensemble
  forecasts. *Monthly Weather Review*, 129, 550–560.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of
  Statistical Learning*, 2nd edn. Springer.
- Jakeman, A. J., Letcher, R. A. and Norton, J. P. (2006). Ten iterative steps
  in development and evaluation of environmental models. *Environmental
  Modelling & Software*, 21, 602–614.
- Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). Leakage in data
  mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge
  Discovery from Data*, 6(4), 15.
- Klemeš, V. (1986). Operational testing of hydrological simulation models.
  *Hydrological Sciences Journal*, 31, 13–24.
- Kuczera, G. and Parent, E. (1998). Monte Carlo assessment of parameter
  uncertainty in conceptual catchment models: the Metropolis algorithm.
  *Journal of Hydrology*, 211, 69–85.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
  observations. *Annals of Statistics*, 17, 1217–1241.
- Laio, F. and Tamea, S. (2007). Verification tools for probabilistic forecasts
  of continuous hydrological variables. *Hydrology and Earth System Sciences*,
  11, 1267–1277.
- Law, A. M. (2015). *Simulation Modeling and Analysis*, 5th edn. McGraw-Hill.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L. (2018).
  Distribution-free predictive inference for regression. *Journal of the
  American Statistical Association*, 113, 1094–1111.
- McInerney, D., Thyer, M., Kavetski, D., Laugesen, R., Tuteja, N. and Kuczera,
  G. (2020). Multi-temporal hydrological residual error modeling for seamless
  subseasonal streamflow forecasting. *Water Resources Research*, 56,
  e2019WR026979.
- Oliveira, R. I., Orenstein, P., Ramos, T. and Romano, J. V. (2024). Split
  conformal prediction and non-exchangeable data. *Journal of Machine Learning
  Research*, 25(225), 1–38.
- Pauli, F., Racugno, W. and Ventura, L. (2011). Bayesian composite marginal
  likelihoods. *Statistica Sinica*, 21, 149–164.
- Piccolroaz, S., Calamita, E., Majone, B., Gallice, A., Siviglia, A. and
  Toffolon, M. (2016). Prediction of river water temperature: a comparison
  between a new family of hybrid models and statistical approaches.
  *Hydrological Processes*, 30, 3901–3917.
- Refsgaard, J. C., van der Sluijs, J. P., Højberg, A. L. and Vanrolleghem, P. A.
  (2007). Uncertainty in the environmental modelling process – a framework and
  guidance. *Environmental Modelling & Software*, 22, 1543–1556.
- Renard, B., Kavetski, D., Kuczera, G., Thyer, M. and Franks, S. W. (2010).
  Understanding predictive uncertainty in hydrologic modeling: the challenge of
  identifying input and structural errors. *Water Resources Research*, 46,
  W05521.
- Ribatet, M., Cooley, D. and Davison, A. C. (2012). Bayesian inference from
  composite likelihoods, with an application to spatial extremes. *Statistica
  Sinica*, 22, 813–845.
- Romano, Y., Patterson, E. and Candès, E. (2019). Conformalized quantile
  regression. *Advances in Neural Information Processing Systems*, 32,
  3543–3553.
- Schoups, G. and Vrugt, J. A. (2010). A formal likelihood function for
  parameter and predictive inference of hydrologic models with correlated,
  heteroscedastic, and non-Gaussian errors. *Water Resources Research*, 46,
  W10531.
- Sorooshian, S. and Dracup, J. A. (1980). Stochastic parameter estimation
  procedures for hydrologic rainfall-runoff models: correlated and
  heteroscedastic error cases. *Water Resources Research*, 16, 430–442.
- Stone, M. (1974). Cross-validatory choice and assessment of statistical
  predictions. *Journal of the Royal Statistical Society B*, 36, 111–147.
- ter Braak, C. J. F. (2006). A Markov chain Monte Carlo version of the genetic
  algorithm Differential Evolution: easy Bayesian computing for real parameter
  spaces. *Statistics and Computing*, 16, 239–249.
- Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water
  temperature as a function of air temperature and discharge. *Environmental
  Research Letters*, 10, 114011.
- US EPA (2009). *Guidance on the Development, Evaluation, and Application of
  Environmental Models*. EPA/100/K-09/003. US Environmental Protection Agency.
- Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a
  Random World*. Springer, New York.
- Vrugt, J. A., ter Braak, C. J. F., Diks, C. G. H., Robinson, B. A., Hyman, J. M.
  and Higdon, D. (2009). Accelerating Markov chain Monte Carlo simulation by
  differential evolution with self-adaptive randomized subspace sampling.
  *International Journal of Nonlinear Sciences and Numerical Simulation*, 10,
  273–290.
- Wilks, D. S. (2011). *Statistical Methods in the Atmospheric Sciences*, 3rd
  edn. Academic Press.
- Zwiers, F. W. and von Storch, H. (1995). Taking serial correlation into account
  in tests of the mean. *Journal of Climate*, 8, 336–351.
