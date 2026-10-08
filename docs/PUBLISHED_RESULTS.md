# Reproducing published air2stream results

pyair2stream has been checked against every published set of air2stream
results whose data are openly available: three papers, by the model's authors
and by an independent group. This page records what was reproduced and the
errors found in the publications themselves, with the evidence for each. It is
written so that a reader can verify every statement from the validation report
([validation/REPORT.md](../validation/REPORT.md)) and the data in this
repository.

| Study | Rivers | Scheme | What pyair2stream reproduces | Errors found in the publication | Check |
|---|---|---|---|---|---|
| Toffolon and Piccolroaz (2015) | 3 Swiss | RK4 | errors: all 30, within rounding. Parameters: its own calibration returns the published values in all 15 cases | the Rhône's calibration and validation periods are misprinted | [V13](../validation/REPORT.md#v13) |
| Piccolroaz et al. (2016) | 3 of 38 Swiss (the 3 distributed) | Crank–Nicolson | errors: all 30, to 0.0005 °C. Parameters: returned by calibration in 10 of 15 cases (the other 5 trade off along flat valleys); 76 of 81 published values inside pyair2stream's 90% intervals | optimiser settings not stated; in 5 of 15 cases the published parameters are not the best fit; validation discharge scaling not stated | [V2](../validation/REPORT.md#v2) |
| Callahan and Moore (2025) | 23 in British Columbia | Crank–Nicolson | simulated series: 45 of 46, day by day, to 0.00013 °C. Parameters: returned by calibration at 1 of 23 stations; where the MCMC converged (19 stations), 137 of 152 published values inside its 90% intervals | one station's calibration record read with its seasonal cycle ten months out of phase; one station's published calibration far from the best fit | [V15](../validation/REPORT.md#v15) |

The errors listed do not affect what pyair2stream computes: in every case the
published numbers are reproduced once the publication's actual procedure is
followed. They matter for anyone repeating these studies, and they show which
details a publication must state for its results to be reproducible.

## 1. Toffolon and Piccolroaz (2015)

> Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water
> temperature as a function of air temperature and discharge. *Environmental
> Research Letters*, 10, 114011. https://doi.org/10.1088/1748-9326/10/11/114011

**Data.** Three Swiss rivers (Mentue, Rhône at Sion, Dischmabach), distributed
with the original code ([data/switzerland/](../data/switzerland/README.md)).
The parameters (supplementary Table 1) and errors (Table 2) are transcribed in
`data/switzerland/published/Toffolon_Piccolroaz_ERL2015.csv`.

**Method as stated.** Calibration by particle swarm optimisation (1000
particles, 1000 iterations) minimising RMSE; the 4th-order Runge–Kutta scheme
with a one-day step; simulated temperatures below 0 °C set to 0 °C.

**Reproduced.** With RK4, all 30 published calibration and validation RMSE
values (5 versions, 3 rivers) are reproduced within their rounding (largest
difference 0.006 °C; the published values have 2 decimals and the parameters
3). pyair2stream's calibration with RK4 returns the published parameters, each
within 1% of its range, in all 15 cases. With Crank–Nicolson the published
errors are missed by up to 0.32 °C: the parameters belong to RK4.

**Error found: the Rhône's periods are misprinted.** Table 1 gives the Rhône's
calibration period as 1984–2003 and its validation period as 2004–2013. With
that split the published errors are missed by up to 0.013 °C, also by versions
3 and 5, which do not use discharge, so the difference cannot come from the
discharge scaling. With 1984–2004 and 2005–2013, the split of the data
distributed with the code (and used in Piccolroaz et al., 2016), all ten Rhône
values are reproduced. The table's periods are therefore a misprint; the
results were computed on the distributed split. Anyone repeating the study with
the stated periods would not reproduce Table 2.

## 2. Piccolroaz et al. (2016)

> Piccolroaz, S., Calamita, E., Majone, B., Gallice, A., Siviglia, A. and
> Toffolon, M. (2016). Prediction of river water temperature: a comparison
> between a new family of hybrid models and statistical approaches.
> *Hydrological Processes*, 30, 3901–3917. https://doi.org/10.1002/hyp.10913

**Data.** 38 Swiss rivers; the data of three are distributed with the code
([data/switzerland/](../data/switzerland/README.md)), with the published
parameters and errors of all 38 (supplementary spreadsheets, in
`data/switzerland/published/`).

**Method as stated.** Calibration by particle swarm optimisation minimising
RMSE; the Crank–Nicolson scheme with a one-day step; two thirds of each record
for calibration, one third for validation.

**Reproduced.** All 30 published calibration and validation RMSE values of the
three distributed rivers, to within 0.0005 °C (the values have 3 decimals).
pyair2stream's calibration matches or improves on the published fit in all 15
cases. The published parameters lie inside pyair2stream's 90% parameter
intervals for 76 of 81 values.

**Errors found.**

1. *The optimiser's settings are not stated.* The paper does not give the
   swarm size or the number of iterations (the 2015 paper does). The original
   program distributed with the paper uses 500 particles and 500 iterations in
   its example, and seeds its random numbers from the clock, so each run
   differs. Run three times exactly as distributed, it returned the published
   parameters every time in only about half of the 15 cases (8 in one
   validation run, 9 in another). The published parameters of
   the other cases cannot be reproduced exactly by anyone, including with the
   original program.
2. *In five cases the published parameters are not the best fit.* For versions
   7 and 8 on the Mentue and versions 3, 7 and 8 on the Rhône, recalibration
   with the same objective, scheme, parameter ranges and data reaches a
   different parameter set; in four of these cases it fits the calibration
   years better (for example, the Mentue's version 8: RMSE 0.633 °C against
   the published 0.645 °C), and the fifth is a tie. The original program,
   scoring both sets itself, agrees. The best fit lies along a flat valley where
   the parameters trade off, and the published sets are points where the
   optimiser stopped. The two sets predict the validation years within
   0.002 °C of each other. The published parameter values of these versions
   should therefore not be interpreted individually.
3. *How discharge was scaled in the validation period is not stated.* The
   original program recomputes `Qmedia`, the mean discharge by which daily
   discharge is scaled, from the validation period's own data, so a parameter
   set fitted with one scaling is validated with another, and the validation
   uses the mean discharge of the period it predicts. The published validation
   errors are reproduced only this way. The effect here is at most 0.009 °C
   (versions 4, 7 and 8, which use discharge). pyair2stream keeps the
   calibration's `Qmedia`.
4. *A minor inconsistency in the code's documentation.* The distributed input
   file uses Crank–Nicolson ("CRN is the suggested choice"), but the example in
   its readme shows RK4. With RK4, 8 of the 15 published parameter sets make the
   simulation unstable.

The evidence for 1, 2 and 4 is V2 parts D, C and E, and for 3 V2 part A.

## 3. Callahan and Moore (2025)

> Callahan, L. and Moore, R. D. (2025). Evaluation of the hybrid air2stream
> model for simulating daily stream temperature during extreme summer heat
> wave and autumn drought conditions. *Hydrological Processes*, 39(1), e70033.
> https://doi.org/10.1002/hyp.70033

> Moore, R. D. and Callahan, L. (2024). Evaluation of the hybrid Air2stream
> model for simulating daily stream temperature during extreme summer heat
> wave and autumn drought conditions - Data sets and scripts (version 1).
> Zenodo. https://doi.org/10.5281/zenodo.14502248 (record:
> https://zenodo.org/records/14502248; licence CC BY 4.0)

**Data.** 23 Water Survey of Canada stations in British Columbia, 2011–2022:
daily air temperature interpolated from the ERA5 reanalysis, discharge,
measured water temperature, the calibrated parameters of the 8-parameter
version, and the simulated water temperature for the calibration (up to 2020)
and validation (2021 and 2022, including the 2021 heat dome and the 2022 autumn
drought) periods ([data/british_columbia/](../data/british_columbia/README.md),
files unchanged). Publishing the simulated series is what made a day-by-day
check possible.

**Reproduced.** Given the published parameters and inputs, pyair2stream
computes the published simulated water temperature for 45 of the 46 records
(23 stations, two periods) to within 0.00013 °C on every day, with the
Crank–Nicolson scheme and each period's own mean discharge as `Qmedia`. The
errors in the paper's heat-dome (25 June to 2 July) and drought (1 September
to 31 October) windows therefore follow; for example, the mean RMSE over the
23 stations in the 2021 heat dome is 1.42 °C, from the published series and
from pyair2stream's alike. pyair2stream's calibration fits each station's
calibration years at least as well as the published parameters (by 0 to
0.15 °C, apart from the station below).

**Parameters.** pyair2stream's calibration returns the published parameter
values (each within 1% of its range) at only 1 of the 23 stations, although it
fits every station at least as well. Whether the published values are
nevertheless consistent with the data was tested as for the 2016 paper: each
station's calibration years give two 90% intervals for every parameter.

- *DE-MCMC intervals*, the parameter values whose fit is close to the best:
  the sampler converged at 19 of the 23 stations (chains of up to 100,000
  steps), and there 137 of the 152 published values lie inside. Of the 15
  outside, 7 belong to station 08HA002 (below) and 1 to station 08GA077, whose
  calibration record was misread (below); the other 7 lie at a bound of the
  parameter ranges (`a4` near −1 at six stations, `a6` near 10 at one). At
  the other 17 stations the published parameters fit the data about as well
  as the best fit.
- *Jackknife intervals*, how far the best fit itself moves when a year is left
  out: they contain 72 of those 152 values. The published sets lie among the
  good fits, but not where a least-squares calibration lands.
- At 4 stations (08FF003, 08HA069, 08KH006, 08MF040) the sampler had not
  converged after 100,000 steps: the chains were long enough (at least 50
  times their autocorrelation time) but split-R̂ stayed at 1.011–1.015, just
  above the 1.01 required. The parameters there are very poorly determined by
  the data. The jackknife intervals there contain 11 of 32 published values.
- The published `a4` lies at or within 0.04 of its lower bound, −1, at 14 of
  the 23 stations: many of the published calibrations stopped on that bound.

Apart from station 08HA002, this is not an error of the publication: many
parameter combinations fit these data almost equally well (equifinality), and
a different calibration can reasonably end elsewhere among them. The
published parameter values should therefore not be interpreted individually;
the predictions, which part A reproduces, are what the model provides.

**Finding: one station's published calibration is far from the best fit.** At
station 08HA002 (Cowichan River at Lake Cowichan, regulated), 7 of the 8
published parameter values lie well outside the region of good fits (for
example, `a5` is 17.0 where the 90% interval is 0.28 to 1.63). Calibrated by
pyair2stream, within the same parameter ranges, the station fits its
calibration years with RMSE 0.72 °C instead of 0.87 °C, and predicts 2021–2022
with 0.76 °C instead of the published 1.09 °C. The dataset does not record how
the published calibration was made (objective, parameter ranges, optimiser and
its settings), so why it stopped there cannot be determined from it; the
station's published parameters and errors, and its share of any averages over
stations, are affected.

**Error found: one calibration record was read with its seasonal cycle ten
months out of phase.** The calibration record of station 08GA077 (Seymour River
below Orchid Creek) starts on 1 November 2012. The original program assumes
that every record starts on 1 January and takes the timing of the seasonal term
from the row number, not the date. The published calibration simulation of this
station differs from pyair2stream's, run with the true dates, by up to 3.3 °C;
read by row as if the record began on 1 January, pyair2stream reproduces it to
0.00001 °C. So the station was calibrated with its seasonal term ten months
out of phase, and its parameters compensate for that. Its validation record
starts on 1 January 2021, so the same parameters were then run in phase: the
published validation RMSE of this station is 1.44 °C. Calibrated by
pyair2stream with the correct dates, the station fits its calibration years
with RMSE 0.84 °C and predicts 2021–2022 with 0.97 °C. The station's published
parameters and errors, and its share of any averages over stations, are
affected; the other 22 stations are not.

The paper's text could not be retrieved automatically (the publisher refuses
automated access), so the comparison is with the published dataset, which
contains the paper's inputs, parameters and simulations; the summary
statistics in the paper's text were not compared.

## Conventions of the original program behind these errors

These are behaviours of the original Fortran program, not stated in the papers,
that the published results depend on. pyair2stream reproduces each when asked
to (that is how the comparisons above were made), but by default does what is
listed on the right ([METHODS §17](METHODS.md#17-differences-from-the-fortran-original)).

| Original program | Consequence | pyair2stream |
|---|---|---|
| assumes every record starts on 1 January; the seasonal timing comes from the row number | a record starting on another date is silently simulated with its seasonal cycle shifted (Callahan and Moore, station 08GA077) | timing from the dates; a calibration or validation record must start on 1 January (except in gap-tolerant mode, which also uses the dates) |
| recomputes `Qmedia` from the validation period's data | parameters are validated with another discharge scaling than they were fitted with, using information from the period predicted | keeps the calibration `Qmedia` |
| PSO with a fixed number of iterations, seeded from the clock | runs cannot be repeated; where parameters trade off, each run returns different parameters | seeded DE with a local search; `random_seed` makes runs identical |
| parameters are specific to the numerical scheme | parameters published for one scheme give other results, or diverge, with another | the scheme is recorded with every calibration (`calibration_metadata.json`), and a FORWARD run with intervals refuses an MCMC chain fitted with another scheme |

## What a publication should state

To make air2stream results reproducible, state: the model version and
numerical scheme; the optimiser, its settings and its random seed (or that it
was seeded from the clock); the calibration and validation dates; how discharge
was scaled in the validation period; and how missing data were handled. Where
possible, publish the inputs, the calibrated parameters and the simulated
series, as Callahan and Moore did.
