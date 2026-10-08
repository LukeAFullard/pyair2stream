# Validation

This folder holds the evidence that pyair2stream is correct and that its results
can be relied on. [REPORT.md](REPORT.md) summarises every check, and each check
has a full report in [reports/](reports/) (question, method, pass criterion,
figures, tables and what the result means).

Every check runs the package the way a user would, through its configuration
files and run modes. Each check states its question, method and pass criterion
before its result.

## What is checked

| Check | Question | Why it matters |
|---|---|---|
| V1 | Does it compute the same temperatures as the original Fortran program, on real, variable inputs? Does it compute the same calibration score, from the same weekly and monthly averages, also when the record has gaps? | Shows the model equations, solution schemes and scoring were translated correctly. |
| V2 | Given the published parameters, does it reproduce the published model errors for every model version on three Swiss rivers? Does its own calibration, done the way the paper did it, fit at least as well (judged by the original program too) and find the published parameters? Do the published parameters lie inside its uncertainty intervals? | Ties the package to the peer-reviewed results of Piccolroaz et al. (2016). |
| V3 | When data are made by the model itself from known parameters, does calibration recover them well enough to predict other years? | Only with a known truth can we tell whether calibration finds the right answer. |
| V4 | Do the 90% uncertainty intervals contain the truth about 90% of the time, with each likelihood the package offers, with ρ (error persistence) measured over days or over weeks, and when calibrating on weekly means? Do intervals at 50%, 80%, 95% and 99% hold too? | An interval is only useful if its stated confidence is honest, at the level the user asks for. |
| V5 | On real rivers, how well does a calibrated model predict years it has not seen, compared with simple alternatives? Do its intervals contain about 90% of real measurements, for single days and for 7-day means, and are they centred on the best fit? How do intervals at other levels (50-99%) do? | Tests the whole approach on real data, not only the code. |
| V6 | How accurate are the numerical schemes, does the package stop runs that go wrong, and does the choice of stable scheme matter? | Rules out numerical error as a source of wrong answers. |
| V7 | Do gaps in the water or air temperature record bias the result? | Real records have gaps. |
| V8 | Does the documented workflow reproduce the calibration exactly? Do scenario comparisons and threshold counts give exact answers where the answer is known? | The tools used to reach a conclusion must be exact. |
| V9 | When the package gives a chance that a yearly statistic (highest daily mean, highest 7-day mean, days above a threshold) exceeded a limit, does that happen as often as it says? On real rivers, do these chances beat going by past years? | This is the answer to a compliance question. |
| V10 | Calibrated only on the coolest (or highest-flow) years, does the model still predict the warmest (or lowest-flow) years, and do its intervals hold? | Limits are breached, and scenarios extrapolate, in exactly those years. |
| V11 | Over many years not used for calibration, do the predicted ranges of yearly statistics hold? Why not, when they do not? Does the cross-validated correction make them hold? | The answer to a compliance question must carry the confidence it states. |
| V12 | Cross-validation hides a year's water temperatures and keeps its air temperature and discharge. Would hiding those inputs too, leaving a buffer around the year, or calibrating on earlier years only change what it reports? | The cross-validation is the evidence for every check of predictions in new years; its design must not flatter the model. |
| V13 | Given the parameters of the first air2stream paper (Toffolon and Piccolroaz, 2015), does it reproduce that paper's errors, and with which numerical scheme? Does its calibration with that scheme return those parameters? | A second published benchmark, from the model's authors, on the same data. |
| V14 | In years not used for calibration, do the prediction intervals hold on the days predicted to be hottest and on the days with the hottest air? | Limits are breached on the hottest days. |
| V15 | Given the parameters and inputs an independent group published for 23 rivers in British Columbia (Callahan and Moore, 2025), does it compute their simulated water temperatures, day by day? Does its own calibration fit at least as well? | Results from other people, rivers and climates, including the 2021 heat dome, compared in full rather than by a summary error. |

## Running it

From the repository root, with the package installed (`pip install -e .`):

```bash
python validation/run_all.py            # full suite, several hours on 4 cores
python validation/run_all.py --quick    # reduced version of every check, about 2 minutes
python validation/run_all.py --only V2 V6
```

With `--only`, the other checks keep their reports from the earlier run.
`REPORT.md` then names the run each report comes from. Since the DE stopping
rule was tightened (0.5.0), V2 and V4 take about an hour each. Where jobs are
limited in length, run the suite in parts with `--only`.

V1 needs `gfortran` and the Fortran source (`git submodule update --init`).
Without them, V1 is reported as not run.

The run rewrites `REPORT.md`, `reports/` (one report per check), `results/`
(every table as CSV) and `figures/`. Scratch files go to `work/`, which is not
kept.

**Will I get the same numbers?** Every random step is seeded. So a rerun with
the same software versions, on the same kind of processor, gives the same
numbers. The report records the versions used. There are two exceptions:

- The model is compiled for the processor it runs on. On another processor,
  the last digits of a calculation can differ. The MCMC checks amplify such
  small differences, so their results then differ by a small random amount.
- In V2 parts D and E, the original Fortran program calibrates. It seeds its
  random numbers from the clock, so those runs differ every time. (Part D
  shows this.)

A rerun of the full suite reproduced every other table exactly.

**The quick run** (`--quick`, also run by CI) checks that every check runs. It
uses only a few replicates or years, which is too few to judge coverage. So in
the quick run, V4, V5 and V14 report their coverage without judging it.

## Data

Most checks use the three Swiss rivers in [`data/switzerland/`](../data/switzerland/README.md),
with the published parameters and model errors of Piccolroaz et al. (2016) and
of Toffolon and Piccolroaz (2015). V15 uses the 23 British Columbia streams of
Callahan and Moore (2025) in [`data/british_columbia/`](../data/british_columbia/README.md),
from their published dataset (https://doi.org/10.5281/zenodo.14502248, CC BY
4.0). Each README gives the sources, periods and licence. The errors found in
the three publications are documented in
[docs/PUBLISHED_RESULTS.md](../docs/PUBLISHED_RESULTS.md).

## What this does and does not show

It shows that:

- the software computes what it claims to;
- calibration finds the right answer when one is known;
- the uncertainty ranges and probabilities hold as often as they state, when
  the model is right.

It also shows how well the model does on real rivers, and where the approach
falls short there:

- Ranges for new years are slightly too narrow (V5).
- The model can be too warm on the hottest days. So uncorrected ranges for
  yearly statistics miss more often than stated. The cross-validated
  correction largely fixes this (V9, V11).
- 99% daily ranges are too narrow, because real errors are sometimes larger
  than the error model allows (V5, V11).
- A version without discharge does not suit a river like the Rhône (V5, V9,
  V10). Its ranges also do not hold on the hottest days (V14).
- Calibrating on weekly or monthly means needs bounds that keep `a2` and `a3`
  at 0 or above (V4).

It does not show that the model suits your river. Check that with your own
data: calibrate on some years and test on others, as V5 does and as the
examples show.
