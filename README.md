# pyair2stream

[![License: CC BY-SA 3.0](https://img.shields.io/badge/License-CC_BY--SA_3.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

`pyair2stream` predicts the **daily mean water temperature of a river**. It
needs daily air temperature and, for some model versions, daily river flow
(discharge). It is a Python version of **air2stream** (Toffolon and
Piccolroaz, 2015), a widely used model. The model is one small equation, based
on the physics of a river's heat budget. You fit it to measurements from your
own site.

You can use it to:

- fill gaps in a water temperature record;
- see how a river's temperature would change with a different flow, for
  example with and without a water abstraction;
- project water temperature under a future climate;
- estimate how likely it is that a temperature limit was exceeded, with an
  uncertainty range.

> pyair2stream is a community port. The authors of the original model did not
> write it and do not maintain it. It gives the same results as the original
> Fortran program ([Is it correct?](#is-it-correct)) and adds new features.

## Install

You need Python 3.9 or newer.

```bash
git clone https://github.com/LukeAFullard/pyair2stream.git
cd pyair2stream
pip install .
pyair2stream --help        # check that it installed
```

## Try it

Run this from the repository's top folder:

```bash
pyair2stream --config examples/01_quickstart/config.yaml
```

It takes under a minute. It fits the model to 2002–2009 data from the Mentue, a
small Swiss river. Then it tests the model on 2010–2012, years the fit did not
use. The results and plots go to `examples/01_quickstart/output/`. The
[example's README](examples/01_quickstart/README.md) explains them.

## Use your own data

There are four steps: prepare the data, write a settings file, run, and read
the results.

### 1. Prepare your data

Make a CSV file with one row per day:

```csv
Date,T_air,T_water,Discharge
2020-01-01,5.2,4.1,12.5
2020-01-02,4.8,,11.8
2020-01-03,6.1,4.0,10.2
```

| Column | Needed? | What it holds |
|---|---|---|
| `Date` | always | the date, for example `2020-01-31` |
| `T_air` | always | daily mean air temperature (°C) |
| `T_water` | always | daily mean measured water temperature (°C); gaps are fine |
| `Discharge` | for versions 4, 7 and 8 | daily mean flow, in any unit; it must be above zero |

The rules:

- Give a row for **every** day. Leave a missing value blank, or write `-999`.
- `T_air` and `Discharge` must have no gaps. If yours have gaps, see
  [example 05](examples/05_gaps/README.md).
- The file must start on 1 January and cover at least a year. Several years
  are better.

Make two files. The model is fitted to the first, the **calibration** file. The
second, the **validation** file, holds a few other years. The model never sees
their water temperatures, so they show how well it predicts.

### 2. Write a settings file

The settings go in a YAML file, for example `config.yaml`:

```yaml
project_name: "my_river"
station_name: "Station_A"
version: 8                 # model version: 3, 4, 5, 7 or 8 (see below)
run_mode: "DE"             # fit the model (Differential Evolution)
random_seed: 42            # gives the same result every time

paths:
  input_data: "data/calibration.csv"
  validation_data: "data/validation.csv"
  output_dir: "output"

parameter_bounds:          # search ranges for the 8 parameters a1..a8
  min: [-5, -5, -5, -1, 0,  0,  0, -1]
  max: [15, 1.5, 5,  1, 20, 10, 1,  5]
```

The bounds above are the original authors' ranges and a good start. Any
setting you leave out has a safe default. The
[User Guide](USER_GUIDE.md#6-configuration-reference) lists every setting.

### 3. Run it

```bash
pyair2stream --config config.yaml
```

The paths in the settings file are relative to the folder you run this from.

### 4. Read the results

Start with these files in the output folder:

| File | What it tells you |
|---|---|
| `goodness_of_fit_validation_*.csv` | how well the model predicts the validation years. **This is the score that matters.** |
| `validation_*.png` | measured and simulated temperature over time. Look for long stretches where the model is too warm or too cool. |
| `bias_by_month_validation_*.png` | the average error in each month. Is the model off in one season? |
| `1_*.out` | the fitted parameters (line 1), then the calibration and validation scores |
| `calibration_metadata.json` | what later runs need: the parameters, the mean discharge and the settings |

How to judge the scores:

- **NSE**: 1 is perfect, and 0 is no better than always guessing the average.
  Above 0.9 is common for this model.
- **RMSE**: the typical daily error, in °C. In example 01 it is 0.63 °C in
  the calibration years and 0.78 °C in the validation years.
- The validation score is usually a little worse than the calibration score.
  If it is much worse, the model does not predict well.

[User Guide §8](USER_GUIDE.md#8-understanding-the-output-files) explains every
output file.

## Which model version?

| Version | Parameters | Uses discharge | Seasonal term |
|:-:|:-:|:-:|:-:|
| 3 | 3 | no | no |
| 4 | 4 | yes | no |
| 5 | 5 | no | yes |
| 7 | 7 | yes | yes |
| 8 | 8 | yes | yes |

Start with version 8 if you have good discharge data, or version 5 if not.
Then try simpler versions. Choose the simplest one that predicts the
validation years well. If the river's summer temperature depends strongly on
its flow, use version 7 or 8. [docs/METHODS.md](docs/METHODS.md#5-the-equation-and-model-versions)
gives the equation.

## Going further

Six worked examples use real data from a Swiss river. Each has a README that
explains the steps and the results ([examples/README.md](examples/README.md)).

| Example | Question |
|---|---|
| [01 Quickstart](examples/01_quickstart/README.md) | Does the model reproduce this river, also in years it was not fitted to? |
| [02 Uncertainty](examples/02_uncertainty/README.md) | What range of temperatures should we expect, and does that range hold? |
| [03 Compliance](examples/03_compliance/README.md) | How likely is it that a temperature limit was exceeded? |
| [04 Scenario](examples/04_scenario/README.md) | What difference would taking 30% of the flow make? |
| [05 Gaps](examples/05_gaps/README.md) | What should I do about missing data? |
| [06 Cross-validation](examples/06_cross_validation/README.md) | Does the model predict every year well? Which version should I use? |

**Was a temperature limit exceeded?** Examples 01 to 03 show the usual route:

1. Check that the model predicts your river well (example 01).
2. Measure its uncertainty with `run_mode: "DE-MCMC"` (example 02).
3. Simulate the period 1,000 times. The share of simulations above the limit
   is the probability that it was exceeded (example 03).
4. Check that answer on the years you calibrated on, and correct it for the
   model's error on the hottest days (cross-validation, example 03).
5. Report a probability with its range and the check, not a yes or no. Work
   through the checklist in [User Guide §14](USER_GUIDE.md#14-checklist-for-results-that-support-a-decision).

The model gives daily **means**. A limit on daily maximum temperature needs an
extra step, which you must justify.

## Documentation

| If you want to... | Read |
|---|---|
| prepare data, choose settings, run the model and read the results | [USER_GUIDE.md](USER_GUIDE.md) |
| follow a worked example | [examples/](examples/README.md) |
| understand the uncertainty ranges and probabilities | [docs/UNCERTAINTY.md](docs/UNCERTAINTY.md) |
| know exactly what the software computes, and its limits | [docs/METHODS.md](docs/METHODS.md) (read §16 before using results for a decision) |
| see the evidence that it works | [validation/REPORT.md](validation/REPORT.md) |
| see which published results it reproduces, and the errors found in them | [docs/PUBLISHED_RESULTS.md](docs/PUBLISHED_RESULTS.md) |
| see what changed between versions | [CHANGELOG.md](CHANGELOG.md) |

## Is it correct?

A [validation suite](validation/README.md) tests this. Its results are in
[validation/REPORT.md](validation/REPORT.md). In short:

- **Same results as the original program.** It computes the same water
  temperatures as the original Fortran program, to 5×10⁻⁶ °C, for every model
  version and solution method ([V1](validation/REPORT.md#v1)).
- **Reproduces published results.** The model's authors published results for
  three Swiss rivers, in 2015 and 2016. Given their parameters, pyair2stream
  reproduces all 30 published errors of each paper
  ([V2](validation/REPORT.md#v2), [V13](validation/REPORT.md#v13)). It also
  reproduces an independent group's simulations for 23 rivers in British
  Columbia, day by day. One of their 46 series matches only when a date error
  in their run is copied ([V15](validation/REPORT.md#v15)).
- **Finds errors in those publications.** Each of the three studies has an
  error or a missing detail. [docs/PUBLISHED_RESULTS.md](docs/PUBLISHED_RESULTS.md)
  lists them, with the evidence.
- **Finds a known answer.** On data made by the model from known parameters,
  the fitted model predicts other years to within 0.04 °C
  ([V3](validation/REPORT.md#v3)).
- **Honest uncertainty ranges.** On such data, 90% ranges contain the truth
  about 90% of the time ([V4](validation/REPORT.md#v4)). On real rivers, in
  years not used for fitting, they held on 85–89.6% of days. So they are
  slightly too narrow for new years ([V5](validation/REPORT.md#v5)).
- **Probabilities need the check.** The model can be too warm on the hottest
  days. So uncorrected ranges for yearly peaks held in only 73–92% of years.
  The cross-validation check and correction brought this to 85–94%
  ([V11](validation/REPORT.md#v11)).
- **Warmer and lower-flow years.** Fitted on the coolest years, the model
  predicted the warmest years almost as well: at most 0.07 °C worse. The same
  held when it was fitted on the highest-flow years and tested on the
  lowest-flow years ([V10](validation/REPORT.md#v10)).
- **Where it falls short.** Four checks do not meet all their criteria: V5,
  V9, V10 and V14. The main reasons are:
  - 95% and 99% ranges were too narrow in some years not used for fitting;
  - version 5 (no discharge) does poorly on the Rhône, a river whose summer
    temperature depends on its flow;
  - on the hottest days of the three Swiss rivers, version 5's 90% ranges held
    on only 84% of days. Version 8's held on 91%.

  The report gives the details.

To run the tests and the validation suite (V1 needs `gfortran`):

```bash
git submodule update --init --recursive
pip install -e . pytest
pytest tests/
python validation/run_all.py --quick     # about 2 minutes; the full suite takes about 2 hours
```

## Differences from the original Fortran

With the same settings, the results match the Fortran. The main differences:

- safer defaults: the stable `CRN` solution method and the DE optimiser;
- clear errors instead of silently wrong numbers, for example for missing days
  or zero flow;
- the mean discharge (`Qmedia`) of the calibration is kept for validation and
  scenario runs, instead of being recomputed;
- new features: uncertainty, scenarios, gaps, cross-validation and more.

The full list is in [docs/METHODS.md §17](docs/METHODS.md#17-differences-from-the-fortran-original).

## Citing

Please cite the original model:

> Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water
> temperature as a function of air temperature and discharge. *Environmental
> Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011

Also record the pyair2stream version you used. There are no tagged releases
yet, so give the git commit (`git rev-parse HEAD`), for example:

> Water temperatures were simulated with pyair2stream
> (https://github.com/LukeAFullard/pyair2stream, commit `<sha>`).

## Help

To report a problem or ask a question, open an issue on
[GitHub](https://github.com/LukeAFullard/pyair2stream/issues). Please include
your settings file, the full error message and the pyair2stream commit.

## License

[CC BY-SA 3.0](LICENSE), the license of the original air2stream code.
