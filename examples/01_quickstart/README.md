# 01 Quickstart: calibrate and test a model

**Question:** can air2stream reproduce the water temperature of the Mentue, a
small Swiss river, from air temperature and discharge? And can it predict
years it was not fitted to?

**Data:** daily air temperature, water temperature and discharge, 2002–2012
([`data/switzerland/`](../../data/switzerland/README.md)). The model is
calibrated (fitted) on 2002–2009 and tested on 2010–2012.

## Run it

From the repository's top folder, in Python:

```python
import pyair2stream

result = pyair2stream.run("examples/01_quickstart/config.yaml")
result.scores["validation"]       # the validation scores: NSE 0.982, RMSE 0.78 °C, ...
result.parameters                 # the fitted parameters a1 ... a8
result.output_dir                 # where the files were written
```

From a terminal, `pyair2stream --config examples/01_quickstart/config.yaml`
does exactly the same. [`run.py`](run.py) runs it and refreshes the figure
below.

The paths in the settings file are relative to the folder you run from (here
the repository's top folder). The run takes under a minute. Near the end, it
prints:

```
DE Finished. Best internal negated objective: -0.987594
L-BFGS-B Finished. Best internal negated objective: -0.987927
Efficiency Index in calibration 0.9879266370887719
Consistency check passed.
```

The settings file, [`config.yaml`](config.yaml), sets:

- the model version: 8, the full model;
- the calibration method: `DE` (Differential Evolution);
- `random_seed: 42`, so you get exactly these numbers.

## Read the results

Everything is written to `examples/01_quickstart/output/`. Start with
`summary.md` (or `summary.html`, the same page for a web browser): the
settings, the data used, the scores, the fitted parameters, every warning, the
figures, and what each other file is.

| | Calibration (2002–2009) | Validation (2010–2012) |
|---|---|---|
| NSE (1 = perfect) | 0.988 | 0.982 |
| RMSE (typical daily error) | 0.63 °C | 0.78 °C |
| Mean absolute error | 0.49 °C | 0.55 °C |

These come from `goodness_of_fit_calibration_*.csv` and
`goodness_of_fit_validation_*.csv`, and are in `result.scores`. **The validation score is the one that
matters**, because the model has not seen those years. It is a little worse
than the calibration score, as expected.

![Observed and simulated water temperature, 2010-2012](figures/validation.png)

*`validation_DE_NSE_Mentue.png`: measured (black) and simulated (orange) water
temperature in the validation years, with air temperature (blue) and
discharge (grey). The residuals (simulated minus measured)
are below. Look for long runs of residuals on one side of zero: they show
periods the model gets wrong. Here that is early 2012, after the river froze.*

Also look at `bias_by_month_validation_DE_NSE_Mentue.png`. It shows the average
error in each month, with a 95% interval. If an interval does not include
zero, the model is too warm or too cool in that month.

The fitted parameters are on the first line of `1_DE_NSE_Mentue_c_1d.out`. They
are also in `calibration_metadata.json`, which later runs reuse (examples
02–04). USER_GUIDE [§8](../../USER_GUIDE.md#8-understanding-the-output-files)
explains every file.

## What it means, and what it does not

- It means: with measured air temperature and discharge, this calibration
  predicts the Mentue's daily mean water temperature in years it was not
  fitted to with a typical error of about 0.8 °C.
- It does not give a range for a single prediction: a best fit has no error
  bars (example [02](../02_uncertainty/README.md) adds them).
- It does not show that the model suits conditions outside 2002–2012, or that
  every year is predicted equally well (example
  [06](../06_cross_validation/README.md) tests each year in turn).

## Use your own data

1. Copy `config.yaml`.
2. Point `paths` at your files. The format is in USER_GUIDE
   [§5](../../USER_GUIDE.md#5-preparing-your-own-data).
3. Change the names.
4. Run `pyair2stream.run("your_config.yaml")`. USER_GUIDE
   [§7.2](../../USER_GUIDE.md#72-from-python) lists what the result holds.

Keep some years back for validation.

## Next

A single best fit has no error bars. Example [02](../02_uncertainty/README.md)
adds them.
