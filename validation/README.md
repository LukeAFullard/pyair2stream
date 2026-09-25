# Validation

This folder holds the evidence that pyair2stream is correct and that its results
can be relied on. The results are in [REPORT.md](REPORT.md).

Every check runs the package the way a user would, through its configuration
files and run modes. Each check states its question, method and pass criterion
before its result.

## What is checked

| Check | Question | Why it matters |
|---|---|---|
| V1 | Does it compute the same temperatures as the original Fortran program, on real, variable inputs? | Shows the model equations and solution schemes were translated correctly. |
| V2 | Given the published parameters, does it reproduce the published model errors for every model version on three Swiss rivers? Does its own calibration fit at least as well, and find the published parameters? | Ties the package to the peer-reviewed results of Piccolroaz et al. (2016). |
| V3 | When data are made by the model itself from known parameters, does calibration recover them well enough to predict other years? | Only with a known truth can we tell whether calibration finds the right answer. |
| V4 | Do the 90% uncertainty intervals contain the truth about 90% of the time? | An interval is only useful if its stated confidence is honest. |
| V5 | On real rivers, how well does a calibrated model predict years it has not seen, compared with simple alternatives? Do its intervals contain about 90% of real measurements? | Tests the whole approach on real data, not only the code. |
| V6 | How accurate are the numerical schemes, does the package stop runs that go wrong, and does the choice of stable scheme matter? | Rules out numerical error as a source of wrong answers. |
| V7 | Do gaps in the water or air temperature record bias the result? | Real records have gaps. |
| V8 | Does the documented workflow reproduce the calibration exactly? Do scenario comparisons and threshold counts give exact answers where the answer is known? | The tools used to reach a conclusion must be exact. |

## Running it

From the repository root, with the package installed (`pip install -e .`):

```bash
python validation/run_all.py            # full suite, about 70 minutes on 4 cores
python validation/run_all.py --quick    # reduced version of every check, about 2 minutes
python validation/run_all.py --only V2 V6
```

V1 needs `gfortran` and the Fortran source (`git submodule update --init`);
without them it is reported as not run. The run rewrites `REPORT.md`,
`results/` (every table as CSV) and `figures/`. Scratch files go to `work/`,
which is not kept. All random steps are seeded, so a rerun on the same software
versions gives the same numbers; the report records the versions used. The one
exception is V2 part D, which runs the original Fortran program's own
calibration: it seeds its random numbers from the clock, so its runs differ each
time (which is what part D shows).

## Data

All checks use the three Swiss rivers in [`data/switzerland/`](../data/switzerland/README.md),
with the published parameters and model errors of Piccolroaz et al. (2016).
That README gives their sources, periods and licence.

## What this does and does not show

It shows that the software computes what it claims to, that calibration finds
the right answer when one is known, and that the uncertainty intervals have the
coverage they state under the conditions tested. It also shows how well the
model does on three real rivers.

It does not show that the model suits your river. Check that with your own data:
calibrate on some years and test on others, as V5 does and as the examples show.
