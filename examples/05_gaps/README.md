# 05 Gaps: what to do with missing data

**Question:** my records have gaps. What should I do?

It depends on which record has them.

## Missing water temperature: nothing to do

The model does not need water temperature to run, only to be scored. Days
without a measurement are simply left out of the score. Leave them empty in
your file (or write `-999`). The validation suite removed up to half the days
at random, whole winters, a whole year, and all but one day a week. The
calibrated model still predicted other years to within 0.06 °C of the truth
([V7](../../validation/REPORT.md#v7)).

## Missing air temperature or discharge: fill or split

The model needs air temperature (and, for versions 4, 7 and 8, discharge) on
every day to run. By default a gap stops the run:

```
ValueError: The series of observed air temperature in ... must be complete. It cannot have gaps or missing data.
```

There are two ways forward.

1. **Fill the gaps**, preferably from a nearby station, or for short gaps by
   straight-line interpolation. Every day is kept.
2. **`gap_tolerant: true`** splits the record at each gap and restarts the
   model after it. The first `warmup_drop_days` (default 15) of each piece are
   not scored, because the model needs time to forget its approximate restart.
   Pieces shorter than `min_segment_days` (default 30) are dropped.
   USER_GUIDE [§10](../../USER_GUIDE.md#10-gap-tolerant-mode) has the details.

## This example

[`run.py`](run.py) removes a three-week gap (July 2005) and ten single days of
air temperature from the Mentue's 2002–2009 record, then calibrates both ways
and tests each on 2010–2012:

```bash
python examples/05_gaps/run.py
```

| Approach | Days scored in calibration | Validation NSE | Validation RMSE |
|---|---|---|---|
| Gaps filled by interpolation ([`filled.yaml`](filled.yaml)) | 2,907 | 0.982 | 0.78 °C |
| Gap-tolerant mode ([`gap_tolerant.yaml`](gap_tolerant.yaml)) | 2,711 | 0.982 | 0.78 °C |

Both give the same predictions. Gap-tolerant mode split the record into 12
pieces (`output/gap_tolerant/gaps_summary.txt`) and lost 196 scored days.

## Which to choose

- **Short, scattered gaps: fill them.** Gap-tolerant mode discards 15 days after
  every gap and drops pieces under 30 days. With gaps every few weeks, it can
  throw away most of the record: in the validation suite, 5% of days missing at
  random left only a third of the days scored ([V7](../../validation/REPORT.md#v7)).
- **Long gaps: use gap-tolerant mode**, or fill from a nearby station.
  Interpolating across weeks invents weather that did not happen.
- Either way, **say what you did**, and how many days were filled or dropped.

## Next

Example [06](../06_cross_validation/README.md) checks how stable the model is
from year to year.
