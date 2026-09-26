# Examples

Worked examples on real data: daily air temperature, water temperature and
discharge of the Mentue, a small Swiss river
([`data/switzerland/`](../data/switzerland/README.md)). Each builds on the one
before. Run them from the repository's top folder. Each `run.py` runs its
example's steps and redraws the figures its README shows.

| Example | Question | Time |
|---|---|---|
| [01 Quickstart](01_quickstart/README.md) | Can the model reproduce this river, including years it was not calibrated on? | < 1 min |
| [02 Uncertainty](02_uncertainty/README.md) | What range of temperatures should we expect, and does the range hold? | 1 min |
| [03 Compliance](03_compliance/README.md) | How likely is it that a temperature limit was exceeded? | 1 min |
| [04 Scenario](04_scenario/README.md) | What difference would abstracting 30% of the flow make? | 1 min |
| [05 Gaps](05_gaps/README.md) | What to do with missing data | 1 min |
| [06 Cross-validation](06_cross_validation/README.md) | Does the model predict every year well? How firmly do the data fix the parameters? Which version to use? | 2 min |

Outputs go to each example's `output/` folder, which is not kept in the
repository. All runs are seeded, so you should get the numbers shown in the
READMEs (up to the last digit on a different computer or library version).
