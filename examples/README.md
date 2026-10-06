# Examples

These worked examples use real data: daily air temperature, water temperature
and discharge of the Mentue, a small Swiss river
([`data/switzerland/`](../data/switzerland/README.md)). Each example builds on
the one before, so work through them in order.

Run them from the repository's top folder. Each example's `run.py` runs all of
its steps and redraws the figures its README shows.

| Example | Question | Time |
|---|---|---|
| [01 Quickstart](01_quickstart/README.md) | Can the model reproduce this river, also in years it was not fitted to? | under 1 min |
| [02 Uncertainty](02_uncertainty/README.md) | What range of temperatures should we expect, and does that range hold? | about 2 min |
| [03 Compliance](03_compliance/README.md) | How likely is it that a temperature limit was exceeded? Do such probabilities hold in years the model was not fitted to? | about 3 min |
| [04 Scenario](04_scenario/README.md) | What difference would taking 30% of the flow make? | 2–3 min |
| [05 Gaps](05_gaps/README.md) | What should I do about missing data? | about 1 min |
| [06 Cross-validation](06_cross_validation/README.md) | Does the model predict every year well? How firmly do the data fix the parameters? Which version should I use? | about 1.5 min |

The times are for a 4-core computer. Outputs go to each example's `output/`
folder, which is not kept in the repository. Every run is seeded, so you should
get the numbers shown in the READMEs. On a different computer or library
version, the last digit can differ.
