# Examples

These worked examples use real data: daily air temperature, water temperature
and discharge of the Mentue, a small Swiss river
([`data/switzerland/`](../data/switzerland/README.md)). Examples 01 to 03 build
on each other, so start with them. The others can be read in any order.

Run them from the repository's top folder. Each example's `run.py` runs all of
its steps and redraws the figures its README shows.

| Example | Question | Time |
|---|---|---|
| [01 Quickstart](01_quickstart/README.md) | Can the model reproduce this river, also in years it was not fitted to? | under 1 min |
| [02 Uncertainty](02_uncertainty/README.md) | What range of temperatures should we expect, and does that range hold? | about 2 min |
| [03 Compliance](03_compliance/README.md) | How likely is it that a temperature limit was exceeded? Do such probabilities hold in years the model was not fitted to? | about 3 min |
| [04 Scenario](04_scenario/README.md) | What difference would taking 30% of the flow make? | 2–3 min |
| [05 Gaps](05_gaps/README.md) | What should I do about missing water temperature, air temperature or discharge? When does gap-tolerant mode work? | about 2 min (the gap study: about 10 min) |
| [06 Cross-validation](06_cross_validation/README.md) | Does the model predict every year well? How firmly do the data fix the parameters? Which version should I use? Which parameters matter most? | about 2 min |
| [07 Preparing data](07_preparing_data/README.md) | How do I turn raw logger files into checked daily input files? | under 1 min |
| [08 Climate](08_climate/README.md) | How much warmer would the river be in a warmer climate, and how often would a limit be exceeded? | about 4 min |

**Which example for which task:**

| Task | Examples |
|---|---|
| Prepare and check your data | 07, then 05 for gaps |
| Decide between filling gaps and gap-tolerant mode | 05 |
| Fit the model and check it predicts well | 01, 06 |
| Fill gaps in a water temperature record | 05 (and 02 for a range) |
| Give a prediction with an uncertainty range | 02 |
| Decide whether a temperature limit was exceeded | 03 |
| Compare a change in flow with the present | 04 |
| Project a warmer climate | 08 |
| Choose a model version; see which parameters matter | 06 |

The times are for a 4-core computer. Outputs go to each example's `output/`
folder, which is not kept in the repository. Every run is seeded, so you should
get the numbers shown in the READMEs. On a different computer or library
version, the last digit can differ.
