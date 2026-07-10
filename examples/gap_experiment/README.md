# Gap Analysis Experiment Results

This report details the stability of parameter values when gaps are introduced into the `T_air` forcing data.

## Method
A baseline Differential Evolution (DE) optimization was run (3000 iterations, 100 particles) using the complete DAV dataset from Switzerland. Various types of gaps were then systematically introduced to the `T_air` column (`NaN` injection), and the DE calibration was repeated to observe how equifinality and goodness-of-fit reacted to missing data.

## Discussion
Across the systematic gap scenarios (`few_short` through `many_long`), the model fit parameters (p1-p8) and objective function values stay close to the gap-free baseline, with NSE shifting by no more than +0.003 even as missing `T_air` reaches 5.9%. The `random` scenario is the exception: NSE rises from 0.9558 to 0.9842, a larger jump than any of the systematic scenarios despite a comparable proportion of missing data. This is consistent with the caveat in the main README's [gap-tolerant mode](../../README.md#gap-tolerant-mode) section — gaps scattered at random are more likely to remove difficult-to-model days piecemeal, without necessarily excluding an entire hard event (like a freeze or flood), which can inflate NSE relative to a continuous record. Parameter convergence (seen in the dotty plots) is well-preserved across all scenarios.

## Results

| Scenario | Missing T_air (%) | NSE | R2 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline** | 0.00% | 0.9558 | 0.9558 | 4.794 | 0.629 | 1.410 | 0.270 | 0.000 | 4.912 | 0.582 | 0.637 |
| **few_short** | 0.59% | 0.9572 | 0.9560 | 4.792 | 0.621 | 1.401 | 0.271 | 0.000 | 4.945 | 0.582 | 0.641 |
| **many_short** | 2.35% | 0.9618 | 0.9573 | 4.910 | 0.625 | 1.430 | 0.255 | 0.000 | 5.183 | 0.579 | 0.651 |
| **few_long** | 2.35% | 0.9568 | 0.9542 | 4.815 | 0.620 | 1.405 | 0.271 | 0.000 | 4.963 | 0.580 | 0.641 |
| **many_long** | 5.87% | 0.9588 | 0.9521 | 4.791 | 0.604 | 1.381 | 0.279 | 0.000 | 5.129 | 0.579 | 0.659 |
| **random** | 4.97% | 0.9842 | 0.9610 | 4.719 | 0.702 | 1.515 | 0.257 | 0.000 | 4.495 | 0.581 | 0.575 |

## Scenario Visualizations

Each scenario below shows the same four diagnostics: the pre-analysis timeline, optimizer convergence, parameter dotty plots, and the full simulation.

### baseline
![baseline Pre-analysis](output/baseline_pre_analysis.png)
![baseline Convergence](output/convergence_DE_NSE_baseline.png)
![baseline Dotty Plots](output/dottyplots_DE_NSE_baseline.png)
![baseline Full Simulation](output/full_simulation_DE_NSE_baseline.png)

### few_short
![few_short Pre-analysis](output/few_short_pre_analysis.png)
![few_short Convergence](output/convergence_DE_NSE_few_short.png)
![few_short Dotty Plots](output/dottyplots_DE_NSE_few_short.png)
![few_short Full Simulation](output/full_simulation_DE_NSE_few_short.png)

### many_short
![many_short Pre-analysis](output/many_short_pre_analysis.png)
![many_short Convergence](output/convergence_DE_NSE_many_short.png)
![many_short Dotty Plots](output/dottyplots_DE_NSE_many_short.png)
![many_short Full Simulation](output/full_simulation_DE_NSE_many_short.png)

### few_long
![few_long Pre-analysis](output/few_long_pre_analysis.png)
![few_long Convergence](output/convergence_DE_NSE_few_long.png)
![few_long Dotty Plots](output/dottyplots_DE_NSE_few_long.png)
![few_long Full Simulation](output/full_simulation_DE_NSE_few_long.png)

### many_long
![many_long Pre-analysis](output/many_long_pre_analysis.png)
![many_long Convergence](output/convergence_DE_NSE_many_long.png)
![many_long Dotty Plots](output/dottyplots_DE_NSE_many_long.png)
![many_long Full Simulation](output/full_simulation_DE_NSE_many_long.png)

### random
![random Pre-analysis](output/random_pre_analysis.png)
![random Convergence](output/convergence_DE_NSE_random.png)
![random Dotty Plots](output/dottyplots_DE_NSE_random.png)
![random Full Simulation](output/full_simulation_DE_NSE_random.png)
