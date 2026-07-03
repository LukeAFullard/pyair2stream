# Gap Analysis Experiment Results

This report details the stability of parameter values when gaps are introduced into the `T_air` forcing data.

## Method
A baseline Differential Evolution (DE) optimization was run (3000 iterations, 100 particles) using the complete DAV dataset from Switzerland. Various types of gaps were then systematically introduced to the `T_air` column (`NaN` injection), and the DE calibration was repeated to observe how equifinality and goodness-of-fit reacted to missing data.

## Discussion
The results show that the pyair2stream model exhibits remarkable robustness to missing forcing data. When gaps are introduced (ranging from short random bursts to extended absences), the objective function values (NSE, R2) and model fit parameters (p1-p8) remain relatively stable compared to the baseline without missing values. The equifinality (seen in the dotty plots) and parameter convergence are well-preserved across the gap scenarios.

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

### baseline
These plots show the pre-analysis timeline, optimizer convergence, parameter dotty plots, and the full simulation.

![baseline Pre-analysis](output/baseline_pre_analysis.png)
![baseline Convergence](output/convergence_DE_NSE_baseline.png)
![baseline Dotty Plots](output/dottyplots_DE_NSE_baseline.png)
![baseline Full Simulation](output/full_simulation_DE_NSE_baseline.png)

### few_short
These plots show the pre-analysis timeline, optimizer convergence, parameter dotty plots, and the full simulation.

![few_short Pre-analysis](output/few_short_pre_analysis.png)
![few_short Convergence](output/convergence_DE_NSE_few_short.png)
![few_short Dotty Plots](output/dottyplots_DE_NSE_few_short.png)
![few_short Full Simulation](output/full_simulation_DE_NSE_few_short.png)

### many_short
These plots show the pre-analysis timeline, optimizer convergence, parameter dotty plots, and the full simulation.

![many_short Pre-analysis](output/many_short_pre_analysis.png)
![many_short Convergence](output/convergence_DE_NSE_many_short.png)
![many_short Dotty Plots](output/dottyplots_DE_NSE_many_short.png)
![many_short Full Simulation](output/full_simulation_DE_NSE_many_short.png)

### few_long
These plots show the pre-analysis timeline, optimizer convergence, parameter dotty plots, and the full simulation.

![few_long Pre-analysis](output/few_long_pre_analysis.png)
![few_long Convergence](output/convergence_DE_NSE_few_long.png)
![few_long Dotty Plots](output/dottyplots_DE_NSE_few_long.png)
![few_long Full Simulation](output/full_simulation_DE_NSE_few_long.png)

### many_long
These plots show the pre-analysis timeline, optimizer convergence, parameter dotty plots, and the full simulation.

![many_long Pre-analysis](output/many_long_pre_analysis.png)
![many_long Convergence](output/convergence_DE_NSE_many_long.png)
![many_long Dotty Plots](output/dottyplots_DE_NSE_many_long.png)
![many_long Full Simulation](output/full_simulation_DE_NSE_many_long.png)

### random
These plots show the pre-analysis timeline, optimizer convergence, parameter dotty plots, and the full simulation.

![random Pre-analysis](output/random_pre_analysis.png)
![random Convergence](output/convergence_DE_NSE_random.png)
![random Dotty Plots](output/dottyplots_DE_NSE_random.png)
![random Full Simulation](output/full_simulation_DE_NSE_random.png)
