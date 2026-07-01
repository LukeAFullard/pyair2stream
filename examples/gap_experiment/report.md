# Gap Analysis Experiment Results

This report details the stability of parameter values when gaps are introduced into the `T_air` forcing data.

## Method
A baseline Differential Evolution (DE) optimization was run (3000 iterations, 100 particles) using the complete DAV dataset from Switzerland. Various types of gaps were then systematically introduced to the `T_air` column (`NaN` injection), and the DE calibration was repeated to observe how equifinality and goodness-of-fit reacted to missing data.

## Results

| Scenario | Missing T_air (%) | NSE | R2 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline** | 0.00% | 0.9558 | 0.9558 | 4.794 | 0.629 | 1.410 | 0.270 | 0.000 | 4.912 | 0.582 | 0.637 |
| **few_short** | 0.59% | 0.9565 | 0.9556 | 4.780 | 0.623 | 1.395 | 0.274 | 0.000 | 4.979 | 0.581 | 0.647 |
| **many_short** | 2.35% | 0.9631 | 0.9576 | 5.014 | 0.615 | 1.429 | 0.269 | 0.000 | 5.340 | 0.578 | 0.672 |
| **few_long** | 2.35% | 0.9581 | 0.9576 | 4.983 | 0.626 | 1.450 | 0.259 | 0.000 | 5.293 | 0.580 | 0.657 |
| **many_long** | 5.87% | 0.9592 | 0.9532 | 4.809 | 0.613 | 1.397 | 0.274 | 0.000 | 5.080 | 0.581 | 0.647 |
| **random** | 4.97% | 0.9851 | 0.9604 | 4.558 | 0.595 | 1.289 | 0.287 | 0.000 | 4.916 | 0.580 | 0.677 |

## Pre-analysis Timelines

### baseline
![baseline Pre-analysis](examples/gap_experiment/output/baseline_pre_analysis.png)

### few_short
![few_short Pre-analysis](examples/gap_experiment/output/few_short_pre_analysis.png)

### many_short
![many_short Pre-analysis](examples/gap_experiment/output/many_short_pre_analysis.png)

### few_long
![few_long Pre-analysis](examples/gap_experiment/output/few_long_pre_analysis.png)

### many_long
![many_long Pre-analysis](examples/gap_experiment/output/many_long_pre_analysis.png)

### random
![random Pre-analysis](examples/gap_experiment/output/random_pre_analysis.png)
