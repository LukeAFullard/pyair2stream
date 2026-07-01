# Gap Analysis Experiment Results

This report details the stability of parameter values when gaps are introduced into the `T_air` forcing data.

## Method
A baseline Differential Evolution (DE) optimization was run (3000 iterations, 100 particles) using the complete DAV dataset from Switzerland. Various types of gaps were then systematically introduced to the `T_air` column (`NaN` injection), and the DE calibration was repeated to observe how equifinality and goodness-of-fit reacted to missing data.

## Results

| Scenario | Missing T_air (%) | NSE | R2 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline** | 0.00% | 0.9558 | 0.9558 | 4.794 | 0.629 | 1.410 | 0.270 | 0.000 | 4.912 | 0.582 | 0.637 |
| **few_short** | 0.59% | 0.9578 | 0.9559 | 4.755 | 0.640 | 1.404 | 0.277 | 0.000 | 4.918 | 0.581 | 0.648 |
| **many_short** | 2.35% | 0.9632 | 0.9577 | 4.724 | 0.622 | 1.364 | 0.302 | 0.000 | 5.290 | 0.581 | 0.695 |
| **few_long** | 2.35% | 0.9576 | 0.9556 | 4.819 | 0.633 | 1.418 | 0.265 | 0.000 | 4.805 | 0.584 | 0.626 |
| **many_long** | 5.87% | 0.9587 | 0.9568 | 4.764 | 0.630 | 1.406 | 0.273 | 0.000 | 4.959 | 0.582 | 0.642 |
| **random** | 4.97% | 0.9859 | 0.9596 | 5.077 | 0.674 | 1.476 | 0.242 | 0.049 | 4.280 | 0.590 | 0.591 |

## Pre-analysis Timelines

### baseline
![baseline Pre-analysis](output/baseline_pre_analysis.png)

### few_short
![few_short Pre-analysis](output/few_short_pre_analysis.png)

### many_short
![many_short Pre-analysis](output/many_short_pre_analysis.png)

### few_long
![few_long Pre-analysis](output/few_long_pre_analysis.png)

### many_long
![many_long Pre-analysis](output/many_long_pre_analysis.png)

### random
![random Pre-analysis](output/random_pre_analysis.png)
