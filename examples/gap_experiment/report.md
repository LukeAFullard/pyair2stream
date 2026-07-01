# Gap Analysis Experiment Results

This report details the stability of parameter values when gaps are introduced into the `T_air` forcing data.

## Method
A baseline Differential Evolution (DE) optimization was run (3000 iterations, 100 particles) using the complete DAV dataset from Switzerland. Various types of gaps were then systematically introduced to the `T_air` column (`NaN` injection), and the DE calibration was repeated to observe how equifinality and goodness-of-fit reacted to missing data.

## Results

| Scenario | Missing T_air (%) | NSE | R2 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline** | 0.00% | 0.9558 | 0.9558 | 4.794 | 0.629 | 1.410 | 0.270 | 0.000 | 4.912 | 0.582 | 0.637 |
| **few_short** | 0.59% | 0.9570 | 0.9563 | 4.751 | 0.646 | 1.425 | 0.268 | 0.000 | 4.824 | 0.582 | 0.631 |
| **many_short** | 2.35% | 0.9628 | 0.9574 | 4.870 | 0.604 | 1.378 | 0.284 | 0.000 | 5.105 | 0.582 | 0.665 |
| **few_long** | 2.35% | 0.9603 | 0.9594 | 4.814 | 0.604 | 1.377 | 0.278 | 0.000 | 5.068 | 0.584 | 0.654 |
| **many_long** | 5.87% | 0.9605 | 0.9560 | 4.692 | 0.618 | 1.370 | 0.288 | 0.000 | 5.122 | 0.583 | 0.670 |
| **random** | 19.98% | 1.0000 | 0.7597 | 1.753 | 0.323 | 1.140 | -0.549 | 7.945 | 8.121 | 0.205 | 3.654 |
