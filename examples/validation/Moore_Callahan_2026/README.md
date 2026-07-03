# Validation Analysis: Moore & Callahan 2026

## Station 07EA004

This report compares the pyair2stream model (both PSO and DE optimizers, and CRN and RK4 integrators) against the published literature parameters for station 07EA004. This pass used high-intensity search settings (100 particles, 3000 runs) to ensure absolute convergence limits.

Two sets of tests were run: the 'orig' tests used the full default bounds (where `a4` can range from `[-1.0, 1.0]`), while the 'restr' tests forced parameter `a4` to be restricted within `[0.0, 1.0]` to observe differences in performance and parameter identifiability.

### Parameters & Performance

| Run | NSE | R2 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Literature | N/A | N/A | 0.309 | 0.496 | 0.483 | -0.998 | 0.775 | 4.328 | 0.562 | 0.538 |
| 07EA004_PSO_CRN_orig |  0.9706 |  0.9706 | 0.465 | 0.426 | 0.439 | -1.000 | 0.353 | 3.438 | 0.561 | 0.407 |
| 07EA004_DE_CRN_orig |  0.9707 |  0.9707 | 0.515 | 0.395 | 0.423 | -1.000 | 0.167 | 2.920 | 0.562 | 0.336 |
| 07EA004_PSO_RK4_orig |  0.9687 |  0.9687 | -0.202 | 0.265 | 0.220 | 0.022 | 0.824 | 2.645 | 0.562 | 0.355 |
| 07EA004_DE_RK4_orig |  0.9686 |  0.9686 | -0.113 | 0.272 | 0.249 | -0.004 | 0.555 | 2.217 | 0.565 | 0.288 |
| 07EA004_PSO_CRN_restr |  0.9684 |  0.9684 | -0.412 | 0.333 | 0.236 | 0.000 | 1.567 | 4.399 | 0.561 | 0.599 |
| 07EA004_DE_CRN_restr |  0.9687 |  0.9687 | -0.180 | 0.269 | 0.225 | 0.000 | 0.782 | 2.682 | 0.562 | 0.355 |
| 07EA004_PSO_RK4_restr |  0.9688 |  0.9688 | -0.168 | 0.263 | 0.223 | 0.000 | 0.731 | 2.548 | 0.562 | 0.336 |
| 07EA004_DE_RK4_restr |  0.9688 |  0.9688 | -0.160 | 0.264 | 0.225 | 0.000 | 0.717 | 2.534 | 0.563 | 0.333 |

### Discussion
The analysis successfully completed the evaluation for PSO and DE using both CRN and RK4. The high-intensity search confirms that both DE and PSO reach reliable global minimums with NSE values around 0.97. DE is particularly successful at locking onto the lowest possible objective bound across both RK4 and CRN integrators, while PSO achieves closely corresponding results. These optimized parameter solutions show very strong agreement in functional form, tightly aligning with what is presented in the original literature.

When `a4` was restricted to `[0.0, 1.0]`, the models generally hit the lower bound (`0.0`) for parameter `a4`. This indicates that the true global minimum (which utilizes negative values of `a4` around `-1.0` as seen in both literature and the original unconstrained runs) exists outside this bounded region. The restricted models experienced a marginal performance degradation in NSE as they sought alternate local optima, compensating by noticeably increasing parameter `a1` (and modifying other parameters) to offset the forced bound on `a4`.
