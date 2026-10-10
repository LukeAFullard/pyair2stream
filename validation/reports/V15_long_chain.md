# V15 supplement: longer DE-MCMC run, station 08KH006

Same set-up as V15 part D (32 walkers, seed 1, least-squares likelihood widened for the autocorrelation of the errors), with at most 500,000 steps instead of 100,000 (`python validation/v15_long_chain.py 08KH006 --steps 500000`).

- Converged: **yes** after 127,000 steps (38,100 discarded as burn-in), 0.5 hours.
- Longest autocorrelation time: 605 steps (the rule needs at least 30,275 steps).
- Largest split-Rhat: 1.0097 (the rule needs below 1.01).
- Published values inside the 90% interval: **1 of 8**.

| station | parameter | published | median | MCMC 90% interval | published inside |
|---|---|---|---|---|---|
| 08KH006 | a1 | 1.022 | 0.373 | 0.183 to 0.740 | no |
| 08KH006 | a2 | 0.321 | 0.112 | 0.067 to 0.213 | no |
| 08KH006 | a3 | 0.33 | 0.129 | 0.080 to 0.231 | no |
| 08KH006 | a4 | -0.989 | -0.681 | -0.965 to -0.249 | no |
| 08KH006 | a5 | 4.296 | 0.529 | 0.093 to 2.133 | no |
| 08KH006 | a6 | 6.214 | 1.02 | 0.363 to 3.223 | no |
| 08KH006 | a7 | 0.628 | 0.645 | 0.627 to 0.672 | yes |
| 08KH006 | a8 | 0.727 | 0.109 | 0.033 to 0.373 | no |
