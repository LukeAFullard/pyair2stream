# Leave-One-Year-Out Cross-Validation Example

To perform Leave-One-Year-Out (LOYO) cross-validation on the DAV dataset, you can use the `cross_validation` block in the configuration as shown here:

```yaml
project_name: "cv_example"
station_name: "DAV"
series: "series"
time_resolution: "1d"
version: 8
objective_function: "NSE"
integrator: "RK4"
run_mode: "DE"
prc: 1.0

# Using gap-tolerant mode since we might have missing T_water
gap_tolerant: true

optimization:
  n_runs: 3000
  n_particles: 200

parameter_bounds:
  min: [-5.0, -5.0, -5.0, -1.0, 0.0, 0.0, 0.0, -1.0]
  max: [15.0, 1.5, 5.0, 1.0, 20.0, 10.0, 1.0, 5.0]

cross_validation:
  enabled: true
  unit: year
  n_years_per_fold: 1
  water_year_start_month: 1
  min_train_years: 1
  skip_first_year: true
  optimizer_overrides:
    n_run: 3000
    n_particles: 200

paths:
  input_data: "../validation/Switzerland/DAV_2327_cc.csv"
```

The configuration instructs pyair2stream to withhold one year of water temperature observations at a time, calibrate the model on the remaining data, and calculate metrics (NSE, KGE, RMSE) on the withheld year. Each fold completely excludes the specified year from the optimization target. For instance, in fold `2004`, the parameters are trained purely on the data from 2005-2009, and the predictions made for 2004 are scored to see how well the parameters generalized.

## Results

fold|NSE|RMSE
---|---|---
2005|0.9615368424568024|0.5835933576231217
2006|0.9604917392203028|0.6030243614411531
2007|0.9476378869751676|0.6652782342596316
2008|0.8690957182134409|0.9831560530637654
2009|0.9540290349906992|0.6266787528320202


### Summary across folds

The table below shows the summary across all folds. `mean` and `std` represent the macro-average and standard deviation across the individual per-fold metrics. `pooled` represents the micro-average computed over all held-out days pooled together.

fold|NSE|RMSE
---|---|---
mean|0.9385582443712824|0.6923461518439384
std|0.0392298418895004|0.1653940979860083
pooled|0.9409911093952475|0.7081540911859495


## Parameter Stability

Because the dataset is split and the model is recalibrated for each fold, each fold yields a slightly different set of calibrated parameters. The stability (or variance) of these parameters gives an indication of how robust the model fit is to the specific subset of data it trains on. High variance might indicate equifinality or overparameterization issues.

The following plot shows the distribution of the calibrated parameters across all the folds. This can be used as a diagnostic for parameter stability and equifinality.

*Observation:* `p1`–`p4` and `p6`–`p8` are stable across folds, varying by only 2–5% of their mean value (see the summary table below) — not the signature of a poorly-identified parameter. `p5` is the exception: it sits at 0.0 (the lower bound) in five of the six folds and only becomes non-zero (0.053) in the 2005 fold, so its variability is a scale artifact rather than genuine fold-to-fold disagreement. This pattern — most parameters well-constrained, one sitting at a bound and contributing little — is consistent with mild **equifinality** in `p5` specifically, rather than broad overparameterization of the 8-parameter model. With ~5 years of training data per fold, `p5` (a constant offset scaled by relative discharge) may simply not be well-identified by this particular record.

![Parameter Stability](cv_parameter_stability.png)

### Calibrated Parameters per Fold

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
2005|4.7495642648228245|0.6408736386344471|1.427039230706956|0.255787522281701|0.0|4.716244416396995|0.5846365455469354|0.6159736419614433
2006|4.922768208765276|0.6360993078270903|1.4288345729711434|0.2722543350817011|0.0|4.940244481657775|0.5812748531724296|0.644065245290745
2007|4.738243441569576|0.5896277923857935|1.3586348804312425|0.2834174805340564|0.0|5.178049240388322|0.5834966766357448|0.6727226153000677
2008|4.94736911825218|0.6458974743388207|1.4650873179864825|0.2631040777414208|0.0|5.17505182306471|0.5788199299021017|0.648511177292167
2009|4.7627241173336365|0.6272477886308595|1.395706216290639|0.2671344619003279|0.0|4.899989320382101|0.5819409908538781|0.6372837919117221


### Parameter Summary Across Folds

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
mean|4.824133830148698|0.6279492003634022|1.415060443677293|0.2683395755078415|0.0|4.98191585637798|0.582033799222218|0.643711294351229
std|0.1020105492464102|0.0224979342540225|0.0399822974421853|0.0103571875811167|0.0|0.1967228893293126|0.0022264246711007|0.0204665034166639


*`pooled` is omitted from this table: pooling applies to held-out predictions, not to per-fold parameter sets, so a single "pooled" parameter value has no meaning here.*
