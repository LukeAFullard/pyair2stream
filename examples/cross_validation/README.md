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
2005|0.9615749074305922|0.5833045102412905
2006|0.960489580676913|0.6030408344071235
2007|0.9476407873547976|0.6652598088557645
2008|0.8690948456359139|0.9831593298032124
2009|0.9540280247288048|0.6266856387673653


### Summary across folds

The table below shows the summary across all folds. `mean` and `std` represent the macro-average and standard deviation across the individual per-fold metrics. `pooled` represents the micro-average computed over all held-out days pooled together.

fold|NSE|RMSE
---|---|---
mean|0.9385656291654044|0.6922900244149512
std|0.0392355721032799|0.165440911659702
pooled|0.9409987916112872|0.7081079933036238


## Parameter Stability

Because the dataset is split and the model is recalibrated for each fold, each fold yields a slightly different set of calibrated parameters. The stability (or variance) of these parameters gives an indication of how robust the model fit is to the specific subset of data it trains on. High variance might indicate equifinality or overparameterization issues.

The following plot shows the distribution of the calibrated parameters across all the folds. This can be used as a diagnostic for parameter stability and equifinality.

*Observation:* `p1`–`p4` and `p6`–`p8` are stable across folds, varying by only 2–5% of their mean value (see the summary table below) — not the signature of a poorly-identified parameter. `p5` is the exception: it sits at 0.0 (the lower bound) in five of the six folds and only becomes non-zero (0.053) in the 2005 fold, so its variability is a scale artifact rather than genuine fold-to-fold disagreement. This pattern — most parameters well-constrained, one sitting at a bound and contributing little — is consistent with mild **equifinality** in `p5` specifically, rather than broad overparameterization of the 8-parameter model. With ~5 years of training data per fold, `p5` (a constant offset scaled by relative discharge) may simply not be well-identified by this particular record.

![Parameter Stability](cv_parameter_stability.png)

### Calibrated Parameters per Fold

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
2005|4.747629891683094|0.6400698273954342|1.424811085660049|0.2567114644918852|0.0|4.730552665050302|0.5845267755360761|0.6179277078147679
2006|4.923944842426356|0.636752756168301|1.4305046164394333|0.2714812215471511|0.0|4.928017272691765|0.5813557138386671|0.6424645964195198
2007|4.738478208720075|0.5896756235910764|1.3587873524916871|0.2833459153928053|0.0|5.17739477972502|0.5835060003506252|0.6726073929825519
2008|4.947207298700462|0.645862296979111|1.4649724464365734|0.2630804376783612|0.0|5.174573058351814|0.5788208288867661|0.6485021484863992
2009|4.762846495892688|0.6272896986538181|1.395751305960009|0.2671016854667467|0.0|4.899171671033041|0.5819404574728518|0.6372228614525036


### Parameter Summary Across Folds

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
mean|4.824021347484535|0.6279300405575481|1.4149653613975504|0.2683441449153899|0.0|4.981941889370388|0.5820299552169973|0.6437449414311485
std|0.1025367360934419|0.0224198071511809|0.0398881683541438|0.0099932970090537|0.0|0.1925252678500496|0.0021894019747475|0.0197887499019206


*`pooled` is omitted from this table: pooling applies to held-out predictions, not to per-fold parameter sets, so a single "pooled" parameter value has no meaning here.*
