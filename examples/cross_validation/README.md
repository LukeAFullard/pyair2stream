# Leave-One-Year-Out Cross-Validation Example

This example demonstrates how to use the `cross_validation` block in the configuration to perform Leave-One-Year-Out (LOYO) cross-validation on the DAV dataset.

The configuration instructs pyair2stream to withhold one year of water temperature observations at a time, calibrate the model on the remaining data, and calculate metrics (NSE, KGE, RMSE) on the withheld year. Each fold completely excludes the specified year from the optimization target. For instance, in fold `2004`, the parameters are trained purely on the data from 2005-2009, and the predictions made for 2004 are scored to see how well the parameters generalized.

## Results

fold|NSE|RMSE
---|---|---
2004|0.9337|0.6915
2005|0.9614|0.5843
2006|0.9605|0.6028
2007|0.9476|0.6653
2008|0.8691|0.9832
2009|0.9540|0.6267


### Summary across folds

The table below shows the summary across all folds. `mean` and `std` represent the macro-average and standard deviation across the individual per-fold metrics. `pooled` represents the micro-average computed over all held-out days pooled together.

fold|NSE|RMSE
---|---|---
mean|0.9377|0.6923
std|0.0351|0.1479
pooled|0.9400|0.7055


## Parameter Stability

Because the dataset is split and the model is recalibrated for each fold, each fold yields a slightly different set of calibrated parameters. The stability (or variance) of these parameters gives an indication of how robust the model fit is to the specific subset of data it trains on. High variance might indicate equifinality or overparameterization issues.

The following plot shows the distribution of the calibrated parameters across all the folds. This can be used as a diagnostic for parameter stability and equifinality.

*Observation:* `p1`–`p4` and `p6`–`p8` are stable across folds, varying by only 2–5% of their mean value (see the summary table below) — not the signature of a poorly-identified parameter. `p5` is the exception: it sits at 0.0 (the lower bound) in five of the six folds and only becomes non-zero (0.053) in the 2005 fold, so its variability is a scale artifact rather than genuine fold-to-fold disagreement. This pattern — most parameters well-constrained, one sitting at a bound and contributing little — is consistent with mild **equifinality** in `p5` specifically, rather than broad overparameterization of the 8-parameter model. With ~5 years of training data per fold, `p5` (a constant offset scaled by relative discharge) may simply not be well-identified by this particular record.

![Parameter Stability](cv_parameter_stability.png)

### Calibrated Parameters per Fold

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
2004|4.7956|0.6103|1.3956|0.2745|0.0000|5.1793|0.5786|0.6458
2005|4.7627|0.6422|1.4477|0.2463|0.0528|4.5483|0.5853|0.5923
2006|4.9450|0.6407|1.4389|0.2702|0.0000|4.8423|0.5815|0.6315
2007|4.7650|0.5930|1.3664|0.2834|0.0000|5.1046|0.5835|0.6632
2008|4.9131|0.6412|1.4543|0.2634|0.0000|5.2780|0.5788|0.6615
2009|4.7675|0.6279|1.3971|0.2671|0.0000|4.8856|0.5819|0.6355


### Parameter Summary Across Folds

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
mean|4.8248|0.6259|1.4167|0.2675|0.0088|4.9730|0.5816|0.6383
std|0.0822|0.0202|0.0353|0.0125|0.0216|0.2676|0.0026|0.0260

*`pooled` is omitted from this table: pooling applies to held-out predictions, not to per-fold parameter sets, so a single "pooled" parameter value has no meaning here.*
