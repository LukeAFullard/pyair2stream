# Leave-One-Year-Out Cross-Validation Example

This example demonstrates how to use the `cross_validation` block in the configuration to perform Leave-One-Year-Out (LOYO) cross-validation on the DAV dataset.

The configuration instructs pyair2stream to withhold one year of water temperature observations at a time, calibrate the model on the remaining data, and calculate metrics (NSE, KGE, RMSE) on the withheld year. Each fold completely excludes the specified year from the optimization target. For instance, in fold `2004`, the parameters are trained purely on the data from 2005-2009, and the predictions made for 2004 are scored to see how well the parameters generalized.

## Results

fold|NSE|RMSE
---|---|---
2004|0.9336491285576962|0.6915588229571206
2005|0.9615500625381984|0.5834930562537592
2006|0.9604867759984554|0.603062237693296
2007|0.9475876906500637|0.6655970384590799
2008|0.8689642775301348|0.983649521569402
2009|0.9540290962984816|0.6266783349563297


## Parameter Stability

Because the dataset is split and the model is recalibrated for each fold, each fold yields a slightly different set of calibrated parameters. The stability (or variance) of these parameters gives an indication of how robust the model fit is to the specific subset of data it trains on. High variance might indicate equifinality or overparameterization issues.

The following plot shows the distribution of the calibrated parameters across all the folds. This can be used as a diagnostic for parameter stability and equifinality.

*Observation:* As seen in the table and plot below, the calibrated parameters fluctuate significantly between folds. This high variance is a classic sign of **equifinality**. Because we are using the 8-parameter version of the model on a relatively short timeframe (only ~5 years of training data per fold), the model is likely overparameterized. The optimizer finds different local minima that fit the training subset well, but the parameter sets themselves aren't uniquely defined.

![Parameter Stability](cv_parameter_stability.png)

### Calibrated Parameters per Fold

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
2004|4.7958523977860885|0.6103479611837682|1.3958257484593206|0.2744671872251145|0.0|5.179072569289614|0.5786109787420093|0.6457128018315996
2005|4.7718609535049925|0.643682404943059|1.433242381418153|0.2560701591024434|0.0|4.656294544073869|0.5846050617894614|0.6081346659152119
2006|4.949758179454187|0.6406186641045637|1.4395742025206777|0.2705589586334236|0.0|4.849900988956806|0.5814428698162843|0.6321911158363488
2007|4.762628972069144|0.5917231136432277|1.3630092415928032|0.284879656384202|0.0|5.1285566871910415|0.5833572324094461|0.6663350281002528
2008|4.940499543230454|0.6361035821691058|1.4555088681857369|0.2620861473338876|0.0|5.288886462241186|0.5788315201743873|0.659637454730465
2009|4.767699229231186|0.62796119906357|1.397220596012217|0.2671147798900077|0.0|4.885621972059985|0.5819419863007271|0.6354759780254924
