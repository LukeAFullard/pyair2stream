# Leave-One-Year-Out Cross-Validation Example

This example demonstrates how to use the `cross_validation` block in the configuration to perform Leave-One-Year-Out (LOYO) cross-validation on the DAV dataset.

The configuration instructs pyair2stream to withhold one year of water temperature observations at a time, calibrate the model on the remaining data, and calculate metrics (NSE, KGE, RMSE) on the withheld year. Each fold completely excludes the specified year from the optimization target. For instance, in fold `2004`, the parameters are trained purely on the data from 2005-2009, and the predictions made for 2004 are scored to see how well the parameters generalized.

## Results

fold|NSE|RMSE
---|---|---
2004|0.9336510325773706|0.6915489003214208
2005|0.961441332404084|0.5843174852899023
2006|0.960523706666189|0.6027803485526382
2007|0.9476407057692268|0.6652603271559584
2008|0.8690747650140415|0.9832347343729628
2009|0.954027961169306|0.6266860719857689


### Summary across folds

The table below shows the summary across all folds. `mean` and `std` represent the macro-average and standard deviation across the individual per-fold metrics. `pooled` represents the micro-average computed over all held-out days pooled together.

fold|NSE|RMSE
---|---|---
mean|0.9377265839333696|0.6923046446131086
std|0.035144857863638|0.1478877321628146
pooled|0.9399833784683193|0.7054903608716758


## Parameter Stability

Because the dataset is split and the model is recalibrated for each fold, each fold yields a slightly different set of calibrated parameters. The stability (or variance) of these parameters gives an indication of how robust the model fit is to the specific subset of data it trains on. High variance might indicate equifinality or overparameterization issues.

The following plot shows the distribution of the calibrated parameters across all the folds. This can be used as a diagnostic for parameter stability and equifinality.

*Observation:* As seen in the table and plot below, the calibrated parameters fluctuate significantly between folds. This high variance is a classic sign of **equifinality**. Because we are using the 8-parameter version of the model on a relatively short timeframe (only ~5 years of training data per fold), the model is likely overparameterized. The optimizer finds different local minima that fit the training subset well, but the parameter sets themselves aren't uniquely defined.

![Parameter Stability](cv_parameter_stability.png)

### Calibrated Parameters per Fold

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
2004|4.795641815383528|0.6103077249857868|1.3956443363706583|0.2745013244964654|0.0|5.179274393222586|0.5786001067577446|0.6458024742405858
2005|4.76271648273271|0.6421840942843177|1.4477240179950464|0.2462690236395343|0.052801103700152|4.548313402748649|0.5853046787790146|0.592330799327366
2006|4.944995137694723|0.6407303151952116|1.4389007914002063|0.2701604544619786|0.0|4.84226851882783|0.5814700874037776|0.6314930784636547
2007|4.76504572642723|0.5930047130797977|1.366381142327927|0.283403569125073|0.0|5.104576818186302|0.5834991918935633|0.6631714080479653
2008|4.913088885611565|0.6412360456560914|1.4543006512795214|0.2634223327818473|0.0|5.278014302772802|0.5787892494336416|0.6615160275044232
2009|4.767484923469988|0.6279217429158289|1.3971252988338208|0.2671041079989953|0.0|4.885608472735765|0.5819411856437816|0.6354772153927758


### Parameter Summary Across Folds

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
mean|4.82482882855329|0.625897439352839|1.41667937303453|0.2674768020839823|0.0088001839500253|4.9730093180823225|0.5816007499852539|0.6382985004961285
std|0.0822215878169828|0.0202465336221937|0.035292119769733|0.012462385825694|0.0215559603201922|0.2675750413810571|0.002621622950265|0.0260051059672355
pooled||||||||
