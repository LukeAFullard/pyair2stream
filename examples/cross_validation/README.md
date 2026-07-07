# Leave-One-Year-Out Cross-Validation Example

This example demonstrates how to use the `cross_validation` block in the configuration to perform Leave-One-Year-Out (LOYO) cross-validation on the DAV dataset using the Differential Evolution (DE) optimizer.

The configuration instructs pyair2stream to withhold one year of water temperature observations at a time, calibrate the model on the remaining data, and calculate metrics (NSE, KGE, RMSE) on the withheld year. Each fold completely excludes the specified year from the optimization target. For instance, in fold `2004`, the parameters are trained purely on the data from 2005-2009, and the predictions made for 2004 are scored to see how well the parameters generalized.

## Results

fold|NSE|RMSE
---|---|---
2004|0.933655954909922|0.6915232473464565
2005|0.9621314551612068|0.5790648210487357
2006|0.960421676702384|0.6035588153755265
2007|0.9476399255921248|0.6652652834769742
2008|0.8689869772214177|0.9835643176756004
2009|0.9540314682955008|0.6266621671393195


## Parameter Stability

Because the dataset is split and the model is recalibrated for each fold, each fold yields a slightly different set of calibrated parameters. The stability (or variance) of these parameters gives an indication of how robust the model fit is to the specific subset of data it trains on. High variance might indicate equifinality or overparameterization issues.

The following plot shows the distribution of the calibrated parameters across all the folds. This can be used as a diagnostic for parameter stability and equifinality.

*Observation:* The parameters show reasonable boundaries indicating structural physical fits thanks to the use of DE optimization. The evaluation metrics hold up exceptionally well across all cross-validation segments, showing strong parameter generalization.

![Parameter Stability](cv_parameter_stability.png)

### Calibrated Parameters per Fold

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
2004|4.7963185200996685|0.6104239726232604|1.3961051365119506|0.2743704157616137|0.0|5.177479626048406|0.5786114968877842|0.6455133769912197
2005|4.73116090614145|0.6313918370847693|1.4064208309606108|0.2665056912874637|0.0085538696487975|4.834720688535748|0.5828028280184135|0.629764859066892
2006|4.959135045374892|0.6427954140919834|1.4460536713500742|0.2683933763752523|0.0|4.817568990144423|0.5817073437419762|0.6275138620665713
2007|4.7654877651975225|0.592995055107803|1.366435272992103|0.2834596411021868|0.0|5.105641179198693|0.5834957845802132|0.6632742545856458
2008|4.940139303859178|0.6360850600067309|1.4553734482427378|0.2621323243531477|0.0|5.28916653098493|0.5788231327749641|0.659727363439466
2009|4.767851489258859|0.6279343674125554|1.3973851450081167|0.2670455133614397|0.0|4.884926597130369|0.5819576182064718|0.6353175846403092
