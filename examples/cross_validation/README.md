# Leave-One-Year-Out Cross-Validation Example

This example demonstrates how to use the `cross_validation` block in the configuration to perform Leave-One-Year-Out (LOYO) cross-validation on the DAV dataset.

The configuration instructs pyair2stream to withhold one year of water temperature observations at a time, calibrate the model on the remaining data, and calculate metrics (NSE, KGE, RMSE) on the withheld year.

## Results

fold|NSE|RMSE
---|---|---
2004|0.563767453314442|1.7732290728379034
2005|0.8641984999274399|1.0965795863372427
2006|0.8508293326955279|1.171743674861908
2007|0.7902174458763086|1.3316172652657845
2008|0.3007509460787757|2.272278059204116
2009|0.8586122173673496|1.099028872811621


## Parameter Stability

Because the dataset is split and the model is recalibrated for each fold, each fold yields a slightly different set of calibrated parameters. The stability (or variance) of these parameters gives an indication of how robust the model fit is to the specific subset of data it trains on. High variance might indicate equifinality or overparameterization issues.

The following plot shows the distribution of the calibrated parameters across all the folds. This can be used as a diagnostic for parameter stability and equifinality.

![Parameter Stability](cv_parameter_stability.png)

### Calibrated Parameters per Fold

fold|p1|p2|p3|p4|p5|p6|p7|p8
---|---|---|---|---|---|---|---|---
2004|3.709557725941516|0.5904376943542933|1.1074179943807392|0.3939060586249784|0.007849279194278|5.69775338223529|0.5510481651046532|1.005732329537366
2005|4.13710675383345|0.4841500859938263|1.5606496339999594|0.218692681490325|0.025211095357837|3.638127772337056|0.5931910321885895|0.4539574259609539
2006|2.7701791537636096|0.2749086368551696|0.9714439035668466|0.4131270138411165|2.112919682445596|4.87781300910739|0.5806309572854597|0.9655793962398448
2007|4.112815835390999|0.5134140450619293|1.6228978482962086|0.2552196968431712|0.0035218454909607|4.349136046173574|0.5993463062686695|0.5243786704834148
2008|1.3060994418651206|0.3589696416079158|0.3085208773984376|0.7742861386503556|4.944113148821146|8.112056237638495|0.5610167004548903|2.068021529091729
2009|2.940502332965747|0.392372950844516|1.1446649908672004|0.3775782708941254|1.7990929900158106|3.976131049173633|0.6000439075825325|0.8259986776481824
