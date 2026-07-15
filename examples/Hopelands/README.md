# Hopelands Water Temperature Analysis Report


## 1. Executive Summary
The `pyair2stream` water temperature model was calibrated on the Hopelands dataset, yielding a Nash-Sutcliffe Efficiency (NSE) of **0.956214**.

## 2. Dataset and Preprocessing
The analysis integrated three primary data sources:
- **Air Temperature**: Originally in Kelvin, converted to Celsius ($T_{Celsius} = T_{Kelvin} - 273.15$).
- **Water Temperature**: Mean daily observations, with outliers (< 0.1°C) excluded.
- **Discharge**: Daily flow observations.

### 2.1. Data Availability and Segment Analysis
The merged timeseries index spans 1972-01-01 to 2026-06-05, reflecting the date range of the source files after merging. Note that the air temperature and discharge records end in 2024 (see §2 above); the trailing rows through mid-2026 come from placeholder dates pre-allocated in the raw water-temperature export and carry no air temperature, discharge, or water temperature values, so they contribute nothing to calibration.
- **T_air missing**: 3.4%
- **T_water missing**: 51.1%
- **Discharge missing**: 36.9%

Despite significant gaps, the **gap-tolerant** mode successfully identified valid segments for model calibration.

![Pre-Analysis Report](pre_analysis_report.png)
*Figure 1: Pre-analysis timeline showing data coverage and identified valid segments (green).*

## 3. Model Calibration (DE-MCMC)
The model was calibrated using a hybrid Differential Evolution (DE) and L-BFGS-B optimization strategy (200 particles, 5000 iterations), followed by Markov Chain Monte Carlo (MCMC) for uncertainty quantification.

### 3.1. Optimization Convergence
![Convergence Plot](output/convergence_DE-MCMC_NSE_Hopelands.png)
*Figure 2: Convergence of objective functions (NSE, R2, MAE) and parameter values during DE optimization.*

### 3.2. Performance Metrics
| Metric | Value |
|--------|-------|
| NSE    | 0.956214 |
| R²     | 0.9513 |
| RMSE   | 0.914  |
| MAE    | 0.705  |

![Calibration Results](output/calibration_DE-MCMC_NSE_Hopelands.png)
*Figure 3: Observed vs. Modeled water temperature for the calibration period, including 90% prediction intervals.*

![Full Simulation](output/full_simulation_DE-MCMC_NSE_Hopelands.png)
*Figure 4: Full simulation timeline showing predicted water temperatures even where observations are missing.*

### 3.3. Parameter Significance and Uncertainty
| Parameter | Mean | 95% CI Lower | 95% CI Upper | Significant |
|-----------|------|--------------|--------------|-------------|
| par_1 | 0.1355 | 0.0844 | 0.1882 | True |
| par_2 | 0.2763 | 0.2669 | 0.2856 | True |
| par_3 | 0.2304 | 0.2212 | 0.2395 | True |
| par_4 | 0.3497 | 0.3306 | 0.3700 | True |
| par_5 | 4.8198 | 4.5544 | 5.0985 | True |
| par_6 | 1.6575 | 1.5638 | 1.7553 | True |
| par_7 | 0.0369 | 0.0347 | 0.0389 | True |
| par_8 | 0.3772 | 0.3573 | 0.3983 | True |


![Dotty Plots](output/dottyplots_DE-MCMC_NSE_Hopelands.png)
*Figure 5: Dotty plots showing the distribution of parameter sets sampled during MCMC.*

![Parameter Correlation](output/parameter_correlation_DE-MCMC_Hopelands.png)
*Figure 6: Correlation matrix between the 8 model parameters.*

### 3.4. Residual Diagnostics
![Residual Diagnostics (Calibration)](output/residual_diagnostics_calibration_DE-MCMC_NSE_Hopelands.png)
*Figure 7: Q-Q plot and Autocorrelation Function (ACF) of the model residuals for the calibration period.*

![Residual Diagnostics (Full Simulation)](output/residual_diagnostics_full_simulation_DE-MCMC_NSE_Hopelands.png)
*Figure 7b: Q-Q plot and Autocorrelation Function (ACF) of the model residuals for the full simulation period.*


### 3.5. Cross-Validation Results
A Leave-One-Year-Out cross-validation was conducted. The table below shows the performance for each held-out year:

fold|NSE|RMSE|n_obs_held_out
---|---|---|---
1999|0.953217744616311|0.8517457448921114|316.0
2000|0.9475985053119684|0.9463921359323728|366.0
2001|0.9605412392249104|0.8329982007783485|365.0
2002|0.9408884703660004|0.816315459996708|358.0
2003|0.9550603506716012|0.8404132271434313|365.0
2004|0.931823568996461|0.9862012215460226|366.0
2005|0.9658953589247404|0.7623911218627593|365.0
2006|0.9492613520953228|0.873836400745936|365.0
2007|0.9330353048985828|1.105237424848338|365.0
2008|0.966854979120545|0.8355651588998632|366.0
2009|0.9220482523910324|1.193114593756219|365.0
2010|0.9378548043785528|1.033985007052204|365.0
2011|0.9569873691469444|0.8701818481077195|365.0
2012|0.9522964504839933|0.8500177507294816|366.0
2013|0.9524410276214011|0.9308963655331022|365.0
2014|0.9457281559365348|0.920300119292216|365.0
2015|0.9492380312057528|1.040171220865041|365.0
2016|0.960556373335429|0.8674272642389248|366.0
2017|0.9257232596667556|1.0785197263316884|348.0
2018|0.9565460103030706|1.0058155353025306|297.0
2019|0.9162194600772507|1.121871544184808|263.0
2020|0.9425020181903954|0.9678235886422444|333.0
2021|0.9493929283888592|0.9538233354402048|365.0
2022|0.946198965847026|0.9124614622900198|365.0
2023|0.8225170142369466|1.535770429581624|361.0
2024|0.7691341329989958|1.121780611332323|92.0


#### Summary across folds

fold|NSE|RMSE|n_obs_held_out
---|---|---|---
mean|0.934983120324438|0.9713483268971632|8943.0
std|0.0435688177378579|0.1603946855343913|
pooled|0.9445621239201136|0.9779333609087328|9703.0


## 4. Sensitivity Analysis
A local One-At-A-Time (OAT) sensitivity analysis was performed to evaluate the impact of each parameter on the simulated water temperature.

![Sensitivity Analysis](output/sensitivity_DE-MCMC_NSE_Hopelands.png)
*Figure 8: Sensitivity index for each model parameter across different perturbation levels.*

## 5. Conclusion
The calibration for the Hopelands station reached an NSE of 0.956 on a dataset with fragmented discharge and temperature records. The parameter estimates are statistically significant, but error margins will be wider when extrapolating across extreme unobserved gaps.

---
*Report updated on 2026-06-05*
