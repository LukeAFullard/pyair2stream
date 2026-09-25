# Swiss river data

Daily air temperature, water temperature and discharge for three Swiss rivers,
used by the examples (`examples/`) and the validation suite (`validation/`).

| File prefix | River, gauge | Regime | Calibration period | Validation period |
|---|---|---|---|---|
| `MAH_2369` | Mentue, Yvonand (La Mauguettaz) | natural, low-land | 2002–2009 | 2010–2012 |
| `SIO_2011` | Rhône, Sion | regulated (hydropower) | 1984–2004 | 2005–2013 |
| `DAV_2327` | Dischmabach, Davos (Kriegsmatte) | snow-fed | 2003–2009 | 2010–2012 |

`MAH` and `DAV` are the MeteoSwiss air-temperature stations; `2369`, `2011` and
`2327` are the FOEN river gauges.

## Files

- `<station>_calibration.csv`, `<station>_validation.csv`: the data in
  pyair2stream's format (`Date, T_air, T_water, Discharge`; empty = missing).
  Produced from `original/` by `python data/switzerland/convert.py`; the values
  are unchanged.
- `original/`: the files exactly as distributed with the original air2stream
  code (`_cc` = calibration, `_cv` = validation; `-999` = missing).
- `published/`: supplementary spreadsheets of Piccolroaz et al. (2016), with the
  published parameters of every air2stream version (`Parameter_values`) and
  the published calibration and validation RMSE (`AIC`). The validation suite
  reproduces these results (`validation/REPORT.md`, V2).

Units: air and water temperature in °C (daily means). Discharge is a daily
mean whose unit the files do not state; its magnitudes (mean about 1.5 for the
Mentue and Dischmabach, 106 for the Rhône) are consistent with FOEN's standard
m³/s. The unit does not affect results, since only the ratio to the mean
discharge enters the model. Air temperature and discharge are complete. Water temperature has a few gaps
(Dischmabach: 2,197 of 2,557 calibration days observed).

## Source and licence

Measured by the Swiss Federal Office for the Environment (FOEN; water
temperature and discharge) and the Swiss Meteorological Institute (MeteoSwiss;
air temperature), as described in Piccolroaz et al. (2016). Distributed with the
air2stream code by its authors under the Creative Commons Attribution-ShareAlike
3.0 licence: https://github.com/spiccolroaz/air2stream (commit `d4834bc`,
folders `Switzerland/` and `References/Main/`). The files here are
byte-identical to those (SHA-256, first 16 characters):

| File | SHA-256 |
|---|---|
| `original/MAH_2369_cc.txt` | `038943e8c16a5c8a` |
| `original/MAH_2369_cv.txt` | `f31d4b360693494f` |
| `original/SIO_2011_cc.txt` | `7c744894234b0933` |
| `original/SIO_2011_cv.txt` | `c65c83f2ced6c499` |
| `original/DAV_2327_cc.txt` | `96aced529be5bdb3` |
| `original/DAV_2327_cv.txt` | `c8485235c40b2ec7` |

If you use these data, cite:

> Piccolroaz, S., Calamita, E., Majone, B., Gallice, A., Siviglia, A. and
> Toffolon, M. (2016). Prediction of river water temperature: a comparison
> between a new family of hybrid models and statistical approaches.
> *Hydrological Processes*, 30, 3901–3917. https://doi.org/10.1002/hyp.10913
