# British Columbia stream data (Callahan and Moore)

Daily air temperature, discharge, measured water temperature, and the water
temperature simulated by air2stream, for 23 Water Survey of Canada (WSC) gauging
stations in British Columbia, with the calibrated parameters of the 8-parameter
version for each station. The validation suite uses them to check that
pyair2stream reproduces another group's published simulations
([V15](../../validation/REPORT.md#v15)).

## Source

The files in `original/` are byte-identical to those published with:

> Callahan, L. and Moore, R. D. (2025). Evaluation of the hybrid air2stream
> model for simulating daily stream temperature during extreme summer heat wave
> and autumn drought conditions. *Hydrological Processes*, 39(1), e70033.
> https://doi.org/10.1002/hyp.70033

in the dataset:

> Moore, R. D. and Callahan, L. (2024). Evaluation of the hybrid Air2stream
> model for simulating daily stream temperature during extreme summer heat wave
> and autumn drought conditions - Data sets and scripts (version 1). Zenodo.
> https://doi.org/10.5281/zenodo.14502248 (all versions:
> https://doi.org/10.5281/zenodo.14502247; record:
> https://zenodo.org/records/14502248)

The dataset is published under the Creative Commons Attribution 4.0
International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/),
which allows it to be redistributed with attribution. The files are unchanged.
The dataset's R scripts are not copied here; they are in the Zenodo record. If
you use these data, cite both the paper and the dataset.

| File | SHA-256 (first 16 characters) | Content |
|---|---|---|
| `original/a2s_8_parameter_values.csv` | `151404389333a4fc` | the calibrated parameters `a1`–`a8` of the 8-parameter version, one row per station |
| `original/stn_hyd_reg.csv` | `2299ab80f86113dd` | station number, name, hydrological regime (snow, rain, hybrid, glacial/hybrid) and regulation status |
| `original/ts_all_58.csv` | `c0a4ca8542122de9` | daily series for every station, below |

Columns of `ts_all_58.csv`, as the dataset describes them: `station_number`
(WSC station number); `year`, `month`, `day`, `date`; `ta`, daily mean air
temperature (°C; interpolated to each station from the ERA5 reanalysis, as the
paper's abstract states); `tw_obs`, daily mean measured water temperature (°C;
`NA` where not measured); `q`, mean daily discharge (m³/s); `period`,
`calibration` (the years up to 2020) or `validation` (2021 and 2022, to 31
October); `tw_sim_8` and `tw_sim_5`, the daily water temperature simulated by
the 8- and 5-parameter versions of air2stream (°C). Only the 8-parameter
version's parameters are published, so only its simulations are checked.

## Stations

23 stations: 14 snow-fed, 5 rain-fed, 2 hybrid and 2 glacial/hybrid; 8 are
regulated. Records start between 2011 and 2016 and end on 31 October 2022. Air
temperature and discharge are complete; water temperature has gaps.

**One record does not start on 1 January.** The calibration record of station
08GA077 (Seymour River below Orchid Creek) starts on 1 November 2012. The
original air2stream program assumes that every record starts on 1 January and
takes the seasonal term's timing from the row number, not the date. The
published simulation of that station's calibration period is reproduced only
when the record is read that way, that is, with its seasonal term ten months
out of phase (V15). pyair2stream takes the timing from the dates and refuses a
calibration record that does not start on 1 January.
