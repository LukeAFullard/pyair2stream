# North-eastern United States river data

Daily air temperature, water temperature and discharge for 30 rivers of New
England and the Mid-Atlantic, used by validation V20 (`validation/REPORT.md`):
the one test of the final method on rivers that played no part in developing
or correcting it.

## How the stations were chosen

By a fixed rule, written in [`download.py`](download.py) before any model was
run on these data, so that the choice cannot depend on how well the model or
its intervals do:

1. every USGS stream gauge in hydrologic regions 01 (New England) and 02
   (Mid-Atlantic) with daily mean water temperature and daily mean discharge;
2. discharge given, and above zero, on every day from 2011 to 2023;
3. at least 300 daily water temperatures in each of at least 5 of the
   calibration years 2011–2020, and at least 300 in 2021–2023;
4. Daymet air temperature at the gauge;
5. of the stations that qualify, 30 drawn at random (`numpy`
   `default_rng(2026)`).

Of 428 candidate gauges, 60 qualified (`candidates.csv` gives every gauge
examined and the reason it was or was not selected). The rule is mechanical:
it keeps urban, regulated and reservoir-fed sites (for example `01362230`, the
outlet of the Shandaken diversion tunnel from the Schoharie Reservoir), as the
Swiss and British Columbia data do.

## Files

- `<site>_calibration.csv` (2011–2020) and `<site>_later.csv` (2021–2023), in
  pyair2stream's format: `Date, T_air, T_water, Discharge`; empty = missing.
- `stations.csv`: the 30 stations, with name, coordinates, drainage area
  (square miles, as USGS gives it) and counts.
- `candidates.csv`: every gauge examined.
- `download.py`: downloads the data and applies the rule
  (`python data/usa_northeast/download.py`). USGS revises provisional data, so
  a new download can differ slightly from these files; the validation uses
  these files.

Units: °C; discharge in m³/s, converted from USGS cubic feet per second.
Air temperature is Daymet's (tmax + tmin) / 2 at the gauge. Daymet has 365
days in every year and leaves out 31 December of leap years; that day is
filled with the mean of 30 December and 1 January. Water temperature and
discharge are USGS daily means as published, approved or provisional,
downloaded on 11 October 2026.

## Sources and licences

- Water temperature and discharge: U.S. Geological Survey, National Water
  Information System (https://waterdata.usgs.gov; https://doi.org/10.5066/F7P55KJN).
  USGS data are in the public domain.
- Air temperature: Thornton, M. M., Shrestha, R., Wei, Y., Thornton, P. E.,
  Kao, S.-C. and Wilson, B. E. (2022). Daymet: Daily Surface Weather Data on a
  1-km Grid for North America, Version 4 R1. ORNL DAAC, Oak Ridge, Tennessee,
  USA. https://doi.org/10.3334/ORNLDAAC/2129. Daymet data are freely available
  without restriction; cite them if you use them.

## Stations

Measured water-temperature days in each file, and SHA-256 (first 16
characters) of each file.

| Site | Name | Calibration days | Later days | `_calibration.csv` | `_later.csv` |
|---|---|---|---|---|---|
| `01018035` | Meduxnekeag River at Lowery Rd. nr Houlton, Maine | 3453 | 1095 | `40671f4afa61a8ec` | `4a7ce5b473671daa` |
| `01091000` | SOUTH BRANCH PISCATAQUOG RIVER NEAR GOFFSTOWN, NH | 3088 | 1026 | `1fc1106d9b2a1c80` | `e55b545627da98c9` |
| `01095375` | QUINAPOXET RIVER AT CANADA MILLS NEAR HOLDEN, MA | 3519 | 1089 | `078cb7931a510c8d` | `01d57b1095c3ce44` |
| `01104370` | STONY BROOK AT VILES STREET, NEAR WESTON, MA | 3653 | 1090 | `6b1c8790a1978834` | `eb50490ad8177332` |
| `01104430` | HOBBS BK BELOW CAMBRIDGE RES NR KENDALL GREEN, MA | 3589 | 1089 | `e0685d34de18994d` | `0847da6482d480c6` |
| `01115170` | MOSWANSICUT STREAM NR NORTH SCITUATE, RI | 3597 | 1080 | `f4a4489ae60b377d` | `17bf1138beddda69` |
| `01129500` | CONNECTICUT RIVER AT NORTH STRATFORD, NH | 3573 | 1095 | `06a68a2a0afcfe08` | `84ccc19a30dd1bf8` |
| `01196500` | QUINNIPIAC RIVER AT WALLINGFORD, CT | 2297 | 816 | `e646e3c0bb8d49fe` | `d7b9328be4cd80f7` |
| `01362230` | DIVERSION FROM SCHOHARIE RESERVOIR NY | 3642 | 1093 | `5cbb14664c1bd0e2` | `02fb51073cb1f15c` |
| `01362500` | ESOPUS CREEK AT COLDBROOK NY | 3629 | 1084 | `6123b94778eb1d86` | `05ac6e9a73cc4548` |
| `01421000` | EAST BRANCH DELAWARE RIVER AT FISHS EDDY NY | 3585 | 1094 | `8b89d5ea4317fcaa` | `a54228350286a597` |
| `01427510` | DELAWARE RIVER AT CALLICOON NY | 3643 | 1086 | `a30aab7b3655ffdf` | `8058149d72765eb3` |
| `01428500` | DELAWARE R ABOVE LACKAWAXEN R NEAR BARRYVILLE NY | 3621 | 1070 | `620f34e200637e5d` | `63b0b7e79d1a119a` |
| `01436690` | NEVERSINK RIVER AT BRIDGEVILLE NY | 3630 | 1084 | `c6be08ec282ea9dd` | `6ca58dda9dbf9f12` |
| `01463500` | Delaware River at Trenton NJ | 3518 | 1082 | `5ac9a82241baa690` | `667a8f16c9ce94fd` |
| `01466500` | McDonalds Branch in Byrne State Forest NJ | 3212 | 1071 | `d4b1fb2b9eb92b5e` | `75391b73babee48f` |
| `01480400` | Birch Run near Wagontown, PA | 3623 | 1095 | `0fe59bfd9bf25ed8` | `49e414bc5bb9b6e3` |
| `01480500` | West Branch Brandywine Creek at Coatesville, PA | 3636 | 1095 | `89e16423b9c76b7e` | `6f46fbb1d90fa34e` |
| `01480617` | West Branch Brandywine Creek at Modena, PA | 3141 | 1089 | `fc1abbff5e4ba5e1` | `593c863ae554c17e` |
| `01481500` | BRANDYWINE CREEK AT WILMINGTON, DE | 3591 | 1095 | `d2cf63bc713a9ec6` | `56378d69d7663f79` |
| `01484525` | MILLSBORO POND OUTLET AT MILLSBORO, DE | 3345 | 1095 | `91be72d7a3c762d9` | `e5d8fb85a8c03be1` |
| `01518700` | Tioga River at Tioga Junction, PA | 1918 | 1089 | `3debaeebf6757951` | `9e89d02bfcc0f088` |
| `01544500` | Kettle Creek at Cross Fork, PA | 3604 | 1094 | `0455030ac8deebe3` | `e2d2e0d4f9ee98e0` |
| `01589290` | SCOTTS LEVEL BRANCH AT ROCKDALE, MD | 1797 | 1087 | `5a556d4084efdc59` | `b04ff39ec5067cb3` |
| `01595800` | NORTH BRANCH POTOMAC RIVER AT BARNUM, WV | 2903 | 1073 | `e25964d8ac6af46b` | `b416eca375e7f919` |
| `01596500` | SAVAGE RIVER NEAR BARTON, MD | 2941 | 1039 | `b6f16e28f106ff44` | `59036931c5d9f332` |
| `01646305` | DEAD RUN AT WHANN AVENUE NEAR MCLEAN, VA | 3544 | 926 | `a42c7dc508972726` | `2270ec5424de6fb4` |
| `01650800` | SLIGO CREEK NEAR TAKOMA PARK, MD | 2748 | 1093 | `08258afbf4eba823` | `ef2bdea1d3e73ff6` |
| `01651800` | WATTS BRANCH AT WASHINGTON, DC | 2839 | 1074 | `8039530099ee231d` | `667e7361ab015258` |
| `02011500` | BACK CREEK NEAR MOUNTAIN GROVE, VA | 3525 | 421 | `db1505af851c167e` | `5ce081b3ff53e53d` |
