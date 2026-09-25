"""
V7 - Missing data.

Real records have gaps. Synthetic data with a known truth (as in V3) are given
typical gap patterns, in the water temperature (which the model only needs for
scoring) and in the air temperature (which the model needs to run). The
calibrated model must still predict other years correctly.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import WORK, Result, Timer, calibrate, mean_discharge, published_params, river_csv, simulate
from v3_recovery import SIGMA, noise, truth_series

VERSION = 5
TOL = 0.15       # °C, prediction error against the noise-free truth


def _pattern(name, dates, rng):
    """Boolean mask of the days removed."""
    n = len(dates)
    if name.endswith("20% of days at random"):
        return rng.random(n) < 0.20
    if name.endswith("50% of days at random"):
        return rng.random(n) < 0.50
    if name.endswith("one day a week"):
        return (np.arange(n) % 7) != 3
    if name.endswith("every December-March"):
        return dates.dt.month.isin([12, 1, 2, 3]).to_numpy()
    if name.endswith("all of 2005"):
        return (dates.dt.year == 2005).to_numpy()
    if name.endswith("5% of days at random") or name.endswith("filled by interpolation"):
        return rng.random(n) < 0.05
    if name.endswith("60 days in summer 2005"):
        return ((dates >= "2005-07-01") & (dates < "2005-08-30")).to_numpy()
    return np.zeros(n, bool)


CASES = [
    ("Water temperature: none missing", "T_water"),
    ("Water temperature: 20% of days at random", "T_water"),
    ("Water temperature: 50% of days at random", "T_water"),
    ("Water temperature: one day a week", "T_water"),
    ("Water temperature: every December-March", "T_water"),
    ("Water temperature: all of 2005", "T_water"),
    ("Air temperature: 5% of days at random", "T_air"),
    ("Air temperature: 60 days in summer 2005", "T_air"),
    ("Air temperature: 5% of days at random, filled by interpolation", "T_air"),
]


def _case(args):
    i, (name, column) = args
    q_cal = mean_discharge(river_csv("MAH_2369", "calibration"))
    par = published_params(VERSION, "MAH_2369")
    tag = f"v7_{i}"
    src, truth_c = truth_series(VERSION, par, "calibration", q_cal, tag=tag)
    _, truth_v = truth_series(VERSION, par, "validation", q_cal, tag=tag)
    rng = np.random.default_rng(700 + i)
    df = src.assign(T_water=np.round(truth_c + noise(len(truth_c), "ar1", rng), 3))
    gone = np.array(_pattern(name, pd.to_datetime(df.Date), rng), dtype=bool)
    gone[0] = False                      # the record must start with a value
    df.loc[gone, column] = np.nan
    gap_tolerant = column == "T_air" and "interpolation" not in name
    if column == "T_air" and not gap_tolerant:
        df["T_air"] = df["T_air"].interpolate(limit_direction="both")
    csv = os.path.join(WORK, f"{tag}_obs.csv")
    df.to_csv(csv, index=False)
    extra = {"Qmedia": q_cal}
    if gap_tolerant:
        extra["gap_tolerant"] = True
    d = calibrate(csv, VERSION, objective="NSE", name=f"{tag}cal", **extra)
    scored = int(np.sum(d.eval_mask & (d.Twat_obs != -999.0)))
    pred = simulate(os.path.join(WORK, f"{tag}_forcing_validation.csv"), VERSION, d.par_best, "CRN", q_cal,
                    name=f"{tag}pred").Twat_mod[365:]
    return {"gap pattern (calibration years 2002-2009)": name,
            "mode": "gap-tolerant" if gap_tolerant else "standard",
            "water temperatures available": int(df.T_water.notna().sum()),
            "days scored in calibration": scored,
            "prediction RMSE vs truth, 2010-2012 (°C)": float(np.sqrt(np.mean((pred - truth_v) ** 2)))}


def run(ctx) -> Result:
    res = Result(
        code="V7", title="Missing data",
        question="Do gaps in the water or air temperature record bias the calibrated model?",
        method=f"Synthetic data as in V3 (version {VERSION}, published Mentue parameters, real Mentue air "
               f"temperature and discharge, AR(1) noise of {SIGMA} °C). Before calibration, days are removed "
               f"from the 2002-2009 record in typical patterns. Gaps in water temperature need no special "
               f"handling: those days are simply not scored. Gaps in air temperature stop the model, so "
               f"either gap-tolerant mode is used (the record is split into pieces, each restarted and its "
               f"first {15} days not scored), or short gaps are filled by straight-line interpolation "
               f"first. Each calibrated model (DE, NSE, CRN) then predicts 2010-2012 from complete forcing "
               f"and is compared with the noise-free truth.",
        criterion=f"Prediction error against the truth at most {TOL} °C for every gap pattern.")
    cases = [CASES[0], CASES[3], CASES[6]] if ctx.quick else CASES
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            rows = list(ex.map(_case, list(enumerate(cases))))
    df = pd.DataFrame(rows)
    err = "prediction RMSE vs truth, 2010-2012 (°C)"
    res.passed = bool((df[err] <= TOL).all())
    res.seconds = t.seconds
    full = df["days scored in calibration"].iloc[0]
    rnd = df[df["gap pattern (calibration years 2002-2009)"] == "Air temperature: 5% of days at random"]
    res.summary = (f"Prediction error against the truth {df[err].min():.3f}-{df[err].max():.3f} °C across "
                   f"{len(df)} gap patterns (noise level {SIGMA} °C).")
    if len(rnd):
        res.summary += (f" Scattered air-temperature gaps in gap-tolerant mode leave only "
                        f"{int(rnd['days scored in calibration'].iloc[0])} of {full} days scored.")
        res.notes.append("Gap-tolerant mode drops every piece of record shorter than min_segment_days (30) and "
                         "does not score the first warmup_drop_days (15) of each piece, so scattered one-day "
                         "gaps in air temperature discard most of the record. Here the prediction stayed "
                         "accurate because the synthetic data follow the model exactly; with real data, fewer "
                         "scored days make the calibration less certain. Filling short air-temperature gaps by "
                         "interpolation (or from a nearby station) keeps every day, at the cost of small errors "
                         "in the filled values; use gap-tolerant mode for long gaps.")
    df[err] = df[err].round(3)
    res.tables.append(("Effect of gaps on the calibrated model", df))
    return res
