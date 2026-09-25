"""
V8 - The end-to-end workflow and the scenario tools give exact answers.

(A) The command-line workflow a user follows (calibrate, then FORWARD runs
from the saved calibration) reproduces the calibration exactly.
(B) A paired scenario comparison recovers an effect whose exact size is known.
(C) The threshold tools agree with an independent calculation.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from common import (AUTHORS_BOUNDS, DE_SETTINGS, REPO, WORK, Result, Timer, load, mean_discharge, quiet,
                    river_csv)

TOL_EXACT = 1e-6       # °C, identical results
TOL_EFFECT = 1e-4      # °C, paired difference against the exact effect


def _cli(cfg: dict, name: str) -> str:
    """Run pyair2stream from the command line, as a user would; returns the output folder."""
    folder = os.path.join(WORK, name)
    os.makedirs(folder, exist_ok=True)
    cfg = {"station_name": "S", "series": "c", **cfg}
    cfg["paths"] = {**cfg["paths"], "output_dir": os.path.join(folder, "out")}
    path = os.path.join(folder, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", path], cwd=REPO, env=env,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cfg["paths"]["output_dir"]


def _part_a():
    """Calibrate version 8 by DE from the command line, then re-run the calibrated model in FORWARD
    mode from the saved calibration_metadata.json, on the calibration and the validation years."""
    cal, val = river_csv("MAH_2369", "calibration"), river_csv("MAH_2369", "validation")
    out = _cli({"version": 8, "integrator": "CRN", "run_mode": "DE", "objective_function": "NSE",
                "random_seed": 1, "optimization": dict(DE_SETTINGS), "parameter_bounds": AUTHORS_BOUNDS,
                "paths": {"input_data": cal, "validation_data": val}}, "v8a_cal")
    meta_path = os.path.join(out, "calibration_metadata.json")
    par = json.load(open(meta_path))["par_best"]
    ref = {"calibration": pd.read_csv(os.path.join(out, "2_DE_NSE_S_cc_1d.csv")),
           "validation": pd.read_csv(os.path.join(out, "3_DE_NSE_S_cv_1d.csv"))}
    rows = []
    for period, csv in (("calibration", cal), ("validation", val)):
        fout = _cli({"version": 8, "integrator": "CRN", "run_mode": "FORWARD", "objective_function": "NSE",
                     "parameters_forward": par,
                     "paths": {"input_data": csv, "calibration_metadata": meta_path}}, f"v8a_fwd_{period}")
        fwd = pd.read_csv(os.path.join(fout, "2_FORWARD_NSE_S_cc_1d.csv"))
        rows.append({"period": period, "days": len(fwd),
                     "max |FORWARD - calibration output| (°C)":
                         float(np.max(np.abs(fwd.Twat_mod.to_numpy() - ref[period].Twat_mod.to_numpy())))})
    return pd.DataFrame(rows)


def _forward(csv, par, q, chain, name, noise, reuse=None, extra_unc=None):
    from pyair2stream.optimization import forward_mode
    fo = {"enable_prediction_intervals": True, "mcmc_chain_path": chain, "n_samples": 200, "random_seed": 3}
    if reuse:
        fo["reuse_sample_indices_from"] = reuse
    cfg = {"version": 5, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": q, "Tice_cover": -100.0,
           "parameters_forward": [float(x) for x in par],
           "uncertainty_options": {"noise_model": noise, "save_ensemble": True, **(extra_unc or {})},
           "forward_options": fo, "paths": {"input_data": csv, "output_dir": os.path.join(WORK, name, "out")}}
    data = load(cfg, name)
    with quiet():
        forward_mode(data)
    return os.path.join(WORK, name, "out", "Forward_Prediction_Ensemble_S_c_1d.npz")


def _part_b():
    """Version 5 responds to a constant air-temperature change dTa with a water-temperature change of
    exactly a2/a3 * dTa on every day (once the start-up has passed). The ice floor is disabled here
    (Tice_cover -100) so that this holds on every day."""
    from pyair2stream.optimization import DE_MCMC_mode
    from pyair2stream import scenario
    cal, val = river_csv("MAH_2369", "calibration"), river_csv("MAH_2369", "validation")
    q = mean_discharge(cal)
    out = os.path.join(WORK, "v8b", "out")
    cfg = {"version": 5, "integrator": "CRN", "run_mode": "DE-MCMC", "objective_function": "NSE",
           "random_seed": 1, "Qmedia": q, "parameter_bounds": AUTHORS_BOUNDS,
           "optimization": {**DE_SETTINGS, "mcmc_walkers": 32, "mcmc_steps": 20000},
           "paths": {"input_data": cal, "output_dir": out}}
    data = load(cfg, "v8b")
    with quiet():
        DE_MCMC_mode(data, seed=1)
    chain = os.path.join(out, "MCMC_chain_S_c_1d.csv")
    warm = pd.read_csv(val)
    warm_csv = os.path.join(WORK, "v8b_plus1.csv")
    warm.assign(T_air=warm.T_air + 1.0).to_csv(warm_csv, index=False)
    rows, ensembles = [], {}
    for noise in ("iid", "ar1"):
        base = _forward(val, data.par_best, q, chain, f"v8b_base_{noise}", noise)
        meta = base.replace(".npz", "_meta.json")
        plus = _forward(warm_csv, data.par_best, q, chain, f"v8b_plus_{noise}", noise, reuse=meta)
        diff = scenario.paired_difference_from_files(plus, base)
        used = json.load(open(meta))["valid_draw_indices"]
        c = pd.read_csv(chain).iloc[used]
        effect = (c["par_2"] / c["par_3"]).to_numpy()[:, None]
        rows.append({"noise model": noise, "draws": diff.shape[0], "days": diff.shape[1],
                     "effect a2/a3 (°C), 5-95% of draws":
                         f"{np.percentile(effect, 5):.3f}-{np.percentile(effect, 95):.3f}",
                     "max |paired difference - a2/a3| (°C)": float(np.nanmax(np.abs(diff - effect)))})
        ensembles[noise] = base
    return pd.DataFrame(rows), ensembles


def _part_c(ensemble_path):
    """scenario.aggregate and scenario.exceedance against an independent calculation."""
    from pyair2stream import scenario
    ens, dates = scenario.load_ensemble(ensemble_path)
    threshold = float(np.nanpercentile(ens, 90))
    rows = []
    weekly = scenario.aggregate(ens, dates, how="mean", freq="7D")
    frame = pd.DataFrame(ens.T, index=dates)
    ref_weekly = frame.groupby((np.arange(len(dates)) // 7)).mean().to_numpy().T
    rows.append({"tool": "aggregate (7-day means)", "cases": weekly.size,
                 "disagreements": int(np.sum(~np.isclose(weekly, ref_weekly, atol=1e-12, equal_nan=True)))})
    for k in (1, 3, 7):
        counts = scenario.exceedance(ens, threshold, consecutive_days=k)
        above = pd.DataFrame(ens > threshold)
        ref = []
        for _, row in above.iterrows():
            runs = row.ne(row.shift()).cumsum()
            lengths = row.groupby(runs).agg(["first", "size"])
            ref.append(int(lengths.loc[lengths["first"] & (lengths["size"] >= k), "size"].sum()))
        rows.append({"tool": f"exceedance, runs of at least {k} day(s)", "cases": len(counts),
                     "disagreements": int(np.sum(counts != np.array(ref)))})
    return pd.DataFrame(rows)


def run(ctx) -> Result:
    res = Result(
        code="V8", title="Workflow and scenario tools give exact answers",
        question="Does the documented workflow reproduce the calibration exactly, does a paired scenario "
                 "comparison give the right answer, and do the threshold tools count correctly?",
        method="(A) Version 8 is calibrated on the Mentue from the command line (DE, NSE, CRN), which writes "
               "calibration_metadata.json. FORWARD runs are then made from the command line with that file "
               "(as USER_GUIDE §12 describes), on the calibration and validation years, and their output "
               "is compared with the calibration's own output. (B) For version 5 (and 3) the model's "
               "response to a constant change in air temperature dTa is known exactly: water temperature "
               "changes by a2/a3 x dTa on every day. Version 5 is calibrated by DE-MCMC on the Mentue; two "
               "FORWARD runs with prediction intervals are made on 2010-2012, one with the real air "
               "temperature and one with it raised by 1 °C, the second reusing the first's parameter "
               "draws; their paired difference (scenario.paired_difference_from_files) is compared, draw "
               "by draw, with that draw's a2/a3. Both noise models are tested. The ice floor is disabled "
               "(Tice_cover -100) so the exact answer holds on every day. (C) scenario.aggregate and "
               "scenario.exceedance are applied to a saved ensemble and compared with an independent "
               "calculation written with pandas.",
        criterion=f"(A) FORWARD output equals the calibration output to within {TOL_EXACT:g} °C. (B) Every "
                  f"draw's paired difference equals its a2/a3 to within {TOL_EFFECT:g} °C on every day, "
                  f"for both noise models (the residual noise cancels). (C) No disagreements.")
    with Timer() as t:
        a = _part_a()
        b, ensembles = _part_b()
        c = _part_c(ensembles["ar1"])
    ok_a = bool((a.iloc[:, -1] <= TOL_EXACT).all())
    ok_b = bool((b.iloc[:, -1] <= TOL_EFFECT).all())
    ok_c = bool((c.disagreements == 0).all())
    res.passed = ok_a and ok_b and ok_c
    res.seconds = t.seconds
    res.summary = (f"(A) FORWARD reproduces the calibration output to {a.iloc[:, -1].max():.0e} °C. "
                   f"(B) Paired differences equal the exact effect to {b.iloc[:, -1].max():.0e} °C for "
                   f"every draw and day. (C) {int(c.disagreements.sum())} disagreements in "
                   f"{int(c.cases.sum())} comparisons.")
    res.notes.append("Part B also shows that the residual noise cancels in a paired difference: its spread "
                     "is the parameter uncertainty of the effect only. This assumes the model's error on a "
                     "given day would be the same under both scenarios.")
    for df in (a, b):
        col = df.columns[-1]
        df[col] = df[col].map(lambda x: f"{x:.1e}")
    res.tables += [("A. Command-line workflow: FORWARD vs calibration output", a),
                   ("B. Paired +1 °C air-temperature scenario vs the exact effect", b),
                   ("C. Threshold tools vs an independent calculation", c)]
    return res
