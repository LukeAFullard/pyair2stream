"""
V15 - Reproduces an independent group's published simulations (British Columbia).

Callahan and Moore (2025, Hydrological Processes 39(1), e70033) calibrated the 8-parameter version
of air2stream for 23 streams in British Columbia and published, for every station, the calibrated
parameters, the inputs, the measured water temperature and their simulated water temperature for
the calibration (up to 2020) and validation (2021-2022) periods (Moore and Callahan, 2024, Zenodo,
https://doi.org/10.5281/zenodo.14502248; data/british_columbia/). Unlike V2 and V13, the rivers,
the people and the code that produced the results are independent of the model's authors, and the
whole simulated series can be compared day by day, not only its error.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DE_SETTINGS, REPO, WORK, Result, Section, Timer, calibrate, load, plot_style,
                    quiet, save_figure, simulate, BLUE, ORANGE, AQUA, INK2, LIGHT_GREY)
from v2_published import MATCH, RANGE, _at_bound, _inside

BC = os.path.join(REPO, "data", "british_columbia", "original")
TOL_SERIES = 0.001          # °C: largest difference allowed between pyair2stream's and the published series
TOL_DE = 0.002              # °C: DE may fit worse than the published parameters by no more than this
HEAT_DOME = ("2021-06-25", "2021-07-02")       # the paper's windows, applied by day of year as in its scripts
DROUGHT = ("2022-09-01", "2022-10-31")
QUICK_STATIONS = ("07EA004", "08GA077")
MCMC_WALKERS, MCMC_STEPS = 32, 100000       # part D: run until converged, at most this many steps (V2 part F:
                                            # 20,000; poorly determined parameters need longer chains here)


def _load():
    ts = pd.read_csv(os.path.join(BC, "ts_all_58.csv"), na_values=["NA"])
    ts["date"] = pd.to_datetime(ts["date"])
    par = pd.read_csv(os.path.join(BC, "a2s_8_parameter_values.csv")).set_index("station")
    info = pd.read_csv(os.path.join(BC, "stn_hyd_reg.csv")).set_index("station_number")
    return ts, par, info


def _write(st, period, rows, from_january=False, start=None, tag=""):
    """The rows as a pyair2stream input file. `from_january`: dates relabelled to start on 1 January of
    the first year, as the original program reads a record (by row, assuming it starts on 1 January).
    `start`: drop the rows before this date. `tag` keeps the files of parallel jobs apart."""
    if start is not None:
        rows = rows[rows.date >= pd.Timestamp(start)]
    dates = (pd.date_range(f"{rows.date.iloc[0].year}-01-01", periods=len(rows), freq="D") if from_january
             else rows.date)
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f"v15{tag}_{st}_{period}{'_jan' if from_january else ''}{'_trim' if start else ''}.csv")
    pd.DataFrame({"Date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"), "T_air": rows.ta.to_numpy(),
                  "T_water": rows.tw_obs.to_numpy(), "Discharge": rows.q.to_numpy()}).to_csv(path, index=False)
    return path, rows


def _simulate(path, par, rows, name):
    """FORWARD run with the period's own mean discharge as Qmedia (as the original program computes it)."""
    q = rows.q[rows.q > 0].mean()
    with quiet():
        d = simulate(path, 8, par, "CRN", float(q), objective="RMS", name=name)
    return np.asarray(d.Twat_mod[365:], float)


def _stats(obs, sim, mask=None):
    m = np.isfinite(obs) & np.isfinite(sim) & (True if mask is None else mask)
    if not m.any():
        return np.nan, np.nan
    e = sim[m] - obs[m]
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(e))


def _windows(rows):
    """Masks of the paper's heat-dome and drought windows (day of year, as in its scripts) for a period."""
    doy = rows.date.dt.dayofyear.to_numpy()
    year = rows.date.dt.year.to_numpy()
    hd = (pd.Timestamp(HEAT_DOME[0]).dayofyear, pd.Timestamp(HEAT_DOME[1]).dayofyear)
    ad = (pd.Timestamp(DROUGHT[0]).dayofyear, pd.Timestamp(DROUGHT[1]).dayofyear)
    return {"heat dome": (doy >= hd[0]) & (doy <= hd[1]) & (year < 2022),
            "autumn drought": (doy >= ad[0]) & (doy <= ad[1]) & (year != 2021)}


def _station(args):
    st, quick = args
    ts, par_table, _ = _load()
    par = [float(par_table.loc[st, f"a{j}"]) for j in range(1, 9)]
    g = ts[ts.station_number == st].sort_values("date").reset_index(drop=True)
    out = {"station": st, "series": [], "de": None}
    for period in ("calibration", "validation"):
        rows = g[g.period == period].reset_index(drop=True)
        path, rows = _write(st, period, rows)
        sim = _simulate(path, par, rows, f"v15_{st}")
        theirs, obs = rows.tw_sim_8.to_numpy(), rows.tw_obs.to_numpy()
        starts_jan = rows.date.iloc[0].month == 1 and rows.date.iloc[0].day == 1
        rec = {"station": st, "period": period, "first day": rows.date.iloc[0].strftime("%Y-%m-%d"),
               "starts on 1 January": starts_jan, "days": len(rows), "measured": int(np.isfinite(obs).sum()),
               "largest difference": float(np.max(np.abs(sim - theirs)))}
        rec["RMSE published"], rec["MBE published"] = _stats(obs, theirs)
        rec["RMSE pyair2stream"], rec["MBE pyair2stream"] = _stats(obs, sim)
        for w, m in _windows(rows).items():
            rec[f"{w}: RMSE published"], _ = _stats(obs, theirs, m)
            rec[f"{w}: RMSE pyair2stream"], _ = _stats(obs, sim, m)
        if not starts_jan:
            path_j, _ = _write(st, period, rows, from_january=True)
            sim_j = _simulate(path_j, par, rows, f"v15_{st}_jan")
            rec["largest difference, read from 1 January"] = float(np.max(np.abs(sim_j - theirs)))
        out["series"].append(rec)

    # Part C: DE calibration on the calibration period (whole years from the first 1 January), RMS objective.
    cal = g[g.period == "calibration"].reset_index(drop=True)
    first = cal.date.iloc[0]
    start = None if (first.month == 1 and first.day == 1) else f"{first.year + 1}-01-01"
    path_c, rows_c = _write(st, "calibration", cal, start=start)
    extra = {"optimization": {"n_run": 40, "n_particles": 8}} if quick else {"optimization": dict(DE_SETTINGS)}
    with quiet():
        d = calibrate(path_c, 8, objective="RMS", integrator="CRN", name=f"v15_de_{st}", **extra)
    de_par = [float(x) for x in d.par_best]
    obs_c = rows_c.tw_obs.to_numpy()
    rmse_de, _ = _stats(obs_c, _simulate(path_c, de_par, rows_c, f"v15_{st}_de"))
    rmse_pub, _ = _stats(obs_c, _simulate(path_c, par, rows_c, f"v15_{st}_pub"))
    val = g[g.period == "validation"].reset_index(drop=True)
    path_v, val = _write(st, "validation", val)
    rmse_de_val, _ = _stats(val.tw_obs.to_numpy(), _simulate(path_v, de_par, val, f"v15_{st}_dev"))
    diff = 100 * float(np.max(np.abs(np.array(de_par) - np.array(par)) / RANGE))
    out["de"] = {"station": st, "calibration from": rows_c.date.iloc[0].strftime("%Y-%m-%d"),
                 "published parameters, calibration RMSE": rmse_pub, "DE, calibration RMSE": rmse_de,
                 "DE minus published": rmse_de - rmse_pub, "DE, validation RMSE": rmse_de_val,
                 "largest parameter difference (% of range)": diff,
                 "same parameters": "yes" if diff <= 100 * MATCH else "no",
                 **{f"a{j + 1}": de_par[j] for j in range(8)}}
    return out


def _calibration_config(st, tag, quick, **extra):
    """Part D: the station's calibration years from the first 1 January, as in part C, with the RMS
    objective, Crank-Nicolson, the authors' parameter ranges and the calibration Qmedia (as V2 part F)."""
    ts, _, _ = _load()
    cal = ts[(ts.station_number == st) & (ts.period == "calibration")].sort_values("date").reset_index(drop=True)
    first = cal.date.iloc[0]
    start = None if (first.month == 1 and first.day == 1) else f"{first.year + 1}-01-01"
    path, rows = _write(st, "calibration", cal, start=start, tag=tag)
    settings = {"n_run": 40, "n_particles": 8} if quick else dict(DE_SETTINGS)
    return {"version": 8, "integrator": "CRN", "objective_function": "RMS", "random_seed": 1,
            "Qmedia": float(rows.q[rows.q > 0].mean()), "parameter_bounds": AUTHORS_BOUNDS,
            "optimization": settings, "paths": {"input_data": path, "output_dir": os.path.join(WORK, tag, "out")},
            **extra}


def _mcmc(args):
    """90% credible interval of every parameter from DE-MCMC with the least-squares likelihood, widened
    for the autocorrelation of the errors (as V2 part F)."""
    st, quick = args
    from pyair2stream.optimization import DE_MCMC_mode
    tag = f"v15_mcmc_{st}"
    cfg = _calibration_config(st, tag, quick, run_mode="DE-MCMC",
                              uncertainty_options={"noise_model": "ar1", "likelihood": "least_squares",
                                                   "strict_convergence": False})
    cfg["optimization"].update({"mcmc_walkers": 16 if quick else MCMC_WALKERS, "mcmc_steps": 600 if quick else MCMC_STEPS})
    data = load(cfg, tag)
    with quiet():
        DE_MCMC_mode(data, seed=1)
    out = os.path.join(WORK, tag, "out")
    meta = json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))
    chain = pd.read_csv(os.path.join(out, "MCMC_chain_S_c_1d.csv"))
    iv = {int(c.split("_")[1]) - 1: tuple(float(x) for x in np.percentile(chain[c], [5, 95])) for c in chain.columns}
    return "mcmc", st, {"intervals": iv, "converged": bool(meta["converged"]), "steps": int(meta["steps_run"]),
                        "tau": meta.get("max_autocorr_time"), "rhat": meta.get("max_split_rhat")}


def _jackknife(args):
    """90% jackknife interval of every parameter from leave-one-year-out cross-validation of the
    calibration years (every year but the first held out once)."""
    st, quick = args
    from pyair2stream.cross_validation import cross_validate
    tag = f"v15_cv_{st}"
    cfg = _calibration_config(st, tag, quick, run_mode="DE",
                              cross_validation={"enabled": True, "unit": "year", "min_train_years": 0,
                                                "skip_first_year": True})
    data = load(cfg, tag)
    with quiet():
        table = cross_validate(data, "DE").set_index("fold")
    folds = [f for f in table.index if str(f).isdigit()]
    iv = {j: (float(table.loc["jackknife_90_lower", f"p{j + 1}"]), float(table.loc["jackknife_90_upper", f"p{j + 1}"]))
          for j in range(8)}
    return "jackknife", st, {"intervals": iv, "folds": len(folds)}


def run(ctx) -> Result:
    res = Result(
        code="V15", title="Reproduces an independent group's published simulations (British Columbia)",
        question="Callahan and Moore (2025) calibrated air2stream's 8-parameter version for 23 streams in British "
                 "Columbia and published their parameters, inputs and simulated water temperatures. Given the "
                 "same parameters and inputs, does pyair2stream compute the same daily water temperatures? Do "
                 "the published errors, including those of the 2021 heat dome and the 2022 autumn drought, "
                 "follow? Does pyair2stream's own calibration fit at least as well?",
        method="The published dataset (Moore and Callahan, 2024, https://doi.org/10.5281/zenodo.14502248, CC BY "
               "4.0; data/british_columbia/, files unchanged) gives, for each of 23 Water Survey of Canada "
               "stations, daily ERA5 air temperature, discharge, measured water temperature and the water "
               "temperature simulated by the 8-parameter version, for the calibration (to 2020) and validation "
               "(2021 to 31 October 2022) periods, and the calibrated parameters. (A) Each station and period is "
               "simulated by pyair2stream with the published parameters (Crank-Nicolson, the period's own mean "
               "discharge as Qmedia, as the original program computes it) and compared day by day with the "
               "published simulation. A record that does not start on 1 January is also run as the original "
               "program reads it: by row, as if it started on 1 January. (B) RMSE and mean bias error (MBE, "
               "simulated minus measured) are computed for each period and for the paper's two windows, as in "
               "its scripts: 25 June to 2 July (heat dome) and 1 September to 31 October (autumn drought), by "
               "day of year, over the calibration years and 2021 (heat dome) or 2022 (drought). (C) Each station "
               "is calibrated by DE (RMS objective, Crank-Nicolson, the authors' parameter ranges of V2, which "
               "contain every published parameter) on its calibration years, from the first 1 January, and the "
               "fit and the parameters compared with the published ones on the same days. (D) How far can the "
               "parameters move and still fit these data? As in V2 part F, each station's calibration years give "
               "two 90% intervals for every parameter around the same least-squares estimator: the jackknife "
               "interval from leave-one-year-out cross-validation (every year but the first held out once), and "
               f"the DE-MCMC credible interval with the least-squares likelihood widened for the autocorrelation "
               f"of the errors ({MCMC_WALKERS} walkers, run until converged, at most {MCMC_STEPS} steps). Whether "
               "each published value lies inside them is recorded.",
        criterion=f"(A) Every published series whose record starts on 1 January is reproduced to within "
                  f"{TOL_SERIES} °C on every day, and a record that does not is reproduced to within "
                  f"{TOL_SERIES} °C when read as the original program reads it. (C) DE fits every calibration "
                  f"period no worse than the published parameters (their RMSE + {TOL_DE} °C). (B), (D), the "
                  f"parameter values found in (C) and the effect of a misread record are reported.")
    _, par_table, info = _load()
    stations = list(QUICK_STATIONS) if ctx.quick else list(par_table.index)
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            # The MCMC runs are the longest: start them all first.
            futures_d = [ex.submit(fn, (st, ctx.quick)) for fn in (_mcmc, _jackknife) for st in stations]
            outs = list(ex.map(_station, [(st, ctx.quick) for st in stations]))
            found = {(kind, st): r for kind, st, r in (f.result() for f in futures_d)}
    res.seconds = t.seconds
    series = pd.DataFrame([r for o in outs for r in o["series"]])
    de = pd.DataFrame([o["de"] for o in outs])
    series["regime"] = series.station.map(info["regime"])
    series["regulation"] = series.station.map(info["status"])

    jan = series[series["starts on 1 January"]]
    other = series[~series["starts on 1 January"]]
    ok_a = bool((jan["largest difference"] <= TOL_SERIES).all())
    if len(other):
        ok_a &= bool((other["largest difference, read from 1 January"] <= TOL_SERIES).all())
    ok_c = bool((de["DE minus published"] <= TOL_DE).all())
    res.passed = ok_a and ok_c

    res.summary = (
        f"(A) Of {len(series)} published series ({len(stations)} stations, two periods), the {len(jan)} whose "
        f"record starts on 1 January are reproduced to within {jan['largest difference'].max():.5f} °C on every "
        f"day. ")
    for _, r in other.iterrows():
        res.summary += (
            f"Station {r['station']}'s {r['period']} record starts on {r['first day']}: with its dates, "
            f"pyair2stream's simulation differs from the published one by up to {r['largest difference']:.2f} °C; "
            f"read by row as if it started on 1 January, as the original program reads it, it is reproduced to "
            f"within {r['largest difference, read from 1 January']:.5f} °C. The published simulation of that "
            f"period therefore has its seasonal term {_months_out(r['first day'])} months out of phase. ")
    res.summary += (
        f"(C) DE fits the calibration years at least as well as the published parameters at "
        f"{int((de['DE minus published'] <= TOL_DE).sum())} of {len(de)} stations (mean difference "
        f"{de['DE minus published'].mean():+.3f} °C), and returns the published parameters (each within "
        f"{MATCH:.0%} of its range) at {int((de['same parameters'] == 'yes').sum())} of {len(de)}.")

    # Part D: are the published parameters inside pyair2stream's 90% intervals?
    names = [f"a{j}" for j in range(1, 9)]
    best_all = de.set_index("station")[names].astype(float)
    detail, summ, plot = [], [], []
    tot = {"mc": 0, "mc_n": 0, "jk": 0, "any": 0, "n": 0}
    for st in stations:
        pub = np.array([float(par_table.loc[st, n]) for n in names])
        best = best_all.loc[st].to_numpy()
        mc, jk = found[("mcmc", st)], found[("jackknife", st)]
        in_mc = in_jk = in_any = 0
        for j in range(8):
            m_lo, m_hi = mc["intervals"][j]
            j_lo, j_hi = jk["intervals"][j]
            a = _inside(pub[j], m_lo, m_hi, j) if mc["converged"] else None
            b = _inside(pub[j], j_lo, j_hi, j)
            in_mc += bool(a)
            in_jk += bool(b)
            in_any += bool(a) or bool(b)
            detail.append({"station": st, "parameter": names[j], "published": f"{pub[j]:.3f}",
                           "pyair2stream best fit": f"{best[j]:.3f}",
                           "MCMC 90% interval": f"{m_lo:.3f} to {m_hi:.3f}",
                           "published inside (MCMC)": "not converged" if a is None else a,
                           "jackknife 90% interval": f"{j_lo:.3f} to {j_hi:.3f}", "published inside (jackknife)": b})
            plot.append({"station": st, "j": j, "mcmc": (pub[j] - m_lo) / (m_hi - m_lo) if m_hi > m_lo else np.nan,
                         "jk": (pub[j] - j_lo) / (j_hi - j_lo) if j_hi > j_lo else np.nan,
                         "mcmc_ok": mc["converged"]})
        tot["mc"] += in_mc if mc["converged"] else 0
        tot["mc_n"] += 8 if mc["converged"] else 0
        tot["jk"] += in_jk
        tot["any"] += in_any
        tot["n"] += 8
        summ.append({"station": st, "MCMC converged (steps)": f"{'yes' if mc['converged'] else 'NO'} ({mc['steps']})",
                     "longest autocorrelation time (steps)": "" if mc["tau"] is None else f"{mc['tau']:.0f}",
                     "largest split-Rhat": "" if mc["rhat"] is None else f"{mc['rhat']:.3f}",
                     "published inside MCMC interval": f"{in_mc} of 8" if mc["converged"] else "not converged",
                     "cross-validation folds": jk["folds"], "published inside jackknife interval": f"{in_jk} of 8",
                     "inside at least one": f"{in_any} of 8"})
    res.summary += (
        f" (D) The published values lie inside pyair2stream's 90% parameter intervals for {tot['mc']} of "
        f"{tot['mc_n']} (MCMC, converged runs) and {tot['jk']} of {tot['n']} (jackknife); {tot['any']} of "
        f"{tot['n']} lie inside at least one.")

    # Tables
    a_cols = ["station", "regime", "regulation", "period", "first day", "days", "measured", "largest difference"]
    if len(other):
        a_cols.append("largest difference, read from 1 January")
    shown_a = series[a_cols].copy()
    shown_a["largest difference"] = shown_a["largest difference"].map(lambda x: f"{x:.2e}")
    if len(other):
        shown_a["largest difference, read from 1 January"] = shown_a["largest difference, read from 1 January"].map(
            lambda x: "" if pd.isna(x) else f"{x:.2e}")
    b_cols = ["station", "period", "RMSE published", "RMSE pyair2stream", "MBE published", "MBE pyair2stream",
              "heat dome: RMSE published", "heat dome: RMSE pyair2stream",
              "autumn drought: RMSE published", "autumn drought: RMSE pyair2stream"]
    shown_b = series[b_cols].copy()
    for c in b_cols[2:]:
        shown_b[c] = shown_b[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    means = []
    groups = [("calibration", "all records"), ("calibration", "records starting on 1 January"),
              ("validation", "all records")]
    for period, which in groups:
        s = series[series.period == period]
        if which != "all records":
            s = s[s["starts on 1 January"]]
        means.append({"period": period, "records": which, "stations": len(s),
                      "mean RMSE, published series": round(float(s["RMSE published"].mean()), 3),
                      "mean RMSE, pyair2stream": round(float(s["RMSE pyair2stream"].mean()), 3),
                      "heat dome, mean RMSE (published)": round(float(s["heat dome: RMSE published"].mean()), 3),
                      "heat dome, mean RMSE (pyair2stream)": round(float(s["heat dome: RMSE pyair2stream"].mean()), 3),
                      "drought, mean RMSE (published)": round(float(s["autumn drought: RMSE published"].mean()), 3),
                      "drought, mean RMSE (pyair2stream)": round(float(s["autumn drought: RMSE pyair2stream"].mean()), 3)})
    means = pd.DataFrame(means)
    shown_c = de.copy()
    for c in ("published parameters, calibration RMSE", "DE, calibration RMSE", "DE, validation RMSE"):
        shown_c[c] = shown_c[c].map(lambda x: f"{x:.4f}")
    shown_c["DE minus published"] = shown_c["DE minus published"].map(lambda x: f"{x:+.4f}")
    shown_c["largest parameter difference (% of range)"] = shown_c["largest parameter difference (% of range)"].map(
        lambda x: f"{x:.1f}")
    for j in range(1, 9):
        shown_c[f"a{j}"] = shown_c[f"a{j}"].map(lambda x: f"{x:.3f}")
    pub_val = series[series.period == "validation"].set_index("station")["RMSE published"]
    shown_c.insert(5, "published, validation RMSE", shown_c.station.map(pub_val).map(lambda x: f"{x:.4f}"))

    res.sections.append(Section(
        "A. The published simulations, day by day",
        "Largest absolute difference, over every day of the period, between pyair2stream's simulation with the "
        "published parameters and the published simulation.",
        figures=[_fig(series)], tables=[("A. Published series reproduced", shown_a)]))
    res.sections.append(Section(
        "B. The published errors, including the heat dome and the drought",
        "RMSE and MBE (°C) of the published series and of pyair2stream's, against the measurements. The windows "
        "follow the paper's scripts. pyair2stream's errors for a record that does not start on 1 January are "
        "those of the record run with its correct dates, so the means are also given without it. The paper's own "
        "summary statistics are not compared here (its text could not be retrieved automatically); these are "
        "computed from its published data.",
        tables=[("B. Mean over stations", means), ("B. By station and period", shown_b)]))
    res.sections.append(Section(
        "C. Calibration by pyair2stream",
        "DE with the RMS objective on each station's calibration years (from the first 1 January), compared with "
        "the published parameters on the same days, with the validation RMSE of the DE parameters. 'Same "
        "parameters': every parameter within 1% of its range of the published value.",
        tables=[("C. DE calibration (RMSE in °C) and parameters", shown_c)]))
    res.sections.append(Section(
        "D. Do the published parameters lie inside pyair2stream's uncertainty intervals?",
        "Two 90% intervals for every parameter, from the same calibration years and least-squares estimator: the "
        "DE-MCMC credible interval (the range of values whose fit is close to the best, widened because the daily "
        "errors are correlated) and the jackknife interval from leave-one-year-out cross-validation (how far the "
        "best fit moves when each year is left out, scaled to a confidence interval). V4 tests both against a "
        "known truth. Counting rules as in V2 part F: a value on a bound of the parameter range counts as inside "
        "an interval reaching to within 1% of the range of that bound, and an MCMC run that did not converge "
        "gives no interval.",
        figures=[_fig_intervals(pd.DataFrame(plot))],
        tables=[("D. Published parameters and 90% intervals: summary by station", pd.DataFrame(summ)),
                ("D. Published parameters and 90% intervals, by parameter", pd.DataFrame(detail))]))

    if len(other):
        for _, r in other.iterrows():
            row = de[de.station == r["station"]].iloc[0]
            res.notes.append(
                f"Station {r['station']}: its calibration record starts on {r['first day']}. The original program "
                "reads a record by row and assumes it starts on 1 January, so the published calibration of this "
                f"station was made with the seasonal term {_months_out(r['first day'])} months out of phase, and "
                "its parameters compensate for that. Its validation record starts on 1 January, so there the "
                "same parameters were run with the seasonal term in phase. Calibrated by pyair2stream with the "
                f"correct dates, the station fits its calibration years with RMSE {row['DE, calibration RMSE']:.3f} °C "
                f"and predicts the validation years with {row['DE, validation RMSE']:.3f} °C, against "
                f"{pub_val.get(r['station'], np.nan):.3f} °C for the published parameters. pyair2stream refuses a "
                "calibration record that does not start on 1 January, and takes the seasonal timing from the dates "
                "in other runs.")
    same = int((de["same parameters"] == "yes").sum())
    conv = [st for st in stations if found[("mcmc", st)]["converged"]]
    dd = pd.DataFrame(detail)
    jk_in = dd["published inside (jackknife)"].astype(bool)
    in_conv = dd.station.isin(conv)
    outside = dd[in_conv & (dd["published inside (MCMC)"].astype(str) == "False")]
    lo_b, hi_b = np.array(AUTHORS_BOUNDS["min"], float), np.array(AUTHORS_BOUNDS["max"], float)
    near = [r for _, r in outside.iterrows()
            if min(abs(float(r.published) - lo_b[int(r.parameter[1]) - 1]),
                   abs(float(r.published) - hi_b[int(r.parameter[1]) - 1])) <= 0.02 * RANGE[int(r.parameter[1]) - 1]]
    note = (f"Parameters: pyair2stream's calibration returns the published parameter values at {same} of {len(de)} "
            f"stations, although it fits the same years at least as well at every station. ")
    if conv:
        note += (f"Where DE-MCMC converged ({len(conv)} stations), {tot['mc']} of {tot['mc_n']} published values lie "
                 f"inside its 90% intervals: there the published parameters fit the data about as well as the best "
                 f"fit does")
        if len(outside):
            note += (f", and of the {len(outside)} outside, {len(near)} lie within 2% of the range of a bound of the "
                     "parameter ranges, where the published calibration stopped")
        note += (f". The jackknife intervals at the same stations contain {int(jk_in[in_conv].sum())} of "
                 f"{int(in_conv.sum())}: they measure how far the best fit itself moves when a year is left out, so "
                 "the published sets lie among the good fits but not where the least-squares estimator lands. ")
    n_nc = len(stations) - len(conv)
    if n_nc:
        note += (f"At {'the other ' if conv else ''}{n_nc} stations the MCMC did not converge within {MCMC_STEPS} steps: the parameters "
                 "are too poorly determined by the data for their distribution to be sampled in that time, and only "
                 f"the jackknife is available ({int(jk_in[~in_conv].sum())} of {int((~in_conv).sum())} published "
                 "values inside). ")
    note += ("Many parameter combinations fit these data almost equally well (equifinality), so individual "
             "parameter values should not be interpreted on their own; the predictions are what part A checks. The "
             "dataset does not record how the published calibration was made (objective, parameter ranges, "
             "optimiser and its settings).")
    res.notes.append(note)
    res.notes.append(
        "Unlike V2 and V13, these results come from another group, other rivers and another climate, and the whole "
        "simulated series is compared day by day, so any difference in the equations, the numerical scheme, the "
        "discharge scaling or the handling of the first year would show. None does, apart from the misread record.")
    return res


def _fig_intervals(plot):
    """Where each published value lies in pyair2stream's 90% intervals: 0 = lower end, 1 = upper end."""
    import matplotlib.pyplot as plt
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    lo_clip, hi_clip = -2.0, 3.0
    rng = np.random.default_rng(0)
    for ax, key, title in ((axes[0], "mcmc", "DE-MCMC 90% interval"), (axes[1], "jk", "Jackknife 90% interval")):
        ax.axhspan(0, 1, color=LIGHT_GREY, alpha=0.35, lw=0, zorder=0)
        g = plot if key == "jk" else plot[plot.mcmc_ok]
        for j in range(8):
            y = g[g.j == j][key].to_numpy()
            y = y[np.isfinite(y)]
            x = j + rng.uniform(-0.2, 0.2, len(y))
            inside = (y >= 0) & (y <= 1)
            ax.scatter(x[inside], y[inside], s=12, color=BLUE, lw=0, zorder=3)
            ax.scatter(x[~inside], np.clip(y[~inside], lo_clip, hi_clip), s=12, color=ORANGE, lw=0, zorder=3)
        ax.set_xticks(range(8), [f"a{j + 1}" for j in range(8)])
        ax.set_ylim(lo_clip - 0.2, hi_clip + 0.2)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Published value's position in the interval")
    fig.tight_layout()
    return (save_figure(fig, "V15_parameter_intervals.png"),
            "Where each station's published parameter value lies in pyair2stream's 90% interval for that parameter "
            "(0: lower end, 1: upper end; shaded: inside). Orange: outside; values beyond the plotted range are "
            "drawn at its edge.")


def _months_out(first_day):
    return (pd.Timestamp(first_day).month - 1) % 12


def _fig(series):
    import matplotlib.pyplot as plt
    plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    s = series.sort_values("largest difference").reset_index(drop=True)
    colours = [BLUE if r["starts on 1 January"] else ORANGE for _, r in s.iterrows()]
    ax1.scatter(range(len(s)), np.maximum(s["largest difference"], 1e-7), s=16, c=colours, zorder=3)
    if "largest difference, read from 1 January" in s:
        for i, r in s.iterrows():
            if not r["starts on 1 January"]:
                ax1.scatter([i], [max(r["largest difference, read from 1 January"], 1e-7)], s=26, marker="s",
                            facecolor="white", edgecolor=ORANGE, zorder=4)
                ax1.annotate(f"{r['station']} {r['period']}:\nwith its dates (filled),\nread from 1 January (open)",
                             (i, r["largest difference"]), xytext=(-8, -4), textcoords="offset points",
                             ha="right", va="top", fontsize=7, color=INK2)
    ax1.axhline(TOL_SERIES, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax1.set_yscale("log")
    ax1.set_xlabel("Published series (station and period), sorted")
    ax1.set_ylabel("Largest daily difference from the published series (°C)")
    ax1.set_title("Each published simulation", fontsize=9)
    for period, colour, marker in (("calibration", BLUE, "o"), ("validation", AQUA, "s")):
        g = series[series.period == period]
        ax2.scatter(g["heat dome: RMSE published"], g["heat dome: RMSE pyair2stream"], s=18, color=colour,
                    marker=marker, label=f"heat dome window, {period}", zorder=3)
    g = series[series.period == "validation"]
    ax2.scatter(g["autumn drought: RMSE published"], g["autumn drought: RMSE pyair2stream"], s=18, color=ORANGE,
                marker="^", label="autumn drought 2022", zorder=3)
    hi = float(np.nanmax(series[["heat dome: RMSE published", "heat dome: RMSE pyair2stream",
                                 "autumn drought: RMSE published", "autumn drought: RMSE pyair2stream"]].to_numpy())) + 0.2
    ax2.plot([0, hi], [0, hi], color=LIGHT_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax2.set_xlim(0, hi)
    ax2.set_ylim(0, hi)
    ax2.set_aspect("equal")
    ax2.set_xlabel("RMSE of the published simulation (°C)")
    ax2.set_ylabel("RMSE of pyair2stream's simulation (°C)")
    ax2.set_title("Errors in the paper's extreme-weather windows", fontsize=9)
    ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return (save_figure(fig, "V15_british_columbia.png"),
            "Left: largest difference between pyair2stream's simulation and each published simulation, with the "
            f"published parameters (dashed: {TOL_SERIES} °C). Blue: records starting on 1 January. Orange: the record "
            "that does not, run with its dates (filled) and read by row from 1 January as the original program "
            "reads it (open). Right: RMSE in the paper's heat-dome and drought windows, published simulation "
            "against pyair2stream's; a point off the line is the misread record, run by pyair2stream with its "
            "correct dates.")
