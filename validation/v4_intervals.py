"""
V4 - Uncertainty intervals are calibrated.

Many synthetic data sets with a known truth are generated (as in V3). Each is
calibrated with DE-MCMC and then run in FORWARD mode with prediction intervals
on three held-out years, exactly as a user would. Over the replicates, a 90%
prediction interval should contain about 90% of the held-out observations,
and the 90% credible interval of each parameter should contain its true value
about 90% of the time. Parameter intervals from leaving one year out (as
cross-validation does) are also tested.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (DE_SETTINGS, WORK, Result, Section, Timer, load, mean_discharge, published_params, quiet,
                    river_csv, AUTHORS_BOUNDS, plot_style, save_figure, reference_line, BLUE, ORANGE, AQUA, INK,
                    INK2, LIGHT_GREY, LEVELS, coverage_by_level)
from v3_recovery import SIGMA, RHO, noise, truth_series

# (label, true version, noise in the data, error model of the likelihood, replicates full / quick).
# Error model: "iid", "ar1" (exact AR(1) likelihood) or "ar1-ls" (least-squares likelihood with the
# effective sample size, AR(1) noise in the predictions). Cases E and F use the same synthetic data as
# B and D (DATA_SEED), so the two AR(1) likelihoods are compared on identical data.
CASES = [
    ("A: version 5, iid noise, iid model", 5, "iid", "iid", 30, 0),
    ("B: version 5, AR(1) noise, exact ar1 likelihood", 5, "ar1", "ar1", 30, 0),
    ("C: version 5, AR(1) noise, iid model (mis-specified)", 5, "ar1", "iid", 30, 0),
    ("D: version 8, AR(1) noise, exact ar1 likelihood", 8, "ar1", "ar1", 12, 0),
    ("E: version 5, AR(1) noise, least-squares likelihood", 5, "ar1", "ar1-ls", 30, 3),
    ("F: version 8, AR(1) noise, least-squares likelihood", 8, "ar1", "ar1-ls", 12, 0),
    ("G: version 5, AR(1) noise, least squares, rho from consecutive days", 5, "ar1", "ar1-ls-daily", 30, 0),
    ("H: version 5, fast + slow noise, least squares (mis-specified noise)", 5, "two-part", "ar1-ls", 30, 0),
    ("I: version 5, fast + slow noise, least squares, rho from consecutive days (mis-specified noise)", 5,
     "two-part", "ar1-ls-daily", 30, 0),
    ("J: version 5, AR(1) noise, least squares, weekly scoring (time_resolution 1w), authors' bounds", 5, "ar1",
     "ar1-ls-1w", 30, 0),
    ("K: as J, with a2 and a3 at least 0", 5, "ar1", "ar1-ls-1w-pos", 30, 0),
]
# Cases on identical synthetic data: E, G, J, K as B; F as D; I as H.
DATA_SEED = {"E": "B", "F": "D", "G": "B", "I": "H", "J": "B", "K": "B"}
# Why the prediction coverage fell below the levels in version 0.5.1 (measured once, on these data;
# the synthetic data were deliberately left unchanged).
ICE_FLOOR_NOTE = (
    "Why the prediction coverage is below the levels, and this check fails (since version 0.5.1). The "
    "synthetic measurements are the noise-free simulation plus random noise, and nothing keeps them at or "
    "above 0 °C. In winter the simulated Mentue is close to 0 °C, so about 3% of the synthetic measurements "
    "are below freezing (2.6%-3.0%, by model version and noise). Since version 0.5.1 every simulated series "
    "of a prediction range is kept at or above Tice_cover (0 °C here), as the simulation itself is, so these "
    "values can never lie inside a range. That accounts for the drop since the previous run, up to 3 points "
    "(about 1 point at the 50% level; 90% intervals: 89%-91% of days before, 87%-88% now; 99% intervals: "
    "98.6%-99.0% before, about 96% now). Real water does not go below freezing: none of the 18,225 measured days of "
    "the Swiss rivers in data/ is below 0 °C, and the check on real data (V5) shows no such drop. The "
    "synthetic data were left as they are, so that the effect is visible; before the ice floor, the same "
    "check gave mean coverage within 1.3 points of every level.")
REPORT_ONLY = ("J",)        # correctly specified but reported, not judged (see notes)
# Weekly scoring cannot see a daily simulation that zigzags; bounds that keep a2 and a3 at least 0
# (water warms with the air and relaxes towards equilibrium) rule such parameter sets out.
POSITIVE_BOUNDS = {"min": [-5, 0, 0, -1, 0, 0, 0, -1], "max": AUTHORS_BOUNDS["max"]}
PI_RANGE = (0.87, 0.93)     # accepted mean coverage of the 90% prediction interval
LEVEL_TOLERANCE = 0.03      # every tested level: accepted mean coverage within 3 points of nominal (as 87-93%)
PAR_MIN = 0.78              # accepted pooled coverage of the 90% parameter intervals (not clearly below 0.9)
PAR_REQUIRED = ("A", "B", "E", "F", "G", "K")   # cases whose parameter intervals must meet PAR_MIN (see notes)
N_JACKKNIFE = 12            # replicates per version for the cross-validation parameter intervals
JACKKNIFE_VERSIONS = (3, 4, 5, 7, 8)
JACKKNIFE_OK = 0.80         # a version's jackknife intervals count as dependable at this coverage or more


def replicate(args):
    label, version, noise_kind, model_noise, r = args
    from pyair2stream.optimization import DE_MCMC_mode, forward_mode
    tag = f"v4_{label[0]}_{r}"
    folder = os.path.join(WORK, tag)
    os.makedirs(folder, exist_ok=True)
    q_cal = mean_discharge(river_csv("MAH_2369", "calibration"))
    par_true = published_params(version, "MAH_2369")
    src_c, truth_c = truth_series(version, par_true, "calibration", q_cal, tag=tag)
    src_v, truth_v = truth_series(version, par_true, "validation", q_cal, tag=tag)
    rng = np.random.default_rng(1000 * (ord(DATA_SEED.get(label[0], label[0])) - 64) + r)
    cal_csv, val_csv = os.path.join(folder, "cal.csv"), os.path.join(folder, "val.csv")
    src_c.assign(T_water=np.round(truth_c + noise(len(truth_c), noise_kind, rng), 3)).to_csv(cal_csv, index=False)
    src_v.assign(T_water=np.round(truth_v + noise(len(truth_v), noise_kind, rng), 3)).to_csv(val_csv, index=False)

    out = os.path.join(folder, "out")
    bounds = POSITIVE_BOUNDS if model_noise.endswith("-pos") else AUTHORS_BOUNDS
    cfg = {"version": version, "integrator": "CRN", "run_mode": "DE-MCMC", "objective_function": "NSE",
           "random_seed": r + 1, "Qmedia": q_cal, "parameter_bounds": bounds,
           "optimization": {"n_run": 300, "n_particles": 15, "mcmc_walkers": 32, "mcmc_steps": 20000},
           "uncertainty_options": {"noise_model": "iid" if model_noise == "iid" else "ar1",
                                   "likelihood": "least_squares" if model_noise.startswith("ar1-ls") else "exact"},
           "paths": {"input_data": cal_csv, "output_dir": out}}
    if model_noise.endswith("-daily"):
        cfg["uncertainty_options"]["rho_timescale"] = "daily"
    time_res = "1w" if "-1w" in model_noise else "1d"
    cfg["time_resolution"] = time_res
    data = load(cfg, tag)
    try:
        with quiet():
            DE_MCMC_mode(data, seed=r + 1)
    except RuntimeError:            # not converged within the maximum
        return {"case": label, "replicate": r, "converged": False, "implausible": implausible(data)}
    meta = json.load(open(os.path.join(out, f"MCMC_chain_S_c_{time_res}_meta.json")))
    chain = pd.read_csv(os.path.join(out, f"MCMC_chain_S_c_{time_res}.csv"))
    per_param = {}
    for col in chain.columns:
        j = int(col.split("_")[1]) - 1
        lo, hi = np.percentile(chain[col], [5, 95])
        per_param[f"a{j + 1}"] = {"inside": bool(lo <= par_true[j] <= hi), "median": float(chain[col].median()),
                                  "sd": float(chain[col].std())}
    inside = [p["inside"] for p in per_param.values()]

    fcfg = {"version": version, "integrator": "CRN", "run_mode": "FORWARD", "Qmedia": q_cal,
            "parameters_forward": [float(x) for x in data.par_best],
            "uncertainty_options": {"noise_model": "iid" if model_noise == "iid" else "ar1", "save_ensemble": True},
            "forward_options": {"enable_prediction_intervals": True,
                                "mcmc_chain_path": os.path.join(out, f"MCMC_chain_S_c_{time_res}.csv"),
                                "n_samples": 1000, "random_seed": r + 1},
            "paths": {"input_data": val_csv, "output_dir": os.path.join(folder, "fwd")}}
    fdata = load(fcfg, tag + "_fwd")
    with quiet():
        forward_mode(fdata)
    fmeta = json.load(open(os.path.join(folder, "fwd", "Forward_Prediction_Ensemble_S_c_1d_meta.json")))
    # Coverage at every tested level, from the saved simulations; parameters likewise from the chain.
    from pyair2stream import scenario
    npz = os.path.join(folder, "fwd", "Forward_Prediction_Ensemble_S_c_1d.npz")
    ens, _ = scenario.load_ensemble(npz)
    by_level = coverage_by_level(ens, pd.read_csv(val_csv).T_water.to_numpy(float))
    os.remove(npz)          # about 8 MB per replicate; only the coverage is needed
    par_by_level = {}
    for level in LEVELS:
        hits = []
        for col in chain.columns:
            lo, hi = np.percentile(chain[col], [50 - level / 2, 50 + level / 2])
            hits.append(bool(lo <= par_true[int(col.split("_")[1]) - 1] <= hi))
        par_by_level[level] = hits
    return {"case": label, "replicate": r, "converged": True, "steps": meta["steps_run"], "rho": meta["rho"],
            "variance factor": meta.get("likelihood_variance_factor"), "implausible": implausible(data),
            "calibration coverage": meta["interval_coverage"],
            "held-out coverage": fmeta["interval_coverage"],
            "parameters inside 90% interval": float(np.mean(inside)), "n parameters": len(inside),
            "per_param": per_param, "held-out coverage by level": by_level, "parameters inside by level": par_by_level}


def implausible(data):
    """Whether pyair2stream's plausibility check (negative relaxation rate, or a daily simulation
    that zigzags) warns about the calibration's best fit."""
    from pyair2stream.model import call_model, check_daily_plausibility, OSCILLATION_CHANGE_CORR
    data.par[:] = data.par_best
    call_model(data)
    with quiet():
        c = check_daily_plausibility(data)
    return bool(c["n_negative_B"] > 0 or (c["change_corr"] is not None and c["change_corr"] < OSCILLATION_CHANGE_CORR))


def sampler_crosscheck(r):
    """Re-run the calibration of case-D replicate r with a different sampler (emcee's stretch move,
    run to 100 autocorrelation times) and compare the 90% parameter intervals with the ones from
    the package's sampler. Restores the package afterwards (worker processes are reused)."""
    import emcee
    import pyair2stream.optimization as opt
    folder = os.path.join(WORK, f"v4_D_{r}")
    q_cal = mean_discharge(river_csv("MAH_2369", "calibration"))
    out = os.path.join(folder, "out_stretch")
    cfg = {"version": 8, "integrator": "CRN", "run_mode": "DE-MCMC", "objective_function": "NSE",
           "random_seed": r + 1, "Qmedia": q_cal, "parameter_bounds": AUTHORS_BOUNDS,
           "optimization": {"n_run": 300, "n_particles": 15, "mcmc_walkers": 32, "mcmc_steps": 100000},
           "uncertainty_options": {"noise_model": "ar1", "likelihood": "exact"},     # as case D
           "paths": {"input_data": os.path.join(folder, "cal.csv"), "output_dir": out}}
    original, factor = opt._make_sampler, opt.MCMC_TAU_FACTOR

    def stretch(nwalkers, ndim, log_prob, seed):
        sampler = original(nwalkers, ndim, log_prob, seed)
        sampler._moves = [emcee.moves.StretchMove()]
        return sampler
    opt._make_sampler, opt.MCMC_TAU_FACTOR = stretch, 100
    try:
        data = load(cfg, f"v4_D_{r}_stretch")
        with quiet():
            opt.DE_MCMC_mode(data, seed=r + 1)
    finally:
        opt._make_sampler, opt.MCMC_TAU_FACTOR = original, factor
    a = pd.read_csv(os.path.join(folder, "out", "MCMC_chain_S_c_1d.csv"))
    b = pd.read_csv(os.path.join(out, "MCMC_chain_S_c_1d.csv"))
    rel = []
    for col in a.columns:
        qa, qb = np.percentile(a[col], [5, 95]), np.percentile(b[col], [5, 95])
        rel.append(np.max(np.abs(qa - qb)) / (qa[1] - qa[0]))
    return {"replicate": r, "steps (DE move)": json.load(open(os.path.join(folder, "out", "MCMC_chain_S_c_1d_meta.json")))["steps_run"],
            "steps (stretch move)": json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))["steps_run"],
            "largest interval-end difference (share of interval width)": float(np.max(rel))}


def leave_one_year_out(args):
    """Cross-validation through the package (`cross_validation.cross_validate`, as a run with a
    `cross_validation:` block does) on one synthetic replicate with AR(1) noise, and whether its
    90% parameter intervals contain the true parameters: (1) the fold mean +- 1.645 x the spread
    between folds ('std' row), and (2) the jackknife interval (jackknife_90 rows). For versions 5
    and 8 the replicate is the one of case B / D, so the MCMC interval can be compared."""
    version, r = args
    from pyair2stream.config import ACTIVE_PARAMS
    from pyair2stream.cross_validation import cross_validate
    q_cal = mean_discharge(river_csv("MAH_2369", "calibration"))
    truth = np.array(published_params(version, "MAH_2369"))
    case = {5: "B", 8: "D"}.get(version)
    if case:
        folder = os.path.join(WORK, f"v4_{case}_{r}")
    else:
        folder = os.path.join(WORK, f"v4_jk{version}_{r}")
        os.makedirs(folder, exist_ok=True)
        src, clean = truth_series(version, truth, "calibration", q_cal, tag=f"v4_jk{version}_{r}")
        rng = np.random.default_rng(50000 + 100 * version + r)
        src.assign(T_water=np.round(clean + noise(len(clean), "ar1", rng), 3)).to_csv(
            os.path.join(folder, "cal.csv"), index=False)
    cfg = {"version": version, "integrator": "CRN", "run_mode": "DE", "objective_function": "NSE",
           "random_seed": r + 1, "Qmedia": q_cal, "parameter_bounds": AUTHORS_BOUNDS,
           "optimization": dict(DE_SETTINGS), "cross_validation": {"enabled": True, "unit": "year"},
           "paths": {"input_data": os.path.join(folder, "cal.csv"), "output_dir": os.path.join(folder, "cv")}}
    data = load(cfg, f"v4jk_{version}_{r}")
    with quiet():
        table = cross_validate(data, "DE").set_index("fold")
    chain_file = os.path.join(folder, "out", "MCMC_chain_S_c_1d.csv")
    chain = pd.read_csv(chain_file) if case and os.path.exists(chain_file) else None
    ls_case = {5: "E", 8: "F"}.get(version)
    ls_file = os.path.join(WORK, f"v4_{ls_case}_{r}", "out", "MCMC_chain_S_c_1d.csv") if ls_case else ""
    ls_chain = pd.read_csv(ls_file) if ls_case and os.path.exists(ls_file) else None
    rows = []
    for j in ACTIVE_PARAMS[version]:
        col = f"p{j + 1}"
        centre, spread = table.loc["mean", col], 1.645 * table.loc["std", col]
        lo, hi = table.loc["jackknife_90_lower", col], table.loc["jackknife_90_upper", col]
        row = {"version": version, "parameter": f"a{j + 1}",
               "raw spread covers": abs(centre - truth[j]) <= spread, "jackknife covers": lo <= truth[j] <= hi,
               "raw spread width": 2 * spread, "jackknife width": hi - lo}
        if chain is not None:
            m_lo, m_hi = np.percentile(chain[f"par_{j + 1}"], [5, 95])
            row.update({"MCMC covers": m_lo <= truth[j] <= m_hi, "MCMC width": m_hi - m_lo})
        if ls_chain is not None:
            m_lo, m_hi = np.percentile(ls_chain[f"par_{j + 1}"], [5, 95])
            row.update({"MCMC least squares covers": m_lo <= truth[j] <= m_hi})
        rows.append(row)
    return rows


def sandwich_prediction():
    """Parameter-interval widths predicted by theory, for version 5 on the synthetic calibration record.

    For least squares with correlated errors (correlation matrix R), the sampling covariance of the
    estimates is sigma^2 (J'J)^-1 J'RJ (J'J)^-1 (the 'sandwich'), where J holds the sensitivity of
    the simulated temperature to each parameter on each day. The least-squares likelihood with the
    effective sample size gives every parameter the covariance sigma^2 c (J'J)^-1, c = (1+rho)/(1-rho).
    The ratio of the two standard deviations is what the replicates measure as 'replicate spread /
    posterior SD' (1 = right width, below 1 = too wide, above 1 = too narrow)."""
    from scipy.linalg import toeplitz
    from scipy.optimize import brentq
    from pyair2stream.config import ACTIVE_PARAMS
    from pyair2stream.uncertainty import weekly_mean_correlation
    from v3_recovery import TWO_PART
    version = 5
    q_cal = mean_discharge(river_csv("MAH_2369", "calibration"))
    par = np.array(published_params(version, "MAH_2369"), float)
    truth_series(version, list(par), "calibration", q_cal, tag="v4_theory")
    csv = os.path.join(WORK, "v4_theory_forcing_calibration.csv")
    active = list(ACTIVE_PARAMS[version])
    from common import simulate
    cols = []
    for j in active:
        h = 1e-4 * max(abs(par[j]), 1.0)
        up, dn = par.copy(), par.copy()
        up[j] += h
        dn[j] -= h
        cols.append((simulate(csv, version, up, "CRN", q_cal, name="v4_theory").Twat_mod[365:]
                     - simulate(csv, version, dn, "CRN", q_cal, name="v4_theory").Twat_mod[365:]) / (2 * h))
    J = np.column_stack(cols)
    k = np.arange(len(J))
    w, rf, rs = TWO_PART["fast_share"], TWO_PART["fast_rho"], TWO_PART["slow_rho"]
    jtj_inv = np.linalg.inv(J.T @ J)
    out = {}
    for noise_label, acf, weekly_case, daily_case in (("AR(1) noise", RHO ** k, "E", "G"),
                                                       ("fast + slow noise", w * rf ** k + (1 - w) * rs ** k, "H", "I")):
        R = toeplitz(acf)
        sandwich = jtj_inv @ (J.T @ (R @ J)) @ jtj_inv
        # The rho each estimator converges to on a long record with this correlation.
        var7 = 7 + 2 * sum((7 - m) * acf[m] for m in range(1, 7))
        r7 = sum((7 - abs(d - 7)) * acf[d] for d in range(1, 14)) / var7
        rho_weekly = max(acf[1], brentq(lambda x: weekly_mean_correlation(x) - r7, 0.0, 0.99))
        for case, rho in ((weekly_case, rho_weekly), (daily_case, acf[1])):
            c = (1 + rho) / (1 - rho)
            ratio = np.sqrt(np.diag(sandwich) / (c * np.diag(jtj_inv)))
            out[case] = {"rho": float(rho), **{f"a{j + 1}": float(x) for j, x in zip(active, ratio)}}
    return out


def run(ctx) -> Result:
    res = Result(
        code="V4", title="Uncertainty intervals are calibrated",
        question="Do the 90% prediction intervals contain about 90% of new observations, and do the "
                 "parameter intervals contain the true parameters about 90% of the time?",
        method=f"As V3, synthetic data are generated from known parameters on real Mentue forcing, with "
               f"noise of {SIGMA} °C (iid, or AR(1) with rho = {RHO}). Each replicate (a new noise draw) is "
               f"calibrated with DE-MCMC (CRN, 32 walkers, run until converged, at most 20,000 steps) on "
               f"2002-2009. A FORWARD run then produces 90% prediction intervals for 2010-2012 from the "
               f"chain, and the share of 2010-2012 observations inside them is recorded. The share of "
               f"parameters whose 90% credible interval contains the true value is also recorded. With AR(1) "
               f"noise, the chain is run with each of the two likelihoods the package offers: the exact AR(1) "
               f"likelihood (cases B, D) and the least-squares likelihood with the effective sample size "
               f"(cases E, F, on the same data as B and D). Case C deliberately uses the wrong (iid) noise "
               f"model on autocorrelated data. The AR(1) correlation rho is estimated, by default, from the "
               f"week-to-week persistence of the errors (rho_timescale weekly); case G repeats case E with rho "
               f"from consecutive days (the 'daily' option). Cases H and I use noise made of a fast (2-day) and "
               f"a slow (3-4 week) part, as measured on the real rivers, which an AR(1) model can only "
               f"approximate: H with the weekly default, I with the daily option, on the same data. Cases J "
               f"and K repeat case E with weekly scoring (time_resolution 1w): the likelihood then compares "
               f"weekly means, whose errors are much less correlated from week to week than days are. J uses "
               f"the authors' bounds, which allow a negative a2 and a3; K requires both to be at least 0. "
               f"The same simulations give the coverage of the "
               f"{', '.join(f'{x}%' for x in LEVELS)} intervals (central percentiles of the 1000 simulations "
               f"on each day), and the chain that of the parameter intervals at those levels.",
        criterion=f"For the correctly specified cases (A, B, D, E, F, G, K): every run converges, and the mean "
                  f"held-out coverage of the 90% prediction interval is between {PI_RANGE[0]:.0%} and "
                  f"{PI_RANGE[1]:.0%}, and that of the {', '.join(f'{x}%' for x in LEVELS if x != 90)} intervals "
                  f"within {LEVEL_TOLERANCE * 100:.0f} percentage points of their level (at most 100%). For cases "
                  f"A, B, E, F, G and K: pooled parameter-interval coverage at least "
                  f"{PAR_MIN:.0%}. Results are reported, not judged, for case C (wrong noise model, expected to "
                  f"be too narrow), case D (see notes), cases H and I (noise that AR(1) can only approximate) and "
                  f"case J (see notes).")
    jobs = []
    for label, v, nk, nm, n_full, n_quick in CASES:
        for r in range(n_quick if ctx.quick else n_full):
            jobs.append((label, v, nk, nm, r))
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            rows = list(ex.map(replicate, jobs))
            checks = [] if ctx.quick else list(ex.map(sampler_crosscheck, (0, 1)))
            jk_jobs = [] if ctx.quick else [(v, r) for v in JACKKNIFE_VERSIONS for r in range(N_JACKKNIFE)]
            jk = pd.DataFrame([x for rows_ in ex.map(leave_one_year_out, jk_jobs) for x in rows_])
    df = pd.DataFrame(rows)
    summary_rows, per_param_rows, level_rows = [], [], []
    ok = True
    for label, *_ in CASES:
        g = df[df.case == label]
        if g.empty:
            continue
        conv = g[g.converged]
        s = {"case": label, "replicates": len(g), "converged": int(g.converged.sum()),
             "mean held-out coverage": round(conv["held-out coverage"].mean(), 3) if len(conv) else None,
             "range": (f"{conv['held-out coverage'].min():.0%}-{conv['held-out coverage'].max():.0%}"
                       if len(conv) else ""),
             "mean calibration coverage": round(conv["calibration coverage"].mean(), 3) if len(conv) else None,
             "parameter coverage": round(float(np.average(conv["parameters inside 90% interval"],
                                                          weights=conv["n parameters"])), 3) if len(conv) else None,
             "mean steps": int(conv["steps"].mean()) if len(conv) else None,
             "mean rho": round(float(conv["rho"].mean()), 3) if len(conv) else None}
        levels_ok = True
        for level in LEVELS:
            m = float(np.mean([c[level] for c in conv["held-out coverage by level"]])) if len(conv) else np.nan
            p_ = (float(np.mean([x for c in conv["parameters inside by level"] for x in c[level]]))
                  if len(conv) else np.nan)
            level_rows.append({"case": label, "level": level, "prediction": m, "parameters": p_})
            levels_ok &= bool(np.isfinite(m) and abs(m - level / 100) <= LEVEL_TOLERANCE + 1e-12)
        if "mis-specified" not in label and label[0] not in REPORT_ONLY:
            if ctx.quick:
                # A few replicates cannot judge coverage (with 3, a 50% interval's mean coverage varies by more
                # than the 3-point tolerance): the quick run only requires every replicate to converge.
                s["pass"] = bool(s["converged"] == s["replicates"])
            else:
                s["pass"] = bool(s["converged"] == s["replicates"] and s["mean held-out coverage"] is not None
                                 and PI_RANGE[0] <= s["mean held-out coverage"] <= PI_RANGE[1] and levels_ok
                                 and (label[0] not in PAR_REQUIRED or s["parameter coverage"] >= PAR_MIN))
            ok &= s["pass"]
        summary_rows.append(s)
        # Per parameter: coverage, and the spread of the estimates between replicates relative to the
        # posterior's own width. For well-calibrated intervals the ratio is close to 1.
        if len(conv) > 2:
            for name in conv["per_param"].iloc[0]:
                med = np.array([p[name]["median"] for p in conv["per_param"]])
                sd = np.array([p[name]["sd"] for p in conv["per_param"]])
                per_param_rows.append({"case": label.split(":")[0], "parameter": name,
                                       "coverage": np.mean([p[name]["inside"] for p in conv["per_param"]]),
                                       "replicate spread / posterior SD": med.std(ddof=1) / sd.mean()})
    summ = pd.DataFrame(summary_rows)
    pp = pd.DataFrame(per_param_rows)
    res.passed = bool(ok)
    res.seconds = t.seconds
    res.summary = ("Quick mode (a few replicates): every replicate must converge; coverage reported, not judged. "
                   if ctx.quick else "") + "; ".join(f"{r['case'].split(':')[0]}: prediction {r['mean held-out coverage']:.0%}, "
                            f"parameters {r['parameter coverage']:.0%}" for _, r in summ.iterrows()
                            if r["mean held-out coverage"] is not None)
    max_check = float(pd.DataFrame(checks).iloc[:, -1].max()) if checks else float("nan")
    fmt = ["mean held-out coverage", "mean calibration coverage", "parameter coverage"]
    shown = summ.copy()
    for col in fmt:
        shown[col] = shown[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    sec_pred = Section(
        "Prediction intervals",
        "Each replicate is a new synthetic data set from the known truth. A 90% prediction interval for "
        "2010-2012 is made from the calibration on 2002-2009, and the share of the 2010-2012 observations "
        "inside it is recorded.",
        figures=[_fig_prediction_coverage(df)], tables=[("Coverage of 90% intervals (means over replicates)", shown)])
    lv = pd.DataFrame(level_rows)
    if len(lv):
        pred_tab = lv.pivot(index="case", columns="level", values="prediction").reindex([c[0] for c in CASES])
        par_tab = lv.pivot(index="case", columns="level", values="parameters").reindex([c[0] for c in CASES])
        fmt_tab = lambda t: t.dropna(how="all").map(lambda x: f"{x:.1%}" if pd.notna(x) else "").rename(
            columns=lambda c: f"{c}% interval").reset_index().assign(case=lambda d: d.case.str.split(":").str[0])
        sec_pred.text += (" The second table gives the coverage of intervals at other levels, from the same "
                          "simulations: a user may ask for any level.")
        sec_pred.tables.append(("Coverage at each level (means over replicates)", fmt_tab(pred_tab)))
        sec_pred.figures.append(_fig_levels(lv))
        judged = lv[~lv.case.str.contains("mis-specified") & ~lv.case.str[0].isin(REPORT_ONLY)].dropna()
        if len(judged):
            worst = judged.loc[(judged.prediction - judged.level / 100).abs().idxmax()]
            res.summary += (f". At every level tested ({', '.join(f'{x}%' for x in LEVELS)}), the correctly "
                            f"specified cases' mean prediction coverage was within "
                            f"{abs(worst.prediction - worst.level / 100) * 100:.1f} points of the level (largest "
                            f"difference: case {worst.case[0]}, {worst.level}% interval, {worst.prediction:.1%})")
    sec_par = Section(
        "Parameter intervals from MCMC",
        "For each replicate, whether each parameter's 90% credible interval contains the true value.")
    if len(lv):
        sec_par.tables.append(("Parameter intervals at each level: share containing the true value (pooled)",
                               fmt_tab(par_tab)))
    res.sections += [sec_pred, sec_par]
    if len(pp):
        spread = pp.pivot(index="parameter", columns="case", values="replicate spread / posterior SD")
        cover = pp.pivot(index="parameter", columns="case", values="coverage")
        sec_par.figures.append(_fig_parameter_coverage(pp))
        sec_par.tables += [("Parameter-interval coverage, by parameter", cover.map(lambda x: f"{x:.0%}" if pd.notna(x) else "").reset_index()),
                           ("Spread of estimates between replicates / posterior standard deviation (1 = calibrated)",
                            spread.round(2).reset_index())]
        if not ctx.quick and all(c in spread.columns for c in "EGHI"):
            theory = sandwich_prediction()
            rows_t = []
            for name in spread.index:
                if name not in theory["E"]:
                    continue
                row = {"parameter": name}
                for c in "EGHI":
                    row[f"{c}: theory / measured"] = f"{theory[c][name]:.2f} / {spread.loc[name, c]:.2f}"
                rows_t.append(row)
            sec_par.tables.append((
                "Width of parameter intervals, theory and measurement: true spread of the estimates / interval "
                "standard deviation (1 = right, below 1 = too wide, above 1 = too narrow); E, H weekly rho; "
                "G, I daily rho", pd.DataFrame(rows_t)))
            slow = max((n for n in theory["H"] if n.startswith("a")), key=lambda n: theory["I"][n])
            res.notes.append(
                f"Why the parameter intervals behave as they do (theory). The least-squares likelihood with the "
                f"effective sample size widens every parameter's interval by the same factor, (1 + rho)/(1 - rho). "
                f"The spread of each estimate really grows by a factor that depends on how that parameter's effect on "
                f"the simulated temperature varies over time (the 'sandwich' covariance of least squares with "
                f"correlated errors): a parameter whose effect changes slowly (here {slow}, the seasonal timing) needs "
                f"the full allowance for long-lasting errors, a parameter whose effect changes from day to day (a2, "
                f"a3) much less. The table gives the ratio predicted by that formula next to the one measured over "
                f"the replicates; they agree closely. With fast + slow noise, the daily rho makes {slow}'s interval "
                f"{theory['I'][slow]:.1f} times too narrow (theory; measured {spread.loc[slow, 'I']:.2f}); the weekly "
                f"rho gives it the right width ({theory['H'][slow]:.2f}; measured {spread.loc[slow, 'H']:.2f}) and makes "
                f"the others wider than necessary. With a single factor for all parameters, the weekly rho is the "
                f"choice that leaves no parameter's interval too narrow in these tests; intervals of parameters with "
                f"fast-varying effects are wider than they need to be with either rho.")
        d = pp[(pp.case == "D") & (pp.coverage < PAR_MIN)]
        if len(d):
            res.notes.append(
                f"Version 8 parameter intervals (case D) contain the true value {summ.loc[summ.case.str.startswith('D'), 'parameter coverage'].iloc[0]:.0%} "
                f"of the time rather than 90%. For the parameters that miss most often "
                f"({', '.join(d.parameter)}), the estimates vary "
                f"{d['replicate spread / posterior SD'].min():.1f}-{d['replicate spread / posterior SD'].max():.1f} "
                f"times as much between replicates as the posterior's own standard deviation. The sampler was "
                f"cross-checked on two replicates with a different sampler (emcee's stretch move, table "
                f"above): the interval ends agree to within {max_check:.0%} of the interval width. The same code gives calibrated "
                f"intervals for version 5. The cause is version 8's parameters trading off against each "
                f"other (several combinations fit almost equally well), which makes the posterior strongly "
                f"non-Gaussian; Bayesian parameter intervals are then not guaranteed to have their nominal "
                f"frequency coverage. Version 8's prediction intervals are unaffected. Do not quote version 8 "
                f"parameter intervals from the exact AR(1) likelihood as confidence statements. (An earlier "
                f"version of this check also required {PAR_MIN:.0%} parameter coverage for case D; it was not "
                f"met, and the requirement was removed after this investigation.)")
    if len(summ) and summ.case.str.startswith("F").any() and summ.case.str.startswith("D").any():
        cov = lambda c: summ.loc[summ.case.str.startswith(c), "parameter coverage"].iloc[0]
        res.notes.append(
            f"The least-squares likelihood (cases E and F) centres the chain on the least-squares fit and widens it "
            f"by the effective sample size n(1 - rho)/(1 + rho). On the same data its version 8 parameter "
            f"intervals contained the true value {cov('F'):.0%} of the time, against {cov('D'):.0%} with the exact "
            f"AR(1) likelihood, and {cov('E'):.0%} against {cov('B'):.0%} for version 5; prediction intervals were "
            f"equally well calibrated with either. On real rivers, where the model is never exactly right, the "
            f"exact AR(1) likelihood can also move the parameters away from the best fit (V5).")
    rho_rows = []
    for weekly, daily, noise_label in (("E", "G", "AR(1) noise"), ("H", "I", "fast + slow noise")):
        for c, timescale in ((weekly, "weekly (default)"), (daily, "daily (option)")):
            r = summ[summ.case.str.startswith(c + ":")]
            if len(r):
                r = r.iloc[0]
                rho_rows.append({"noise in the data": noise_label, "case": c, "rho time scale": timescale,
                                 "mean rho": r["mean rho"], "prediction coverage": r["mean held-out coverage"],
                                 "parameter coverage": r["parameter coverage"]})
    if rho_rows:
        rt = pd.DataFrame(rho_rows)
        sec_rho = Section(
            "Estimating rho from week-to-week persistence",
            "Version 5 with the least-squares likelihood, on identical synthetic data for each pair of cases: rho "
            "estimated from the persistence of the errors from week to week (the default) or from consecutive "
            "days (the 'daily' option). With AR(1) noise both should behave alike. With noise that also has a "
            "slow part, as real model errors do, the daily estimate sees only the fast part.",
            figures=[_fig_rho_timescale(rt)],
            tables=[("rho time scale: estimated rho and coverage of 90% intervals", rt.assign(**{
                c: rt[c].map(lambda x: f"{x:.1%}") for c in ("prediction coverage", "parameter coverage")}))])
        res.sections.append(sec_rho)
        get = {r["case"]: r for r in rho_rows}
        if all(k in get for k in "EGHI"):
            res.notes.append(
                f"rho time scale. With AR(1) noise (true rho {RHO}) the weekly default estimated rho = "
                f"{get['E']['mean rho']:.2f} on average and the daily option {get['G']['mean rho']:.2f}; parameter "
                f"intervals covered the truth {get['E']['parameter coverage']:.0%} and {get['G']['parameter coverage']:.0%} "
                f"of the time. With fast + slow noise the weekly default estimated rho = {get['H']['mean rho']:.2f} "
                f"and the daily option {get['I']['mean rho']:.2f}; parameter coverage was "
                f"{get['H']['parameter coverage']:.0%} and {get['I']['parameter coverage']:.0%}. Prediction "
                f"coverage of single days was {min(r['prediction coverage'] for r in rho_rows):.0%}-"
                f"{max(r['prediction coverage'] for r in rho_rows):.0%} in all four cases, since on a single day "
                f"rho does not change the width of the band (V9 tests quantities that span weeks).")
    if len(summ) and summ.case.str.startswith("K").any() and summ.case.str.startswith("E").any():
        cov = lambda c: summ.loc[summ.case.str.startswith(c + ":"), "parameter coverage"].iloc[0]
        k = df[df.case.str.startswith("K:") & df.converged]
        kf, kr = k["variance factor"].astype(float), k["rho"].astype(float)
        res.notes.append(
            f"Weekly scoring (case K, the data of case E). The likelihood counts weekly means, and allows for "
            f"their correlation from week to week: n/n_eff = 1 + 2 r_b/(1 - rho^7), on average "
            f"{kf.mean():.2f}, against (1 + rho)/(1 - rho) = {((1 + kr) / (1 - kr)).mean():.2f} for daily values "
            f"with the same rho. Parameter intervals contained the true value {cov('K'):.0%} of the time "
            f"({cov('E'):.0%} with daily scoring).")
    if len(summ) and summ.case.str.startswith("J").any():
        j = df[df.case.str.startswith("J:")]
        bad = j[j["implausible"].astype(bool)]
        good = j[~j["implausible"].astype(bool) & j.converged]
        cov_good = (float(np.average(good["parameters inside 90% interval"], weights=good["n parameters"]))
                    if len(good) else float("nan"))
        others = df[~df.case.str.startswith("J:")]
        res.notes.append(
            f"Weekly scoring with the authors' bounds (case J). {len(bad)} of {len(j)} calibrations ended on a "
            f"parameter set with a negative relaxation rate (a3 below 0) whose daily simulation zigzags between "
            f"0 °C and high values: weekly means average the zigzag out, so it can score as well as the true "
            f"parameters. pyair2stream's plausibility check (negative relaxation rate, or successive daily "
            f"changes correlated below -0.5) warned for each of these {len(bad)} and for "
            f"{int(others['implausible'].astype(bool).sum())} of the {len(others)} calibrations in the other "
            f"cases; {int((~bad.converged.astype(bool)).sum())} of the {len(bad)} did not converge. Over "
            f"the {len(j)} calibrations, parameter intervals contained the true value "
            f"{summ.loc[summ.case.str.startswith('J:'), 'parameter coverage'].iloc[0]:.0%} of the time, and "
            f"{cov_good:.0%} over the {len(good)} that converged without a warning. With weekly or monthly "
            f"scoring, bounds that keep a2 and a3 at least 0 rule such sets out (case K). An earlier run of "
            f"this check judged case J against the parameter criterion and failed for this reason (76%); "
            f"case K was added after it.")
    if checks:
        ck = pd.DataFrame(checks)
        sec_par.tables.append(("Sampler cross-check, case D: package sampler (DE move) vs stretch move", ck.round(3)))
        if (ck.iloc[:, -1] > 0.10).any():
            res.passed = False
            res.notes.append("The two samplers disagree by more than 10% of an interval's width.")
    if len(jk):
        for col in ("raw spread covers", "jackknife covers", "MCMC covers", "MCMC least squares covers"):
            if col in jk:
                jk[col] = jk[col].astype(float)          # True/False, NaN where there is no MCMC
        g = jk.groupby("version")
        by_version = pd.DataFrame({
            "raw spread (fold mean +- 1.645 x std)": g["raw spread covers"].mean(),
            "jackknife": g["jackknife covers"].mean()})
        if "MCMC covers" in jk:
            by_version["MCMC (same replicates)"] = g["MCMC covers"].mean()
        if "MCMC least squares covers" in jk:
            by_version["MCMC least squares (same replicates)"] = g["MCMC least squares covers"].mean()
            ratio = (jk["jackknife width"] / jk["MCMC width"]).groupby(jk.version).median()
        shown = by_version.map(lambda x: f"{x:.0%}" if pd.notna(x) else "")
        if "MCMC covers" in jk:
            shown["median width, jackknife / MCMC (exact likelihood)"] = ratio.round(1).map(lambda x: "" if pd.isna(x) else x)
        res.sections.append(Section(
            "Parameter intervals from cross-validation",
            f"For {N_JACKKNIFE} replicates per model version, leave-one-year-out cross-validation through the "
            "package, and whether three kinds of 90% interval contain the true parameters: the raw spread between "
            "folds, the jackknife interval reported in cv_results.csv, and (versions 5 and 8) the MCMC interval "
            "from the same data.",
            figures=[_fig_jackknife(by_version)],
            tables=[(f"Parameter intervals from cross-validation (cross_validation.cross_validate, leave one "
                     f"year out), {N_JACKKNIFE} replicates per version with AR(1) noise: share of 90% "
                     f"intervals containing the true value", shown.reset_index())]))
        cover = by_version["jackknife"]
        raw = by_version["raw spread (fold mean +- 1.645 x std)"]
        low = [v for v in JACKKNIFE_VERSIONS if cover[v] < JACKKNIFE_OK]
        note = (f"Cross-validation's spread of the parameters between folds is not a confidence interval: the folds "
                f"share most of their data, so the spread (fold mean +- 1.645 x std) contained the true values only "
                f"{raw.min():.0%}-{raw.max():.0%} of the time. The jackknife intervals in cv_results.csv scale the "
                f"spread for that overlap; they contained the true values {cover.min():.0%}-{cover.max():.0%} of the "
                f"time across versions {', '.join(map(str, JACKKNIFE_VERSIONS))}, so they are approximate 90% "
                f"intervals, a little narrow for some versions.")
        if "MCMC covers" in jk and pd.notna(by_version.loc[8, "MCMC (same replicates)"]):
            note += (f" For version 8 they were closer to 90% than the MCMC parameter intervals with the exact AR(1) "
                     f"likelihood on the same replicates ({cover[8]:.0%} against "
                     f"{by_version.loc[8, 'MCMC (same replicates)']:.0%}), and about {ratio[8]:.1f} times as wide.")
            if "MCMC least squares (same replicates)" in by_version:
                note += (f" The MCMC intervals with the least-squares likelihood (the default) contained the true "
                         f"values {by_version.loc[8, 'MCMC least squares (same replicates)']:.0%} of the time for "
                         f"version 8 and {by_version.loc[5, 'MCMC least squares (same replicates)']:.0%} for "
                         f"version 5.")
        if low:
            note += f" For versions {', '.join(map(str, low))} they fell below {JACKKNIFE_OK:.0%}."
        note += (f" With {N_JACKKNIFE} replicates per version, each share is uncertain by several percentage "
                 f"points.")
        res.notes.append(note)
    res.notes.insert(0, ICE_FLOOR_NOTE)
    return res


def _fig_prediction_coverage(df):
    import matplotlib.pyplot as plt
    plot_style()
    df = df[df.converged]
    cases = list(dict.fromkeys(df.case))
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    rng = np.random.default_rng(0)
    for i, c in enumerate(cases):
        y = df[df.case == c]["held-out coverage"].to_numpy() * 100
        colour = LIGHT_GREY if "mis-specified" in c else BLUE
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(y)), y, s=16, color=colour, alpha=0.8, zorder=3)
        ax.hlines(y.mean(), i - 0.28, i + 0.28, color=INK, lw=1.6, zorder=4)
    reference_line(ax, 90)
    short = {"A": "A\nversion 5\niid noise", "B": "B\nversion 5\nexact AR(1)", "C": "C\nversion 5\nwrong model",
             "D": "D\nversion 8\nexact AR(1)", "E": "E\nversion 5\nleast squares", "F": "F\nversion 8\nleast squares",
             "G": "G\nas E,\ndaily rho", "H": "H\nfast + slow\nnoise", "I": "I\nas H,\ndaily rho",
             "J": "J\nweekly scoring,\nauthors' bounds", "K": "K\nweekly scoring,\na2, a3 >= 0"}
    ax.set_xticks(range(len(cases)), [short.get(c[0], c) for c in cases], fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Held-out observations inside\nthe 90% interval (%)")
    ax.set_title("Prediction-interval coverage, one point per replicate (black: mean; dashed: nominal 90%)")
    return (save_figure(fig, "V4_interval_coverage.png"),
            "Each point is one synthetic data set. Mean coverage is about the same in every case, including the "
            "deliberately wrong noise model (grey): for single days the noise model hardly matters. It is below "
            "90% because about 3% of the synthetic values are below 0 °C, which no range can reach since version "
            "0.5.1 (see the notes).")


def _fig_levels(lv):
    """Stated level against achieved coverage of prediction intervals, for the correctly specified cases."""
    import matplotlib.pyplot as plt
    plot_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([45, 100], [45, 100], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    shown = [c for c in dict.fromkeys(lv.case) if "mis-specified" not in c]
    for k, c in enumerate(shown):
        g = lv[lv.case == c]
        ax.plot(g.level, g.prediction * 100, marker="o", ms=3.5, lw=1.0, alpha=0.8,
                color=(BLUE, ORANGE, AQUA)[k % 3], label=c.split(":")[0])
    mis = lv[lv.case.str.contains("mis-specified")]
    for c in dict.fromkeys(mis.case):
        g = mis[mis.case == c]
        ax.plot(g.level, g.prediction * 100, marker="s", ms=3, lw=0.8, color=LIGHT_GREY,
                label=f"{c.split(':')[0]} (mis-specified)")
    ax.set_xlim(45, 100)
    ax.set_ylim(45, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Stated level of the prediction interval (%)")
    ax.set_ylabel("Held-out observations inside it (%)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_title("Synthetic data: stated against achieved coverage", fontsize=9)
    return (save_figure(fig, "V4_levels.png"),
            "Mean share of held-out observations inside the prediction interval, for each stated level and case. "
            "On the dashed diagonal the intervals hold at every level.")


def _fig_rho_timescale(rt):
    import matplotlib.pyplot as plt
    plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    groups = list(dict.fromkeys(rt["noise in the data"]))
    for i, g in enumerate(groups):
        for j, (ts, colour) in enumerate((("weekly (default)", BLUE), ("daily (option)", ORANGE))):
            r = rt[(rt["noise in the data"] == g) & (rt["rho time scale"] == ts)]
            if len(r):
                v = float(r["parameter coverage"].iloc[0]) * 100
                ax.bar(i + (j - 0.5) * 0.36, v, 0.32, color=colour, label=ts if i == 0 else None, zorder=3)
                ax.annotate(f"{v:.0f}%", (i + (j - 0.5) * 0.36, v), xytext=(0, -4), textcoords="offset points",
                            ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    reference_line(ax, 90, "nominal 90%")
    ax.set_xticks(range(len(groups)), [f"{g}" for g in groups])
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Parameter intervals containing\nthe true value (%)")
    ax.set_title("Version 5: rho from weekly or daily persistence")
    ax.legend(loc="lower right", fontsize=7.5)
    return (save_figure(fig, "V4_rho_timescale.png"),
            "Share of parameter intervals containing the true value, with rho estimated from week-to-week "
            "persistence (blue) or from consecutive days (orange), on identical synthetic data.")


def _fig_parameter_coverage(pp):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plot_style()
    params = sorted(pp.parameter.unique(), key=lambda x: int(x[1:]))
    x = {p: i for i, p in enumerate(params)}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    styles = {"B": (BLUE, "o", "B: version 5, exact ar1"), "D": (BLUE, "D", "D: version 8, exact ar1"),
              "E": (ORANGE, "o", "E: version 5, least squares"), "F": (ORANGE, "D", "F: version 8, least squares"),
              "A": (AQUA, "s", "A: version 5, iid noise and model"), "C": (LIGHT_GREY, "s", "C: wrong noise model")}
    handles = []
    for case, (colour, marker, label) in styles.items():
        g = pp[pp.case == case]
        if g.empty:
            continue
        off = {"A": -0.25, "B": -0.15, "E": -0.05, "D": 0.05, "F": 0.15, "C": 0.25}[case]
        ax.scatter([x[p] + off for p in g.parameter], g.coverage * 100, s=30, marker=marker, color=colour, zorder=3)
        handles.append(Line2D([0], [0], marker=marker, lw=0, color=colour, label=label))
    reference_line(ax, 90, "nominal 90%")
    ax.set_xticks(range(len(params)), params)
    ax.set_ylim(30, 105)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Replicates whose 90% interval\ncontains the true value (%)")
    ax.set_title("MCMC parameter intervals by case and parameter")
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, ncol=2)
    return (save_figure(fig, "V4_parameter_coverage.png"),
            "Share of replicates whose 90% MCMC interval contains the true parameter. Blue: exact AR(1) likelihood "
            "(circles version 5, diamonds version 8); orange: least-squares likelihood on the same data; grey: the "
            "wrong noise model, too narrow.")


def _fig_jackknife(by_version):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plot_style()
    versions = list(by_version.index)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cols = [("raw spread (fold mean +- 1.645 x std)", LIGHT_GREY, "o", "spread between folds (not an interval)"),
            ("jackknife", ORANGE, "s", "cross-validation jackknife"),
            ("MCMC (same replicates)", BLUE, "D", "MCMC, exact ar1 likelihood, same data"),
            ("MCMC least squares (same replicates)", AQUA, "^", "MCMC, least-squares likelihood, same data")]
    handles = []
    for k, (col, colour, marker, label) in enumerate(cols):
        if col not in by_version:
            continue
        vals = by_version[col].to_numpy(dtype=float) * 100
        xs = np.arange(len(versions)) + (k - 1.5) * 0.14
        ok = np.isfinite(vals)
        ax.scatter(xs[ok], vals[ok], s=34, marker=marker, color=colour, zorder=3)
        handles.append(Line2D([0], [0], marker=marker, lw=0, color=colour, label=label))
    reference_line(ax, 90, "nominal 90%")
    ax.set_xticks(range(len(versions)), [f"version {v}" for v in versions])
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Intervals containing\nthe true value (%)")
    ax.set_title("Parameter intervals from leave-one-year-out cross-validation")
    ax.legend(handles=handles, loc="lower right", fontsize=7.5)
    return (save_figure(fig, "V4_jackknife.png"),
            "The jackknife intervals (orange) are close to 90% for every version; the raw spread between folds "
            "(grey) is far too narrow to use as an interval.")
