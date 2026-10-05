"""
V1 - Same results as the original Fortran on real inputs.

The unit tests compare against the Fortran on a synthetic series with constant
discharge, which leaves the discharge terms (theta**a4, a5, a8) untested. This
check uses real, variable Swiss discharge and air temperature.

Part A compares the daily simulations. Part B compares what calibration
optimises: the score (RMS, NSE, KGE) and the weekly and monthly averages it is
computed from, including how incomplete weeks and months are handled (prc).
"""

import itertools
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (REPO, WORK, VERSIONS, RIVERS, Result, Section, published_params, river_csv, mean_discharge,
                    simulate, Timer, plot_style, save_figure, reference_line, BLUE, ORANGE, INK2, MUTED,
                    LIGHT_GREY)

INTEGRATORS = ("EUL", "RK2", "RK4", "CRN")     # the ones the Fortran implements
TOL = 1e-5                                     # °C; the Fortran prints 6 decimals
PLAUSIBLE = 100.0                              # beyond this a run has diverged

# Part B. The Fortran prints scores with 6 decimals and averages with 5.
OBJECTIVES = ("RMS", "NSE", "KGE")
RESOLUTIONS = ("1d", "1w", "2w", "1m")
PRC = (0.6, 0.9)                               # 0.6 is the original program's example setting
TOL_SCORE = 1e-6
TOL_AVERAGE = 1e-5
B_VERSION = 8                                  # every term of the model, including discharge
UPSTREAM = os.path.join(REPO, "fortran", "upstream")


def fortran_available() -> bool:
    return shutil.which("gfortran") is not None and os.path.exists(
        os.path.join(REPO, "fortran", "upstream", "src", "AIR2STREAM_MAIN.f90"))


def run(ctx) -> Result:
    res = Result(
        code="V1", title="Same results as the original Fortran",
        question="Does pyair2stream compute the same daily water temperatures as the original Fortran "
                 "air2stream when both are given real, variable inputs? And does it compute the same "
                 "calibration score, from the same weekly and monthly averages, including when the record "
                 "has gaps?",
        method="The Fortran source (git submodule) is compiled with gfortran. (A) For each model version and "
               "each integrator the Fortran implements, both programs simulate the Mentue calibration "
               "record (8 years of real air temperature and discharge) with the published parameters "
               "(Piccolroaz et al., 2016). Daily outputs are compared. (B) For each river, both programs score "
               "the published version 8 parameters (FORWARD mode) on the calibration and validation records, "
               "with each objective (RMS, NSE, KGE) and time resolution (daily, 1 week, 2 weeks, 1 month). This "
               "is done on the records as distributed and on copies with deliberate gaps in the water "
               "temperature (15% of days at random, ten days in a row, a whole month, a month with 55% of its "
               "days and one with 70%), with the share of days a week or month needs (prc) set to 0.6 and "
               "0.9. Each program reads the same values from its own input format. The scores and the "
               "weekly or monthly averages each program writes are compared.",
        criterion=f"(A) On every day where both stay below {PLAUSIBLE:.0f} °C, they differ by at most {TOL:g} °C; "
                  f"and they diverge (exceed it) on exactly the same days. (B) In every case the same weeks or "
                  f"months are scored, the scores differ by at most {TOL_SCORE:g} and the averages by at most "
                  f"{TOL_AVERAGE:g} °C (the precision of the Fortran's printed output).")
    if not fortran_available():
        res.summary = "Skipped: gfortran or the Fortran submodule is not available."
        return res

    sys.path.insert(0, REPO)
    from tests.fortran_runner import run_fortran_model

    with Timer() as t:
        src = pd.read_csv(river_csv("MAH_2369", "calibration"))
        n = len(src)
        # The Fortran runner writes its input starting 2000-01-01 with only the first
        # day's water temperature (4 °C); the Python run is given exactly the same.
        dates = pd.date_range("2000-01-01", periods=n, freq="D")
        csv = os.path.join(WORK, "v1_input.csv")
        os.makedirs(WORK, exist_ok=True)
        tw = np.full(n, np.nan)
        tw[0] = 4.0
        pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "T_air": src.T_air, "T_water": tw,
                      "Discharge": src.Discharge}).to_csv(csv, index=False)
        qmedia = mean_discharge(csv)
        rows, example = [], None
        for v in VERSIONS:
            par = published_params(v, "MAH_2369")
            for integ in INTEGRATORS:
                py = simulate(csv, v, par, integ, qmedia, name="v1").Twat_mod[365:]
                ft = run_fortran_model(v, integ, n, src.T_air.values, src.Discharge.values, par, qmedia, 4.0)
                py_ok = np.isfinite(py) & (np.abs(py) < PLAUSIBLE)
                ft_ok = np.isfinite(ft) & (np.abs(ft) < PLAUSIBLE)
                both = py_ok & ft_ok
                max_diff = float(np.max(np.abs(py[both] - ft[both]))) if both.any() else np.nan
                if (v, integ) == (8, "CRN"):
                    example = (dates, py.copy(), np.asarray(ft, float).copy())
                rows.append({"version": v, "integrator": integ, "days compared": int(both.sum()),
                             "days diverged (Python)": int((~py_ok).sum()),
                             "days diverged (Fortran)": int((~ft_ok).sum()),
                             "same divergence days": bool(np.array_equal(py_ok, ft_ok)),
                             "max |difference| (°C)": max_diff})
        b, b_example = _part_b(ctx)
    df = pd.DataFrame(rows)
    ok = (df["same divergence days"] & ((df["max |difference| (°C)"] <= TOL) | (df["days compared"] == 0))).all()
    ok_b = bool((b["same weeks or months scored"] & (b["largest score difference"] <= TOL_SCORE)
                 & (b["largest average difference (°C)"] <= TOL_AVERAGE)).all())
    res.passed = bool(ok) and ok_b
    res.seconds = t.seconds
    n_div = int((df["days diverged (Python)"] > 0).sum())
    res.summary = (f"(A) All {len(df)} version × integrator cases agree to within "
                   f"{df['max |difference| (°C)'].max():.1e} °C" if ok else "(A) Some cases disagree (see table)")
    if n_div:
        res.summary += (f"; in {n_div} cases (explicit integrators) both programs diverge on the same days "
                        "with these parameters, which were calibrated with CRN (see V6)")
    res.summary += "."
    if ok_b:
        res.summary += (f" (B) In all {len(b)} combinations of river, record, objective, time resolution and prc, "
                        f"both programs score the same weeks or months and give the same score (largest "
                        f"difference {b['largest score difference'].max():.0e}) and averages (largest difference "
                        f"{b['largest average difference (°C)'].max():.0e} °C), with and without gaps.")
    else:
        res.summary += " (B) Some scoring cases disagree (see table B)."
    res.sections.append(Section(
        "A. Python and Fortran, day by day",
        "Both programs simulated the same eight years of real Mentue forcing with the published parameters. "
        "The figures show the largest daily difference in each case and, for the full model with the "
        "default scheme, the two simulations themselves.",
        figures=_figures(df, example),
        tables=[("A. Python vs Fortran, Mentue 2002-2009, published parameters", df)]))
    res.sections.append(Section(
        "B. Scores and weekly and monthly averages",
        "Calibration maximises a score computed from daily values or from weekly or monthly averages. A week or "
        "month is averaged only if at least the share prc of its days have an observation; otherwise it is "
        "left out. Both programs scored the same parameters on the same data, with and without deliberate "
        "gaps. 'Blocks scored' counts the days, weeks or months that enter the score.",
        figures=[_figure_b(b, b_example)],
        tables=[("B. Scores and averages, pyair2stream vs Fortran (published version 8 parameters)", b)]))
    if ok_b:
        res.notes.append("Part B: the two programs agree on which weeks and months have enough data, on their "
                         "averages and on the score, so a calibration at weekly or monthly resolution optimises "
                         "the same quantity in both. (A week or month without any observation is never scored "
                         "in either.)")
    return res


# --- Part B ------------------------------------------------------------------------

def _with_gaps(df, seed):
    """A copy of a record with deliberate gaps in the water temperature."""
    df = df.copy()
    d = pd.to_datetime(df.Date)
    rng = np.random.default_rng(seed)
    tw = df.T_water.to_numpy(float).copy()
    tw[rng.random(len(tw)) < 0.15] = np.nan                                  # scattered single days
    y0 = d.dt.year.iloc[0]
    if d.dt.year.nunique() >= 6:
        tw[((d.dt.year == y0 + 3) & (d.dt.month == 7)).to_numpy()] = np.nan   # a whole month
        m = ((d.dt.year == y0 + 4) & (d.dt.month == 2)).to_numpy()
        tw[m & (rng.random(len(tw)) < 0.45)] = np.nan                        # about 55% of a month left
        m = ((d.dt.year == y0 + 5) & (d.dt.month == 10)).to_numpy()
        tw[m & (rng.random(len(tw)) < 0.30)] = np.nan                        # about 70% of a month left
        tw[800:810] = np.nan                                                 # ten days in a row
    else:
        tw[((d.dt.year == y0 + 1) & (d.dt.month == 8)).to_numpy()] = np.nan   # a whole month
    df["T_water"] = tw
    return df


def _write_inputs(df, csv, txt):
    """The same values for both programs: a CSV for pyair2stream and the Fortran's text format."""
    df.to_csv(csv, index=False, float_format="%.3f")
    d = pd.to_datetime(df.Date)
    with open(txt, "w") as f:
        for y, m, dd, ta, w, q in zip(d.dt.year, d.dt.month, d.dt.day, df.T_air, df.T_water.fillna(-999.0),
                                      df.Discharge):
            f.write(f"{y} {m} {dd} {ta:.3f} {w:.3f} {q:.3f}\n")


def _fortran_forward(folder, st, obj, res, prc, par):
    """The original program in FORWARD mode: scores (calibration, validation) and its output series."""
    from tests.fortran_runner import _build_fortran_binary
    air, water = st.split("_")
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(_build_fortran_binary(), os.path.join(d, "air2stream"))
        os.makedirs(os.path.join(d, "Switzerland"))
        for f in (f"{st}_cc.txt", f"{st}_cv.txt"):
            shutil.copy(os.path.join(folder, f), os.path.join(d, "Switzerland", f))
        shutil.copy(os.path.join(UPSTREAM, "Switzerland", "parameters.txt"), os.path.join(d, "Switzerland"))
        shutil.copy(os.path.join(UPSTREAM, "PSO.txt"), d)
        with open(os.path.join(d, "Switzerland", "parameters_forward.txt"), "w") as f:
            f.write(" ".join(f"{x:.12f}" for x in par) + "\n")
        with open(os.path.join(d, "input.txt"), "w") as f:
            f.write(f"! Main input\nSwitzerland\n{air}\n{water}\nc\n{res}\n{B_VERSION}\n0\n{obj}\nCRN\n"
                    f"FORWARD\n{prc:.2f}\n10\n-999\n")
        # FORWARD pauses after its own consistency message; the input line lets it continue.
        subprocess.run(["./air2stream"], cwd=d, input="go\n", capture_output=True, text=True, check=True)
        out = os.path.join(d, "Switzerland", f"output_{B_VERSION}")
        scores = open(os.path.join(out, f"1_FORWARD_{obj}_{st}_c_{res}.out")).read().split()
        cols = ["y", "m", "d", "Tair", "obs", "mod", "obs_agg", "mod_agg", "Q"]
        series = [pd.read_csv(os.path.join(out, f"{k}_FORWARD_{obj}_{st}_c{p}_{res}.out"), sep=r"\s+",
                              header=None, names=cols) for k, p in (("2", "c"), ("3", "v"))]
        return float(scores[-2]), float(scores[-1]), series


def _b_case(args):
    st, record, obj, res, prc = args
    folder = os.path.join(WORK, f"v1b_{st}_{'gaps' if record == 'with gaps' else 'orig'}")
    par = published_params(B_VERSION, st)
    f_cal, f_val, f_series = _fortran_forward(folder, st, obj, res, prc, par)
    row = {"river": RIVERS[st], "record": record, "objective": obj, "time resolution": res,
           "prc": prc if res != "1d" else ""}
    worst_score, worst_avg, same, example = 0.0, 0.0, True, None
    for period, f_score, f in (("calibration", f_cal, f_series[0]), ("validation", f_val, f_series[1])):
        csv = os.path.join(folder, f"{st}_{period[0]}.csv")
        q = pd.read_csv(csv).Discharge.to_numpy(float)
        # Each program takes the mean discharge of the file it reads (here, of each period's file).
        data = simulate(csv, B_VERSION, par, "CRN", float(q[q != -999].mean()), objective=obj,
                        name=f"v1b_{st}_{record[:4]}_{obj}_{res}_{prc}_{period[0]}", time_resolution=res,
                        prc=float(prc))
        from pyair2stream.model import funcobj
        p_score = float(funcobj(data))
        fo, fm = f.obs_agg.to_numpy(), f.mod_agg.to_numpy()
        po, pm = data.Twat_obs_agg, data.Twat_mod_agg
        f_pos, p_pos = fo != -999, po != -999
        both = f_pos & p_pos
        same &= bool(np.array_equal(f_pos, p_pos))
        worst_score = max(worst_score, abs(f_score - p_score))
        if both.any():
            worst_avg = max(worst_avg, float(np.max(np.abs(fo[both] - po[both]))),
                            float(np.max(np.abs(fm[both] - pm[both]))))
        row[f"{period}: blocks scored"] = int(p_pos.sum())
        row[f"{period}: score, Fortran"] = f_score
        row[f"{period}: score, pyair2stream"] = round(p_score, 6)
        if period == "calibration" and (st, record, obj, res, prc) == ("MAH_2369", "with gaps", "RMS", "1m", 0.6):
            obs_daily = np.where(data.Twat_obs == -999.0, np.nan, data.Twat_obs)[365:]
            example = (pd.to_datetime(pd.read_csv(csv).Date).to_numpy(), obs_daily, fo[365:], po[365:])
    row["same weeks or months scored"] = same
    row["largest score difference"] = worst_score
    row["largest average difference (°C)"] = worst_avg
    return row, example


def _part_b(ctx):
    stations = ["MAH_2369"] if ctx.quick else list(RIVERS)
    for st in stations:
        for record in ("as distributed", "with gaps"):
            folder = os.path.join(WORK, f"v1b_{st}_{'gaps' if record == 'with gaps' else 'orig'}")
            os.makedirs(folder, exist_ok=True)
            for period, seed in (("calibration", 1), ("validation", 2)):
                df = pd.read_csv(river_csv(st, period))
                if record == "with gaps":
                    df = _with_gaps(df, seed)
                _write_inputs(df, os.path.join(folder, f"{st}_{period[0]}.csv"),
                              os.path.join(folder, f"{st}_c{period[0]}.txt"))
    jobs = []
    for st, record, obj, res in itertools.product(stations, ("as distributed", "with gaps"), OBJECTIVES, RESOLUTIONS):
        # prc only matters for weeks and months with missing days
        for prc in (PRC if (record == "with gaps" and res != "1d") else PRC[:1]):
            jobs.append((st, record, obj, res, prc))
    with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
        out = list(ex.map(_b_case, jobs))
    example = next((e for _, e in out if e is not None), None)
    return pd.DataFrame([r for r, _ in out]), example


def _figure_b(b, example):
    import matplotlib.pyplot as plt
    plot_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    if example is not None:
        dates, obs, f_avg, p_avg = example
        win = (dates >= np.datetime64("2004-10-01")) & (dates < np.datetime64("2006-12-01"))
        a1.plot(dates[win], obs[win], color=LIGHT_GREY, lw=0.8, label="daily observations (with gaps)")
        fk, pk = win & (f_avg != -999), win & (p_avg != -999)
        a1.scatter(dates[fk], f_avg[fk], s=70, color=BLUE, alpha=0.45, lw=0, label="monthly mean, Fortran", zorder=3)
        a1.scatter(dates[pk], p_avg[pk], s=12, color=ORANGE, label="monthly mean, pyair2stream", zorder=4)
        months = pd.period_range(pd.Timestamp(dates[win][0]), pd.Timestamp(dates[win][-1]), freq="M")
        shaded = False
        for m in months:
            in_m = win & (dates >= np.datetime64(m.start_time)) & (dates <= np.datetime64(m.end_time))
            if not (p_avg[in_m] != -999).any():
                a1.axvspan(m.start_time, m.end_time, color=LIGHT_GREY, alpha=0.35, lw=0,
                           label=None if shaded else "month left out (fewer than 60% of days observed)")
                shaded = True
        import matplotlib.dates as mdates
        a1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        a1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        a1.set_ylabel("Water temperature (°C)")
        a1.set_title("Mentue with gaps, monthly means (prc 0.6)")
        a1.legend(loc="upper left", fontsize=7)
    labels, x = [], 0
    for res in RESOLUTIONS:
        for obj in OBJECTIVES:
            g = b[(b["time resolution"] == res) & (b.objective == obj)]
            a2.scatter(np.full(len(g), x), np.maximum(g["largest score difference"], 1e-9), s=16, color=BLUE,
                       alpha=0.7, zorder=3)
            labels.append(f"{obj} {res}")
            x += 1
    reference_line(a2, 5e-7, "rounding of the Fortran's printed score")
    reference_line(a2, TOL_SCORE, f"pass limit {TOL_SCORE:g}")
    a2.set_yscale("log")
    a2.set_ylim(1e-9, 1e-4)
    a2.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    a2.set_ylabel("|score, Python - Fortran|")
    a2.set_title("Score difference, every case")
    return (save_figure(fig, "V1_scores.png"),
            "Left: with gaps in the record, both programs average the same months and leave out the same ones "
            "(shaded). Right: the largest difference in the score, for every river, "
            "record and prc, is within the rounding of the Fortran's printed output.")


def _figures(df, example):
    import matplotlib.pyplot as plt
    plot_style()
    figs = []
    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels = [f"v{v} {i}" for v, i in zip(df.version, df.integrator)]
    x = np.arange(len(df))
    y = df["max |difference| (°C)"].to_numpy()
    ax.scatter(x, y, s=28, color=BLUE, zorder=3, label="largest difference on any day")
    div = df["days diverged (Python)"] > 0
    ax.scatter(x[div], y[div], s=80, facecolors="none", edgecolors=ORANGE, lw=1.4, zorder=4,
               label="both programs diverge on the same days (excluded from the comparison)")
    reference_line(ax, TOL, f"pass limit {TOL:g} °C")
    reference_line(ax, 5e-6, "rounding of the Fortran's printed output (5e-6 °C)")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e-4)
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("|Python - Fortran| (°C)")
    ax.set_title("Largest daily difference between pyair2stream and the original Fortran")
    ax.legend(loc="lower left", fontsize=7.5)
    figs.append((save_figure(fig, "V1_max_difference.png"),
                 "Every model version and scheme agrees with the Fortran to the last digit the Fortran prints."))
    if example is not None:
        _, py, ft = example
        n = 365
        day = np.arange(1, n + 1)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        a1.plot(day, ft[:n], color=BLUE, lw=3, alpha=0.5, label="original Fortran")
        a1.plot(day, py[:n], color=ORANGE, lw=1.0, label="pyair2stream")
        a1.set_ylabel("Water temperature (°C)")
        a1.set_title("Version 8, Crank-Nicolson, first year of the Mentue record")
        a1.legend(loc="upper left")
        a2.plot(day, py[:n] - ft[:n], color=INK2, lw=0.8)
        a2.set_xlabel("Day of the record (Mentue forcing, 2002)")
        a2.set_ylabel("Difference (°C)")
        a2.set_ylim(-1e-5, 1e-5)
        a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        figs.append((save_figure(fig, "V1_timeseries.png"),
                     "The two simulations lie on top of each other; their difference (lower panel) is "
                     "below 0.00001 °C every day."))
    return figs
