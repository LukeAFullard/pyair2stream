"""
Draw the figures of docs/UNCERTAINTY.md.

    python docs/figures/make_uncertainty_figures.py

Uses the outputs of examples 03 and 04 (running them first if they are missing,
a few minutes) and the validation results in validation/results/. Writes
docs/figures/U*.png.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "validation"))
os.environ.setdefault("MPLBACKEND", "Agg")

from common import (plot_style, BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, AXIS, SURFACE, LIGHT_GREY,  # noqa: E402
                    load, quiet)

import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch   # noqa: E402
from scipy.stats import norm, binom, t as student_t             # noqa: E402

from pyair2stream import scenario        # noqa: E402
from pyair2stream.uncertainty import generate_ar1_noise        # noqa: E402
from pyair2stream.post_processing import gap_aware_acf          # noqa: E402

EX03 = os.path.join(REPO, "examples", "03_compliance", "output")
EX04 = os.path.join(REPO, "examples", "04_scenario", "output")
LIGHT_BLUE = "#cde2fb"


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def ensure_examples():
    if not os.path.exists(os.path.join(EX03, "check", "cv_yearly_statistics.csv")):
        subprocess.run([sys.executable, "examples/03_compliance/run.py"], cwd=REPO, check=True)
    if not os.path.exists(os.path.join(EX04, "abstraction", "Forward_Prediction_Ensemble_Mentue_c_1d.npz")):
        subprocess.run([sys.executable, "examples/04_scenario/run.py"], cwd=REPO, check=True)


def calibration_residuals():
    """Daily residuals (simulated - measured) of example 03's calibration, 2002-2009."""
    d = pd.read_csv(os.path.join(EX03, "calibration", "2_DE-MCMC_NSE_Mentue_cc_1d.csv"))
    dates = pd.to_datetime(dict(year=d.Year, month=d.Month, day=d.Day))
    obs = d.Twat_obs.where(d.Twat_obs != -999.0)
    return pd.Series((d.Twat_mod - obs).to_numpy(), index=dates)


def meta():
    return json.load(open(os.path.join(EX03, "calibration", "MCMC_chain_Mentue_c_1d_meta.json")))


# --- U01: the pipeline, as a sketch ---------------------------------------------------------------

def u01_pipeline():
    plot_style()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, text, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=SURFACE, ec=colour, lw=1.6))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.55, text, ha="center", va="top", fontsize=7.8, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=INK2))

    top, bottom = 2.55, 0.25
    box(0.1, top, 2.0, 1.8, "1. Calibrate", "DE finds the\nbest-fitting\nparameters\n(run_mode DE-MCMC)", BLUE)
    box(2.5, top, 2.0, 1.8, "2. Parameter\nuncertainty", "\n\nMCMC: every\nparameter set\nthat fits the data", BLUE)
    box(4.9, top, 2.0, 1.8, "3. Error model", "size σ and\npersistence ρ of the\nmodel's daily errors\n(AR(1))", BLUE)
    box(7.3, top, 3.6, 1.8, "4. 1000 simulated series", "each with its own parameter set\nand its own day-to-day error\n(FORWARD, save_ensemble)", BLUE)
    box(7.3, bottom, 3.6, 1.9, "5. Your statistic", "computed in each series: a daily\nvalue, a 7-day mean, a yearly peak,\ndays above a limit; the share above\nthe limit is the probability", ORANGE)
    box(3.9, bottom, 3.0, 1.9, "6. Check", "cross-validation: did the\nstated ranges hold in years\nthe model was not fitted to?\n(cv_*.csv)", AQUA)
    box(0.1, bottom, 3.4, 1.9, "7. Correct and report", "yearly statistics: correct for\nthe bias found in step 6;\nreport the probability, its range\nand the check", AQUA)
    arrow(2.1, top + 0.9, 2.5, top + 0.9)
    arrow(4.5, top + 0.9, 4.9, top + 0.9)
    arrow(6.9, top + 0.9, 7.3, top + 0.9)
    arrow(9.1, top, 9.1, bottom + 1.9)
    arrow(7.3, bottom + 0.95, 6.9, bottom + 0.95)
    arrow(3.9, bottom + 0.95, 3.5, bottom + 0.95)
    save(fig, "U01_pipeline.png")


# --- U02: where the width comes from ---------------------------------------------------------------

def u02_sources():
    """A summer of example 03: the parameter-only band (MCMC draws, almost no added error) inside the
    full band (parameters plus day-to-day error)."""
    work_name = "docs_u02"
    chain = os.path.join(EX03, "calibration", "MCMC_chain_Mentue_c_1d.csv")
    cfg = {"station_name": "Mentue", "series": "c", "version": 8, "integrator": "CRN", "run_mode": "FORWARD",
           "uncertainty_options": {"save_ensemble": True},
           "forward_options": {"enable_prediction_intervals": True, "mcmc_chain_path": chain, "n_samples": 1000,
                               "random_seed": 42, "residual_sigma": 1e-9},
           "paths": {"input_data": os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv"),
                     "calibration_metadata": os.path.join(EX03, "calibration", "calibration_metadata.json")}}
    from pyair2stream.optimization import forward_mode
    data = load(cfg, work_name)
    with quiet():
        forward_mode(data)
    out = data.folder
    par_only, dates = scenario.load_ensemble(os.path.join(out, "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
    full, _ = scenario.load_ensemble(os.path.join(EX03, "prediction", "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
    obs = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv")).T_water.to_numpy()
    m = (dates >= "2011-06-01") & (dates <= "2011-09-30")
    plot_style()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    lo, hi = np.percentile(full[:, m], [5, 95], axis=0)
    plo, phi = np.percentile(par_only[:, m], [5, 95], axis=0)
    ax.fill_between(dates[m], lo, hi, color=LIGHT_BLUE, lw=0, label="90% band: parameters + daily error")
    ax.fill_between(dates[m], plo, phi, color=BLUE, alpha=0.55, lw=0, label="90% band: parameters only")
    ax.plot(dates[m], np.median(full[:, m], axis=0), color=BLUE, lw=1.2, label="median of the simulations")
    ax.scatter(dates[m], obs[m], s=7, color=INK, zorder=4, label="measured (not used)")
    ax.set_ylabel("Water temperature (°C)")
    ax.set_title("Mentue, summer 2011, predicted from 2002-2009: most of the width is the model's daily error",
                 fontsize=9.5)
    ax.legend(loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.32))
    w_full, w_par = float(np.mean(hi - lo)), float(np.mean(phi - plo))
    ax.annotate(f"average width: {w_full:.1f} °C with daily error, {w_par:.2f} °C from parameters alone",
                (0.01, 0.95), xycoords="axes fraction", fontsize=8, color=INK2, va="top")
    save(fig, "U02_sources.png")
    return w_full, w_par


# --- U03-U05: errors persist ---------------------------------------------------------------------

def u03_persistence(res, sigma, rho_week):
    plot_style()
    year = res["2006"]
    rng = np.random.default_rng(3)
    n = len(year)
    iid = rng.normal(0, sigma, n)
    ar = generate_ar1_noise(n, sigma, rho_week, [(0, n - 1)], rng)
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, series, title, colour in zip(
            axes, (year.to_numpy(), iid, ar),
            ("Real errors of the model (Mentue, 2006): runs of warm and cool days",
             f"Independent random errors of the same size (σ = {sigma:.2f} °C): no runs",
             f"AR(1) errors of the same size, ρ = {rho_week:.2f}: runs, as in the real errors"),
            (INK, ORANGE, BLUE)):
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.plot(year.index, series, color=colour, lw=0.9)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("°C")
    axes[0].set_ylim(-2.5, 2.5)
    fig.suptitle("Simulated minus measured water temperature", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, "U03_errors_persist.png")


def u04_autocorrelation(res, rho_day, rho_week):
    plot_style()
    lags, acf, _ = gap_aware_acf(res, 60)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(lags, acf, color=INK, marker="o", ms=3, lw=1, label="real errors (Mentue 2002-2009)")
    k = np.arange(0, 61)
    ax.plot(k, rho_day ** k, color=ORANGE, lw=1.6, label=f"AR(1), ρ = {rho_day:.2f} from consecutive days ('daily')")
    ax.plot(k, rho_week ** k, color=BLUE, lw=1.6, label=f"AR(1), ρ = {rho_week:.2f} from week-to-week ('weekly', default)")
    ax.set_xlabel("Days apart")
    ax.set_ylabel("Correlation of the errors")
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.1, 1.0)
    ax.legend(fontsize=7.5)
    ax.set_title("How long an error lasts: a fast part (days) and a slow part (weeks)", fontsize=9.5)
    save(fig, "U04_autocorrelation.png")


def u05_windows(res, sigma, rho_day, rho_week):
    """Typical error of an average over m days: real, and as each error model implies."""
    plot_style()
    windows = np.array([1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90])

    def ar1_sd(rho, m):
        k = np.arange(1, m)
        return sigma * np.sqrt((1 + 2 * np.sum((1 - k / m) * rho ** k)) / m)
    real = []
    for m in windows:
        r = res.rolling(m).mean()
        complete = res.notna().rolling(m).sum() == m
        real.append(float(np.sqrt(np.mean(r[complete] ** 2))))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(windows, real, color=INK, marker="o", ms=4, lw=1.4, label="real errors")
    ax.plot(windows, [ar1_sd(rho_week, m) for m in windows], color=BLUE, lw=1.6,
            label=f"AR(1), ρ = {rho_week:.2f} (default)")
    ax.plot(windows, [ar1_sd(rho_day, m) for m in windows], color=ORANGE, lw=1.6, label=f"AR(1), ρ = {rho_day:.2f} ('daily')")
    ax.plot(windows, sigma / np.sqrt(windows), color=MUTED, lw=1.6, ls=(0, (4, 2)), label="independent errors")
    ax.set_xscale("log")
    ax.set_xticks(windows[[0, 2, 4, 6, 8, 10, 11]], [str(x) for x in windows[[0, 2, 4, 6, 8, 10, 11]]])
    ax.set_xlabel("Averaging period (days)")
    ax.set_ylabel("Typical error of the average (°C)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5)
    ax.set_title("Errors that persist do not average out: a 7-day or 30-day mean stays uncertain", fontsize=9.5)
    save(fig, "U05_averaging.png")


# --- U06: the band is not the statistic ------------------------------------------------------------

def u06_band_vs_statistic():
    plot_style()
    ens, dates = scenario.load_ensemble(os.path.join(EX03, "prediction", "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
    obs = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv")).T_water.to_numpy()
    y = dates.year == 2011
    counts = scenario.year_statistics(ens, dates, threshold=18.0)[2011]["days above threshold"]
    lo, hi = np.percentile(ens[:, y], [5, 95], axis=0)
    band_lo, band_hi = int((lo > 18).sum()), int((hi > 18).sum())
    measured = int((obs[y] > 18).sum())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    m = (dates >= "2011-06-01") & (dates <= "2011-09-30")
    ax = axes[0]
    lo_s, hi_s = np.percentile(ens[:, m], [5, 95], axis=0)
    ax.fill_between(dates[m], lo_s, hi_s, color=LIGHT_BLUE, lw=0, label="daily 90% band")
    for k in range(3):
        ax.plot(dates[m], ens[k, m], lw=0.8, color=(BLUE, ORANGE, AQUA)[k], alpha=0.9,
                label="three of the 1000 simulated series" if k == 0 else None)
    ax.axhline(18, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.text(dates[m][2], 18.1, "18 °C", fontsize=7.5, color=INK2, va="bottom")
    ax.set_ylabel("Water temperature (°C)")
    ax.set_title("Summer 2011: each simulated series wanders inside the band", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower center")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax = axes[1]
    ax.hist(counts, bins=np.arange(counts.min() - 0.5, counts.max() + 1.5, 1), color=BLUE, alpha=0.7)
    q5, q95 = np.percentile(counts, [5, 95])
    ax.axvspan(q5 - 0.5, q95 + 0.5, color=LIGHT_BLUE, alpha=0.5, lw=0, zorder=0)
    top = ax.get_ylim()[1]
    for x, text, colour, ha in ((band_lo, "band's lower\nedge above 18", ORANGE, "left"),
                                (band_hi, "band's upper\nedge above 18", ORANGE, "right"),
                                (measured, "measured", INK, "left")):
        ax.axvline(x, color=colour, lw=2)
        ax.text(x, top * (0.62 if text == "measured" else 0.95), (" " if ha == "left" else "") + text
                + (" " if ha == "right" else ""), fontsize=7.5, color=colour, ha=ha, va="top")
    ax.set_xlabel("Days above 18 °C in 2011")
    ax.set_ylabel("Simulations")
    ax.set_title(f"Counted series by series: 90% range {q5:.0f}-{q95:.0f} days", fontsize=9)
    save(fig, "U06_band_vs_statistic.png")
    return band_lo, band_hi, float(q5), float(q95), measured


# --- U07: what a level means -----------------------------------------------------------------------

def u07_levels():
    plot_style()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.linspace(-3.6, 3.6, 600)
    ax.set_xlim(-3.7, 6.2)
    ax.plot(x, norm.pdf(x), color=INK, lw=1.4)
    levels = (99, 95, 90, 80, 50)
    shades = ("#eef4fd", "#d9e8fb", "#bcd6f7", "#8fb8ee", "#4f8fe0")
    for level, c in zip(levels, shades):
        z = norm.ppf(0.5 + level / 200)
        xx = x[np.abs(x) <= z]
        ax.fill_between(xx, 0, norm.pdf(xx), color=c, lw=0)
    for k, level in enumerate(levels):
        z = norm.ppf(0.5 + level / 200)
        yy = 0.43 - 0.055 * k
        ax.annotate("", (-z, yy), (z, yy), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
        miss = 100 - level
        days = "1 day in 2" if level == 50 else f"1 day in {round(100 / miss):d}"
        ax.text(2.75, yy, f"{level}%: ±{z:.2f} σ, misses {days}", va="center", fontsize=7.8, color=INK2)
    ax.set_yticks([])
    ax.set_xticks(range(-3, 4), [f"{k:+d} σ" if k else "best\nestimate" for k in range(-3, 4)])
    ax.set_ylim(0, 0.48)
    ax.set_title("A central range of L%: the measured value should fall outside it (100 − L)% of the time",
                 fontsize=9.5)
    ax.spines["left"].set_visible(False)
    save(fig, "U07_levels.png")


# --- U08: the PIT, the measured value's place among the simulations ---------------------------------

def u08_pit():
    plot_style()
    rng = np.random.default_rng(8)
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.6, 0.1, 1, 1, 1], hspace=0.65, wspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    sims = rng.normal(20.2, 0.5, 1000)
    measured = 19.95
    p = float(np.mean(sims < measured))
    bins = np.arange(18.6, 21.81, 0.1)
    ax.hist(sims[sims < measured], bins=bins, color=BLUE, alpha=0.8, label=f"below the measured value: {p:.0%}")
    ax.hist(sims[sims >= measured], bins=bins, color=LIGHT_GREY, alpha=0.9, label="above")
    ax.axvline(measured, color=INK, lw=2)
    ax.text(measured, ax.get_ylim()[1] * 0.55, " measured", fontsize=8, color=INK)
    ax.set_xlabel("A yearly statistic in 1000 simulations (°C)")
    ax.set_ylabel("Simulations")
    ax.set_title(f"One held-out year: PIT = {p:.2f}", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    cases = (("Right: flat", 1.0, 0.0, AQUA), ("Too narrow: U-shaped", 1.6, 0.0, ORANGE),
             ("Too wide: humped", 0.6, 0.0, ORANGE), ("Biased (too warm)", 1.0, -0.7, ORANGE))
    for k, (title, spread, shift, colour) in enumerate(cases):
        ax = fig.add_subplot(gs[k // 2, 2 + k % 2])
        truth = rng.normal(shift, spread, 3000)
        pits = norm.cdf(truth)
        ax.hist(pits, bins=10, range=(0, 1), color=colour, alpha=0.8)
        ax.axhline(300, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([])
        ax.set_xlabel("PIT", fontsize=7.5)
    ax = fig.add_subplot(gs[:, 4])
    ax.axis("off")
    ax.text(0, 0.95, "Many held-out years:\nthe PITs should spread\nevenly between 0 and 1\n(dashed line).\n\n"
            "Share of years with\nPIT between 0.05 and 0.95\n= coverage of the 90%\nrange.", fontsize=8,
            color=INK2, va="top")
    fig.suptitle("The PIT: where the measured value falls among the simulations", y=1.02, fontsize=10)
    save(fig, "U08_pit.png")


# --- U09: what chance alone does --------------------------------------------------------------------

def u09_chance():
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, n in zip(axes, (10, 48)):
        k = np.arange(0, n + 1)
        pmf = binom.pmf(k, n, 0.9)
        lo, hi = binom.interval(0.95, n, 0.9)
        colours = [BLUE if lo <= x <= hi else LIGHT_GREY for x in k]
        ax.bar(k, pmf, color=colours, width=0.8)
        ax.set_xlim(max(-0.5, n * 0.6), n + 0.6)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(f"Years inside a correct 90% range, out of {n}")
        ax.set_ylabel("Chance")
        ax.set_title(f"{n} years: {int(lo)} to {int(hi)} inside is expected by chance (95%)", fontsize=9)
    fig.suptitle("Even a perfect 90% range does not contain exactly 90% of a few years", y=1.03, fontsize=10)
    fig.tight_layout()
    save(fig, "U09_chance.png")


# --- U10: the model's bias on the hottest days ------------------------------------------------------

def u10_bias_at_peaks():
    plot_style()
    t = pd.read_csv(os.path.join(REPO, "validation", "results", "V11_3.csv"))
    t = t[t.statistic.isin(["highest daily mean", "highest 7-day mean"])].copy()
    t["mean"] = t["measured minus predicted median, mean"].str.split().str[0].astype(float)
    ci = t["95% CI"].str.split(" to ")
    t["lo"], t["hi"] = ci.str[0].astype(float), ci.str[1].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, stat in zip(axes, ("highest daily mean", "highest 7-day mean")):
        g = t[t.statistic == stat].reset_index(drop=True)
        ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
        for i, r in g.iterrows():
            colour = ORANGE if r.lo > 0 or r.hi < 0 else BLUE
            ax.plot([r.lo, r.hi], [i, i], color=colour, lw=2)
            ax.scatter(r["mean"], i, color=colour, s=28, zorder=3)
        ax.set_yticks(range(len(g)), [f"{r.river}, version {r.version}" for _, r in g.iterrows()], fontsize=8)
        ax.set_xlabel("Measured minus predicted median (°C)")
        ax.set_title(stat, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("Years held out by cross-validation: the model's average error in the yearly peak, with 95% "
                 "interval (orange: biased)", y=1.02, fontsize=9.5)
    fig.tight_layout()
    save(fig, "U10_bias_at_peaks.png")


# --- U11: the correction ----------------------------------------------------------------------------

def u11_correction():
    plot_style()
    check = pd.read_csv(os.path.join(EX03, "check", "cv_yearly_statistics.csv"))
    dev = check[check.statistic == "highest 7-day mean"].set_index("year").deviation
    ens, dates = scenario.load_ensemble(os.path.join(EX03, "prediction", "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
    peak = scenario.year_statistics(ens, dates, threshold=18.0)[2011]["highest 7-day mean"]
    corrected = scenario.correct_statistic(peak, dev, seed=2011)
    obs = pd.read_csv(os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv"), parse_dates=["Date"])
    measured = float(obs[obs.Date.dt.year == 2011].T_water.rolling(7).mean().max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    ax.axhline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.scatter(dev.index, dev.values, color=INK, s=24, zorder=3, label="one held-out year")
    n = len(dev)
    half = student_t.ppf(0.975, n - 1) * dev.std(ddof=1) / np.sqrt(n)
    ax.axhspan(dev.mean() - half, dev.mean() + half, color=ORANGE, alpha=0.2, lw=0)
    ax.axhline(dev.mean(), color=ORANGE, lw=1.6, label=f"mean {dev.mean():+.2f} °C, with 95% interval")
    ax.set_xlabel("Year held out (calibrated on the others)")
    ax.set_ylabel("Measured minus predicted\nmedian, highest 7-day mean (°C)")
    ax.set_title("Step 1, the check: the model puts the peak too high", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    bins = np.arange(17.5, 22.51, 0.2)
    ax.hist(peak, bins=bins, color=LIGHT_GREY, alpha=0.9, label=f"uncorrected: P(> 20 °C) = {np.mean(peak > 20):.2f}")
    ax.hist(corrected, bins=bins, color=BLUE, alpha=0.6, label=f"corrected: P(> 20 °C) = {np.mean(corrected > 20):.2f}")
    ax.axvline(20, color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.axvline(measured, color=INK, lw=2)
    ax.text(20, ax.get_ylim()[1] * 0.97, " limit", fontsize=8, color=INK2, va="top")
    ax.text(measured, ax.get_ylim()[1] * 0.85, "measured ", fontsize=8, color=INK, ha="right", va="top")
    ax.set_xlabel("Highest 7-day mean of 2011 (°C)")
    ax.set_ylabel("Simulations")
    ax.set_title("Step 2: shift every simulation by that mean, with its uncertainty", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "U11_correction.png")
    return float(dev.mean()), float(half), float(np.mean(peak > 20)), float(np.mean(corrected > 20))


# --- U12: paired scenarios ----------------------------------------------------------------------------

def u12_paired():
    plot_style()
    a, dates = scenario.load_ensemble(os.path.join(EX04, "abstraction", "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
    b, _ = scenario.load_ensemble(os.path.join(EX04, "baseline", "Forward_Prediction_Ensemble_Mentue_c_1d.npz"))
    summer = (dates.year == 2011) & dates.month.isin([6, 7, 8])
    ma, mb = a[:, summer].mean(axis=1), b[:, summer].mean(axis=1)
    paired = ma - mb
    rng = np.random.default_rng(12)
    unpaired = ma - rng.permutation(mb)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    bins = np.linspace(min(ma.min(), mb.min()), max(ma.max(), mb.max()), 40)
    ax.hist(mb, bins=bins, color=BLUE, alpha=0.6, label="measured flow")
    ax.hist(ma, bins=bins, color=ORANGE, alpha=0.6, label="30% of the flow abstracted")
    ax.set_xlabel("Mean water temperature, June-August 2011 (°C)")
    ax.set_ylabel("Simulations")
    ax.set_title("Each scenario alone: the two ranges overlap almost entirely", fontsize=9)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)))
    for i, (vals, label, colour) in enumerate(((paired, "paired: same parameters\nand same daily error", AQUA),
                                               (unpaired, "unpaired (wrong)", LIGHT_GREY))):
        lo, med, hi = np.percentile(vals, [5, 50, 95])
        ax.plot([lo, hi], [i, i], color=colour, lw=8, solid_capstyle="butt")
        ax.scatter(med, i, color=INK, s=22, zorder=3)
        fmt = lambda v: "0.00" if abs(v) < 0.005 else f"{v:+.2f}"
        ax.text(hi + 0.04, i, f"{fmt(lo)} to {fmt(hi)} °C", va="center", fontsize=8, color=INK2)
    ax.set_yticks([0, 1], ["paired: same parameters\nand same daily error", "unpaired (wrong)"], fontsize=8)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(-0.8, 1.3)
    ax.set_xlabel("Change in the June-August mean (°C), 90% range and median")
    ax.set_title("The difference, simulation by simulation: the change is clear", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "U12_paired.png")
    return [float(x) for x in np.percentile(paired, [5, 50, 95])], [float(x) for x in np.percentile(unpaired, [5, 95])]


# --- U13: parameter sets that fit almost equally well ---------------------------------------------------

def u13_parameters():
    plot_style()
    chain = pd.read_csv(os.path.join(EX03, "calibration", "MCMC_chain_Mentue_c_1d.csv"))
    best = [float(x) for x in open(os.path.join(EX03, "calibration", "1_DE-MCMC_NSE_Mentue_c_1d.out")).readline().split()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.scatter(chain.par_2, chain.par_3, s=3, color=BLUE, alpha=0.25, lw=0, label="parameter sets from MCMC")
    ax.scatter(best[1], best[2], s=60, color=ORANGE, marker="*", zorder=3, label="best fit (DE)")
    ax.set_xlabel("a2 (effect of air temperature)")
    ax.set_ylabel("a3 (relaxation)")
    r = np.corrcoef(chain.par_2, chain.par_3)[0, 1]
    ax.set_title(f"a2 and a3 trade off (correlation {r:.2f})", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax = axes[1]
    lo, hi = np.percentile(chain.par_7, [5, 95])
    ax.hist(chain.par_7, bins=40, color=BLUE, alpha=0.7)
    ax.axvspan(lo, hi, color=LIGHT_BLUE, alpha=0.6, lw=0, zorder=0)
    ax.axvline(best[6], color=ORANGE, lw=2)
    ax.set_xlabel("a7 (timing of the seasonal term, fraction of a year)")
    ax.set_ylabel("Parameter sets")
    ax.set_title(f"a7: 90% interval {lo:.3f} to {hi:.3f}", fontsize=9)
    save(fig, "U13_parameters.png")


# --- U14: why n days are not n independent pieces of information ------------------------------------------

def u14_effective_n(rho_day, rho_week, n_days):
    plot_style()
    rho = np.linspace(0, 0.97, 300)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(rho, (1 - rho) / (1 + rho), color=BLUE, lw=1.8)
    for r, label, ytext in ((rho_day, "'daily' ρ", 0.22), (rho_week, "'weekly' ρ (default)", 0.10)):
        f = (1 - r) / (1 + r)
        ax.scatter(r, f, color=ORANGE, s=30, zorder=3)
        ax.annotate(f"{label} = {r:.2f}: {n_days} days count as {n_days * f:.0f}", (r, f), xytext=(0.03, ytext),
                    textcoords="data", fontsize=8, color=INK2, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xlabel("Persistence of the errors from one day to the next (ρ)")
    ax.set_ylabel("Independent days per day measured")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.set_title("Persistent errors carry less information: the effective sample size", fontsize=9.5)
    save(fig, "U14_effective_n.png")


def main():
    ensure_examples()
    res = calibration_residuals()
    m = meta()
    sigma, rho_week = float(m["sigma"]), float(m["rho"])
    from pyair2stream.uncertainty import estimate_ar1_rho
    ok = res.notna().to_numpy()
    rho_day = estimate_ar1_rho(np.nan_to_num(res.to_numpy()), np.where(ok, 0.0, -999.0), ok, [(0, len(res) - 1)])
    numbers = {"sigma": sigma, "rho_week": rho_week, "rho_day": rho_day, "n_days": int(res.notna().sum())}
    u01_pipeline()
    numbers["band_width_full"], numbers["band_width_parameters"] = u02_sources()
    u03_persistence(res, sigma, rho_week)
    u04_autocorrelation(res, rho_day, rho_week)
    u05_windows(res, sigma, rho_day, rho_week)
    numbers["band_count_lo"], numbers["band_count_hi"], numbers["count_q5"], numbers["count_q95"], \
        numbers["count_measured"] = u06_band_vs_statistic()
    u07_levels()
    u08_pit()
    u09_chance()
    u10_bias_at_peaks()
    numbers["dev_mean"], numbers["dev_half"], numbers["p_uncorrected"], numbers["p_corrected"] = u11_correction()
    numbers["paired"], numbers["unpaired"] = u12_paired()
    u13_parameters()
    u14_effective_n(rho_day, rho_week, numbers["n_days"])
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
