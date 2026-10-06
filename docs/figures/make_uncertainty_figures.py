"""
Draw the figures of docs/UNCERTAINTY.md.

    python docs/figures/make_uncertainty_figures.py          # every figure
    python docs/figures/make_uncertainty_figures.py --rho    # only those of the weekly rho (U15-U19)

Uses the outputs of examples 03 and 04 (running them first if they are missing,
a few minutes), the validation results in validation/results/, and, for U16 and
U19, calibrations of versions 5 and 8 on the three Swiss rivers (a few
minutes). Writes docs/figures/U*.png and prints the numbers the documents quote.
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

from common import (plot_style, BLUE, ORANGE, AQUA, YELLOW, INK, INK2, MUTED, GRID, AXIS, SURFACE, LIGHT_GREY,  # noqa: E402
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


# --- U15-U19: the weekly rho, how it is made and what it changes -------------------------------------

RIVERS = (("Mentue", "MAH_2369"), ("Rhône", "SIO_2011"), ("Dischmabach", "DAV_2327"))
WINDOWS = np.array([1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90])


def rho_estimates(res):
    """The package's two estimates of rho from a daily residual series with NaN gaps: from consecutive days,
    and from consecutive weeks; and the default, the larger of the two."""
    from pyair2stream.uncertainty import estimate_ar1_rho, estimate_ar1_rho_weekly
    ok = res.notna().to_numpy()
    e, obs, seg = np.nan_to_num(res.to_numpy()), np.where(ok, 0.0, -999.0), [(0, len(res) - 1)]
    r_day = estimate_ar1_rho(e, obs, ok, seg)
    r_week_only = estimate_ar1_rho_weekly(e, obs, ok, seg)
    return r_day, r_week_only, max(r_day, r_week_only)


def week_correlation(res, w=7):
    """Measured correlation between the mean errors of a w-day window and of the next w-day window
    (every starting day, complete windows only), as the weekly estimate measures it for w = 7."""
    means = res.rolling(w).mean().to_numpy()
    full = (res.notna().rolling(w).sum() == w).to_numpy()
    ok = full[:-w] & full[w:]
    return float(np.corrcoef(means[:-w][ok], means[w:][ok])[0, 1])


def alternative_rhos(res, sigma):
    """Two other single-rho choices, for comparison: the rho that reproduces the persistence of 30-day
    windows (as the weekly estimate does for 7-day windows), and the rho that reproduces the size of the
    7-day mean error."""
    from scipy.optimize import brentq
    from pyair2stream.uncertainty import weekly_mean_correlation
    r30 = week_correlation(res, 30)
    rho_month = brentq(lambda x: weekly_mean_correlation(x, 30) - r30, 1e-6, 0.999) if r30 > 0 else 0.0
    real7 = real_mean_sd(res, 7)
    rho_size = brentq(lambda x: ar1_mean_sd(sigma, x, 7) - real7, 0.0, 0.999)
    return rho_month, rho_size


def ar1_mean_sd(sigma, rho, m):
    """Standard deviation of the mean of m consecutive days of AR(1) errors."""
    k = np.arange(1, m)
    return sigma * np.sqrt((1 + 2 * np.sum((1 - k / m) * rho ** k)) / m)


def real_mean_sd(res, m):
    """Root-mean-square of the real errors averaged over m consecutive measured days."""
    r = res.rolling(m).mean()
    full = res.notna().rolling(m).sum() == m
    return float(np.sqrt(np.mean(r[full] ** 2)))


def river_residuals():
    """Daily calibration errors of versions 5 and 8 on the three Swiss rivers (DE, seed 42; a few minutes)."""
    from common import calibrate, river_csv, daily
    out = {}
    for river, station in RIVERS:
        for version in (5, 8):
            data = calibrate(river_csv(station, "calibration"), version, seed=42, name=f"docs_rho_{station}_{version}")
            obs, sim = daily(data)
            out[(river, version)] = pd.Series(sim - obs)
    return out


def u15_rho_how(res, rho_day, rho_week_only, rho_week, r_week):
    """Diagram: the same daily errors read day to day and week to week, the two estimates, and where rho is used."""
    plot_style()
    fig = plt.figure(figsize=(11, 6.6))
    ax = fig.add_axes([0.07, 0.53, 0.9, 0.4])
    span = res["2006-06-05":"2006-07-30"]                       # eight weeks, Monday to Sunday
    days = span.index
    half = pd.Timedelta(hours=12)
    for k in range(0, 8, 2):
        ax.axvspan(days[7 * k] - half, days[7 * k + 6] + half, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(days, span, color=INK, lw=0.9, marker="o", ms=2.6, label="daily error (simulated − measured)")
    means = []
    for k in range(8):
        w = span.iloc[7 * k:7 * k + 7]
        means.append((w.index[3], float(w.mean())))
        ax.plot([w.index[0] - half, w.index[-1] + half], [w.mean()] * 2, color=BLUE, lw=3.2, solid_capstyle="butt",
                label="mean error of each 7-day window" if k == 0 else None)
    i = 16
    ax.annotate("", (days[i + 1], span.iloc[i + 1]), (days[i], span.iloc[i]),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, connectionstyle="arc3,rad=-0.6"))
    ax.text(days[i], max(span.iloc[i], span.iloc[i + 1]) + 0.25, "'daily':\nday → next day", color=ORANGE,
            fontsize=8, ha="center", va="bottom")
    (x0, y0), (x1, y1) = means[4], means[5]
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6,
                                                        connectionstyle="arc3,rad=-0.35"))
    ax.text(x0 + pd.Timedelta(days=3.5), max(y0, y1) + 0.45, "'weekly':\nweek → next week", color=BLUE, fontsize=8,
            ha="center", va="bottom")
    ax.set_ylabel("Error (°C)")
    lim = float(np.nanmax(np.abs(span))) + 0.9
    ax.set_ylim(-lim, lim)
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title("The same errors, read two ways (Mentue, June-July 2006)", fontsize=10, loc="left")

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.43])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, colour, title, text):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=SURFACE, ec=colour,
                                    lw=1.8))
        ax.text(x + 0.18, y + h - 0.16, title, ha="left", va="top", fontsize=9.5, weight="bold", color=INK)
        ax.text(x + 0.18, y + h - 0.52, text, ha="left", va="top", fontsize=8.3, color=INK2, linespacing=1.4)

    box(0.15, 1.72, 5.9, 1.18, ORANGE, "'daily' estimate",
        f"Correlate each day's error with the next day's: {rho_day:.2f}.\nSo ρ = {rho_day:.2f}.")
    box(0.15, 0.1, 5.9, 1.45, BLUE, "'weekly' estimate",
        f"Correlate each 7-day mean error with the mean of the next 7 days\n(windows starting on every day, "
        f"not only Mondays): {r_week:.2f}.\n"
        f"Find the AR(1) whose 7-day means correlate {r_week:.2f}:\nρ = {rho_week_only:.2f}.")
    box(6.75, 0.1, 4.1, 2.8, AQUA, "Default: the larger of the two",
        f"ρ = {rho_week:.2f} ('weekly', the default)\n\nρ then sets:\n• how wide the parameter ranges are\n"
        "   (the effective number of days)\n• how long the random errors added to\n   every simulated series last\n"
        "• the same, in the cross-validation check")
    for y in (2.31, 0.83):
        ax.add_patch(FancyArrowPatch((6.08, y), (6.72, 1.5), arrowstyle="-|>", mutation_scale=13, lw=1.3, color=INK2))
    save(fig, "U15_rho_how.png")


def u16_rho_conversion(rho_day, rho_week_only, r_week, cases):
    """Left: the week-to-week correlation of an AR(1)'s 7-day means against rho, and how the measured value is
    converted. Right: on every river and version, the measured week-to-week correlation against what the daily rho
    implies."""
    from pyair2stream.uncertainty import weekly_mean_correlation, MAX_RHO
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    rho = np.linspace(0, MAX_RHO, 400)
    ax.plot(rho, [weekly_mean_correlation(x) for x in rho], color=INK, lw=1.8,
            label="AR(1): correlation of consecutive 7-day means")
    implied = weekly_mean_correlation(rho_day)
    ax.plot([rho_day, rho_day, 0], [0, implied, implied], color=ORANGE, lw=1.2, ls=(0, (4, 2)))
    ax.scatter(rho_day, implied, color=ORANGE, s=34, zorder=3)
    ax.annotate(f"daily ρ = {rho_day:.2f} implies\nconsecutive weeks correlate {implied:.2f}", (rho_day, implied),
                xytext=(0.05, 0.42), fontsize=8, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7))
    ax.plot([0, rho_week_only, rho_week_only], [r_week, r_week, 0], color=BLUE, lw=1.2, ls=(0, (4, 2)))
    ax.scatter(rho_week_only, r_week, color=BLUE, s=34, zorder=3)
    ax.annotate(f"measured: {r_week:.2f}\n→ weekly ρ = {rho_week_only:.2f}", (rho_week_only, r_week),
                xytext=(0.42, 0.78), fontsize=8, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("ρ, persistence from one day to the next")
    ax.set_ylabel("Correlation of consecutive 7-day mean errors")
    ax.set_title("Converting the week-to-week correlation to ρ (Mentue, version 8)", fontsize=9.5, loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axes[1]
    labels = []
    for i, (name, c) in enumerate(cases.items()):
        y = len(cases) - 1 - i
        ax.plot([c["r_week_if_daily"], c["r_week"]], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter(c["r_week_if_daily"], y, color=ORANGE, s=40, zorder=3, label="if the errors were AR(1) with the daily ρ" if i == 0 else None)
        ax.scatter(c["r_week"], y, color=INK, s=40, zorder=3, label="measured" if i == 0 else None)
        labels.append(name)
    ax.set_yticks(range(len(cases)), labels[::-1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(cases) - 0.3 + 0.9)
    ax.set_xlabel("Correlation of consecutive 7-day mean errors")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", fontsize=7.5, ncol=1)
    ax.set_title("Every river: errors persist from week to week\nfar more than the daily ρ implies", fontsize=9.5,
                 loc="left")
    fig.tight_layout()
    save(fig, "U16_rho_conversion.png")


def u17_rho_series(res, sigma, rho_day, rho_week):
    """Two years of real errors and of AR(1) errors with each rho (the same random numbers), with their 30-day
    means: the same daily size, but over a month the daily rho averages the errors away."""
    plot_style()
    span = res["2006":"2007"]
    n = len(span)
    shocks = np.random.default_rng(17).standard_normal(n)

    def ar1(rho):
        x = np.empty(n)
        x[0] = sigma * shocks[0]
        for t in range(1, n):
            x[t] = rho * x[t - 1] + sigma * np.sqrt(1 - rho ** 2) * shocks[t]
        return x
    m = 30
    month_real = real_mean_sd(res, m)
    rows = ((span.to_numpy(), INK, f"Real errors (Mentue). Typical size: {sigma:.2f} °C a day, "
                                   f"{month_real:.2f} °C for a 30-day mean (2002-2009)"),
            (ar1(rho_day), ORANGE, f"AR(1) with the daily ρ = {rho_day:.2f}: same daily size, but 30-day means only "
                                   f"{ar1_mean_sd(sigma, rho_day, m):.2f} °C: a month averages the errors away"),
            (ar1(rho_week), BLUE, f"AR(1) with the weekly ρ = {rho_week:.2f} (default): same daily size, 30-day means "
                                  f"{ar1_mean_sd(sigma, rho_week, m):.2f} °C: drifts for weeks, like the real errors"))
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True, sharey=True)
    for ax, (series, colour, title) in zip(axes, rows):
        s = pd.Series(series, index=span.index)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.plot(s.index, s, color=colour, lw=0.5, alpha=0.4, label="daily")
        ax.plot(s.index, s.rolling(m, center=True).mean(), color=colour, lw=2.2, label="30-day mean")
        ax.set_title(title, fontsize=8.8, loc="left")
        ax.set_ylabel("°C")
    axes[0].legend(loc="upper right", fontsize=7.5, ncol=2)
    axes[0].set_ylim(-2.5, 2.5)
    fig.suptitle("Same size of daily error, different persistence: over weeks and months the daily ρ is too calm",
                 y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, "U17_rho_series.png")
    return {"sd30_real": month_real, "sd30_daily": ar1_mean_sd(sigma, rho_day, m),
            "sd30_weekly": ar1_mean_sd(sigma, rho_week, m)}


def evidence_rows():
    """Weekly against daily rho in the validation suite, read from validation/results/ (V4, V5, V9).
    Each row: (label, check, weekly (low, high), daily (low, high)), coverage of nominal 90% ranges in %."""
    res = os.path.join(REPO, "validation", "results")
    pct = lambda s: pd.to_numeric(s.astype(str).str.rstrip("%"))
    v4 = pd.read_csv(os.path.join(res, "V4_1.csv"))
    v4["code"] = v4["case"].str[0]
    par = v4.set_index("code")["parameter coverage"].pipe(pct)
    a7 = pd.read_csv(os.path.join(res, "V4_4.csv")).set_index("parameter").loc["a7"].pipe(lambda s: pct(s))
    v5 = pd.read_csv(os.path.join(res, "V5_2.csv"))
    week5, day5 = v5[v5["noise model"] == "ar1-ls"], v5[v5["noise model"] == "ar1-ls-daily"]
    v9a = pd.read_csv(os.path.join(res, "V9_1.csv"))
    fs_week = pct(v9a[v9a.case == "version 5, fast + slow noise"]["inside 90% range"])
    fs_day = pct(v9a[v9a.case == "version 5, fast + slow noise, rho from consecutive days"]["inside 90% range"])
    v9b = pd.read_csv(os.path.join(res, "V9_3.csv"))
    yr_week = pct(v9b[v9b["rho time scale"].str.startswith("weekly")]["inside 90% range"])
    yr_day = pct(v9b[v9b["rho time scale"].str.startswith("daily")]["inside 90% range"])
    span = lambda s: (float(s.min()), float(s.max()))
    return [
        ("Synthetic AR(1) errors:\nparameter ranges", "V4", span(par[["E"]]), span(par[["G"]])),
        ("Synthetic fast + slow errors:\nparameter ranges, all", "V4", span(par[["H"]]), span(par[["I"]])),
        ("Synthetic fast + slow errors:\nrange of a7 (seasonal timing)", "V4", span(a7[["H"]]), span(a7[["I"]])),
        ("Synthetic fast + slow errors:\nyearly peaks and counts", "V9", span(fs_week), span(fs_day)),
        ("Real rivers, new years:\ndaily values", "V5", span(pct(week5.coverage)), span(pct(day5.coverage))),
        ("Real rivers, new years:\n7-day means", "V5", span(pct(week5["7-day mean coverage"])),
         span(pct(day5["7-day mean coverage"]))),
        ("Real rivers, new years:\nyearly peaks and counts", "V9", span(yr_week), span(yr_day)),
    ]


def u18_rho_evidence(rows):
    plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axvspan(85, 95, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(90, color=INK2, lw=1, ls=(0, (4, 3)))
    for i, (label, check, week, day) in enumerate(rows):
        y = len(rows) - 1 - i
        for (lo, hi), dy, colour, name in ((week, 0.14, BLUE, "weekly ρ (default)"), (day, -0.14, ORANGE, "daily ρ")):
            if hi - lo < 0.6:
                ax.scatter((lo + hi) / 2, y + dy, color=colour, s=40, zorder=3, label=name if i == 0 else None)
            else:
                ax.plot([lo, hi], [y + dy] * 2, color=colour, lw=6, solid_capstyle="round", zorder=3,
                        label=name if i == 0 else None)
        ax.text(101, y, check, fontsize=8, color=INK2, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8)
    ax.set_xlim(50, 103)
    ax.set_xlabel("Share inside the 90% range (%): the dashed line is the target; the grey band is ±5 points")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Where the two choices differ, the weekly ρ holds its 90% ranges; where they agree, it costs nothing",
                 fontsize=9.5, loc="left")
    save(fig, "U18_rho_evidence.png")


def u19_rho_scales(cases):
    """For every river and version: how well each single-rho choice reproduces the typical error of an
    m-day mean (the band spans the six river and version cases; the line is their median)."""
    plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axhspan(0.9, 1.1, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
    choices = (("ratio_month", AQUA, "ρ matched to month-to-month persistence"),
               ("ratio_week", BLUE, "ρ matched to week-to-week persistence ('weekly', default)"),
               ("ratio_size", YELLOW, "ρ matched to the size of a 7-day mean error"),
               ("ratio_day", ORANGE, "ρ from consecutive days ('daily')"))
    for key, colour, label in choices:
        r = np.array([c[key] for c in cases.values()])
        ax.fill_between(WINDOWS, r.min(axis=0), r.max(axis=0), color=colour, alpha=0.22, lw=0)
        ax.plot(WINDOWS, np.median(r, axis=0), color=colour, lw=2, label=label)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 7, 14, 30, 60, 90], ["1", "3", "7", "14", "30", "60", "90"])
    ax.set_xlabel("Averaging period (days)")
    ax.set_ylabel("Typical error of the average:\nAR(1) ÷ real")
    ax.set_ylim(0.5, 1.5)
    ax.text(1.05, 1.43, "above 1: ranges wider than needed (cautious)", fontsize=8, color=INK2)
    ax.text(1.05, 0.53, "below 1: ranges too narrow (over-confident)", fontsize=8, color=INK2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=7.5)
    ax.set_title("Three rivers, versions 5 and 8: of the single-ρ choices, the weekly one stays closest to the real "
                 "errors\nfrom a day to three months (bands: the six cases; lines: their median)", fontsize=9.5,
                 loc="left")
    save(fig, "U19_rho_scales.png")


def rho_figures(res, sigma):
    """Draw U15-U19; return the numbers the documents quote."""
    from pyair2stream.uncertainty import weekly_mean_correlation
    rho_day, rho_week_only, rho_week = rho_estimates(res)
    r_week = week_correlation(res)
    cases = {}
    for (river, version), e in river_residuals().items():
        d, wo, w = rho_estimates(e)
        s = float(np.sqrt(np.nanmean(e ** 2)))
        real = [real_mean_sd(e, m) for m in WINDOWS]
        month, size = alternative_rhos(e, s)
        cases[f"{river}, version {version}"] = {
            "rho_day": d, "rho_week": w, "rho_month": month, "rho_size": size,
            "r_week": week_correlation(e), "r_week_if_daily": weekly_mean_correlation(d),
            **{f"ratio_{k}": [ar1_mean_sd(s, rho, m) / r for m, r in zip(WINDOWS, real)]
               for k, rho in (("day", d), ("week", w), ("month", month), ("size", size))}}
    u15_rho_how(res, rho_day, rho_week_only, rho_week, r_week)
    u16_rho_conversion(rho_day, rho_week_only, r_week, cases)
    sd30 = u17_rho_series(res, sigma, rho_day, rho_week)
    rows = evidence_rows()
    u18_rho_evidence(rows)
    u19_rho_scales(cases)
    pick = lambda key, m: [round(c[key][list(WINDOWS).index(m)], 2) for c in cases.values()]
    return {"mentue_rho_day": rho_day, "mentue_rho_week_only": rho_week_only, "mentue_rho_week": rho_week,
            "mentue_r_week": r_week, "mentue_r_week_if_daily": weekly_mean_correlation(rho_day),
            "mentue_sd7": {"real": real_mean_sd(res, 7), "daily": ar1_mean_sd(sigma, rho_day, 7),
                           "weekly": ar1_mean_sd(sigma, rho_week, 7)},
            "mentue_sd30": sd30,
            "cases": {k: {key: round(c[key], 3) for key in ("rho_day", "rho_week", "rho_month", "rho_size", "r_week",
                                                             "r_week_if_daily")} for k, c in cases.items()},
            **{key: {m: (min(pick(key, m)), max(pick(key, m))) for m in (1, 3, 7, 14, 30, 60, 90)}
               for key in ("ratio_day", "ratio_week", "ratio_month", "ratio_size")},
            "evidence": [(r[0].replace("\n", " "), r[1], r[2], r[3]) for r in rows]}


def main():
    if sys.argv[1:] == ["--rho"]:          # only the figures of the weekly rho (U15-U19)
        ensure_examples()
        m = meta()
        print(json.dumps(rho_figures(calibration_residuals(), float(m["sigma"])), indent=1, ensure_ascii=False))
        return
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
    numbers["rho"] = rho_figures(res, sigma)
    print(json.dumps(numbers, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
