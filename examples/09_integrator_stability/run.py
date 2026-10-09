"""
Run example 09: numerical stability of the five integrators, for every model version.

    python examples/09_integrator_stability/run.py

No calibration: it uses the parameters published by the original model's authors for the
three Swiss rivers, for every model version (3, 4, 5, 7, 8). Those of Piccolroaz et al. (2016)
were calibrated with Crank-Nicolson (CRN), those of Toffolon and Piccolroaz (2015) with RK4.

Each part prints the numbers of one section of the README:
  §2 The B series (compute_B_series) of every river and version: B against time and discharge.
  §3 What one step of each integrator does to a difference in water temperature (its
     amplification factor), from the formula and measured through the package itself.
  §4 The B-series (Butcher series) of each method: elementary weights of the rooted trees up to
     order 4, the order conditions, and the stability functions they sum to.
  §5 Every integrator with all 30 published parameter sets, against a fine-step solution of the
     same equation, with the checks a FORWARD run makes.
  §6 A calibration that learned RK4's behaviour near its stability limit (Dischmabach, version 5).
  §7 Accuracy when stable: how far each method lags an equilibrium that warms steadily.
  §8 Scenario flows: discharge 0.1 to 3 times the record, with Qmedia kept fixed.
  §9 The package's B check, and the largest growth of a difference over any stretch of days,
     from the B series alone.
Writes tables to output/ and the README's figures to figures/.
"""
import contextlib
import dataclasses
import io
import os
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numba import njit

from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import (NumericalDivergenceError, STABILITY_LIMITS, STABILITY_MAX_GROWTH, call_model,
                                check_numerical_divergence, compute_B_series, largest_growth, stability_report,
                                step_amplification, warn_on_stability)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data", "switzerland")
OUT = os.path.join(HERE, "output")
FIG = os.path.join(HERE, "figures")
for folder in (OUT, FIG, os.path.join(OUT, "configs"), os.path.join(OUT, "scenarios")):
    os.makedirs(folder, exist_ok=True)

RIVERS = {"MAH_2369": "Mentue", "SIO_2011": "Rhône", "DAV_2327": "Dischmabach"}
WATER_ID = {"MAH_2369": 2369, "SIO_2011": 2011, "DAV_2327": 2327}
VERSIONS = (3, 4, 5, 7, 8)
SCHEMES = ("CRN", "EXP", "RK4", "RK2", "EUL")
EXPLICIT = ("RK4", "RK2", "EUL")
SOURCES = {"2016": "calibrated with CRN (Piccolroaz et al., 2016)",
           "2015": "calibrated with RK4 (Toffolon and Piccolroaz, 2015)"}
SUBSTEPS = 96          # fine-step reference: 96 RK4 sub-steps per day (15 minutes), as in validation V6
PLAUSIBLE = 60.0       # degC, the package's default max_plausible_twat
GOOD, POOR = 0.3, 1.0  # degC RMS difference from the reference: follows the equation / clearly wrong

# One colour per integrator, in every figure (the categorical palette of validation/common.py).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREEN, VIOLET = "#008300", "#4a3aa7"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOUR = {"CRN": BLUE, "EXP": ORANGE, "RK4": AQUA, "RK2": YELLOW, "EUL": MAGENTA}
MARKER = {"CRN": "o", "EXP": "s", "RK4": "D", "RK2": "^", "EUL": "v"}
VERSION_COLOUR = {8: INK, 7: GREEN, 4: VIOLET, 5: MUTED, 3: MUTED}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}

matplotlib.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "legend.frameon": False,
    "legend.fontsize": 8})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def limit_line(ax, value, label, axis="y", **kw):
    """A dashed reference line (a stability limit), labelled at its end."""
    (ax.axhline if axis == "y" else ax.axvline)(value, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    if label:
        if axis == "y":
            ax.annotate(label, (1, value), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                        textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK2, **kw)
        else:
            ax.annotate(label, (value, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                        textcoords="offset points", ha="left", va="top", fontsize=7, color=INK2, **kw)


# --- Published parameters and running the package as a user would --------------------------

_BOOK = openpyxl.load_workbook(os.path.join(DATA, "published", "Piccolroaz_etal_HP2016-Parameter_values.xlsx"),
                               data_only=True)
_TABLE_2015 = pd.read_csv(os.path.join(DATA, "published", "Toffolon_Piccolroaz_ERL2015.csv"))


def parameters(source, version, station):
    """The published daily-resolution parameters a1..a8 (0 where the version does not use one)."""
    if source == "2015":
        r = _TABLE_2015[(_TABLE_2015.station == station) & (_TABLE_2015.version == version)].iloc[0]
        return [0.0 if pd.isna(r[f"a{i}"]) else float(r[f"a{i}"]) for i in range(1, 9)]
    rows = list(_BOOK[f"a2s_{version}"].iter_rows(values_only=True))
    header = rows[1]
    for r in rows[2:]:
        if r[1] == WATER_ID[station]:
            par, c = [0.0] * 8, 3
            while c < len(header) and header[c] is not None:      # the first block: daily resolution
                par[int(header[c][1]) - 1] = float(r[c])
                c += 1
            return par
    raise KeyError(station)


def calibration_csv(station):
    return os.path.join(DATA, f"{station}_calibration.csv")


def mean_discharge(csv):
    q = pd.read_csv(csv).Discharge
    return float(q[q > 0].mean())


def load(csv, version, par, integrator, qmedia, name, **extra):
    """A FORWARD settings file for `par`, loaded with read_calibration and read_Tseries."""
    cfg = {"project_name": "stability", "station_name": name, "series": "c", "version": version,
           "integrator": integrator, "run_mode": "FORWARD", "objective_function": "RMS",
           "parameters_forward": [float(x) for x in par], "Qmedia": float(qmedia),
           "paths": {"input_data": os.path.relpath(csv, REPO), "output_dir": os.path.relpath(OUT, REPO)}, **extra}
    path = os.path.join(OUT, "configs", f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    cwd = os.getcwd()
    os.chdir(REPO)                       # paths in a settings file are relative to where you run from
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            data = read_calibration(path)
            read_Tseries(data, "c")
    finally:
        os.chdir(cwd)
    return data


def simulate(data):
    """Simulate with the checks a FORWARD run makes: the B check before, the divergence check
    after. A real run that is stopped by the B check does not simulate; here it is simulated
    anyway, to show what it would have given. Returns 'none', 'warned' or 'stopped'."""
    response, printed = "none", io.StringIO()
    with contextlib.redirect_stdout(printed):
        try:
            warn_on_stability(data, data.stability_error_fraction)
        except NumericalDivergenceError:
            response = "stopped"
        call_model(data)
        if response == "none":
            try:
                check_numerical_divergence(data, data.max_plausible_twat)
            except NumericalDivergenceError:
                response = "stopped"
    if response == "none" and "stability limit" in printed.getvalue():
        response = "warned"
    return response


@njit
def _rhs(version, p, ta, q, tw, t, qmedia):
    a1, a2, a3, a4, a5, a6, a7, a8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    season = np.cos(2.0 * np.pi * (t - a7))
    if version == 3:
        return a1 + a2 * ta - a3 * tw
    if version == 5:
        return a1 + a2 * ta - a3 * tw + a6 * season
    th = q / qmedia
    if version == 4:
        return (a1 + a2 * ta - a3 * tw) / th ** a4
    if version == 7:
        return a1 + a2 * ta - a3 * tw + th * (a5 + a6 * season - a8 * tw)
    return (a1 + a2 * ta - a3 * tw + th * (a5 + a6 * season - a8 * tw)) / th ** a4


@njit
def _reference(version, p, tair, q, tt, tw0, qmedia, tice, m):
    n = tair.shape[0]
    out = np.empty(n)
    out[0] = tw0
    h = 1.0 / m
    for j in range(n - 1):
        tw = out[j]
        for k in range(m):
            s0, s1 = k * h, (k + 1) * h
            sm = 0.5 * (s0 + s1)
            dta, dq = tair[j + 1] - tair[j], q[j + 1] - q[j]
            k1 = _rhs(version, p, tair[j] + s0 * dta, q[j] + s0 * dq, tw, tt[j] + s0 / 365.0, qmedia)
            k2 = _rhs(version, p, tair[j] + sm * dta, q[j] + sm * dq, tw + 0.5 * h * k1, tt[j] + sm / 365.0, qmedia)
            k3 = _rhs(version, p, tair[j] + sm * dta, q[j] + sm * dq, tw + 0.5 * h * k2, tt[j] + sm / 365.0, qmedia)
            k4 = _rhs(version, p, tair[j] + s1 * dta, q[j] + s1 * dq, tw + h * k3, tt[j] + s1 / 365.0, qmedia)
            tw = tw + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if tw < tice:
                tw = tice
        out[j + 1] = tw
    return out


def reference(data):
    """The same equation solved with 96 RK4 sub-steps per day: air temperature, discharge and
    time vary linearly within each day; the ice floor is applied at every sub-step (validation V6
    uses the same solver). With the published parameters it is exact to about 1e-13 degC where
    the exact solution is known (V6, part A)."""
    tw0 = data.Twat_obs[0] if data.Twat_obs[0] != -999.0 else 4.0       # the start call_model uses
    return _reference(data.version, np.asarray(data.par, float), data.Tair, data.Q, data.tt,
                      float(tw0), float(data.Qmedia), float(data.Tice_cover), SUBSTEPS)


# --- What one step does to a difference, with B the same every day -----------------------------

def R_formula(scheme, B):
    """The amplification factor with B the same on every day: the stability function R(z), z = -B."""
    z = -np.asarray(B, float)
    return {"CRN": (1 + z / 2) / (1 - z / 2), "EXP": np.exp(z), "RK4": 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24,
            "RK2": 1 + z + z ** 2 / 2, "EUL": 1 + z}[scheme]


RK4_LIMIT = float(STABILITY_LIMITS["RK4"])
_b = np.linspace(2.7, 2.9, 200001)
RK4_LIMIT_EXACT = float(_b[np.argmin(np.abs(R_formula("RK4", _b) - 1))])
mentue = calibration_csv("MAH_2369")
QM_MENTUE = mean_discharge(mentue)

# --- §2 The B series of every river and version ------------------------------------------------
print("§2 The B series: B (per day) on the calibration record, published parameters")
records, bseries, loaded = [], {}, {}
for station, river in RIVERS.items():
    csv = calibration_csv(station)
    qm = mean_discharge(csv)
    for source in SOURCES:
        for v in VERSIONS:
            par = parameters(source, v, station)
            d = load(csv, v, par, "CRN", qm, f"{station}_{source}_v{v}_CRN")
            B = compute_B_series(d)[365:]       # the first 365 days are the warm-up copy of the first year
            bseries[station, source, v] = B
            loaded[station, source, v] = d
            records.append({"river": river, "parameters": source, "version": v, "a3": par[2], "a4": par[3],
                            "a8": par[7], "B min": B.min(), "B median": np.median(B),
                            "B 99th percentile": np.percentile(B, 99), "B max": B.max(),
                            "days B > 2 (%)": 100 * np.mean(B > 2.0),
                            "days B > 2.785 (%)": 100 * np.mean(B > RK4_LIMIT)})
bstats = pd.DataFrame(records)
pd.set_option("display.width", 200)
print(bstats.round(3).to_string(index=False))
bstats.round(4).to_csv(os.path.join(OUT, "B_series_statistics.csv"), index=False)


def fig_b_year(year=2004):
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.4), sharex="col", gridspec_kw={"height_ratios": [2.2, 1]})
    for k, (station, river) in enumerate(RIVERS.items()):
        dates = pd.to_datetime(pd.read_csv(calibration_csv(station)).Date)
        m = (dates.dt.year == year).to_numpy()
        ax = axes[0, k]
        for v in (8, 7, 5):
            ax.plot(dates[m], bseries[station, "2016", v][m], color=VERSION_COLOUR[v], lw=1.6 if v == 8 else 1.3,
                    label=f"version {v}")
        limit_line(ax, 2.0, None)
        limit_line(ax, RK4_LIMIT, None)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 60)
        ax.set_title(river, loc="left")
        ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
        qm = loaded[station, "2016", 8].Qmedia
        theta = pd.read_csv(calibration_csv(station)).Discharge.to_numpy() / qm
        axes[1, k].plot(dates[m], theta[m], color=INK2, lw=1.1)
        axes[1, k].set_yscale("log")
        axes[1, k].set_ylim(0.04, 30)
        axes[1, k].set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
        axes[1, k].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axes[1, k].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    axes[0, 0].set_ylabel("B, per day (log)")
    axes[1, 0].set_ylabel("Discharge / mean\n(θ, log)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=INK2, lw=0.8, ls=(0, (4, 3))))
    labels.append("stability limits: 2 (EUL, RK2) and 2.785 (RK4)")
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"The B series through {year}, with the 2016 parameters (calibrated with CRN): constant for "
                 f"version 5, following discharge for versions 7 and 8", x=0.01, y=1.0, ha="left", fontsize=10.5)
    fig.align_ylabels(axes[:, 0])
    save(fig, "b_series_through_a_year.png")


def B_of_theta(v, par, theta):
    """B as a function of the scaled discharge theta, the formula of compute_B_series."""
    a3, a4, a8 = par[2], par[3], par[7]
    theta = np.asarray(theta, float)
    if v in (3, 5):
        return np.full(theta.shape, a3)
    return {4: a3 / theta ** a4, 7: a3 + a8 * theta}.get(v, (a3 + a8 * theta) / theta ** a4)


def fig_b_theta():
    theta = np.logspace(np.log10(0.03), np.log10(30), 400)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (station, river) in zip(axes, RIVERS.items()):
        q = pd.read_csv(calibration_csv(station)).Discharge.to_numpy() / loaded[station, "2016", 8].Qmedia
        lo, hi = np.percentile(q, [1, 99])
        ax.axvspan(lo, hi, color=MUTED, alpha=0.12, lw=0)
        ax.axvspan(q.min(), q.max(), color=MUTED, alpha=0.07, lw=0)
        for v in (8, 7, 4, 5):
            par = parameters("2016", v, station)
            ax.plot(theta, B_of_theta(v, par, theta), color=VERSION_COLOUR[v], lw=1.7 if v == 8 else 1.3,
                    label={8: "version 8: (a3 + a8·θ) / θ^a4", 7: "version 7: a3 + a8·θ", 4: "version 4: a3 / θ^a4",
                           5: "versions 3 and 5: a3 (version 5 shown)"}[v])
        limit_line(ax, 2.0, None)
        limit_line(ax, RK4_LIMIT, None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.03, 30)
        ax.set_ylim(0.1, 200)
        ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
        ax.set_yticks([0.1, 1, 2, 10, 100], ["0.1", "1", "2", "10", "100"])
        ax.set_title(river, loc="left")
        ax.set_xlabel("Discharge / mean discharge (θ)")
        ax.annotate("1–99% of\nrecorded days", (np.sqrt(lo * hi), 0.13), ha="center", fontsize=7, color=INK2)
    axes[0].set_ylabel("B, per day")
    axes[0].annotate("RK4 limit 2.785\nEUL, RK2 limit 2", (0.033, 2.9), fontsize=7, color=INK2, va="bottom")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("B against discharge, 2016 parameters: the model version decides the shape", x=0.01, y=1.01,
                 ha="left", fontsize=10.5)
    save(fig, "b_against_discharge.png")


def fig_b_published():
    labels = [(s, v) for s in RIVERS for v in VERSIONS]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    for i, (station, v) in enumerate(labels):
        y = len(labels) - 1 - i
        for source, dy, colour in (("2016", 0.17, COLOUR["CRN"]), ("2015", -0.17, COLOUR["RK4"])):
            B = bseries[station, source, v]
            p5, p50, p95 = np.percentile(B, [5, 50, 95])
            ax.plot([B.min(), B.max()], [y + dy] * 2, color=colour, lw=1.0, solid_capstyle="round")
            ax.plot([p5, p95], [y + dy] * 2, color=colour, lw=4.5, solid_capstyle="round")
            ax.scatter([p50], [y + dy], s=22, color=SURFACE, edgecolor=colour, lw=1.4, zorder=3)
    limit_line(ax, 2.0, None, axis="x")
    limit_line(ax, RK4_LIMIT, "RK4: 2.785", axis="x")
    ax.annotate("EUL, RK2: 2", (2.0, 1), xycoords=("data", "axes fraction"), xytext=(-3, -2),
                textcoords="offset points", ha="right", va="top", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 50)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20, 50], ["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_yticks(range(len(labels))[::-1], [f"{RIVERS[s]} v{v}" for s, v in labels], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("B on each day of the calibration record (per day, log scale)")
    ax.legend(handles=[Line2D([0], [0], color=COLOUR["CRN"], lw=4.5, label="2016 parameters, calibrated with CRN"),
                       Line2D([0], [0], color=COLOUR["RK4"], lw=4.5, label="2015 parameters, calibrated with RK4"),
                       Line2D([0], [0], color=INK2, lw=4.5, label="thick: 5–95% of days; thin: all days"),
                       Line2D([0], [0], marker="o", lw=0, markerfacecolor=SURFACE, markeredgecolor=INK2,
                              label="median")],
              loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=2)
    ax.set_title("The B series of every published parameter set: calibration with RK4 kept B under RK4's limit",
                 loc="left", fontsize=10)
    save(fig, "b_series_published.png")


fig_b_year()
fig_b_theta()
fig_b_published()

# --- §3 What one step does: the amplification factor ---------------------------------------
print("\n§3 What one step does to a difference in water temperature (B the same every day)")
B_TABLE = (0.5, 1.0, 1.6, 2.0, 2.5, 2.785, 3.0, 5.0)
measured = {}
for scheme in SCHEMES:
    for B in np.arange(0.25, 6.01, 0.25):
        # Version 3 with a1 = a2 = 0 has no forcing at all: each day multiplies the water
        # temperature by R. The ice floor is switched off so that negative values survive.
        d = load(mentue, 3, [0, 0, B, 0, 0, 0, 0, 0], scheme, QM_MENTUE, "unit_step", Tice_cover=-1e300)
        with contextlib.redirect_stdout(io.StringIO()):
            call_model(d)
        measured[scheme, round(B, 3)] = d.Twat_mod[1] / d.Twat_mod[0]
table = pd.DataFrame({"B (per day)": B_TABLE, "equation exp(-B)": np.exp(-np.array(B_TABLE))})
for scheme in SCHEMES:
    table[scheme] = R_formula(scheme, B_TABLE)
print(table.round(3).to_string(index=False))
table.round(4).to_csv(os.path.join(OUT, "amplification_factors.csv"), index=False)
largest = max(abs(measured[s, B] - R_formula(s, B)) for s, B in measured)
print(f"Measured through the package's integrators at {len(measured)} (method, B) pairs: largest difference "
      f"from the formula {largest:.1e}.")
bb = np.linspace(0, 3, 300001)
for scheme in ("RK4", "RK2"):
    r = R_formula(scheme, bb)
    print(f"{scheme}: smallest amplification factor {r.min():.3f} (at B = {bb[np.argmin(r)]:.2f}); it never "
          f"damps a difference by more than {1 - r.min():.0%} per day.")
print(f"RK4's limit: R(-B) = 1 at B = {RK4_LIMIT_EXACT:.4f} (the package uses {RK4_LIMIT}).")

# B changing from day to day: version 8 with the forcing switched off (a1 = a2 = a5 = a6 = 0, so A = 0) on each
# river's real discharge. Then each day multiplies the water temperature by exactly R_j (step_amplification).
worst, days = 0.0, 0
for station in RIVERS:
    csv = calibration_csv(station)
    par = parameters("2016", 8, station)
    no_forcing = [0.0, 0.0, par[2], par[3], 0.0, 0.0, par[6], par[7]]
    for scheme in SCHEMES:
        d = load(csv, 8, no_forcing, scheme, mean_discharge(csv), f"factors_{station}_{scheme}", Tice_cover=-1e300)
        with contextlib.redirect_stdout(io.StringIO()):
            call_model(d)
        tw = d.Twat_mod
        ok = (np.abs(tw[:-1]) > 1e-200) & (np.abs(tw[:-1]) < 1e200) & np.isfinite(tw[1:])
        R = step_amplification(d)
        worst = max(worst, float(np.max(np.abs(tw[1:][ok] / tw[:-1][ok] - R[ok]) / np.maximum(1.0, np.abs(R[ok])))))
        days += int(ok.sum())
print(f"B changing with discharge (version 8, three rivers, five methods): step_amplification matches the "
      f"integrators on {days} steps, largest relative difference {worst:.1e}.")


def fig_amplification():
    B = np.linspace(0, 6, 601)
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.9), sharey=True)
    for ax, scheme in zip(axes, SCHEMES):
        ax.axhspan(1, 3, color=STATUS["critical"], alpha=0.07, lw=0)
        ax.axhspan(-3, -1, color=STATUS["critical"], alpha=0.07, lw=0)
        ax.axhspan(-1, 0, color=MUTED, alpha=0.08, lw=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.plot(B, np.exp(-B), color=MUTED, lw=1.2, label="the equation, exp(−B)")
        r = R_formula(scheme, B)
        ax.plot(B, np.where(np.abs(r) < 3, r, np.nan), color=COLOUR[scheme], lw=2, label=scheme)
        pts = [(Bv, Rv) for (s, Bv), Rv in measured.items() if s == scheme and abs(Rv) < 1.55]
        ax.scatter(*zip(*pts), s=16, facecolor=SURFACE, edgecolor=COLOUR[scheme], lw=1.1, zorder=3,
                   label="measured: one step of the package")
        lim = {"RK4": RK4_LIMIT, "RK2": 2.0, "EUL": 2.0}.get(scheme)
        if lim:
            limit_line(ax, lim, f"limit {lim:g}", axis="x")
        ax.set_title({"CRN": "CRN (the default)"}.get(scheme, scheme), loc="left")
        ax.set_xlim(0, 6)
        ax.set_ylim(-1.55, 1.55)
        ax.set_xlabel("B (per day)")
    axes[0].set_ylabel("Amplification factor R\n(one day's effect on a difference)")
    for y, text in ((1.27, "grows"), (0.5, "decays"), (-0.5, "decays, changes\nsign every day"),
                    (-1.27, "grows, changes\nsign every day")):
        axes[-1].annotate(text, (5.85, y), ha="right", va="center", fontsize=7, color=INK2)
    handles = [Line2D([0], [0], color=MUTED, lw=1.2, label="the equation, exp(−B)"),
               Line2D([0], [0], color=INK2, lw=2, label="the method's R (colour: method)"),
               Line2D([0], [0], marker="o", lw=0, markerfacecolor=SURFACE, markeredgecolor=INK2,
                      label="measured: one step of the package's integrator"),
               Patch(color=STATUS["critical"], alpha=0.15, label="|R| > 1: unstable")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.15))
    fig.suptitle("What one daily step does to a difference in water temperature", x=0.01, y=1.03, ha="left",
                 fontsize=10.5)
    save(fig, "amplification.png")


fig_amplification()

# --- §4 The B-series (Butcher series) of each method ------------------------------------------
# A tree is a tuple of its root's subtrees: () is a single vertex.
TREES = [(), ((),), ((), ()), (((),),), ((), (), ()), ((), ((),)), (((), ()),), ((((),),),)]
TREE_NAMES = ["τ", "[τ]", "[τ,τ]", "[[τ]]", "[τ,τ,τ]", "[τ,[τ]]", "[[τ,τ]]", "[[[τ]]]"]
F = Fraction
TABLEAUX = {   # (A, b) of each method as a Runge-Kutta method; c = A 1
    "EUL": ([[F(0)]], [F(1)]),
    "RK2": ([[F(0), F(0)], [F(1), F(0)]], [F(1, 2), F(1, 2)]),
    "CRN": ([[F(0), F(0)], [F(1, 2), F(1, 2)]], [F(1, 2), F(1, 2)]),
    "RK4": ([[F(0)] * 4, [F(1, 2), F(0), F(0), F(0)], [F(0), F(1, 2), F(0), F(0)], [F(0), F(0), F(1), F(0)]],
            [F(1, 6), F(1, 3), F(1, 3), F(1, 6)]),
}


def order(tree):
    return 1 + sum(order(t) for t in tree)


def gamma(tree):
    g = order(tree)
    for t in tree:
        g *= gamma(t)
    return g


def stage_weights(tree, A):
    """g(t)_i: the product over the root's subtrees u of (A g(u))_i; Phi(t) = b . g(t)."""
    g = [F(1)] * len(A)
    for u in tree:
        gu = stage_weights(u, A)
        Agu = [sum(A[i][j] * gu[j] for j in range(len(A))) for i in range(len(A))]
        g = [g[i] * Agu[i] for i in range(len(A))]
    return g


def phi(tree, method):
    A, b = TABLEAUX[method]
    return sum(bi * gi for bi, gi in zip(b, stage_weights(tree, A)))


def chain(k):
    t = ()
    for _ in range(k - 1):
        t = (t,)
    return t


def frac(x):
    return str(x) if x.denominator != 1 else str(x.numerator)


print("\n§4 B-series: elementary weights Phi(t) of the rooted trees up to order 4; the exact flow has 1/gamma(t)")
rows = []
for tree, name in zip(TREES, TREE_NAMES):
    row = {"tree": name, "order": order(tree), "gamma": gamma(tree), "exact 1/gamma": frac(F(1, gamma(tree))),
           "tall": tree == chain(order(tree))}
    for m in ("EUL", "RK2", "CRN", "RK4"):
        row[m] = frac(phi(tree, m))
    rows.append(row)
trees = pd.DataFrame(rows)
print(trees.to_string(index=False))
trees.to_csv(os.path.join(OUT, "butcher_trees.csv"), index=False)
for m in ("EUL", "RK2", "CRN", "RK4"):
    ok = [phi(t, m) == F(1, gamma(t)) for t in TREES]
    p = 0
    while p < 4 and all(ok[i] for i, t in enumerate(TREES) if order(t) <= p + 1):
        p += 1
    print(f"  {m}: satisfies every order condition up to order {p}")
print("Coefficients of z^k in the stability function R(z) (Phi of the tall tree with k vertices) against 1/k!:")
coef = pd.DataFrame({"k": range(1, 7), "exact 1/k!": [frac(F(1, int(np.prod(range(1, k + 1))))) for k in range(1, 7)]})
for m in ("EUL", "RK2", "CRN", "RK4"):
    coef[m] = [frac(phi(chain(k), m)) for k in range(1, 7)]
print(coef.to_string(index=False))
coef.to_csv(os.path.join(OUT, "stability_function_coefficients.csv"), index=False)


def draw_tree(ax, tree, x0, y0, width, colour, dy=0.45):
    """Butcher's convention: the root at the bottom."""
    ax.scatter([x0], [y0], s=34, color=colour, zorder=3, edgecolor=SURFACE, lw=1.2)
    n = len(tree)
    for k, sub in enumerate(tree):
        x = x0 + (k - (n - 1) / 2) * width / max(n, 1)
        ax.plot([x0, x], [y0, y0 + dy], color=colour, lw=1.3, zorder=2)
        draw_tree(ax, sub, x, y0 + dy, width / max(n, 1) * 0.9, colour, dy)


def fig_trees():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    for k, (tree, name) in enumerate(zip(TREES, TREE_NAMES)):
        tall = tree == chain(order(tree))
        colour = INK if tall else MUTED
        x = k * 1.25
        if tall:
            ax.add_patch(plt.Rectangle((x - 0.55, -0.62), 1.1, 2.75, color=BLUE, alpha=0.08, lw=0))
        draw_tree(ax, tree, x, 0, 0.75, colour)
        ax.text(x, -0.3, name, ha="center", va="top", fontsize=8, color=INK)
        ax.text(x, 2.05, f"order {order(tree)}\nγ = {gamma(tree)}", ha="center", va="top", fontsize=7, color=INK2)
    ax.text(-0.55, -0.85, "Shaded: the tall trees. For the test equation dTw/dt = −B·Tw only they survive, and "
            "their weights are the coefficients of the stability function R(z).",
            fontsize=7.5, color=INK2, va="top")
    ax.set_xlim(-0.7, 9.6)
    ax.set_ylim(-1.1, 2.15)
    ax.axis("off")
    ax.set_title("Rooted trees up to order 4: the terms of a B-series", loc="left", fontsize=10.5)
    save(fig, "butcher_trees.png")


def fig_regions():
    x, y = np.meshgrid(np.linspace(-6.2, 1.2, 741), np.linspace(-3.6, 3.6, 721))
    z = x + 1j * y
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), sharey=True)
    for ax, scheme in zip(axes, SCHEMES):
        r = np.abs({"CRN": (1 + z / 2) / (1 - z / 2), "EXP": np.exp(z), "RK4": 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24,
                    "RK2": 1 + z + z ** 2 / 2, "EUL": 1 + z}[scheme])
        ax.contourf(x, y, r <= 1, levels=[0.5, 1.5], colors=[COLOUR[scheme]], alpha=0.18)
        ax.contour(x, y, r, levels=[1], colors=[COLOUR[scheme]], linewidths=1.2)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=AXIS, lw=0.8)
        lim = {"RK4": RK4_LIMIT_EXACT, "RK2": 2.0, "EUL": 2.0}.get(scheme, 6.2)
        ax.plot([-lim, 0], [0, 0], color=COLOUR[scheme], lw=3.2, solid_capstyle="butt", zorder=3)
        if lim < 6:
            ax.plot([-6.2, -lim], [0, 0], color=STATUS["critical"], lw=3.2, solid_capstyle="butt", zorder=3)
            ax.annotate(f"−{lim:.4g}", (-lim, 0), xytext=(0, -11), textcoords="offset points", ha="center",
                        fontsize=7, color=INK2)
        ax.set_title({"CRN": "CRN (the default)"}.get(scheme, scheme), loc="left")
        ax.set_xlim(-6.2, 1.2)
        ax.set_ylim(-3.6, 3.6)
        ax.set_aspect("equal")
        ax.set_xlabel("Re z")
    axes[0].set_ylabel("Im z")
    handles = [Patch(color=INK2, alpha=0.25, label="|R(z)| ≤ 1: stable"),
               Line2D([0], [0], color=INK2, lw=3, label="z = −B·Δt, B > 0: where this model lives (stable part)"),
               Line2D([0], [0], color=STATUS["critical"], lw=3, label="its unstable part")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Stability regions in the complex plane; the model only uses the negative real axis", x=0.01,
                 ha="left", fontsize=10.5)
    save(fig, "stability_regions.png")


fig_trees()
fig_regions()

# --- §5 Every integrator with every published parameter set ------------------------------------
print("\n§5 Every integrator with the 30 published parameter sets, against the fine-step reference")
runs, sims, refs = [], {}, {}
for station, river in RIVERS.items():
    csv = calibration_csv(station)
    qm = mean_discharge(csv)
    for source in SOURCES:
        for v in VERSIONS:
            par = parameters(source, v, station)
            ref = reference(loaded[station, source, v])
            refs[station, source, v] = ref
            obs = loaded[station, source, v].Twat_obs[365:].astype(float)
            obs[obs == -999.0] = np.nan
            seen = np.isfinite(obs)
            rmse_ref = float(np.sqrt(np.mean((ref[365:][seen] - obs[seen]) ** 2)))
            for scheme in SCHEMES:
                d = load(csv, v, par, scheme, qm, f"{station}_{source}_v{v}_{scheme}")
                response = simulate(d)
                sim = d.Twat_mod[365:].copy()
                sims[station, source, v, scheme] = sim
                diverged = bool(np.any(~np.isfinite(sim) | (sim > PLAUSIBLE)))
                diff = np.inf if diverged else float(np.sqrt(np.mean((sim - ref[365:]) ** 2)))
                runs.append({
                    "river": river, "parameters": source, "version": v, "integrator": scheme,
                    "RMS difference from reference (degC)": diff,
                    "RMSE against measurements (degC)": np.inf if diverged else
                    float(np.sqrt(np.mean((sim[seen] - obs[seen]) ** 2))),
                    "RMSE of the reference (degC)": rmse_ref,
                    "days B above the limit (%)": 100 * stability_report(d)["frac_exceeding"],
                    "log10 largest growth": largest_growth(d)["log10_growth"],
                    "days at the 0 degC floor": int(np.sum(sim <= d.Tice_cover)) if not diverged else -1,
                    "days at the floor, reference": int(np.sum(ref[365:] <= d.Tice_cover)),
                    "diverged": diverged, "package": response})
runs = pd.DataFrame(runs)
runs.round(4).to_csv(os.path.join(OUT, "published_parameter_runs.csv"), index=False)


def outcome(diff):
    return "good" if diff <= GOOD else ("warning" if diff <= POOR else "critical")


for source in SOURCES:
    r = runs[runs.parameters == source]
    grid = r.pivot_table(index=["river", "version"], columns="integrator", sort=False,
                         values="RMS difference from reference (degC)").reindex(columns=list(SCHEMES))
    print(f"\nRMS difference from the reference (degC), parameters {SOURCES[source]}; inf = diverged")
    print(grid.round(3).to_string())
    for scheme in SCHEMES:
        s = r[r.integrator == scheme]
        print(f"  {scheme}: follows the equation (<= {GOOD} degC) in {int((s['RMS difference from reference (degC)'] <= GOOD).sum())}"
              f" of 15, diverged in {int(s.diverged.sum())}; package stopped {int((s.package == 'stopped').sum())}, "
              f"warned {int((s.package == 'warned').sum())}")
stable = runs[(runs["log10 largest growth"] == 0) | runs.integrator.isin(["CRN", "EXP"])]
print("\nRMS difference from the reference of the stable runs: CRN and EXP always, the others where no difference "
      "can grow (degC):")
for scheme in SCHEMES:
    x = stable.loc[stable.integrator == scheme, "RMS difference from reference (degC)"]
    print(f"  {scheme}: {x.min():.3f} to {x.max():.3f} over {len(x)} runs")
crn = runs[runs.integrator == "CRN"]
print(f"CRN: the largest growth of a difference over a stretch of days is {10 ** crn['log10 largest growth'].max():.2f} "
      f"times ({crn.loc[crn['log10 largest growth'].idxmax(), 'river']}, version "
      f"{crn.loc[crn['log10 largest growth'].idxmax(), 'version']}); it is never stopped.")
wrong_unstopped = runs[(runs["RMS difference from reference (degC)"] > POOR) & (runs.package != "stopped")]
print(f"\nRuns more than {POOR} degC from the reference that the package did not stop:"
      + (" none" if wrong_unstopped.empty else ""))
if not wrong_unstopped.empty:
    print(wrong_unstopped[["river", "parameters", "version", "integrator", "RMS difference from reference (degC)",
                           "days B above the limit (%)", "log10 largest growth", "days at the 0 degC floor",
                           "days at the floor, reference", "package"]].round(2).to_string(index=False))
inaccurate = runs[(runs["RMS difference from reference (degC)"] > GOOD) & (runs.package != "stopped")]
print(f"Runs more than {GOOD} degC from the reference that the package did not stop "
      f"({len(inaccurate)}; largest growth up to {10 ** inaccurate['log10 largest growth'].max():.1f} times):")
print(inaccurate[["river", "parameters", "version", "integrator", "RMS difference from reference (degC)",
                  "log10 largest growth", "package"]].round(2).to_string(index=False))


def fig_outcomes():
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0), sharey=True)
    rows = [(r, v) for r in RIVERS.values() for v in VERSIONS]
    for ax, source in zip(axes, SOURCES):
        r = runs[runs.parameters == source].set_index(["river", "version", "integrator"])
        for i, (river, v) in enumerate(rows):
            y = len(rows) - 1 - i
            for j, scheme in enumerate(SCHEMES):
                row = r.loc[(river, v, scheme)]
                diff = row["RMS difference from reference (degC)"]
                ax.add_patch(plt.Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, color=STATUS[outcome(diff)],
                                           alpha=0.22, lw=0))
                text = "div." if not np.isfinite(diff) else f"{diff:.2f}"
                mark = {"warned": " W", "stopped": " S"}.get(row.package, "")
                ax.text(j + 0.5, y + 0.5, text + mark, ha="center", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, len(SCHEMES))
        ax.set_ylim(0, len(rows))
        ax.set_xticks(np.arange(len(SCHEMES)) + 0.5, SCHEMES)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(rows))[::-1] + 0.5, [f"{river} v{v}" for river, v in rows])
        ax.grid(False)
        for side in ("left", "bottom", "top"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{source} parameters, {SOURCES[source].split(' (')[0]}", loc="left", pad=22)
    handles = [Patch(color=STATUS["good"], alpha=0.35, label=f"follows the equation (≤ {GOOD} °C RMS)"),
               Patch(color=STATUS["warning"], alpha=0.35, label=f"inaccurate ({GOOD}–{POOR:g} °C)"),
               Patch(color=STATUS["critical"], alpha=0.35, label=f"wrong (> {POOR:g} °C) or diverged (div.)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02))
    fig.text(0.5, -0.005, "Numbers: RMS difference (°C) from a fine-step solution of the same equation. "
             "W: the package warned; S: the package stopped the run (B check or divergence check).",
             ha="center", fontsize=7.5, color=INK2)
    fig.suptitle("Every integrator with every published parameter set, on each river's calibration record",
                 x=0.01, ha="left", fontsize=10.5)
    save(fig, "every_integrator_every_version.png")


fig_outcomes()

# --- §6 A calibration that learned RK4's behaviour -----------------------------------------------
print("\n§6 Dischmabach, version 5, parameters calibrated with RK4 (2015)")
st, v = "DAV_2327", 5
par = parameters("2015", v, st)
dates = pd.to_datetime(pd.read_csv(calibration_csv(st)).Date)
d = loaded[st, "2015", v]
obs = d.Twat_obs[365:].astype(float)
obs[obs == -999.0] = np.nan
rk4, exact = sims[st, "2015", v, "RK4"], refs[st, "2015", v][365:]
crn16 = sims[st, "2016", v, "CRN"]
a1, a2, a3, a4, a5, a6, a7, a8 = par
equilibrium = ((a1 + a2 * d.Tair + a6 * np.cos(2 * np.pi * (d.tt - a7))) / a3)[365:]
print(f"a3 = {a3} = B on every day; RK4's limit {RK4_LIMIT_EXACT:.4f}; R_RK4 = {R_formula('RK4', a3):.3f}, "
      f"exp(-B) = {np.exp(-a3):.3f}, R_CRN = {R_formula('CRN', a3):.3f}")
monthly = []
for month in range(1, 13):
    m = (dates.dt.month == month).to_numpy() & np.isfinite(obs)
    monthly.append({"month": month, "RMSE RK4": np.sqrt(np.mean((rk4[m] - obs[m]) ** 2)),
                    "RMSE equation (reference)": np.sqrt(np.mean((exact[m] - obs[m]) ** 2)),
                    "mean measured": obs[m].mean(), "mean RK4": rk4[m].mean(), "mean equation": exact[m].mean(),
                    "days equilibrium below 0 degC (%)": 100 * np.mean(equilibrium[(dates.dt.month == month).to_numpy()] < 0)})
monthly = pd.DataFrame(monthly)
print(monthly.round(2).to_string(index=False))
monthly.round(3).to_csv(os.path.join(OUT, "dischmabach_v5_by_month.csv"), index=False)
seen = np.isfinite(obs)
for name, s in (("RK4", rk4), ("the equation (reference)", exact), ("2016 parameters with CRN", crn16)):
    print(f"  {name}: RMSE {np.sqrt(np.mean((s[seen] - obs[seen]) ** 2)):.3f} degC, mean error "
          f"{np.mean(s[seen] - obs[seen]):+.3f} degC")


def fig_rk4_calibration():
    m = ((dates >= "2005-11-01") & (dates <= "2006-05-31")).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(dates[m], equilibrium[m], color=MUTED, lw=0.9, label="equilibrium A/B of the equation")
    ax.plot(dates[m], exact[m], color=INK, lw=1.6, label="the equation solved exactly (fine steps)")
    ax.plot(dates[m], rk4[m], color=COLOUR["RK4"], lw=1.6, label="RK4, one-day steps")
    ax.scatter(dates[m], obs[m], s=9, color=INK2, zorder=3, label="measured")
    ax.set_ylim(-4, 9)
    ax.set_ylabel("Water temperature (°C)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.annotate("Equilibrium below 0 °C: the exact solution sits on the 0 °C floor",
                (pd.Timestamp("2006-02-01"), -3.6), fontsize=7.5, color=INK2, ha="center", va="center")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Dischmabach, version 5, parameters calibrated with RK4 (a3 = B = 2.768, 0.6% under RK4's limit):\n"
                 "RK4 keeps the winter water above 0 °C; the same equation solved exactly does not", loc="left",
                 fontsize=10)
    save(fig, "rk4_calibration_near_its_limit.png")


fig_rk4_calibration()


def fig_failure_modes():
    cases = [("SIO_2011", "2016", 3, "EUL", "1990-06-01", "1990-07-15", 2.0,
              "EUL, Rhône, version 3 (B = 2.56 on every day): flips between 0 °C and 25 °C"),
             ("DAV_2327", "2016", 8, "RK4", "2006-03-15", "2006-11-15", RK4_LIMIT,
              "RK4, Dischmabach, version 8: at the 0 °C floor for five months, while B is above 2.785"),
             ("SIO_2011", "2015", 7, "RK2", "2000-09-15", "2000-11-15", 2.0,
              "RK2, Rhône, version 7 (2015 parameters): falls to 0 °C in a flood, with B above 2 on {n} days")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.6))
    for ax, (station, source, v, scheme, a, b, lim, title) in zip(axes, cases):
        dts = pd.to_datetime(pd.read_csv(calibration_csv(station)).Date)
        m = ((dts >= a) & (dts <= b)).to_numpy()
        B = bseries[station, source, v]
        above = m & (B > lim)
        for day in dts[above]:
            ax.axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12), color=STATUS["critical"],
                       alpha=0.08, lw=0)
        o = loaded[station, source, v].Twat_obs[365:].astype(float)
        o[o == -999.0] = np.nan
        ax.plot(dts[m], refs[station, source, v][365:][m], color=INK, lw=1.5, label="the equation (fine steps)")
        ax.plot(dts[m], np.clip(sims[station, source, v, scheme][m], -5, 60), color=COLOUR[scheme], lw=1.5,
                label=scheme)
        ax.scatter(dts[m], o[m], s=8, color=INK2, zorder=3, label="measured")
        row = runs[(runs.river == RIVERS[station]) & (runs.parameters == source) & (runs.version == v)
                   & (runs.integrator == scheme)].iloc[0]
        response = {"stopped": "the package stops this run", "warned": "the package only warns",
                    "none": "no message"}[row.package]
        ax.set_title(f"{title.format(n=int(above.sum()))} — {response}", loc="left", fontsize=9)
        ax.set_ylabel("°C")
        if (dts[m].iloc[-1] - dts[m].iloc[0]).days < 100:
            ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 15)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(loc="upper right", ncol=3, fontsize=7.5)
    axes[0].set_ylim(-1, 33)
    axes[1].set_ylim(-1, 14)
    axes[2].set_ylim(-1, 13)
    fig.text(0.01, -0.01, "Shaded: days on which B is above the method's limit.", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, "failure_modes.png")


fig_failure_modes()

# --- §7 Stable is not the same as accurate: a steadily warming equilibrium ----------------------
print("\n§7 Equilibrium warming steadily by 0.02 degC/day: lag behind the equation's own solution (hours)")
RAMP = 0.02
ramp_csv = os.path.join(OUT, "steady_warming.csv")
n = 730
pd.DataFrame({"Date": pd.date_range("2001-01-01", periods=n).strftime("%Y-%m-%d"), "T_air": RAMP * np.arange(n),
              "T_water": [0.0] + [np.nan] * (n - 1), "Discharge": 1.0}).to_csv(ramp_csv, index=False)
lag = []
for B in (0.25, 0.5, 1.0, 1.5, 2.5, 5.0, 20.0):
    row = {"B (per day)": B, "the equation lags its equilibrium by (h)": 24 / B}
    for scheme in SCHEMES:
        # Version 3 with a1 = 0 and a2 = a3 = B: the equilibrium is the air temperature itself.
        d = load(ramp_csv, 3, [0, B, B, 0, 0, 0, 0, 0], scheme, 1.0, f"ramp_{scheme}", Tice_cover=-1e300)
        with contextlib.redirect_stdout(io.StringIO()):
            call_model(d)
        exact_end = d.Tair[-1] - RAMP / B                     # the equation's solution, long after the start
        error = d.Twat_mod[-1] - exact_end
        row[scheme] = -24 * error / RAMP if np.isfinite(error) and abs(error) < 1e3 else np.nan
    lag.append(row)
lag = pd.DataFrame(lag)
print("Extra lag of each method behind the equation, in hours (negative: ahead; nan: unstable)")
print(lag.round(1).assign(**{"B (per day)": lag["B (per day)"]}).to_string(index=False))
lag.round(2).to_csv(os.path.join(OUT, "steady_warming_lag.csv"), index=False)

# --- §8 Scenario flows --------------------------------------------------------------------------
print("\n§8 Scenario flows: discharge multiplied by a factor, Qmedia kept at the calibration value")
FACTORS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
SCENARIO_CASES = (("MAH_2369", 8), ("SIO_2011", 8))
scen = []
for station, v in SCENARIO_CASES:
    par = parameters("2015", v, station)
    record = pd.read_csv(calibration_csv(station))
    qm = mean_discharge(calibration_csv(station))
    for factor in FACTORS:
        csv = os.path.join(OUT, "scenarios", f"{station}_flow_x{factor}.csv")
        record.assign(Discharge=record.Discharge * factor).to_csv(csv, index=False)
        ref = None
        for scheme in SCHEMES:
            d = load(csv, v, par, scheme, qm, f"scenario_{station}_x{factor}_{scheme}")
            response = simulate(d)
            if ref is None:
                ref = reference(d)[365:]
            sim = d.Twat_mod[365:]
            diverged = bool(np.any(~np.isfinite(sim) | (sim > PLAUSIBLE)))
            B = compute_B_series(d)[365:]
            scen.append({"river": RIVERS[station], "version": v, "flow factor": factor, "integrator": scheme,
                         "B max": B.max(), "RMS difference from reference (degC)":
                             np.inf if diverged else float(np.sqrt(np.mean((sim - ref) ** 2))),
                         "days B above the limit (%)": 100 * stability_report(d)["frac_exceeding"],
                         "log10 largest growth": largest_growth(d)["log10_growth"],
                         "package": response})
scen = pd.DataFrame(scen)
scen.round(4).to_csv(os.path.join(OUT, "scenario_flows.csv"), index=False)
for (river, v), g in scen.groupby(["river", "version"], sort=False):
    t = g.pivot_table(index="flow factor", columns="integrator", values="RMS difference from reference (degC)",
                      sort=False)[list(SCHEMES)]
    p = g.pivot_table(index="flow factor", columns="integrator", values="package", aggfunc="first",
                      sort=False)[list(SCHEMES)]
    bmax = g.groupby("flow factor").first()["B max"]
    shown = t.round(2).astype(str).replace("inf", "div.") + p.replace({"none": "", "warned": " W", "stopped": " S"})
    shown.insert(0, "B max", bmax.round(2))
    print(f"\n{river}, version {v}, parameters calibrated with RK4 (2015); RMS difference from the reference (degC); "
          f"W warned, S stopped")
    print(shown.to_string())
c = scen[scen.integrator == "CRN"]
print(f"\nCRN in every scenario: a difference grows at most {10 ** c['log10 largest growth'].max():.2f} times over a "
      f"stretch of days, and the run is at most {c['RMS difference from reference (degC)'].max():.2f} degC from the "
      f"fine-step solution.")


def fig_scenarios():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    top = 30
    for ax, (station, v) in zip(axes, SCENARIO_CASES):
        g = scen[(scen.river == RIVERS[station]) & (scen.version == v)]
        ax.axhspan(top * 0.62, top * 1.6, color=STATUS["critical"], alpha=0.07, lw=0)
        ax.annotate("diverged", (0.55, top), fontsize=7.5, color=INK2, va="center", ha="center")
        for scheme in SCHEMES:
            s = g[g.integrator == scheme].sort_values("flow factor")
            y = s["RMS difference from reference (degC)"].to_numpy()
            yy = np.where(np.isfinite(y), y, top)
            ax.plot(s["flow factor"], yy, color=COLOUR[scheme], lw=1.6, marker=MARKER[scheme], ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=scheme)
        ax.axvline(1.0, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.annotate("recorded\nflows", (1.0, 0.006), xytext=(3, 0), textcoords="offset points", fontsize=7,
                    color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(FACTORS, [f"{f:g}" for f in FACTORS])
        ax.minorticks_off()
        ax.set_ylim(0.005, top * 1.6)
        ax.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
        ax.set_xlabel("Discharge × factor (Qmedia fixed)")
        par = parameters("2015", v, station)
        ax.set_title(f"{RIVERS[station]}, version {v} (a4 = {par[3]:g}, a8 = {par[7]:g})", loc="left")
    axes[0].set_ylabel("RMS difference from the\nfine-step solution (°C)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), ncol=5, fontsize=7.5, handlelength=1.5,
                   columnspacing=1.0)
    fig.suptitle("Scenario flows with parameters calibrated with RK4: stable on the recorded flows is not stable "
                 "on other flows", x=0.01, ha="left", fontsize=10.5)
    save(fig, "scenario_flows.png")


fig_scenarios()

# --- §9 The B check, and the growth over the worst stretch ---------------------------------------
print("\n§9 The package's B check (share of days above the limit) and the largest growth over a stretch")
ex = runs[runs.integrator.isin(EXPLICIT)].copy()
ex["outcome"] = ex["RMS difference from reference (degC)"].map(outcome)
for label, rows in (("no growth (factor <= 1)", ex["log10 largest growth"] == 0),
                    ("growth up to 10x", (ex["log10 largest growth"] > 0) & (ex["log10 largest growth"] <= 1)),
                    ("growth 10x to 1000x", (ex["log10 largest growth"] > 1) & (ex["log10 largest growth"] <= 3)),
                    ("growth above 1000x", ex["log10 largest growth"] > 3)):
    g = ex[rows]
    print(f"  {label:26s} {len(g):3d} runs: follow the equation {int((g.outcome == 'good').sum()):2d}, inaccurate "
          f"{int((g.outcome == 'warning').sum()):2d}, wrong or diverged {int((g.outcome == 'critical').sum()):2d}"
          f" (EUL among the inaccurate: {int(((g.outcome == 'warning') & (g.integrator == 'EUL')).sum())})")
low_share = ex[(ex["days B above the limit (%)"] > 0) & (ex["days B above the limit (%)"] <= 10)]
print(f"  Runs with B above the limit on 0-10% of days (the B check warns, but does not stop): {len(low_share)}; "
      f"their largest growth ranges from 10^{low_share['log10 largest growth'].min():.1f} to "
      f"10^{low_share['log10 largest growth'].max():.1f}")
print(low_share[["river", "parameters", "version", "integrator", "days B above the limit (%)",
                 "log10 largest growth", "RMS difference from reference (degC)", "package"]].round(2).to_string(index=False))


def fig_growth():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    top = 30
    y = ex["RMS difference from reference (degC)"].to_numpy()
    y = np.where(np.isfinite(y), y, top)
    rng = np.random.default_rng(9)
    jitter = rng.uniform(-0.12, 0.12, len(y))
    for ax, xname in zip(axes, ("days B above the limit (%)", "log10 largest growth")):
        ax.axhspan(top * 0.62, top * 1.6, color=STATUS["critical"], alpha=0.07, lw=0)
        ax.axhspan(0.001, GOOD, color=STATUS["good"], alpha=0.06, lw=0)
        for scheme in EXPLICIT:
            k = (ex.integrator == scheme).to_numpy()
            x = ex[xname].to_numpy()[k]
            x = np.where(x > 0, x, jitter[k])          # spread the runs at zero sideways, so all are visible
            ax.scatter(x, y[k], s=30, marker=MARKER[scheme], color=COLOUR[scheme], edgecolor=SURFACE,
                       lw=0.8, zorder=3, label=scheme)
        ax.set_yscale("log")
        ax.set_ylim(0.005, top * 1.6)
        ax.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    axes[0].annotate("diverged", (0.05, top), fontsize=7.5, color=INK2, va="center")
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlim(-0.3, 130)
    axes[0].set_xticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    limit_line(axes[0], 10, "stop above 10%", axis="x")
    axes[0].set_xlabel("Days with B above the method's limit (%)\n(the package's check)")
    limit_line(axes[1], np.log10(STABILITY_MAX_GROWTH), "stop above ×100", axis="x")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlim(-0.3, 5000)
    axes[1].set_xticks([0, 1, 10, 100, 1000], ["none", "×10", "×10¹⁰", "×10¹⁰⁰", "×10¹⁰⁰⁰"])
    axes[1].set_xlabel("Largest growth of a difference over any stretch of days\n(from the B series alone)")
    axes[0].set_ylabel("RMS difference from the\nfine-step solution (°C)")
    axes[1].legend(loc="center right", fontsize=7.5)
    fig.suptitle("RK4, RK2 and EUL with the 30 published parameter sets: what predicts a wrong run?", x=0.01,
                 ha="left", fontsize=10.5)
    save(fig, "what_predicts_a_wrong_run.png")


fig_growth()
print(f"\nFigures written to {os.path.relpath(FIG, REPO)}/, tables to {os.path.relpath(OUT, REPO)}/")
