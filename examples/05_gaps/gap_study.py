"""
Example 05, part 2: when does gap-tolerant mode work, and when does it not?

    python examples/05_gaps/gap_study.py

About ten minutes on a 4-core computer. Calibrates the Mentue's 2002-2009 record about 130 times,
each time with air temperature removed in a different way, and compares the predictions of
2010-2012 with those of the calibration on the complete record:

1. How fast the model forgets its restart after a gap. This sets how long the warm-up must be.
2. Long gaps (a month, a quarter, half a year and a year, each at four places in the record):
   gap-tolerant mode against filling the gap with a straight line or with the seasonal average.
3. Scattered gaps (1-20% of days missing at random): gap-tolerant mode with warm-ups of 15, 7,
   4 and 0 days, against filling with a straight line.

Water temperature is kept on every day: only the air temperature is missing.
Writes the tables to output/study/ and the README's figures to figures/.
"""
import contextlib
import io
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pyair2stream.io import compute_doy_climatology, read_calibration, read_Tseries
from pyair2stream.model import aggregation, call_model, compute_B_series, statis
from pyair2stream.optimization import DE_mode

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "output", "study")
FIG = os.path.join(HERE, "figures")
CAL = os.path.join(REPO, "data", "switzerland", "MAH_2369_calibration.csv")
VAL = os.path.join(REPO, "data", "switzerland", "MAH_2369_validation.csv")

SETTINGS = {"project_name": "gap_study", "station_name": "Mentue", "series": "c", "time_resolution": "1d",
            "version": 8, "integrator": "CRN", "objective_function": "NSE", "run_mode": "DE",
            "parameter_bounds": {"min": [-5, -5, -5, -1, 0, 0, 0, -1], "max": [15, 1.5, 5, 1, 20, 10, 1, 5]}}
SEED = 42

LENGTHS = (("a month", 30), ("a quarter", 91), ("half a year", 182), ("a year", 365))
STARTS = ("2003-07-01", "2005-01-01", "2006-04-01", "2007-10-01")    # summer, winter, spring, autumn
LONG_METHODS = ("gap-tolerant", "straight line", "seasonal average")

SHARES = (0.01, 0.02, 0.05, 0.10, 0.20)
DRAWS = 3
WARMUPS = ((15, 30), (7, 14), (4, 8), (0, 1))       # (warmup_drop_days, min_segment_days)

COLOURS = {"gap-tolerant": "tab:blue", "straight line": "tab:orange", "seasonal average": "tab:green"}
WARMUP_COLOURS = ("#08306b", "#2171b5", "#6baed6", "#b0cfe8")       # 15, 7, 4, 0 days: dark to light


def warmup_name(warmup, min_segment):
    return f"gap-tolerant, warm-up {warmup} d"


# --- Calibration (runs in worker processes) -----------------------------------------------------
def calibrate(job):
    """DE calibration of one input file; returns the best parameters and the days scored."""
    name, csv, extra, seed = job
    folder = os.path.join(OUT, "runs", name)
    os.makedirs(folder, exist_ok=True)
    cfg = {**SETTINGS, **extra, "random_seed": seed, "paths": {"input_data": csv, "output_dir": folder}}
    path = os.path.join(folder, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            data = read_calibration(path)
            read_Tseries(data, "c")
            aggregation(data)
            statis(data)
            DE_mode(data, seed=seed)
    except ValueError as err:            # e.g. no stretch long enough in gap-tolerant mode
        return {"name": name, "par": None, "scored": 0, "error": str(err).splitlines()[0]}
    scored = int(np.sum(data.eval_mask & (data.Twat_obs != -999.0)))
    return {"name": name, "par": data.par_best.copy(), "scored": scored, "qmedia": float(data.Qmedia),
            "error": ""}


def load(csv, name, **extra):
    """Load a file for simulation only (FORWARD), as a run would."""
    folder = os.path.join(OUT, "runs", name)
    os.makedirs(folder, exist_ok=True)
    cfg = {**SETTINGS, "run_mode": "FORWARD", "parameters_forward": [0.0] * 8,
           "paths": {"input_data": csv, "output_dir": folder}, **extra}
    path = os.path.join(folder, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    with contextlib.redirect_stdout(io.StringIO()):
        data = read_calibration(path)
        read_Tseries(data, "c")
    return data


# --- Gaps and fills --------------------------------------------------------------------------------
def straight_line(cal, gone):
    return cal.T_air.mask(gone).interpolate(limit_direction="both")


def seasonal_average(cal, gone):
    """Each missing day gets the mean air temperature of that day of the year in the other years,
    smoothed over 15 days."""
    doy = cal.Date.dt.dayofyear.clip(upper=365)
    mean = cal.T_air[~gone].groupby(doy[~gone]).mean().reindex(range(1, 366)).interpolate()
    wrapped = pd.concat([mean.iloc[-7:], mean, mean.iloc[:7]])
    smooth = wrapped.rolling(15, center=True).mean().iloc[7:-7]
    smooth.index = range(1, 366)
    return cal.T_air.where(~gone, doy.map(smooth))


def write_case(cal, name, gone, method):
    """Write the calibration file for one gap and method; returns (csv, extra settings)."""
    if method == "straight line":
        df, extra = cal.assign(T_air=straight_line(cal, gone)), {}
    elif method == "seasonal average":
        df, extra = cal.assign(T_air=seasonal_average(cal, gone)), {}
    else:
        warmup, min_segment = method
        df = cal.assign(T_air=cal.T_air.mask(gone))
        extra = {"gap_tolerant": True, "warmup_drop_days": warmup, "min_segment_days": min_segment}
    csv = os.path.join(OUT, "inputs", f"{name}.csv")
    df.to_csv(csv, index=False, date_format="%Y-%m-%d")
    return csv, extra


# --- 1. How fast is the restart forgotten? --------------------------------------------------------
def restart_memory(par, qmedia, days=20):
    """Restart the model on every measured day of 2002-2009 and follow the difference from the
    continuous simulation. Returns {start: array (restarts x days+1)}."""
    data = load(CAL, "memory", Qmedia=qmedia)
    data.par[:] = par
    call_model(data)
    continuous = data.Twat_mod.copy()
    B = compute_B_series(data)[365:]
    compute_doy_climatology(data)
    data.gap_tolerant = True
    measured = data.Twat_obs.copy()
    out = {"measured water temperature": [], "day-of-year average": []}
    for offset in range(days + 1):
        starts = np.arange(365 + offset, data.n_tot - days - 1, days + 1)
        starts = starts[measured[starts] != -999.0]
        data.segments = [(int(s), int(s) + days) for s in starts]
        for kind in out:
            data.Twat_obs[:] = measured
            if kind == "day-of-year average":
                data.Twat_obs[starts] = -999.0          # no measurement: start from the average
            call_model(data)
            out[kind] += [np.abs(data.Twat_mod[s:s + days + 1] - continuous[s:s + days + 1]) for s in starts]
    data.Twat_obs[:] = measured
    return {k: np.array(v) for k, v in out.items()}, B


def main():
    os.makedirs(os.path.join(OUT, "inputs"), exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    cal = pd.read_csv(CAL, parse_dates=["Date"])
    n = len(cal)

    # --- The jobs ------------------------------------------------------------------------------------
    jobs, cases = [], []
    for seed in (SEED, 1, 2):          # the complete record; other seeds show the optimizer's own scatter
        jobs.append((f"complete_seed{seed}", CAL, {}, seed))
    for label, days in LENGTHS:
        for start in STARTS:
            gone = ((cal.Date >= start) & (cal.Date < pd.Timestamp(start) + pd.Timedelta(days=days))).to_numpy()
            for method in LONG_METHODS:
                name = f"long_{days}d_{start}_{method.replace(' ', '_')}"
                csv, extra = write_case(cal, name, gone, (15, 30) if method == "gap-tolerant" else method)
                jobs.append((name, csv, extra, SEED))
                cases.append({"study": "long", "gap": label, "days": days, "start": start, "method": method,
                              "name": name, "missing": int(gone.sum())})
    for share in SHARES:
        for draw in range(DRAWS):
            rng = np.random.default_rng(1000 * draw + round(share * 100))
            gone = rng.random(n) < share
            gone[[0, -1]] = False
            for method in ("straight line",) + WARMUPS:
                label = method if isinstance(method, str) else warmup_name(*method)
                name = f"scattered_{round(share * 100)}pc_draw{draw}_{label.replace(' ', '_').replace(',', '')}"
                csv, extra = write_case(cal, name, gone, method)
                jobs.append((name, csv, extra, SEED))
                cases.append({"study": "scattered", "share": share, "draw": draw, "method": label, "name": name,
                              "missing": int(gone.sum())})

    print(f"Running {len(jobs)} calibrations ...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = {r["name"]: r for r in pool.map(calibrate, jobs)}
    qmedia = results[f"complete_seed{SEED}"]["qmedia"]
    assert all(r["qmedia"] == qmedia for r in results.values() if r["par"] is not None)

    # --- Predictions of 2010-2012 ------------------------------------------------------------------
    val = load(VAL, "validation", Qmedia=qmedia)
    observed = np.where(val.Twat_obs[365:] == -999.0, np.nan, val.Twat_obs[365:])

    def predict(par):
        val.par[:] = par
        call_model(val)
        return val.Twat_mod[365:].copy()

    def rmse(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))

    reference = predict(results[f"complete_seed{SEED}"]["par"])
    complete_scored = results[f"complete_seed{SEED}"]["scored"]
    complete_rmse = rmse(reference, observed)
    seed_change = [rmse(predict(results[f"complete_seed{s}"]["par"]), reference) for s in (1, 2)]
    print(f"\nComplete record: {complete_scored} days scored; 2010-2012 RMSE {complete_rmse:.3f} °C. Other DE seeds "
          f"change the 2010-2012 predictions by {max(seed_change):.3f} °C at most (the optimizer's own scatter).")

    rows = []
    for case in cases:
        r = results[case["name"]]
        row = {**case, "days scored": r["scored"], "share of measured days scored (%)":
               round(100 * r["scored"] / complete_scored, 1), "error": r["error"]}
        if r["par"] is not None:
            pred = predict(r["par"])
            row["2010-2012 RMSE (°C)"] = round(rmse(pred, observed), 3)
            row["change in 2010-2012 predictions (°C)"] = round(rmse(pred, reference), 3)
            row.update({f"a{i + 1}": round(float(v), 4) for i, v in enumerate(r["par"])})
        rows.append(row)
    table = pd.DataFrame(rows).drop(columns="name")
    long = table[table.study == "long"].dropna(axis=1, how="all").drop(columns="study")
    scattered = table[table.study == "scattered"].dropna(axis=1, how="all").drop(columns="study")
    long.to_csv(os.path.join(OUT, "long_gaps.csv"), index=False)
    scattered.to_csv(os.path.join(OUT, "scattered_gaps.csv"), index=False)

    measures = ["2010-2012 RMSE (°C)", "change in 2010-2012 predictions (°C)"]
    long_summary = long.groupby(["days", "gap", "method"], sort=False)[measures].agg(["mean", "max"]).round(3)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nLong gaps (mean and largest over the four placements):\n" + long_summary.to_string())
    scattered_summary = (scattered.groupby(["share", "method"], sort=False)
                         [["share of measured days scored (%)"] + measures].mean().round(3))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nScattered gaps (mean of the draws):\n" + scattered_summary.to_string())
    failed = scattered[scattered.error != ""]
    if len(failed):
        print("\nRuns that stopped:\n" + failed[["share", "draw", "method", "error"]].to_string(index=False))

    # --- 1. Restart memory ---------------------------------------------------------------------------
    memory, B = restart_memory(results[f"complete_seed{SEED}"]["par"], qmedia)
    needed = int(np.ceil(3.0 / np.median(B)))
    print(f"\nDecay rate B: median {np.median(B):.2f} per day; three relaxation times = {3 / np.median(B):.1f} days "
          f"(the run suggests warmup_drop_days of at least {needed}).")
    for kind, a in memory.items():
        print(f"Restart from the {kind} ({len(a)} restarts), mean difference from the continuous run by day:")
        print("  " + "  ".join(f"day {k}: {a[:, k].mean():.2f}" for k in (0, 1, 2, 3, 4, 5, 7, 10, 15)))
    pd.DataFrame({f"{stat} difference, restart from the {kind} (°C)": f(a, axis=0)
                  for kind, a in memory.items()
                  for stat, f in (("mean", np.mean), ("95th percentile",
                                                       lambda x, axis: np.percentile(x, 95, axis=axis)))}
                 ).rename_axis("days after restart").round(4).to_csv(os.path.join(OUT, "restart_memory.csv"))

    # --- Figures -------------------------------------------------------------------------------------
    figure_memory(memory, needed)
    figure_long(long, complete_rmse)
    figure_scattered(scattered, complete_rmse)


def figure_memory(memory, needed):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    days = np.arange(next(iter(memory.values())).shape[1])
    for (kind, a), colour in zip(memory.items(), ("tab:blue", "tab:purple")):
        ax.plot(days, a.mean(axis=0), "o-", color=colour, ms=4, label=f"restart from the {kind}: mean")
        ax.plot(days, np.percentile(a, 95, axis=0), "--", color=colour, lw=1, label="... 95% of restarts below")
    ax.axvline(needed, color="black", lw=0.8)
    ax.text(needed + 0.2, 2.6, f"{needed} days: the warm-up\nthe run suggests", fontsize=8, va="top")
    ax.axvline(15, color="tab:gray", lw=0.8, ls=":")
    ax.text(15.2, 2.6, "15 days:\nthe default", fontsize=8, va="top", color="dimgray")
    ax.set_xlim(-0.3, days[-1] + 0.3)
    ax.set_ylim(0, 3.5)
    ax.set_xticks(range(0, days[-1] + 1, 2))
    ax.set_xlabel("Days after the restart")
    ax.set_ylabel("Difference from a run\nwithout the gap (°C)")
    ax.set_title("After a gap, the model forgets its restart within a few days", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.98), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "restart_memory.png"), dpi=130)
    plt.close(fig)


def figure_long(long, complete_rmse):
    measure = "2010-2012 RMSE (°C)"
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [label for label, _ in LENGTHS]
    for k, method in enumerate(LONG_METHODS):
        x0 = np.arange(len(labels)) + (k - 1) * 0.25
        sub = long[long.method == method]
        means = [sub[sub.gap == label][measure].mean() for label in labels]
        for x, label in zip(x0, labels):
            vals = sub[sub.gap == label][measure].to_numpy()
            ax.plot(np.full(len(vals), x), vals, "o", color=COLOURS[method], ms=4, alpha=0.6, mfc="none")
        ax.plot(x0, means, "_", color=COLOURS[method], ms=16, mew=2.5, label=method)
    ax.axhline(complete_rmse, color="black", lw=0.8, ls="--")
    ax.text(len(labels) - 0.55, complete_rmse, "  complete record", fontsize=8, va="bottom", ha="right")
    ax.set_xticks(np.arange(len(labels)), [f"{label} missing" for label in labels])
    ax.set_ylabel("Error in 2010-2012 (RMSE, °C)")
    ax.set_title("A long gap in air temperature: gap-tolerant mode against filling it", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", title="the gap was", title_fontsize=8)
    ax.set_ylim(bottom=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "long_gaps.png"), dpi=130)
    plt.close(fig)


def figure_scattered(scattered, complete_rmse):
    methods = [warmup_name(*w) for w in WARMUPS] + ["straight line"]
    colours = dict(zip(methods, WARMUP_COLOURS + ("tab:orange",)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method in methods:
        sub = scattered[scattered.method == method].groupby("share")
        x = 100 * np.array(sorted(sub.groups))
        style = dict(color=colours[method], ms=4, label=method)
        axes[0].plot(x, sub["share of measured days scored (%)"].mean().to_numpy(), "o-", **style)
        axes[1].plot(x, sub["2010-2012 RMSE (°C)"].mean().to_numpy(), "o-", **style)
    axes[0].set_ylabel("Measured days scored (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("How much of the record is used", fontsize=10)
    axes[1].axhline(complete_rmse, color="black", lw=0.8, ls="--")
    axes[1].text(20, complete_rmse, "complete record ", fontsize=8, va="bottom", ha="right")
    axes[1].set_ylabel("Error in 2010-2012 (RMSE, °C)")
    axes[1].set_title("How well the calibrated model predicts other years", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Days missing at random (%)")
        ax.set_xticks(100 * np.array(SHARES))
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scattered_warmup.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
