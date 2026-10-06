"""
V14 - Do the prediction intervals hold on the hottest days?

Temperature limits are breached on the hottest days, so an interval that holds on average but not
there would mislead a compliance assessment. This check measures, in years the model was not
calibrated on, how often the measured temperature falls inside the predicted interval on the
days predicted to be hottest, the days with the hottest air, and in summer.

Which days count as hottest must be decided from what is known before the measurement (the
predicted temperature, or the air temperature). Choosing days by their measured temperature picks
days whose error happened to be positive, so even for a correct interval the measurement then lies
above it more often than stated, and below it less often; the check shows this with a perfect-model
reference, in which one simulated series plays the part of the measurements.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AUTHORS_BOUNDS, DE_SETTINGS, RIVERS, Result, Section, Timer, accepted_real, load, quiet,
                    plot_style, save_figure, BLUE, ORANGE, AQUA, YELLOW, INK2, LIGHT_GREY)

VERSIONS = (5, 8)
LEVELS = (50, 80, 90, 95, 99)
JUDGED_LEVELS = (80, 90, 95)
HOT_SHARE = 0.10            # the hottest 10% of a river's held-out days
N_SIMULATIONS = 1000
SUBSETS = ("all days", "summer (June-August)", "hottest 10% by prediction", "hottest 10% by air temperature",
           "hottest 10% by measurement")
JUDGED = ("hottest 10% by prediction", "hottest 10% by air temperature")


def _job(args):
    """Leave-one-year-out cross-validation of one river and version (settings of V11), then, for each
    held-out day and 7-day mean, whether the measurement and a perfect-model stand-in fall inside the
    central range of the fold's simulations at each level."""
    st, version, quick = args
    from v11_yearly_check import _full_record
    from pyair2stream.cross_validation import _fold_ensemble, cross_validate
    csv = _full_record(st)
    full = pd.read_csv(csv)
    q = full.Discharge
    tair = pd.Series(full.T_air.to_numpy(), index=pd.to_datetime(full.Date))
    settings = {"n_run": 40, "n_particles": 8} if quick else dict(DE_SETTINGS)
    cfg = {"version": version, "integrator": "CRN", "run_mode": "DE", "objective_function": "NSE",
           "random_seed": 1, "optimization": settings, "parameter_bounds": AUTHORS_BOUNDS,
           "Qmedia": float(q[q > 0].mean()), "paths": {"input_data": csv},
           "cross_validation": {"enabled": True, "unit": "year", "min_train_years": 0, "skip_first_year": True}}
    data = load(cfg, f"v14_{st}_{version}")
    if quick:
        years = sorted(set(data.date[365:, 0]))
        data.cross_validation.min_train_years = max(0, len(years) - 4)
    with quiet():
        _, folds = cross_validate(data, "DE", return_folds=True)
    rows = []
    for r in folds:
        rng = np.random.default_rng([list(RIVERS).index(st), version, r.fold_id])
        obs, ens = _fold_ensemble(r, "ar1", N_SIMULATIONS, rng)
        pred = np.where(np.isfinite(obs), r.sim_held_out, np.nan)
        truth, others = ens[0], ens[1:]          # perfect-model stand-in: one simulation as the "measurement"
        for scale, o, t, e, p in (("daily", obs, truth, others, pred),
                                  ("7-day", *(_week(x) for x in (obs, truth)), _week_ens(others), _week(pred))):
            ok = np.isfinite(o)
            if not ok.any():
                continue
            bounds = {lev: np.percentile(e[:, ok], [50 - lev / 2, 50 + lev / 2], axis=0) for lev in LEVELS}
            dates = r.dates_held_out[ok]
            frame = pd.DataFrame({"river": RIVERS[st], "version": version, "scale": scale, "date": dates,
                                  "measured": o[ok], "predicted": p[ok], "perfect": t[ok],
                                  "air": (tair.reindex(dates).to_numpy() if scale == "daily"
                                          else tair.rolling(7).mean().reindex(dates).to_numpy())})
            for lev, (lo, hi) in bounds.items():
                frame[f"in {lev}"] = (o[ok] >= lo) & (o[ok] <= hi)
                frame[f"perfect in {lev}"] = (t[ok] >= lo) & (t[ok] <= hi)
                if lev == 90:
                    frame["above 90"], frame["below 90"] = o[ok] > hi, o[ok] < lo
                    frame["perfect above 90"], frame["perfect below 90"] = t[ok] > hi, t[ok] < lo
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _week(x):
    """7-day moving mean, defined only where all 7 days are present (within one held-out year)."""
    return pd.Series(x).rolling(7).mean().to_numpy()


def _week_ens(e):
    return pd.DataFrame(e.T).rolling(7).mean().to_numpy().T


def _subsets(g):
    """Masks of each subset of one river's held-out days (or 7-day means), with thresholds from that river."""
    top = 1 - HOT_SHARE
    return {"all days": np.ones(len(g), bool),
            "summer (June-August)": g.date.dt.month.isin([6, 7, 8]).to_numpy(),
            "hottest 10% by prediction": (g.predicted >= g.predicted.quantile(top)).to_numpy(),
            "hottest 10% by air temperature": (g.air >= g.air.quantile(top)).to_numpy(),
            "hottest 10% by measurement": (g.measured >= g.measured.quantile(top)).to_numpy()}


def _perfect_subsets(g):
    """The same subsets for the perfect-model stand-in: 'by measurement' selects on its own values."""
    s = _subsets(g)
    s["hottest 10% by measurement"] = (g.perfect >= g.perfect.quantile(1 - HOT_SHARE)).to_numpy()
    return s


def _coverage(df):
    """Share inside at each level, by version, scale and subset, pooled over rivers; and the perfect-model
    reference."""
    rows = []
    for (version, scale), g in df.groupby(["version", "scale"], sort=True):
        masks = {name: [] for name in SUBSETS}
        pmasks = {name: [] for name in SUBSETS}
        parts = []
        for _, h in g.groupby("river", sort=True):
            h = h.reset_index(drop=True)
            for name, m in _subsets(h).items():
                masks[name].append(m)
            for name, m in _perfect_subsets(h).items():
                pmasks[name].append(m)
            parts.append(h)
        h = pd.concat(parts, ignore_index=True)
        for name in SUBSETS:
            m, pm = np.concatenate(masks[name]), np.concatenate(pmasks[name])
            row = {"version": version, "scale": scale, "subset": name, "values": int(m.sum())}
            for lev in LEVELS:
                row[f"{lev}%"] = float(h.loc[m, f"in {lev}"].mean())
                row[f"perfect model {lev}%"] = float(h.loc[pm, f"perfect in {lev}"].mean())
            for side in ("above", "below"):
                row[f"{side} the 90% interval"] = float(h.loc[m, f"{side} 90"].mean())
                row[f"perfect model {side} the 90% interval"] = float(h.loc[pm, f"perfect {side} 90"].mean())
            row["mean error (predicted - measured)"] = float((h.predicted[m] - h.measured[m]).mean())
            row["RMSE"] = float(np.sqrt(np.mean((h.predicted[m] - h.measured[m]) ** 2)))
            rows.append(row)
    return pd.DataFrame(rows)


def run(ctx) -> Result:
    from v11_yearly_check import _full_record
    res = Result(
        code="V14", title="Prediction intervals on the hottest days",
        question="Limits are breached on the hottest days. In years the model was not calibrated on, do the "
                 "prediction intervals hold on the days predicted to be hottest, and on the days with the hottest "
                 "air, as well as they do on average?",
        method="For each Swiss river (calibration and validation files joined) and versions 5 and 8, "
               "leave-one-year-out cross-validation with the settings of V11 (DE, authors' bounds, Qmedia fixed) "
               f"gives each held-out year's simulation and its fold's error model (sigma and rho of the training "
               f"years). {N_SIMULATIONS} series are made from each (AR(1) error, as a FORWARD run does; parameter "
               "uncertainty not included), and for each measured day and each 7-day moving mean it is recorded "
               "whether the measurement lies inside the central 50%, 80%, 90%, 95% and 99% ranges. Shares inside "
               "are pooled over the three rivers for: all days; summer (June-August); the hottest 10% of each "
               "river's held-out days by predicted temperature; by air temperature (7-day mean air temperature "
               "for 7-day means); and by measured temperature. For the 90% interval the shares above and below "
               "it are given separately. Choosing days by their measured temperature is a biased selection: a "
               "perfect-model reference repeats every share with one simulated series in place of the "
               "measurements, which shows what a correct interval gives for each way of choosing days.",
        criterion=f"Pooled over the rivers, for each version, the shares of daily values inside the "
                  f"{', '.join(f'{x}%' for x in JUDGED_LEVELS)} ranges on the hottest 10% of days by prediction "
                  f"and by air temperature lie within the accepted range for real rivers (a miss rate between "
                  f"half and 1.5 times the stated one, as in V5). Summer, 7-day means and selection by "
                  f"measurement are reported, not judged.")
    stations = ["MAH_2369"] if ctx.quick else list(RIVERS)
    versions = (8,) if ctx.quick else VERSIONS
    jobs = [(st, v, ctx.quick) for st in stations for v in versions]
    jobs.sort(key=lambda j: j[0] != "SIO_2011")
    for st in stations:
        _full_record(st, write=True)
    with Timer() as t:
        with ProcessPoolExecutor(max_workers=ctx.workers) as ex:
            df = pd.concat(list(ex.map(_job, jobs)), ignore_index=True)
    res.seconds = t.seconds
    cov = _coverage(df)

    ok, judged_rows = True, []
    for r in cov[(cov.scale == "daily") & cov.subset.isin(JUDGED)].to_dict("records"):
        for lev in JUDGED_LEVELS:
            lo, hi = accepted_real(lev)
            share = r[f"{lev}%"]
            inside = lo <= share <= hi
            ok &= bool(inside)
            judged_rows.append({"version": r["version"], "subset": r["subset"], "level": f"{lev}%",
                                "share inside": f"{share:.1%}", "accepted": f"{lo:.1%}-{hi:.1%}",
                                "within": "yes" if inside else "no"})
    # The criterion needs every river and year: a quick run (one river, a few years) reports the shares
    # but does not judge them, as V5 skips its coverage criterion in quick mode.
    res.passed = True if ctx.quick else bool(ok)
    judged = pd.DataFrame(judged_rows)

    shown = cov.copy()
    for c in [c for c in shown.columns if c.endswith("%") or c.endswith("interval")]:
        shown[c] = shown[c].map(lambda x: f"{x:.1%}")
    shown["mean error (predicted - measured)"] = shown["mean error (predicted - measured)"].map(lambda x: f"{x:+.2f} °C")
    shown["RMSE"] = shown["RMSE"].map(lambda x: f"{x:.2f} °C")
    main_cols = ["version", "scale", "subset", "values"] + [f"{lev}%" for lev in LEVELS] + \
                ["above the 90% interval", "below the 90% interval", "mean error (predicted - measured)", "RMSE"]
    ref_cols = ["version", "scale", "subset"] + [f"perfect model {lev}%" for lev in LEVELS] + \
               ["perfect model above the 90% interval", "perfect model below the 90% interval"]

    by_river = []
    for (river, version), g in df[df.scale == "daily"].groupby(["river", "version"], sort=True):
        g = g.reset_index(drop=True)
        for name, m in _subsets(g).items():
            if name not in JUDGED + ("all days",):
                continue
            by_river.append({"river": river, "version": version, "subset": name, "days": int(m.sum()),
                             "inside 90%": f"{g.loc[m, 'in 90'].mean():.1%}",
                             "above": f"{g.loc[m, 'above 90'].mean():.1%}",
                             "below": f"{g.loc[m, 'below 90'].mean():.1%}",
                             "mean error (predicted - measured)": f"{(g.predicted[m] - g.measured[m]).mean():+.2f} °C",
                             "RMSE": f"{np.sqrt(np.mean((g.predicted[m] - g.measured[m]) ** 2)):.2f} °C"})
    by_river = pd.DataFrame(by_river)

    def share(version, scale, subset, lev, perfect=False):
        row = cov[(cov.version == version) & (cov.scale == scale) & (cov.subset == subset)]
        return float(row[f"perfect model {lev}%" if perfect else f"{lev}%"].iloc[0])

    def side(version, subset, which, perfect=False):
        row = cov[(cov.version == version) & (cov.scale == "daily") & (cov.subset == subset)]
        return float(row[f"perfect model {which} the 90% interval" if perfect else f"{which} the 90% interval"].iloc[0])

    vs = sorted(set(cov.version))
    res.summary = (
        ("Quick mode (one river, a few years): shares reported, criterion not judged. " if ctx.quick else "")
        + "Share of held-out daily measurements inside the 90% interval, pooled over the rivers: "
        + "; ".join(f"version {v}: all days {share(v, 'daily', 'all days', 90):.1%}, hottest 10% by prediction "
                    f"{share(v, 'daily', 'hottest 10% by prediction', 90):.1%}, by air temperature "
                    f"{share(v, 'daily', 'hottest 10% by air temperature', 90):.1%}, summer "
                    f"{share(v, 'daily', 'summer (June-August)', 90):.1%}" for v in vs)
        + ". On the hottest days chosen by their measured temperature, the measurement lay above the 90% "
        "interval on " + ", ".join(f"{side(v, 'hottest 10% by measurement', 'above'):.1%}" for v in vs)
        + " and below it on " + ", ".join(f"{side(v, 'hottest 10% by measurement', 'below'):.1%}" for v in vs)
        + " of days (version " + ", ".join(str(v) for v in vs) + "); a perfect model gives "
        + ", ".join(f"{side(v, 'hottest 10% by measurement', 'above', True):.1%} above and "
                    f"{side(v, 'hottest 10% by measurement', 'below', True):.1%} below" for v in vs)
        + " on the days it would choose this way (5% each on fairly chosen days).")
    res.sections.append(Section(
        "Coverage by subset of days",
        "Each share is the fraction of measured held-out values inside the central range of the simulations. "
        "The perfect-model columns repeat the calculation with one simulated series in place of the "
        "measurements: what a correct interval gives for each way of choosing days.",
        figures=[_fig(cov)],
        tables=[("Judged: hottest days by prediction and by air temperature, daily values", judged),
                ("Share inside each range, pooled over the rivers", shown[main_cols]),
                ("Perfect-model reference: share inside each range when the 'measurement' is one of the "
                 "simulations", shown[ref_cols]),
                ("By river: daily values inside the 90% interval and mean error", by_river)]))

    for v in vs:
        hot = cov[(cov.version == v) & (cov.scale == "daily") & (cov.subset == "hottest 10% by prediction")].iloc[0]
        allr = cov[(cov.version == v) & (cov.scale == "daily") & (cov.subset == "all days")].iloc[0]
        gap = allr["90%"] - hot["90%"]
        text = (f"Version {v}: on the days predicted to be hottest the 90% interval held {hot['90%']:.1%} of the "
                f"time, against {allr['90%']:.1%} on all days; the measurement lay above it on "
                f"{hot['above the 90% interval']:.1%} and below it on {hot['below the 90% interval']:.1%} of those "
                f"days (5% each if right). The model's RMSE on those days was {hot['RMSE']:.2f} °C, against "
                f"{allr['RMSE']:.2f} °C on all days, and its mean error {hot['mean error (predicted - measured)']:+.2f} °C "
                f"(positive: predicted too warm).")
        if gap <= -0.02 and hot["RMSE"] < allr["RMSE"]:
            text += (" The interval is wider than needed there: its width comes from the error size over the "
                     "whole year, and the model's errors are smaller on the hottest days. That is cautious for "
                     "a single day, but it spreads the probability of exceeding a limit more widely than the "
                     "data justify.")
        elif gap >= 0.02:
            text += (" The interval holds less often there than on average: the model's error on its hottest "
                     "predictions is not its typical error, as V11 found for yearly peaks. Check intervals on hot "
                     "days at your site.")
        res.notes.append(text)
    ref_above = [side(v, "hottest 10% by measurement", "above", True) for v in vs]
    res.notes.append(
        "Checking an interval only on the days when the measured temperature was highest (for example, the "
        "days a limit was exceeded) is a biased test: those days were chosen partly because their error was "
        "positive, so even for a correct interval the measurement lies above it more often than stated and "
        f"below it less often. Here a perfect model had the 'measurement' above its 90% interval on "
        f"{min(ref_above):.0%}-{max(ref_above):.0%} of such days, instead of 5%. To check intervals on hot days, "
        "choose the days by the prediction or by the air temperature, as here.")
    return res


def _fig(cov):
    import matplotlib.pyplot as plt
    plot_style()
    daily = cov[cov.scale == "daily"]
    vs = sorted(set(daily.version))
    fig, axes = plt.subplots(1, len(vs), figsize=(4.6 * len(vs), 4.3), sharey=True, squeeze=False)
    styles = {"all days": (INK2, "-", "o"), "summer (June-August)": (AQUA, "-", "o"),
              "hottest 10% by prediction": (BLUE, "-", "o"), "hottest 10% by air temperature": (ORANGE, "-", "s"),
              "hottest 10% by measurement": (YELLOW, (0, (2, 2)), "^")}
    for ax, v in zip(axes[0], vs):
        ax.plot([45, 100], [45, 100], color=LIGHT_GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for name, (colour, ls, marker) in styles.items():
            row = daily[(daily.version == v) & (daily.subset == name)].iloc[0]
            ax.plot(LEVELS, [100 * row[f"{lev}%"] for lev in LEVELS], color=colour, ls=ls, marker=marker, ms=3.5,
                    lw=1.3, label=name)
        ref = daily[(daily.version == v) & (daily.subset == "hottest 10% by measurement")].iloc[0]
        ax.plot(LEVELS, [100 * ref[f"perfect model {lev}%"] for lev in LEVELS], color=YELLOW, ls=(0, (1, 1.5)),
                lw=1.0, label="by measurement, perfect model")
        ax.set_xlim(45, 100)
        ax.set_ylim(30, 100)
        ax.set_xlabel("Stated level of the interval (%)")
        ax.set_title(f"Version {v}", fontsize=9)
    axes[0][0].set_ylabel("Held-out days inside the interval (%)")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return (save_figure(fig, "V14_hot_days.png"),
            "Share of held-out daily measurements inside the interval at each stated level, by subset of days. On "
            "the grey diagonal the interval holds. Days chosen by their measured temperature (dashed) are a "
            "biased selection: even a perfect model (dotted) departs from the diagonal there.")
