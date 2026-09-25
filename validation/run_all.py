"""
Run the validation suite and write validation/REPORT.md.

    python validation/run_all.py            # full suite (about 15 minutes on 4 cores)
    python validation/run_all.py --quick    # reduced version of every check (about 2 minutes)
    python validation/run_all.py --only V2 V6

The report, its tables (validation/results/*.csv) and figures
(validation/figures/*.png) are written from scratch on every run. Scratch files
go to validation/work/ (not committed).
"""

import argparse
import datetime
import importlib
import os
import platform
import subprocess
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402

from common import FIGURES, REPO, RESULTS, Result   # noqa: E402

CHECKS = [("V1", "v1_fortran"), ("V2", "v2_published"), ("V3", "v3_recovery"), ("V4", "v4_intervals"),
          ("V5", "v5_real_rivers"), ("V6", "v6_numerics"), ("V7", "v7_gaps"), ("V8", "v8_workflow")]


def git_state() -> str:
    """The commit being validated, and whether tracked files differ from it."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    dirty = " (with uncommitted changes)" if git("status", "--porcelain", "--untracked-files=no") else ""
    return f"commit {git('rev-parse', '--short', 'HEAD')}{dirty}"


def environment(quick: bool, seconds: float, state: str) -> list:
    import emcee, numba, scipy
    import pyair2stream
    return [
        ("Date", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("pyair2stream", f"{pyair2stream.__version__}, {state} at the start of the run"),
        ("Mode", "quick (reduced)" if quick else "full"),
        ("Run time", f"{seconds / 60:.1f} minutes"),
        ("Python", platform.python_version()),
        ("Packages", f"numpy {np.__version__}, scipy {scipy.__version__}, pandas {pd.__version__}, "
                     f"emcee {emcee.__version__}, numba {numba.__version__}"),
        ("Machine", f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs"),
    ]


def md_table(df: pd.DataFrame) -> str:
    def cell(x):
        if x is None:
            return ""
        if isinstance(x, float):
            return "" if np.isnan(x) else f"{x:.4g}"
        if isinstance(x, (bool, np.bool_)):
            return "yes" if x else "no"
        return str(x).replace("|", "\\|")
    head = "| " + " | ".join(str(c).replace("|", "\\|") for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def status(r: Result) -> str:
    return "not run" if r.passed is None else ("PASS" if r.passed else "FAIL")


# --- figures ------------------------------------------------------------------

def _save(fig, name):
    import matplotlib.pyplot as plt
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def figures(r: Result) -> None:
    import matplotlib.pyplot as plt
    data = getattr(r, "figure_data", None)
    if data is None:
        return
    if r.code == "V2":
        fig, ax = plt.subplots(figsize=(5, 5))
        for col, label, m in (("pyair2stream cal", "calibration period", "o"),
                              ("pyair2stream val", "validation period", "s")):
            pub = data["published cal" if "cal" in col else "published val"]
            ax.scatter(pub, data[col], marker=m, label=label, s=28)
        lim = [0, float(max(data["published cal"].max(), data["published val"].max())) * 1.1]
        ax.plot(lim, lim, color="grey", lw=1)
        ax.set(xlim=lim, ylim=lim, xlabel="Published RMSE (°C)", ylabel="pyair2stream RMSE (°C)",
               title="Published parameters: RMSE reproduced")
        ax.legend()
        r.figures.append((_save(fig, "V2_published_rmse.png"),
                          "Each point is one river, model version and period; all lie on the 1:1 line."))
    elif r.code == "V4":
        df = data[data.converged]
        cases = list(dict.fromkeys(df.case))
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, c in enumerate(cases):
            y = df[df.case == c]["held-out coverage"] * 100
            ax.scatter(np.full(len(y), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(y)), y, s=16)
            ax.hlines(y.mean(), i - 0.25, i + 0.25, color="black")
        ax.axhline(90, color="grey", ls="--", lw=1)
        ax.set_xticks(range(len(cases)), [c.split(":")[0] for c in cases])
        ax.set(ylabel="Held-out observations inside\nthe 90% interval (%)",
               title="Prediction-interval coverage, one point per replicate")
        r.figures.append((_save(fig, "V4_interval_coverage.png"),
                          "Cases as in the table. Black line: mean over replicates. Dashed: the nominal 90%."))
    elif r.code == "V5":
        a, b = data
        fig, ax = plt.subplots(figsize=(8, 3.6))
        rivers = list(dict.fromkeys(a.river))
        labels = list(dict.fromkeys(a.version.astype(str)))
        w = 0.8 / len(labels)
        for k, lab in enumerate(labels):
            vals = [a[(a.river == rv) & (a.version.astype(str) == lab)].RMSE.mean() for rv in rivers]
            name = f"version {lab}" if lab.isdigit() else lab
            ax.bar(np.arange(len(rivers)) + k * w, vals, w, label=name,
                   color=None if lab.isdigit() else ("0.55" if "day" in lab else "0.8"))
        ax.set_xticks(np.arange(len(rivers)) + 0.4 - w / 2, rivers)
        ax.set(ylabel="RMSE on validation years (°C)", title="Predicting years not used for calibration")
        ax.legend(ncol=2, fontsize=8, frameon=False)
        r.figures.append((_save(fig, "V5_validation_rmse.png"),
                          "Lower is better. Grey bars: the two simple alternatives."))
        if b is not None and len(b) and "7-day mean coverage" in b:
            b = b[b.converged]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, col, title in zip(axes, ("coverage", "7-day mean coverage"),
                                      ("Daily temperatures", "7-day mean temperatures")):
                for k, (noise, colour) in enumerate((("iid", "0.6"), ("ar1", "tab:blue"))):
                    g = b[b["noise model"] == noise]
                    x = np.arange(len(g)) + (k - 0.5) * 0.38
                    ax.bar(x, g[col] * 100, 0.38, color=colour, label=f"noise_model: {noise}")
                ax.axhline(90, color="black", ls="--", lw=1)
                g = b[b["noise model"] == "iid"]
                ax.set_xticks(np.arange(len(g)), [f"{rv} v{v}" for rv, v in zip(g.river, g.version)], fontsize=8,
                              rotation=30, ha="right")
                ax.set(title=title, ylim=(0, 100))
            axes[0].set_ylabel("Validation observations inside\nthe 90% interval (%)")
            handles, names = axes[0].get_legend_handles_labels()
            fig.legend(handles, names, frameon=False, fontsize=8, ncol=2, loc="lower center",
                       bbox_to_anchor=(0.5, -0.17))
            r.figures.append((_save(fig, "V5_interval_coverage.png"),
                              "Dashed: the nominal 90%. For 7-day means only AR(1) noise comes close."))


# --- report -------------------------------------------------------------------

def write_report(results, env, quick):
    lines = ["# pyair2stream validation report", "",
             "Generated by `validation/run_all.py`. Each check states its question, method and pass "
             "criterion before its result. See `validation/README.md` for what each check covers and why.", ""]
    if quick:
        lines += ["> **Quick mode:** every check ran in a reduced form. Use the full run for evidence.", ""]
    lines += ["## Summary", "", "| Check | Question | Result |", "|---|---|---|"]
    for r in results:
        lines.append(f"| [{r.code}](#{r.code.lower()}) | {r.title} | **{status(r)}** |")
    lines += ["", "## Environment", "", "| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in env], ""]
    for r in results:
        lines += [f"## {r.code}", "", f"### {r.title}", "", f"**Question.** {r.question}", "",
                  f"**Method.** {r.method}", "", f"**Pass criterion.** {r.criterion}", "",
                  f"**Result: {status(r)}.** {r.summary}", f"(Run time {r.seconds / 60:.1f} minutes.)", ""]
        for i, (title, df) in enumerate(r.tables, 1):
            name = f"{r.code}_{i}.csv"
            df.to_csv(os.path.join(RESULTS, name), index=False)
            lines += [f"**{title}** ([csv](results/{name}))", "", md_table(df), ""]
        for fname, caption in r.figures:
            lines += [f"![{caption}](figures/{fname})", "", f"*{caption}*", ""]
        for n in r.notes:
            lines += [f"> {n}", ""]
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="reduced version of every check")
    ap.add_argument("--only", nargs="+", metavar="CODE", help="run only these checks, e.g. V2 V6")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="parallel processes")
    args = ap.parse_args()
    ctx = types.SimpleNamespace(quick=args.quick, workers=args.workers)
    selected = [(c, m) for c, m in CHECKS if not args.only or c in {o.upper() for o in args.only}]
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    for folder in (RESULTS, FIGURES):
        for f in os.listdir(folder):
            if not args.only or f.split("_")[0] in {c for c, _ in selected}:
                os.remove(os.path.join(folder, f))
    results = []
    state = git_state()
    t0 = datetime.datetime.now()
    for code, module in selected:
        print(f"{code} ...", flush=True)
        try:
            r = importlib.import_module(module).run(ctx)
        except Exception:
            r = Result(code=code, title=module, question="", method="", criterion="", passed=False,
                       summary="The check stopped with an error:\n\n```\n" + traceback.format_exc() + "```")
        try:
            figures(r)
        except Exception:
            r.notes.append("Figure failed: " + traceback.format_exc(limit=1))
        print(f"{code} {status(r)} ({r.seconds / 60:.1f} min): {r.summary}", flush=True)
        results.append(r)
    env = environment(args.quick, (datetime.datetime.now() - t0).total_seconds(), state)
    write_report(results, env, args.quick)
    failed = [r.code for r in results if r.passed is False]
    print("All checks passed." if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
