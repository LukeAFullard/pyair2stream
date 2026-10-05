"""
Run the validation suite and write validation/REPORT.md.

    python validation/run_all.py            # full suite (about 105 minutes on 4 cores)
    python validation/run_all.py --quick    # reduced version of every check (about 2 minutes)
    python validation/run_all.py --only V2 V6

The summary (validation/REPORT.md), one full report per check
(validation/reports/V*.md), their tables (validation/results/*.csv) and figures
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
          ("V5", "v5_real_rivers"), ("V6", "v6_numerics"), ("V7", "v7_gaps"), ("V8", "v8_workflow"),
          ("V9", "v9_exceedance"), ("V10", "v10_split_sample"),
          ("V11", "v11_yearly_check")]


def git_state() -> str:
    """The commit being validated, and whether tracked files differ from it."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    # The run's own outputs (REPORT.md, reports/, results/, figures/) are not part of what is validated.
    dirty = " (with uncommitted changes)" if git(
        "status", "--porcelain", "--untracked-files=no", "--", ".", ":!validation/REPORT.md",
        ":!validation/reports", ":!validation/results", ":!validation/figures") else ""
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


# --- reports ------------------------------------------------------------------

REPORTS = os.path.join(HERE, "reports")


def _table_lines(r: Result, title: str, df: pd.DataFrame, counter: list, prefix: str) -> list:
    counter[0] += 1
    name = f"{r.code}_{counter[0]}.csv"
    df.to_csv(os.path.join(RESULTS, name), index=False)
    return [f"**{title}** ([csv]({prefix}results/{name}))", "", md_table(df), ""]


def as_parts(text: str) -> list:
    """Text with labelled parts '(A) ... (B) ...' as an introduction plus one bullet per part."""
    import re
    pieces = [p for p in re.split(r"\s*(?=\([A-H]\)\s)", text.strip()) if p]
    if len(pieces) < 3:
        return [text, ""]
    intro, parts = (pieces[0], pieces[1:]) if not pieces[0].startswith("(") else ("", pieces)
    return ([intro, ""] if intro else []) + [f"- {p}" for p in parts] + [""]


def write_check_report(r: Result, env_line: str) -> None:
    """reports/<code>.md: the full report of one check."""
    lines = [f"# {r.code}. {r.title}", "",
             f"[Validation summary](../REPORT.md) · {env_line} · run time {r.seconds / 60:.1f} minutes", "",
             f"**Result: {status(r)}.**", "", *as_parts(r.summary)]
    for heading, text in (("Question", r.question), ("Method", r.method), ("Pass criterion", r.criterion)):
        if text:
            lines += [f"## {heading}", "", *as_parts(text)]
    counter = [0]
    if r.sections:
        lines += ["## Results", ""]
    for s in r.sections:
        lines += [f"### {s.title}", ""]
        if s.text:
            lines += [s.text, ""]
        for fname, caption in s.figures:
            lines += [f"![{caption}](../figures/{fname})", "", f"*{caption}*", ""]
        for title, df in s.tables:
            lines += _table_lines(r, title, df, counter, "../")
    if r.notes:
        lines += ["## What this means", ""]
        for n in r.notes:
            lines += [n, ""]
    with open(os.path.join(REPORTS, f"{r.code}.md"), "w") as f:
        f.write("\n".join(lines))


def write_report(results, env, quick):
    """REPORT.md: the summary, with each check's headline figure and a link to its full report."""
    lines = ["# pyair2stream validation report", "",
             "Generated by `validation/run_all.py`. This page summarises each check; its full report "
             "(question, method, pass criterion, figures, tables and what the result means) is linked "
             "below it. See `validation/README.md` for why each check matters.", ""]
    if quick:
        lines += ["> **Quick mode:** every check ran in a reduced form. Use the full run for evidence.", ""]
    lines += ["## Summary", "", "| Check | Question | Result | Full report |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| [{r.code}](#{r.code.lower()}) | {r.title} | **{status(r)}** | "
                     f"[reports/{r.code}.md](reports/{r.code}.md) |")
    lines += ["", "## Environment", "", "| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in env], ""]
    env_line = next(v for k, v in env if k == "pyair2stream").split(" at the start")[0]
    env_line = f"pyair2stream {env_line}, {dict(env)['Date']}"
    for r in results:
        write_check_report(r, env_line)
        lines += [f"## {r.code}", "", f"### {r.title}", "", f"**Question.** {r.question}", "",
                  f"**Result: {status(r)}.**", "", *as_parts(r.summary)]
        headline = next((f for s in r.sections for f in s.figures), None)
        if headline:
            lines += [f"![{headline[1]}](figures/{headline[0]})", "", f"*{headline[1]}*", ""]
        lines += [f"**Full report: [reports/{r.code}.md](reports/{r.code}.md)**", ""]
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="reduced version of every check")
    ap.add_argument("--only", nargs="+", metavar="CODE", help="run only these checks, e.g. V2 V6")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="parallel processes")
    args = ap.parse_args()
    ctx = types.SimpleNamespace(quick=args.quick, workers=args.workers)
    state = git_state()
    selected = [(c, m) for c, m in CHECKS if not args.only or c in {o.upper() for o in args.only}]
    for folder in (RESULTS, FIGURES, REPORTS):
        os.makedirs(folder, exist_ok=True)
        for f in os.listdir(folder):
            if not args.only or f.split("_")[0].split(".")[0] in {c for c, _ in selected}:
                os.remove(os.path.join(folder, f))
    results = []
    t0 = datetime.datetime.now()
    for code, module in selected:
        print(f"{code} ...", flush=True)
        try:
            r = importlib.import_module(module).run(ctx)
        except Exception:
            r = Result(code=code, title=module, question="", method="", criterion="", passed=False,
                       summary="The check stopped with an error:\n\n```\n" + traceback.format_exc() + "```")
        print(f"{code} {status(r)} ({r.seconds / 60:.1f} min): {r.summary}", flush=True)
        results.append(r)
    env = environment(args.quick, (datetime.datetime.now() - t0).total_seconds(), state)
    write_report(results, env, args.quick)
    failed = [r.code for r in results if r.passed is False]
    print("All checks passed." if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
