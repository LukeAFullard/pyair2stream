"""
V15 supplement - a longer DE-MCMC run for British Columbia stations whose sampler did not converge
within V15's 100,000 steps.

Same calibration data, likelihood, walkers and seed as V15 part D (`v15_british_columbia._mcmc`);
only the step limit is raised (default 500,000). The sampler still stops as soon as the convergence
rule is met (steps >= 50 x autocorrelation time and split-Rhat < 1.01). Results go to
validation/results/V15_long_chain_<station>.csv and validation/reports/V15_long_chain.md; V15's own
results are not changed.

    python validation/v15_long_chain.py 08KH006 [--steps 500000]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import WORK, load                                        # noqa: E402
from v2_published import _inside                                     # noqa: E402
from v15_british_columbia import MCMC_WALKERS, _calibration_config, _load   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run(st, steps):
    from pyair2stream.optimization import DE_MCMC_mode
    tag = f"v15_long_chain_{st}"
    cfg = _calibration_config(st, tag, False, run_mode="DE-MCMC",
                              uncertainty_options={"noise_model": "ar1", "likelihood": "least_squares",
                                                   "strict_convergence": False})
    cfg["optimization"].update({"mcmc_walkers": MCMC_WALKERS, "mcmc_steps": steps})
    data = load(cfg, tag)
    t0 = time.time()
    DE_MCMC_mode(data, seed=1)
    hours = (time.time() - t0) / 3600
    out = os.path.join(WORK, tag, "out")
    meta = json.load(open(os.path.join(out, "MCMC_chain_S_c_1d_meta.json")))
    chain = pd.read_csv(os.path.join(out, "MCMC_chain_S_c_1d.csv"))
    _, par, _ = _load()
    pub = par.loc[st, [f"a{j + 1}" for j in range(8)]].to_numpy(float)
    rows = []
    for j, c in enumerate(chain.columns):
        lo, hi = np.percentile(chain[c], [5, 95])
        rows.append({"station": st, "parameter": f"a{j + 1}", "published": round(pub[j], 3),
                     "median": round(float(np.median(chain[c])), 3),
                     "MCMC 90% interval": f"{lo:.3f} to {hi:.3f}",
                     "published inside": "yes" if _inside(pub[j], lo, hi, j) else "no"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", f"V15_long_chain_{st}.csv"), index=False)
    n_in = int((df["published inside"] == "yes").sum())
    md = [f"# V15 supplement: longer DE-MCMC run, station {st}", "",
          f"Same set-up as V15 part D ({MCMC_WALKERS} walkers, seed 1, least-squares likelihood widened for "
          f"the autocorrelation of the errors), with at most {steps:,} steps instead of 100,000 "
          f"(`python validation/v15_long_chain.py {st} --steps {steps}`).", "",
          f"- Converged: **{'yes' if meta['converged'] else 'NO'}** after {meta['steps_run']:,} steps "
          f"({meta['burnin']:,} discarded as burn-in), {hours:.1f} hours.",
          f"- Longest autocorrelation time: {meta['max_autocorr_time']:.0f} steps "
          f"(the rule needs at least {50 * meta['max_autocorr_time']:,.0f} steps).",
          f"- Largest split-Rhat: {meta['max_split_rhat']:.4f} (the rule needs below 1.01).",
          f"- Published values inside the 90% interval: **{n_in} of 8**.", "",
          "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns),
          *("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)), ""]
    path = os.path.join(HERE, "reports", "V15_long_chain.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("station")
    ap.add_argument("--steps", type=int, default=500000)
    a = ap.parse_args()
    run(a.station, a.steps)
