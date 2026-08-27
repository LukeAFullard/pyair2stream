"""
Regression tests for docs/audit/03_objective_function_and_masks.md.

Two masks control which days count: `data.segments` (contiguous valid-forcing
blocks) and `data.eval_mask` (days eligible for scoring, built by
`detect_segments`/`prepare_evaluation`). Before this fix:

- Defect A: in gap-tolerant mode, `statis()` summed over every window
  `aggregation()` emitted, while `funcobj()` additionally skipped windows
  failing `eval_mask` -- so NSE/KGE/R2 were computed from mismatched samples.
- Defect B: `eval_mask` was only ever built when `gap_tolerant: true`, so it
  stayed `None` for the whole default (non-gap-tolerant) workflow, and the
  MCMC likelihood's own daily `Twat_obs != -999.0` mask then double-counted
  the warm-up block (a verbatim copy of year one) as real observations.

These tests check that:
1. `n_dat` after `aggregation()` equals the number of windows `funcobj()`
   actually scores, and the reported NSE matches an independently computed
   NSE on that exact matched subset, to 1e-10.
2. The reported KGE matches an independently computed KGE on the exact
   matched (eval_mask-and-observed) subset, for two different
   `warmup_drop_days` values -- i.e. the alignment holds regardless of which
   value is chosen, not just by accident for one.
3. `data.eval_mask` is not None after a non-gap-tolerant data load, and
   excludes the warm-up block.
4. The MCMC likelihood's N (recorded in the sidecar as `n_valid_pairs`)
   equals the number of real (non-warm-up) observations, not `n_tot`.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from pyair2stream.config import CommonData
from pyair2stream.io import read_Tseries
from pyair2stream.model import prepare_evaluation, aggregation, statis, call_model, funcobj
from pyair2stream.optimization import DE_MCMC_mode

PAR0 = [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]


def _build_gap_tolerant_data(warmup_drop_days, n_real_days=200, min_segment_days=30):
    n_tot = 365 + n_real_days
    data = CommonData()
    data.version = 8
    data.gap_tolerant = True
    data.warmup_drop_days = warmup_drop_days
    data.min_segment_days = min_segment_days
    data.mod_num = 'CRN'
    data.time_res = '1d'
    data.n_tot = n_tot
    data.Tair = 15.0 + 5.0 * np.sin(np.linspace(0, 8 * np.pi, n_tot))
    data.Q = np.full(n_tot, 10.0)
    data.Qmedia = 10.0
    data.Tice_cover = 0.0
    data.par = np.array(PAR0, dtype=np.float64)
    data.flag_par = np.ones(8, dtype=bool)
    data.date = np.zeros((n_tot, 3), dtype=np.int32)
    data.tt = np.array([((i % 365) + 1) / 365.0 for i in range(n_tot)])
    # Observations exist for the entire real record, INCLUDING inside each
    # segment's warmup_drop_days -- this is exactly what makes Defect A bite.
    data.Twat_obs = np.full(n_tot, -999.0)
    data.Twat_obs[365:] = 12.0 + 3.0 * np.sin(np.linspace(0, 4 * np.pi, n_real_days))
    data.Twat_mod = np.zeros(n_tot)
    return data


class TestEvalMaskAndObjective(unittest.TestCase):
    def test_nse_matches_honest_nse_on_matched_subset(self):
        data = _build_gap_tolerant_data(warmup_drop_days=15)
        data.fun_obj = 'NSE'

        prepare_evaluation(data)
        aggregation(data)
        statis(data)
        call_model(data)
        reported_nse = funcobj(data)

        matched = (data.Twat_obs != -999.0) & data.eval_mask & (data.Twat_mod != -999.0)
        obs = data.Twat_obs[matched]
        mod = data.Twat_mod[matched]
        mean_obs = obs.mean()
        honest_nse = 1.0 - np.sum((obs - mod) ** 2) / np.sum((obs - mean_obs) ** 2)

        # n_dat (aggregation's window count) must equal the number of windows
        # funcobj actually scores, i.e. the matched subset -- and both at 1d
        # resolution mean the same thing: one window per scored day.
        self.assertEqual(data.n_dat, int(matched.sum()))
        self.assertEqual(int(np.sum(data.Twat_mod_agg[365:] != -999.0)), data.n_dat)

        self.assertAlmostEqual(reported_nse, honest_nse, places=10)

    def test_kge_matches_honest_kge_regardless_of_warmup_drop_days(self):
        for warmup_drop_days in (10, 40):
            data = _build_gap_tolerant_data(warmup_drop_days=warmup_drop_days)
            data.fun_obj = 'KGE'

            prepare_evaluation(data)
            aggregation(data)
            statis(data)
            call_model(data)
            reported_kge = funcobj(data)

            matched = (data.Twat_obs != -999.0) & data.eval_mask & (data.Twat_mod != -999.0)
            obs = data.Twat_obs[matched]
            mod = data.Twat_mod[matched]
            mean_obs, mean_mod = obs.mean(), mod.mean()
            std_obs, std_mod = obs.std(ddof=1), mod.std(ddof=1)
            covar = np.sum((obs - mean_obs) * (mod - mean_mod)) / (len(obs) - 1)
            honest_kge = 1.0 - np.sqrt(
                (std_mod / std_obs - 1.0) ** 2
                + (mean_mod / mean_obs - 1.0) ** 2
                + (covar / (std_mod * std_obs) - 1.0) ** 2
            )

            self.assertAlmostEqual(
                reported_kge, honest_kge, places=9,
                msg=f"KGE/honest-KGE mismatch at warmup_drop_days={warmup_drop_days}",
            )

    def test_eval_mask_set_and_excludes_warmup_after_non_gap_tolerant_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            import pandas as pd
            csv_path = os.path.join(tmp, 'data.csv')
            n_days = 400
            dates = pd.date_range('2001-01-01', periods=n_days, freq='D')
            df = pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'T_air': 10.0 + 12.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.0),
                'Discharge': np.full(n_days, 10.0),
                'T_water': 8.0 + 6.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.0),
            })
            df.to_csv(csv_path, index=False)

            data = CommonData()
            data.runmode = 'DE'
            data._input_data_path_cal = csv_path

            read_Tseries(data, 'c')

            self.assertIsNotNone(data.eval_mask)
            self.assertEqual(data.eval_mask[:365].sum(), 0)
            self.assertTrue(data.eval_mask[365:].all())

    def test_mcmc_likelihood_n_excludes_warmup_block(self):
        n_days = 400
        n_tot = n_days + 365
        data = CommonData()
        data.version = 8
        data.gap_tolerant = False
        data.mod_num = 'CRN'
        data.time_res = '1d'
        data.fun_obj = 'NSE'
        data.n_tot = n_tot
        data.Tair = 15.0 + 5.0 * np.sin(np.linspace(0, 8 * np.pi, n_tot))
        data.Q = np.full(n_tot, 10.0)
        data.Qmedia = 10.0
        data.Tice_cover = 0.0
        data.par = np.array(PAR0, dtype=np.float64)
        data.par_best = data.par.copy()
        # Fix every parameter except a1 so MCMC has only 1 active dimension (fast).
        data.parmin = data.par.copy()
        data.parmax = data.par.copy()
        data.parmin[0] = 0.5
        data.parmax[0] = 1.5
        data.flag_par = np.ones(8, dtype=bool)
        data.date = np.zeros((n_tot, 3), dtype=np.int32)
        data.tt = np.array([((i % 365) + 1) / 365.0 for i in range(n_tot)])
        data.Twat_obs = np.full(n_tot, -999.0)
        data.Twat_obs[365:] = 12.0 + 3.0 * np.sin(np.linspace(0, 4 * np.pi, n_days))
        data.Twat_mod = np.zeros(n_tot)

        data.n_particles = 4
        data.n_run = 2
        data.mcmc_walkers = 4
        data.mcmc_steps = 3
        data.mineff_index = -1e30

        prepare_evaluation(data)
        aggregation(data)
        statis(data)

        with tempfile.TemporaryDirectory() as tmp:
            data.folder = tmp
            data.station = 'test'
            data.series = 'c'
            DE_MCMC_mode(data, seed=1)

            sidecar_path = os.path.join(tmp, "MCMC_chain_test_c_1d_meta.json")
            with open(sidecar_path) as f:
                meta = json.load(f)

        # N must be the number of REAL observations (index >= 365), not n_tot
        # (which double-counts the warm-up copy of year one -- report 03, Defect B).
        self.assertEqual(meta['n_valid_pairs'], n_days)
        self.assertNotEqual(meta['n_valid_pairs'], n_tot)


if __name__ == '__main__':
    unittest.main()
