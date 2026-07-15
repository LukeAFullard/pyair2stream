import sys

filepath = 'pyair2stream/cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

old_loop = """
    for label, idx in folds:
        orig_twat, orig_tair, orig_q = _mask_fold(data, idx)
        try:"""

new_loop = """
    for label, idx in folds:
        orig_twat, orig_tair, orig_q, w_idx, orig_w_twat, orig_w_tair, orig_w_q = _mask_fold(data, idx)
        try:"""

if old_loop in content:
    content = content.replace(old_loop, new_loop)
else:
    print("Could not find old_loop")

old_inline_restore = """
            # Restore forcing variables (Tair, Q) before forward simulation
            # so the model integrates through the held-out window properly
            data.Tair[idx] = orig_tair
            data.Q[idx] = orig_q
"""

new_inline_restore = """
            # Restore forcing variables (Tair, Q) before forward simulation
            # so the model integrates through the held-out window properly
            if data.gap_tolerant:
                data.Tair[idx] = orig_tair
                data.Q[idx] = orig_q
                if w_idx.size > 0:
                    data.Tair[w_idx] = orig_w_tair
                    data.Q[w_idx] = orig_w_q
"""

if old_inline_restore in content:
    content = content.replace(old_inline_restore, new_inline_restore)
else:
    print("Could not find old_inline_restore")

old_finally = """
            results.append(FoldResult(
                fold_id=len(results),
                label=label,
                held_out_start=start_date,
                held_out_end=end_date,
                n_obs_held_out=int(np.sum(orig_twat != MISSING_DATA_SENTINEL)),
                par_best=data.par_best.copy(),
                nse=nse,
                kge=kge,
                rmse=rmse,
                obs_held_out=orig_twat.copy(),
                sim_held_out=sim[idx].copy(),
            ))
        finally:
            _restore_fold(data, idx, orig_twat, orig_tair, orig_q)"""

new_finally = """
            results.append(FoldResult(
                fold_id=len(results),
                label=label,
                held_out_start=start_date,
                held_out_end=end_date,
                n_obs_held_out=int(np.sum(orig_twat != MISSING_DATA_SENTINEL)),
                par_best=data.par_best.copy(),
                nse=nse,
                kge=kge,
                rmse=rmse,
                obs_held_out=orig_twat.copy(),
                sim_held_out=sim[idx].copy(),
            ))
        finally:
            _restore_fold(data, idx, orig_twat, orig_tair, orig_q, w_idx, orig_w_twat, orig_w_tair, orig_w_q)"""

if old_finally in content:
    content = content.replace(old_finally, new_finally)
else:
    print("Could not find old_finally")

with open(filepath, 'w') as f:
    f.write(content)
print("Successfully patched driver loops")
