import sys

filepath = 'pyair2stream/cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

old_mask = """def _mask_fold(data: CommonData, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"
    Set Twat_obs, Tair, and Q to the configured missing value for the given rows and return the original
    values (for scoring + restoration). By masking the forcing data as well, gap_tolerant
    mode will correctly segment the ODE integration, preventing catastrophic state drift
    over long missing target windows.

    Note: In default whole-series mode (gap_tolerant=False), a held-out year is properly
    excluded from the objective (Twat_obs = -999.0), but the ODE still free-integrates
    through the held-out window using actual forcing data with no restart. This is generally
    fine as the model is fairly mean-reverting, but represents an asymmetry compared to
    the segmented restart behavior of gap-tolerant mode.
    \"\"\"
    orig_twat = data.Twat_obs[idx].copy()
    orig_tair = data.Tair[idx].copy()
    orig_q = data.Q[idx].copy()

    data.Twat_obs[idx] = MISSING_DATA_SENTINEL

    if data.gap_tolerant:
        data.Tair[idx] = MISSING_DATA_SENTINEL
        data.Q[idx] = MISSING_DATA_SENTINEL

    return orig_twat, orig_tair, orig_q"""

new_mask = """def _mask_fold(data: CommonData, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"
    Set Twat_obs, Tair, and Q to the configured missing value for the given rows and return the original
    values (for scoring + restoration). By masking the forcing data as well, gap_tolerant
    mode will correctly segment the ODE integration, preventing catastrophic state drift
    over long missing target windows.

    Note: In default whole-series mode (gap_tolerant=False), a held-out year is properly
    excluded from the objective (Twat_obs = -999.0), but the ODE still free-integrates
    through the held-out window using actual forcing data with no restart. This is generally
    fine as the model is fairly mean-reverting, but represents an asymmetry compared to
    the segmented restart behavior of gap-tolerant mode.
    \"\"\"
    orig_twat = data.Twat_obs[idx].copy()
    orig_tair = data.Tair[idx].copy()
    orig_q = data.Q[idx].copy()

    data.Twat_obs[idx] = MISSING_DATA_SENTINEL

    if data.gap_tolerant:
        data.Tair[idx] = MISSING_DATA_SENTINEL
        data.Q[idx] = MISSING_DATA_SENTINEL

    # Also handle warm-up block if year 1 is masked
    w_mask = (idx >= 365) & (idx < 730)
    w_idx = idx[w_mask] - 365

    orig_w_twat = data.Twat_obs[w_idx].copy() if w_idx.size > 0 else np.array([])
    orig_w_tair = data.Tair[w_idx].copy() if w_idx.size > 0 else np.array([])
    orig_w_q = data.Q[w_idx].copy() if w_idx.size > 0 else np.array([])

    if w_idx.size > 0:
        data.Twat_obs[w_idx] = MISSING_DATA_SENTINEL
        if data.gap_tolerant:
            data.Tair[w_idx] = MISSING_DATA_SENTINEL
            data.Q[w_idx] = MISSING_DATA_SENTINEL

    return orig_twat, orig_tair, orig_q, w_idx, orig_w_twat, orig_w_tair, orig_w_q"""


old_restore = """def _restore_fold(data: CommonData, idx: np.ndarray, orig_twat: np.ndarray, orig_tair: np.ndarray, orig_q: np.ndarray) -> None:
    \"\"\"
    Restore the original forcing data and observations to the global CommonData
    arrays after a cross-validation fold has finished.

    This ensures subsequent folds start with a clean slate without reloading
    the datasets from disk.
    \"\"\"
    data.Twat_obs[idx] = orig_twat
    data.Tair[idx] = orig_tair
    data.Q[idx] = orig_q"""

new_restore = """def _restore_fold(data: CommonData, idx: np.ndarray, orig_twat: np.ndarray, orig_tair: np.ndarray, orig_q: np.ndarray, w_idx: np.ndarray, orig_w_twat: np.ndarray, orig_w_tair: np.ndarray, orig_w_q: np.ndarray) -> None:
    \"\"\"
    Restore the original forcing data and observations to the global CommonData
    arrays after a cross-validation fold has finished.

    This ensures subsequent folds start with a clean slate without reloading
    the datasets from disk.
    \"\"\"
    data.Twat_obs[idx] = orig_twat
    if data.gap_tolerant:
        data.Tair[idx] = orig_tair
        data.Q[idx] = orig_q

    if w_idx.size > 0:
        data.Twat_obs[w_idx] = orig_w_twat
        if data.gap_tolerant:
            data.Tair[w_idx] = orig_w_tair
            data.Q[w_idx] = orig_w_q"""

if old_mask in content:
    content = content.replace(old_mask, new_mask)
else:
    print("Could not find old_mask")

if old_restore in content:
    content = content.replace(old_restore, new_restore)
else:
    print("Could not find old_restore")

with open(filepath, 'w') as f:
    f.write(content)
print("Successfully patched _mask_fold and _restore_fold")
