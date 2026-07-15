import sys

filepath = 'pyair2stream/cross_validation.py'

with open(filepath, 'r') as f:
    content = f.read()

# Replace block in build_folds
old_block = """    folds = []
    for block in blocks:
        mask = np.isin(wy, block)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        label = str(block[0]) if len(block) == 1 else f"{block[0]}-{block[-1]}"
        folds.append((label, idx))
    return folds"""

new_block = """    folds = []
    for block in blocks:
        mask = np.isin(wy, block)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        valid_obs_count = np.sum(data.Twat_obs[idx] != MISSING_DATA_SENTINEL)
        if valid_obs_count < cv_config.min_valid_obs:
            continue

        label = str(block[0]) if len(block) == 1 else f"{block[0]}-{block[-1]}"
        folds.append((label, idx))
    return folds"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(filepath, 'w') as f:
        f.write(content)
    print("Successfully updated build_folds in pyair2stream/cross_validation.py")
else:
    print("Could not find old_block in pyair2stream/cross_validation.py")
