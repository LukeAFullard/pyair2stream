import sys

filepath = 'pyair2stream/cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

old_loop_start = """    results: list[FoldResult] = []

    from .io import compute_doy_climatology, compute_qmedia

    for label, idx in folds:"""

new_loop_start = """    results: list[FoldResult] = []

    from .io import compute_doy_climatology, compute_qmedia

    orig_par = data.par.copy() if data.par is not None else None
    orig_par_best = data.par_best.copy() if data.par_best is not None else None

    try:
        for label, idx in folds:"""

old_loop_body = """
    for label, idx in folds:
        orig_twat, orig_tair, orig_q, w_idx, orig_w_twat, orig_w_tair, orig_w_q = _mask_fold(data, idx)
        try:"""

new_loop_body = """
        for label, idx in folds:
            orig_twat, orig_tair, orig_q, w_idx, orig_w_twat, orig_w_tair, orig_w_q = _mask_fold(data, idx)
            try:"""

if old_loop_start in content:
    content = content.replace(old_loop_start, new_loop_start)
else:
    print("Could not find old_loop_start")

content_lines = content.splitlines(True)
inside_loop = False
indent = "    "
try_block_indent = "        "

# Doing a simple string replacement based on indentation for the loop content.
# Actually let's use ast or string matching carefully.
