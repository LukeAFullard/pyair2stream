import sys

filepath = 'pyair2stream/cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

# Instead of indenting the whole loop, I can just save the parameters at the start and restore them at the end.
old_block = """    results: list[FoldResult] = []

    from .io import compute_doy_climatology, compute_qmedia"""

new_block = """    results: list[FoldResult] = []

    from .io import compute_doy_climatology, compute_qmedia

    orig_par = data.par.copy() if data.par is not None else None
    orig_par_best = data.par_best.copy() if data.par_best is not None else None"""

if old_block in content:
    content = content.replace(old_block, new_block)
else:
    print("Could not find old_block")

old_return = """    if data.gap_tolerant:
        compute_qmedia(data)
        compute_doy_climatology(data)

    return results"""

new_return = """    if data.gap_tolerant:
        compute_qmedia(data)
        compute_doy_climatology(data)

    if orig_par is not None:
        data.par[:] = orig_par[:]
    if orig_par_best is not None:
        data.par_best[:] = orig_par_best[:]

    return results"""

if old_return in content:
    content = content.replace(old_return, new_return)
else:
    print("Could not find old_return")

with open(filepath, 'w') as f:
    f.write(content)
print("Successfully patched par restore")
