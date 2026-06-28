with open("pyair2stream/model_numba.py", "r") as f:
    lines = f.read()

search_block = """
    elif fun_obj_type == 1: # KGE
        if valid_n_dat < 2:
            return -999.0, Twat_mod_agg
"""
replace_block = """
    elif fun_obj_type == 1: # KGE
        if valid_n_dat < 2:
            return -999.0, Twat_mod_agg, nse, r2, mae
"""

lines = lines.replace(search_block, replace_block)

with open("pyair2stream/model_numba.py", "w") as f:
    f.write(lines)
