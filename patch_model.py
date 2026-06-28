with open("pyair2stream/model.py", "r") as f:
    lines = f.read()

search_block = """
    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=np.bool_)

    ind, Twat_mod_agg = fast_funcobj(
        data.n_dat, data.n_tot, data.I_inf, data.I_pos, data.Twat_mod, data.Twat_obs_agg,
        eval_mask, fun_obj_type, data.mean_obs, data.TSS_obs, data.std_obs
    )

    data.Twat_mod_agg = Twat_mod_agg
"""

replace_block = """
    eval_mask = data.eval_mask if data.eval_mask is not None else np.ones(data.n_tot, dtype=np.bool_)

    ind, Twat_mod_agg, current_nse, current_r2, current_mae = fast_funcobj(
        data.n_dat, data.n_tot, data.I_inf, data.I_pos, data.Twat_mod, data.Twat_obs_agg,
        eval_mask, fun_obj_type, data.mean_obs, data.TSS_obs, data.std_obs
    )

    data.Twat_mod_agg = Twat_mod_agg
    data.current_nse = current_nse
    data.current_r2 = current_r2
    data.current_mae = current_mae
"""

lines = lines.replace(search_block, replace_block)

with open("pyair2stream/model.py", "w") as f:
    f.write(lines)
