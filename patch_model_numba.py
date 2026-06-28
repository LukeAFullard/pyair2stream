with open("pyair2stream/model_numba.py", "r") as f:
    lines = f.read()

import re

search_block = """
    if valid_n_dat == 0:
        return -999.0, Twat_mod_agg

    if fun_obj_type == 0: # NSE
        TSS = 0.0
        for k in range(valid_n_dat):
            TSS += (agg_mod_valid[k] - agg_obs_valid[k]) ** 2

        if TSS_obs == 0.0:
            if TSS == 0.0:
                return 1.0, Twat_mod_agg
            else:
                return -1e30, Twat_mod_agg
        return 1.0 - TSS / TSS_obs, Twat_mod_agg
"""

replace_block = """
    if valid_n_dat == 0:
        return -999.0, Twat_mod_agg, -999.0, -999.0, -999.0

    # Calculate NSE, R2, MAE for all evaluations
    sum_mod = 0.0
    sum_abs_err = 0.0
    TSS = 0.0
    for k in range(valid_n_dat):
        sum_mod += agg_mod_valid[k]
        sum_abs_err += abs(agg_mod_valid[k] - agg_obs_valid[k])
        TSS += (agg_mod_valid[k] - agg_obs_valid[k]) ** 2

    mean_mod = sum_mod / valid_n_dat
    mae = sum_abs_err / valid_n_dat

    if TSS_obs == 0.0:
        if TSS == 0.0:
            nse = 1.0
        else:
            nse = -1e30
    else:
        nse = 1.0 - TSS / TSS_obs

    # R2
    TSS_mod = 0.0
    covar_mod = 0.0
    for k in range(valid_n_dat):
        TSS_mod += (agg_mod_valid[k] - mean_mod) ** 2
        covar_mod += (agg_mod_valid[k] - mean_mod) * (agg_obs_valid[k] - mean_obs)

    if TSS_mod == 0.0 or TSS_obs == 0.0:
        r2 = -999.0
    else:
        r = covar_mod / np.sqrt(TSS_mod * TSS_obs)
        r2 = r ** 2

    if fun_obj_type == 0: # NSE
        return nse, Twat_mod_agg, nse, r2, mae
"""

lines = lines.replace(search_block, replace_block)

search_block2 = """
        if std_obs == 0 or std_mod == 0:
            return -999.0, Twat_mod_agg

        return 1.0 - np.sqrt((std_mod / std_obs - 1.0)**2 + (mean_mod / mean_obs - 1.0)**2 + (covar_mod / (std_mod * std_obs) - 1.0)**2), Twat_mod_agg

    elif fun_obj_type == 2: # RMS
        TSS = 0.0
        for k in range(valid_n_dat):
            TSS += (agg_mod_valid[k] - agg_obs_valid[k]) ** 2
        return -np.sqrt(TSS / valid_n_dat), Twat_mod_agg

    return -999.0, Twat_mod_agg
"""

replace_block2 = """
        if std_obs == 0 or std_mod == 0:
            return -999.0, Twat_mod_agg, nse, r2, mae

        kge = 1.0 - np.sqrt((std_mod / std_obs - 1.0)**2 + (mean_mod / mean_obs - 1.0)**2 + (covar_mod / (std_mod * std_obs) - 1.0)**2)
        return kge, Twat_mod_agg, nse, r2, mae

    elif fun_obj_type == 2: # RMS
        TSS = 0.0
        for k in range(valid_n_dat):
            TSS += (agg_mod_valid[k] - agg_obs_valid[k]) ** 2
        rms = -np.sqrt(TSS / valid_n_dat)
        return rms, Twat_mod_agg, nse, r2, mae

    return -999.0, Twat_mod_agg, nse, r2, mae
"""

lines = lines.replace(search_block2, replace_block2)

with open("pyair2stream/model_numba.py", "w") as f:
    f.write(lines)
