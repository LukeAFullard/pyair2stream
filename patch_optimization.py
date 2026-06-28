with open("pyair2stream/optimization.py", "r") as f:
    lines = f.read()

import re

# PSO_mode
pso_search = """
        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(p_vals) + [eff_index]
            history.append(row)
"""
pso_replace = """
        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(p_vals) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
            history.append(row)
"""
lines = lines.replace(pso_search, pso_replace)

pso_search2 = """
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index"])
"""
pso_replace2 = """
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index", "NSE", "R2", "MAE"])
"""
lines = lines.replace(pso_search2, pso_replace2)


# LH_mode
lh_search = """
        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(p_vals) + [eff_index]
            history.append(row)
"""
lh_replace = """
        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(p_vals) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
            history.append(row)
"""
lines = lines.replace(lh_search, lh_replace)

lh_search2 = """
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index"])
"""
lh_replace2 = """
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index", "NSE", "R2", "MAE"])
"""
lines = lines.replace(lh_search2, lh_replace2)

# DE_mode
de_search = """
        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(p_vals) + [eff_index]
            history.append(row)
"""
de_replace = """
        if not np.isnan(eff_index) and eff_index >= data.mineff_index:
            row = list(p_vals) + [eff_index, data.current_nse, data.current_r2, data.current_mae]
            history.append(row)
"""
lines = lines.replace(de_search, de_replace)

de_search2 = """
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index"])
"""
de_replace2 = """
    df = pd.DataFrame(history, columns=[f"par_{j+1}" for j in range(n_par)] + ["eff_index", "NSE", "R2", "MAE"])
"""
lines = lines.replace(de_search2, de_replace2)

with open("pyair2stream/optimization.py", "w") as f:
    f.write(lines)
