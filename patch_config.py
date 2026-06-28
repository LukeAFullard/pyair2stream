with open("pyair2stream/config.py", "r") as f:
    lines = f.read()

search_block = """
    # Allocatable arrays - Float (np.float64)
"""

replace_block = """
    # Tracking metrics across evaluations
    current_nse: np.float64 = np.float64(-999.0)
    current_r2: np.float64 = np.float64(-999.0)
    current_mae: np.float64 = np.float64(-999.0)

    # Allocatable arrays - Float (np.float64)
"""

lines = lines.replace(search_block, replace_block)

with open("pyair2stream/config.py", "w") as f:
    f.write(lines)
