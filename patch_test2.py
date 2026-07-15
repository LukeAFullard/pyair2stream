import sys

filepath = 'tests/test_cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

old_str = "dummy_data.mod_num = \"RK4\""
new_str = "dummy_data.mod_num = \"RK4\"\n    dummy_data.fun_obj = \"NSE\"\n    dummy_data.objective_function = \"NSE\""

if old_str in content:
    content = content.replace(old_str, new_str)
    with open(filepath, 'w') as f:
        f.write(content)
    print("Successfully updated test_run_leave_one_year_out_cv_gap_tolerant")
else:
    print("Could not find old_str")
