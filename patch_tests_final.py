import sys

filepath = 'tests/test_cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

# Fix dummy_data initialization for tests
import re
content = re.sub(r'def test_build_folds_single_year\(dummy_data\):\n',
r'def test_build_folds_single_year(dummy_data):\n    dummy_data.Twat_obs = np.ones(dummy_data.n_tot)\n', content)

content = re.sub(r'def test_build_folds_n_years\(dummy_data\):\n',
r'def test_build_folds_n_years(dummy_data):\n    dummy_data.Twat_obs = np.ones(dummy_data.n_tot)\n', content)

# Fix unpack issue for _mask_fold in test_cross_validation_leak_prevention
old_unpack = "orig_twat, orig_tair, orig_q = _mask_fold(dummy_data, fold_idx)"
new_unpack = "orig_twat, orig_tair, orig_q, w_idx, orig_w_twat, orig_w_tair, orig_w_q = _mask_fold(dummy_data, fold_idx)"
if old_unpack in content:
    content = content.replace(old_unpack, new_unpack)

old_restore_call = "_restore_fold(dummy_data, fold_idx, orig_twat, orig_tair, orig_q)"
new_restore_call = "_restore_fold(dummy_data, fold_idx, orig_twat, orig_tair, orig_q, w_idx, orig_w_twat, orig_w_tair, orig_w_q)"
if old_restore_call in content:
    content = content.replace(old_restore_call, new_restore_call)

with open(filepath, 'w') as f:
    f.write(content)
print("Successfully patched tests")
