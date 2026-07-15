import sys

filepath = 'pyair2stream/cross_validation.py'
with open(filepath, 'r') as f:
    content = f.read()

old_inline_restore = """            # Restore forcing variables (Tair, Q) before forward simulation
            # so the model integrates through the held-out window properly
            if data.gap_tolerant:
                data.Tair[idx] = orig_tair
                data.Q[idx] = orig_q
                if w_idx.size > 0:
                    data.Tair[w_idx] = orig_w_tair
                    data.Q[w_idx] = orig_w_q"""

new_inline_restore = """            # Restore forcing variables (Tair, Q) before forward simulation
            # so the model integrates through the held-out window properly.
            # (Note: _restore_fold also restores forcing, but doing it here is required
            # for the forward simulation. _restore_fold is idempotent.)
            if data.gap_tolerant:
                data.Tair[idx] = orig_tair
                data.Q[idx] = orig_q
                if w_idx.size > 0:
                    data.Tair[w_idx] = orig_w_tair
                    data.Q[w_idx] = orig_w_q"""

if old_inline_restore in content:
    content = content.replace(old_inline_restore, new_inline_restore)
    with open(filepath, 'w') as f:
        f.write(content)
    print("Successfully patched inline restore comment")
else:
    print("Could not find old_inline_restore")
