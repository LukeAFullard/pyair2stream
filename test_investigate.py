# Ah! Wait!
# In the original log:
# 1  2006       NaN       NaN           365.0
#
# So 2006 had 365.0 held out observations, but NSE was NaN!
# Why would NSE be NaN?
# Because the SIMULATED values were all -999.0!
# Why were they -999.0? Because detect_segments dropped the segment for 2006 for some reason!
#
# But wait, with my fix, 2006 NSE is NO LONGER NaN! It is -5.74!
# So my fix FIXED 2006!
# Why did it fix 2006?
# Look at my change to redundant forcing restore:
#
# BEFORE:
#            data.Tair[idx] = orig_tair
#            data.Q[idx] = orig_q
#            if data.gap_tolerant:
#                data.segments = None
#                detect_segments(data)
#
# Wait, before, it ALWAYS restored data.Tair[idx] = orig_tair and Q[idx] = orig_q.
# Then it called detect_segments(data).
# But wait, it did this BEFORE calling aggregation(data) and statis(data)!
#
# AFTER MY FIX:
#            if data.gap_tolerant:
#                data.Tair[idx] = orig_tair
#                data.Q[idx] = orig_q
#
# I encapsulated it behind `if data.gap_tolerant`. Since gap_tolerant is TRUE for DAV, this is identical!
# Wait! Did I change something else?
# Let's look at the `_restore_fold` change.
# BEFORE:
# _restore_fold(data, idx, orig_twat, orig_tair, orig_q)
# it restored Tair and Q explicitly.
#
# Did I accidentally break something?
# No, let's look at the actual parameters calibrated in the previous log vs my log!
#
# PREVIOUS LOG:
# 0    2005     2005-01-01   2005-12-31  ...  7.915363  0.027144  0.974313
# MY LOG:
# 0    2005     2005-01-01   2005-12-31  ...  8.174009  0.712901  1.552033
#
# THE CALIBRATED PARAMETERS ARE DIFFERENT!
# If the parameters are different, then the NSE is different.
# Why are the calibrated parameters different?
# Because the OPTIMIZER converged to a different spot!
# Why did the optimizer converge to a different spot?
# Because I copied and restored `data.par` and `data.par_best`!
#
# User point 6:
# "data.par / data.par_best left at last fold's values after CV... harmless since main.py returns immediately after CV... but a latent trap"
# Wait! If `data.par_best` is NOT restored between folds...
# Does that mean Fold N uses Fold N-1's `par_best` as its starting point / initial particle in PSO/DE???
# YES! The optimizers in `pyair2stream` likely seed their initial population using the current `data.par`!
# By restoring `data.par` to its pre-CV state at the VERY END of `run_leave_one_year_out_cv`...
# WAIT! Did I restore `data.par` *inside* the loop, or at the *end* of the function?
# Let's check my code for step 6!
