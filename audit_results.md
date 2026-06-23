# Audit of python_port_feasibility.md Claims

I have reviewed the `python_port_feasibility.md` file against the actual `air2stream` Fortran codebase and verified the following 8 claims.

**1. PSO global best initialised from wrong array**
*Claim:* Line 71: `CALL best(fit, k, foptim)` passes `fit` (still all zeros) instead of `fitbest` (populated just above). `gbest` is set from a random particle rather than the best-performing one, corrupting the first iteration's global attractor.
*Verification Status:* **True**. In `AIR2STREAM_RUNMODE.f90` lines 66-70 populate `fitbest`. Then, line 71 reads: `CALL best(fit,k,foptim)`. The `fit` array has not been populated yet, meaning `gbest` will be initialized incorrectly.

**2. PSO convergence check is permanently dead**
*Claim:* Line 145: `IF (norm .lt. 0.0)` — `norm` is a square root and can never be negative. The early-exit mechanism never fires regardless of convergence.
*Verification Status:* **True**. In `AIR2STREAM_RUNMODE.f90` lines 144-145, `norm=SQRT(norm)` calculates the square root (which is non-negative), followed immediately by `IF (norm .lt. 0.0) THEN`. This block is unreachable.

**3. Version 8 parameters never zeroed**
*Claim:* Lines 81–86: the block labelled `!air2stream with 8 parameters` is guarded by `IF (version == 4)`. When `version == 8`, parameters 5–8 are not zeroed, but the block intended to do so is skipped.
*Verification Status:* **True**. In `AIR2STREAM_READ.f90` lines 81-86, there is a block:
```fortran
IF (version == 4) THEN                                      !air2stream with 8 parameters
     parmin(5)=0;    parmax(5)=0;    flag_par(5)=.false.;
     parmin(6)=0;    parmax(6)=0;    flag_par(6)=.false.;
     parmin(7)=0;    parmax(7)=0;    flag_par(7)=.false.;
     parmin(8)=0;    parmax(8)=0;    flag_par(8)=.false.;
END IF
```
This comes after the actual `version == 4` block and is clearly meant for `version == 8` but incorrectly checks `version == 4`.

**4. LH file handle never closed**
*Claim:* Line 239: `! CLOSE(10)` is commented out. In LATHYP mode the binary output file is never explicitly closed, risking incomplete/unflushed output.
*Verification Status:* **True**. In `AIR2STREAM_RUNMODE.f90` line 239 (within `SUBROUTINE LH_mode`), the code has `!    CLOSE(10)`, leaving the file unclosed.

**5. `funcobj` requires `I_pos`/`I_inf` indirection — not directly vectorisable**
*Claim:* Before computing NSE/KGE/RMS, the subroutine averages `Twat_mod` over variable-width aggregation windows using the `I_pos` index array and `I_inf` range matrix. A direct `np.mean((obs - mod)**2)` is incorrect.
*Verification Status:* **True**. In `AIR2STREAM_SUBROUTINES.f90` lines 150-157, `Twat_mod_agg` is calculated using nested loops over `I_inf(i,1)` to `I_inf(i,2)` indexing `Twat_mod(I_pos(j))`, making direct vectorization of `Twat_mod` non-trivial without recreating this aggregation correctly.

**6. `Tice_cover` floor clamp not mentioned**
*Claim:* Line 85 clamps every output step: `Twat_mod[j+1] = max(Twat_mod[j+1], Tice_cover)`. Omitting this in Python will silently produce wrong temperatures whenever the model goes below the ice threshold.
*Verification Status:* **True**. In `AIR2STREAM_SUBROUTINES.f90` line 85, after all numerical integrations, `Twat_mod(j+1)=MAX(Twat_mod(j+1),Tice_cover);` correctly enforces the ice threshold.

**7. Initial PSO evaluations excluded from dotty-plot output**
*Claim:* The `WRITE(10)` block (line 106) is only reached during the main `n_run` iteration loop. The initial particle evaluations (lines 66–70) are never written to the binary output file.
*Verification Status:* **True**. In `AIR2STREAM_RUNMODE.f90`, the initial run of parameters on lines 66-70 sets `fitbest` but does not `WRITE(10)`. Writing only occurs in the `DO i=1,n_run` loop starting at line 76 (write on line 106).

**8. PSO random re-seeding per iteration makes results non-reproducible**
*Claim:* Line 77 calls `random_seed()` inside every iteration, re-seeding from system time. Results cannot be reproduced even with the same input. Python's `np.random` does not replicate this behaviour by default.
*Verification Status:* **True**. In `AIR2STREAM_RUNMODE.f90` line 77, right inside `DO i=1,n_run`, `CALL random_seed()` is executed, reseeding the RNG on every outer iteration, inherently breaking reproducibility.
