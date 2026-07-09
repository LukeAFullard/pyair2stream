# Fortran reference source: provenance and patch notice

`fortran/upstream/` is a git submodule pointing at the original `air2stream`
Fortran implementation:

- Repository: https://github.com/spiccolroaz/air2stream
- Authors: Sebastiano Piccolroaz and Marco Toffolon (University of Trento)
- License: [Creative Commons Attribution-ShareAlike 3.0](https://github.com/spiccolroaz/air2stream/blob/main/LICENSE)
- Pinned commit: see `.gitmodules` / `git -C fortran/upstream rev-parse HEAD`

It is used **only** by the test suite, to numerically validate
`pyair2stream`'s Python/Numba reimplementation against the original Fortran
model (see `tests/fortran_runner.py`, `tests/test_golden.py`,
`tests/test_physics_golden.py`). It is not required to run `pyair2stream`
itself.

## Why a patch is needed

The upstream source was written for Intel Fortran on Windows and uses a
couple of Intel-specific extensions that `gfortran` does not implement:

| File | Upstream (Intel Fortran) | Patched (gfortran) | Reason |
|---|---|---|---|
| `AIR2STREAM_READ.f90` | `USE ifport` / `makedirqq(folder)` | commented out / `EXECUTE_COMMAND_LINE('mkdir -p ...')` | `ifport` and `makedirqq` are Intel-only |
| `AIR2STREAM_READ.f90` | `WRITE(2,'(<n_par>(F10.5,1x))')` | `WRITE(2,*)` | angle-bracket repeat-count format specifier is an Intel extension |
| `AIR2STREAM_RUNMODE.f90` | `form='binary'` | `form='unformatted',access='stream'` | `form='binary'` is an Intel extension; the ISO Fortran equivalent is stream access |
| `AIR2STREAM_SUBROUTINES.f90` | `WRITE(11,'(<n_par>(f10.6,1x))')` | `WRITE(11,*)` | same angle-bracket format issue |

None of these changes affect the governing equations, numerical integrators,
or model physics — only file I/O mechanics needed to compile with a
different (non-Intel) compiler. The full, human-readable diff is in
[`gfortran-portability.patch`](gfortran-portability.patch), applied
automatically by `tests/fortran_runner.py` at test time; the submodule
checkout itself is never modified.

This satisfies the CC BY-SA "clearly label changes" requirement for
adaptations of the licensed work: the changes are isolated in a single patch
file, documented here, and never presented as the original.

## Updating the pinned commit

If you want to track a newer upstream commit:

```bash
cd fortran/upstream
git fetch
git checkout <new-commit-sha>
cd ../..
git add fortran/upstream
```

Then re-run the test suite. If upstream has changed any of the four patched
regions, `patch` will fail loudly with a clear error (see
`_build_fortran_binary` in `tests/fortran_runner.py`) rather than silently
compiling something different — at that point the patch context needs a
manual update.
