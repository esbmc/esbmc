# Bug fixing

Protocols for fixing memory-safety / undefined-behaviour bugs and for landing
multi-patch fixes.

## Regression Tests for Memory/UB Bugs

When fixing a memory-safety or undefined-behaviour bug in C/C++ code:

1. Before applying the fix, write a regression test that reproduces the bug
   under sanitizers (ASan, UBSan, or MSan as appropriate; TSan for data races).
2. Compile and run the regression test, and confirm it fails on the unfixed code
   — either via a clear sanitizer diagnostic or by tripping an embedded `assert`
   — so the failure mode is reproducible end-to-end, not just inferred.
3. Apply the fix and re-run the compiled test; confirm it now passes cleanly
   (assertion holds and no sanitizer diagnostic).
4. Skip this step for pure logic bugs, build/config issues, or non-C/C++ work —
   sanitizers do not apply.

If sanitizers do not reproduce the bug (e.g. timing-dependent races,
allocator-dependent use-after-free, MSan without instrumented dependencies,
optimisation-dependent UB, or input coverage gaps):

1. Try a different sanitizer (ASan ↔ TSan ↔ MSan ↔ UBSan) and vary build flags
   (`-O0` vs `-O2`, `_GLIBCXX_DEBUG`, `MALLOC_PERTURB_`,
   `ASAN_OPTIONS=detect_stack_use_after_return=1`).
2. If still not reproducible under sanitizers, fall back to ESBMC
   (`esbmc-verifier` agent) to obtain a deterministic witness.
3. As a last resort, write a regression test that reproduces the observable
   symptom (wrong output, assertion, crash) without relying on a sanitizer
   diagnostic, and note in the commit message why sanitizer-based reproduction
   was not feasible.

## Incremental Patch Testing

When a fix involves multiple patches (e.g. N1, N2), apply and test them one at a
time:

1. Apply patch N1, then run the relevant tests to check whether the bug is
   fixed.
2. If fixed, stop — do not apply further patches.
3. If not fixed, apply patch N2 and test again. Repeat until the bug is resolved
   or all patches are exhausted.
4. Do not apply all patches at once before testing.
