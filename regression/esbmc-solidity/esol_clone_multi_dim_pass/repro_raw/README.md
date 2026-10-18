# Multi-dim clone KNOWNBUG — raw-C/C++ repros

Both programs mirror the same shape as the parent `contract.sol`
(multi-dim outer array + fresh outer alloc_array + per-slot arrcpy of
inner rows + read through nested pointer).  Both PASS in plain C/C++
mode, proving that the backend (goto-symex + any SMT solver in our
support set) handles the pattern correctly.

```
esbmc raw_u256_c.c   --unwind 4 --no-unwinding-assertions --no-standard-checks --force-malloc-success --bitwuzla
esbmc raw_u256_cpp.cpp --unwind 4 --no-unwinding-assertions --no-standard-checks --force-malloc-success --bitwuzla
esbmc raw_u256_cpp_sol_pattern.cpp --unwind 3 --no-unwinding-assertions --no-standard-checks --force-malloc-success --bitwuzla
```

`raw_u256_cpp_sol_pattern.cpp` was added 2026-04-20 to test the
hypothesis that Solidity's `cpp_new + temp-stack-ctor + *new_ptr = tmp`
emission pattern itself is the bug.  It reproduces that pattern
byte-identically in raw C++ (no explicit `ctor(base)` on the heap
pointer, direct struct ASSIGN for both `*new_ptr = tmp` and `*c = *base`)
and still PASSES.  So the emission pattern is NOT the root cause —
something else about the Solidity frontend's contract-struct setup
triggers the failure.  See contract.sol header comment for disproven
hypotheses and remaining suspects.

The Solidity version of the same shape FAILS under `--bound --cvc5`
(and `--bitwuzla` where it hits a different symptom — null inner
pointer in `_ESBMC_element_null_check` inside arrcpy).  The regression
test.desc here is KNOWNBUG; it flips to CORE once the Solidity-frontend
specific trigger is resolved.
