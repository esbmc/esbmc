---
title: "Development Update: End of August 2026"
date: 2026-08-26T18:00:00+01:00
draft: false
tags:
  - ESBMC
  - FormalVerification
  - ModelChecking
  - OpenSource
---

A short follow-up to the [late-August update](/news/development-update-late-august-2026):
roughly 80 pull requests landed on `master` between 22 and 26 August. Here are
the ones a user will notice.

**Bitwuzla now uses its native floating-point theory.**
[#7022](https://github.com/esbmc/esbmc/pull/7022) implements the SMT
floating-point theory on the Bitwuzla backend — the default solver — so floats
are encoded with `fp.add` and friends instead of ESBMC's own bit-vector
lowering. On symbolic-input FP queries it measured 1.2–13× faster, and 303
FP-touching regression tests agree on every verdict. `--fp2bv` opts back into
the old encoding; `fmod`/`remainder` keep a bit-vector round-trip because
`fp.rem` is two orders of magnitude slower on the solver side. One known gap:
the theory cannot represent the sign of a NaN
([#7021](https://github.com/esbmc/esbmc/issues/7021)).

**A cross-run proof cache.**
[#7143](https://github.com/esbmc/esbmc/pull/7143) adds `--proof-cache <dir>`,
which keys every discharged claim on a hash of its sliced SSA cone, the build
and the options that affect what is verified, and skips the solver on a hit.
Only proved (UNSAT) claims are stored; `--proof-cache-verify` solves anyway and
reports any mismatch. The full write-up is on the new
[Proof cache](/docs/proof-cache) page.

**Verdicts and property tables that were missing.**
A plain BMC run under `--show-cex` recorded no verdict and printed
"0 properties failed" above `VERIFICATION FAILED`
([#7258](https://github.com/esbmc/esbmc/pull/7258)); `--k-induction-parallel`
skipped the per-property table — and with it the CWE rows — that sequential
`--k-induction` prints ([#7262](https://github.com/esbmc/esbmc/pull/7262)). An
unknown `--function` on a `.goto` binary now reports an error rather than
aborting ([#7263](https://github.com/esbmc/esbmc/pull/7263)), and a goto binary
produced by another tool prints in C syntax instead of crashing the
counterexample printer ([#7264](https://github.com/esbmc/esbmc/pull/7264)). The
SV-COMP wrapper was taught to read the per-property table introduced by #7064,
which had been turning ~2500 correct-false verdicts into `Unknown`
([#7250](https://github.com/esbmc/esbmc/pull/7250)).

**`--dead-code-check` reports the dead direction and fewer false advisories.**
The advisory named the guard of the *live* branch; it now names the direction
that is actually unreachable
([#7295](https://github.com/esbmc/esbmc/pull/7295)). It no longer reports
CWE-561 on `if (1)` ([#7277](https://github.com/esbmc/esbmc/pull/7277)), on
code reachable only through a call inside a constant-folded Python `assert`
([#7279](https://github.com/esbmc/esbmc/pull/7279)), or on the `IndexError` /
`KeyError` / `ZeroDivisionError` guards and exception-propagation edges the
Python frontend inserts itself
([#7307](https://github.com/esbmc/esbmc/pull/7307)).

**Pointer arithmetic below an object.**
Three stacked fixes ([#7222](https://github.com/esbmc/esbmc/pull/7222),
[#7224](https://github.com/esbmc/esbmc/pull/7224),
[#7228](https://github.com/esbmc/esbmc/pull/7228)) make `p - k` on a `void *`
move in the right direction, report a `p[-1]` underflow that a wrapping offset
used to hide, and compare pointer offsets signed so that `p >= b` is false for
`p = b - 1`. A harness that walks `for (p = end; p >= begin; p--)` over a struct
or scalar now reports out of bounds, as it already did over an array. Every
multi-byte heap write on a big-endian target was byte-reversed against the read
that stitched it back ([#7276](https://github.com/esbmc/esbmc/pull/7276)).

**C++ base subobjects under ESBMC's own layout.**
The thunk adapting a `Base*` receiver and both arms of `dynamic_cast` sized
their adjustment with clang's record layout, but ESBMC lays classes out itself,
so under the Itanium primary-base rule dispatch and member access disagreed on
which object they were touching
([#7243](https://github.com/esbmc/esbmc/pull/7243)). A derived-to-base
conversion onto a non-first base under a virtual base is now displaced, so the
base's methods, constructor and destructor address the right storage
([#7241](https://github.com/esbmc/esbmc/pull/7241),
[#7292](https://github.com/esbmc/esbmc/pull/7292)). `string_view` became a
literal type and gained its `std::string` conversion
([#7215](https://github.com/esbmc/esbmc/pull/7215),
[#7253](https://github.com/esbmc/esbmc/pull/7253)), and `timegm` is modelled
beside `mktime` ([#7273](https://github.com/esbmc/esbmc/pull/7273)).

**Python: closures, dunders and dynamic typing.**
A nested `def` reading an enclosing function's scalar is bound to a captured
cell rather than an unconstrained frame local
([#7297](https://github.com/esbmc/esbmc/pull/7297)); writing `c[0] += 1` from
a nested function mutates the enclosing list
([#7225](https://github.com/esbmc/esbmc/pull/7225),
[#7227](https://github.com/esbmc/esbmc/pull/7227)). `Callable[[A], R]`
annotations carry their signature, so a call through the value is no longer
nondet ([#7298](https://github.com/esbmc/esbmc/pull/7298)). Explicit
`__getitem__`, `__len__` and `__contains__` calls dispatch to their operators
([#7316](https://github.com/esbmc/esbmc/pull/7316)). Dynamic typing now covers
`elif` chains and `-`/`/` between two tagged operands
([#7281](https://github.com/esbmc/esbmc/pull/7281)). Call-site parameter
inference iterates to a fixpoint, fixing a spurious CWE-469 on a comparison of
two unannotated parameters ([#7257](https://github.com/esbmc/esbmc/pull/7257)),
and a comprehension target that shadows an outer variable gets its own name
([#7235](https://github.com/esbmc/esbmc/pull/7235)).

**Python: `sorted()` and friends stop lying.**
`sorted()`, `min()` and `max()` with a `key=` the preprocessor cannot fold used
to drop the key silently and answer in natural order; they are now refused
with a named error ([#7313](https://github.com/esbmc/esbmc/pull/7313)).
`sorted()`, `reversed()` and `list()` keep their argument's element type, so a
list of tuples still unpacks after sorting
([#7310](https://github.com/esbmc/esbmc/pull/7310)), and slicing a dict view
keeps its element types too
([#7318](https://github.com/esbmc/esbmc/pull/7318)). A subscript on the
right-hand side of an assignment emits one bounds check rather than two — a 20%
drop in symex assignments on a double-subscript loop
([#7309](https://github.com/esbmc/esbmc/pull/7309)).

**NumPy views.**
Row views (`a[0]`, `a[-1]`), column views (`a[:, j]`), strided and reversed 1-D
slices (`a[::2]`, `a[::-1]`), `np.diagonal`, `np.ravel` and `a.flat[i]` are
now pointer-backed views onto the base buffer, so writes through either side
are observed by the other and `len`, `.shape` and `.ndim` report the view's own
extent ([#7193](https://github.com/esbmc/esbmc/pull/7193),
[#7261](https://github.com/esbmc/esbmc/pull/7261)). `np.trace` and
`np.fill_diagonal` reuse the same offset arithmetic.

The website documentation has been updated to match. As always, the full list is
in the [commit history](https://github.com/esbmc/esbmc/commits/master/) — and if
you hit a bug or a missing feature, we would love an
[issue report](https://github.com/esbmc/esbmc/issues).
