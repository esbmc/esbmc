---
title: "Development Update: Early September 2026"
date: 2026-09-04T09:00:00+01:00
draft: false
tags:
  - ESBMC
  - FormalVerification
  - ModelChecking
  - OpenSource
---

Following the [end-of-August update](/news/development-update-end-of-august-2026),
about 120 pull requests landed on `master` between 26 August and 4 September.
ESBMC is now at **v8.5** ([#7379](https://github.com/esbmc/esbmc/pull/7379)).
Here are the changes a user will notice.

**A Linux ARM64 release asset.**
[#7229](https://github.com/esbmc/esbmc/pull/7229) adds `esbmc-linux-armv8.zip`
as a fourth canonical release asset, built statically against LLVM 22, so it
runs on any aarch64 Linux userspace. An armv8 leg now runs per PR as well. Four
features are unavailable on that build — `--32`, Solidity, CVC5 and
`--goto-contractor` — and interval analysis differs from x86_64; the
[Setup](/docs/setup#arm64-builds) page lists each with its tracking issue.

**`--loop-invariant-check` stops reporting proofs it never made.**
The mode replaced a loop with `havoc; assume(I)` and then dropped every claim
after the loop, so programs with real bugs verified. A sweep of the guard and
loop shapes found three causes, each fixed
([#7482](https://github.com/esbmc/esbmc/pull/7482)): the havoc landed after a
guard's own side effects, a `do`-`while` back edge was cut with `assume(false)`,
and a loop whose only writes go through a pointer left the modified-variable
analysis with nothing to havoc. Six loop shapes that reported `VERIFICATION
SUCCESSFUL` on a false post-loop assertion now reach a verdict instead — three
more came out of review — and a loop the schema cannot havoc soundly is left to
the unwinder rather than proved. Which verdict depends on the next change.

**A weak invariant is `UNKNOWN`, not `FAILED`.**
Every claim downstream of that havoc is checked against an over-approximation,
so a correct but weak invariant produced a counterexample that no execution can
reach. Such a claim is now reported as unknown, with the reason attached
([#7491](https://github.com/esbmc/esbmc/pull/7491)):

```
  PASSED   [main.assertion.1]  line 11  loop invariant base case
  PASSED   [main.assertion.2]  line 11  loop invariant inductive step
  UNKNOWN  [main.assertion.3]  line 16  assertion s == 3 (loop invariant too
           weak to prove this claim: the counterexample is against the havoc
           abstraction, not a reachable state of the program)

** 0 of 3 properties failed, 2 passed, 1 unknown
VERIFICATION UNKNOWN
```

`FAILED` is kept for a claim ahead of every havoc, for the invariant's own
inductive step and assigns-compliance check, for a declined loop, and for an
outermost loop's base case — an inner loop's base case sits downstream of the
outer havoc and is downgraded with the rest. The consequence is worth stating:
this mode can no longer report a bug after an annotated loop; plain BMC and
`--k-induction` remain the modes that refute.

**Andersen points-to replaces the GOTO-level value-set analysis.**
[#6623](https://github.com/esbmc/esbmc/pull/6623) moves GCSE and k-induction's
pointer-array-write resolver onto an inclusion-based whole-program points-to
analysis that abstains per query instead of disabling itself wholesale. Making
those consumers actually run surfaced four latent bugs, the worst of which was
not GCSE-specific: a pass that inserts instructions handed loop analysis stale
location numbers, so a forward branch could be misread as a back edge and
rewritten to `assume(!guard)`. Under `--gcse` that silently turned failing
programs into proofs; on ReachSafety-ECA it accounted for 48 wrong answers,
now zero.

**Two more soundness fixes.** The LD front end collected networks only from
ladder bodies, so a POU written in `ST`, `FBD`, `SFC` or `IL` left the scan cycle
empty and every property held vacuously; such a body is now rejected
([#7365](https://github.com/esbmc/esbmc/pull/7365)). And `__int128` escaped the
usual arithmetic conversions on the default clang-c path, while its rank sat
above pointers and every floating type, so `pointer + __int128` and
`float + __int128` flattened the wrong operand
([#7343](https://github.com/esbmc/esbmc/pull/7343)).

**Loops that never terminated under default flags now decide.**
A census of trip-count spellings found clusters where the bound never folded, so
the loop unwound forever: a bound spelled `&a[4]` rather than `a + 4`
([#7346](https://github.com/esbmc/esbmc/pull/7346),
[#7391](https://github.com/esbmc/esbmc/pull/7391),
[#7394](https://github.com/esbmc/esbmc/pull/7394)), a descending pointer walk
where the ascending twin folded ([#7395](https://github.com/esbmc/esbmc/pull/7395)),
`(char *)&a[4] - (char *)&a[0]` and `&p.b - &p.a`
([#7392](https://github.com/esbmc/esbmc/pull/7392)), a bound from `offsetof`
([#7393](https://github.com/esbmc/esbmc/pull/7393)), `sizeof(a) / sizeof(a[0])`
under `--no-simplify` ([#7400](https://github.com/esbmc/esbmc/pull/7400)), a null
guard on the address of an object ([#7441](https://github.com/esbmc/esbmc/pull/7441)),
a struct written through a nested member
([#7463](https://github.com/esbmc/esbmc/pull/7463)), a constant element of a
multi-dimensional array ([#7341](https://github.com/esbmc/esbmc/pull/7341)), and a
union carrying a nondeterministic symbol or read across same-width members
([#7446](https://github.com/esbmc/esbmc/pull/7446),
[#7447](https://github.com/esbmc/esbmc/pull/7447)). A callee that re-checks a
condition the caller's path already decided no longer forks both sides
([#7450](https://github.com/esbmc/esbmc/pull/7450)), and an index a struct field
carries stays on the constant-offset path instead of degenerating to a
whole-object `byte_extract` — 242 s of symex on the issue's reproducer becomes
0.001 s ([#7317](https://github.com/esbmc/esbmc/pull/7317)).

**Operational-model cost that no longer grows with `--unwind`.**
A screening test — hold the program constant and raise the bound — found six
models paying for paths that cannot execute. `len()` now folds
([#7440](https://github.com/esbmc/esbmc/pull/7440)), `list.extend()` and list
slicing copy at the frontend's constant element width
([#7435](https://github.com/esbmc/esbmc/pull/7435),
[#7436](https://github.com/esbmc/esbmc/pull/7436)), `list.remove()` splits its
search from its shift ([#7361](https://github.com/esbmc/esbmc/pull/7361)),
`list.sort()` dispatches on the type flag rather than a field read
([#7432](https://github.com/esbmc/esbmc/pull/7432)), `std::vector::reserve()`
reallocates in place so `size()` and `capacity()` stay decidable
([#7437](https://github.com/esbmc/esbmc/pull/7437)), and a `memcpy` whose length
is not constant is bounded by the operands' own widths instead of falling into a
byte loop ([#7438](https://github.com/esbmc/esbmc/pull/7438)). `str.replace()`,
`find`, `rfind`, `index` and `rindex` constant-fold on constant operands, which
also removed a spurious `VERIFICATION FAILED`
([#7373](https://github.com/esbmc/esbmc/pull/7373),
[#7374](https://github.com/esbmc/esbmc/pull/7374),
[#7375](https://github.com/esbmc/esbmc/pull/7375)).

**Python: `key=`, dynamic typing and `unittest`.**
`min()` and `max()` with a `key=` the preprocessor cannot fold now lower to an
explicit scan, so the key is really applied over a list with symbolic elements
([#7314](https://github.com/esbmc/esbmc/pull/7314)); `sorted()` folds a key
spelled as a module-level function or `d.__getitem__`
([#7348](https://github.com/esbmc/esbmc/pull/7348)) and is lowered when it is a
`for` loop's iterable ([#7322](https://github.com/esbmc/esbmc/pull/7322),
[#7368](https://github.com/esbmc/esbmc/pull/7368)); shapes neither route can
lower keep the refusal rather than dropping the key
([#7358](https://github.com/esbmc/esbmc/pull/7358)). Dynamic typing gained
addition between two tagged scalars
([#7329](https://github.com/esbmc/esbmc/pull/7329)) and ordered comparisons
against a literal and between two tagged operands, with `True == 1` now holding
([#7364](https://github.com/esbmc/esbmc/pull/7364),
[#7513](https://github.com/esbmc/esbmc/pull/7513)). `unittest.main()` runs the
tests it discovers, in CPython's order, instead of being a no-op that reached
verification with zero claims ([#7390](https://github.com/esbmc/esbmc/pull/7390)),
and a base class bound by `from module import Base` resolves its inherited
methods ([#7396](https://github.com/esbmc/esbmc/pull/7396)). Also: a lambda
parameter is typed from its call site
([#7385](https://github.com/esbmc/esbmc/pull/7385)), a function name in a list
literal decays to a pointer rather than dumping core
([#7387](https://github.com/esbmc/esbmc/pull/7387)), a string method's result is
no longer sized by the call's first argument
([#7382](https://github.com/esbmc/esbmc/pull/7382)), a nested-list subscript stays
an lvalue ([#7367](https://github.com/esbmc/esbmc/pull/7367)), and a tuple element
type crosses the call boundary
([#7370](https://github.com/esbmc/esbmc/pull/7370)). A new consensus suite runs
ESBMC over the Ethereum specification functions `ethcheck` checks
([#7444](https://github.com/esbmc/esbmc/pull/7444)); it immediately caught two
crashes ([#7471](https://github.com/esbmc/esbmc/pull/7471)).

**NumPy: descriptors and returned arrays.**
A fixed-shape view now carries its own shape and strides, so 2-D transpose,
`swapaxes`/`moveaxis`, contiguous reshape, squeeze/expand_dims and a read-only
`broadcast_to` alias the base buffer, iterate with `np.nditer`, and materialise
through `np.copy` or `tolist()` ([#7323](https://github.com/esbmc/esbmc/pull/7323)). A user function
may return an array — a constructor result, a parameter, a view, or a descriptor
call over a parameter — and the remaining `ndarray` methods (`tolist`, `any`,
`all`, the reductions, `argmin`/`argmax`, `sort`, `argsort`, `searchsorted`)
accept a concrete variable rather than only an inline literal
([#7386](https://github.com/esbmc/esbmc/pull/7386)).

**C++ operational models.**
`std::unordered_multimap` ([#7334](https://github.com/esbmc/esbmc/pull/7334)),
`std::filesystem::directory_iterator` with the `[fs.path.decompose]` members
([#7350](https://github.com/esbmc/esbmc/pull/7350)), the container-level
relational operators for `list`, `set`, `multiset`, `map` and `multimap`
([#7340](https://github.com/esbmc/esbmc/pull/7340)), the `Allocator` template
parameter on `list` and `deque`
([#7492](https://github.com/esbmc/esbmc/pull/7492)), `deque::back()` returning a
reference ([#7330](https://github.com/esbmc/esbmc/pull/7330)), the missing
`[meta.trans]` aliases and `remove_all_extents`
([#7337](https://github.com/esbmc/esbmc/pull/7337)), and the remaining C99 math
functions in namespace `std` ([#7339](https://github.com/esbmc/esbmc/pull/7339)).
`list`'s iterator moved to namespace scope so argument-dependent lookup reaches
the element type, as libc++ spells it
([#7342](https://github.com/esbmc/esbmc/pull/7342)).

**Contracts.**
`__ESBMC_old(ptr[j])` under a quantifier works on a bare `int r[N]` parameter
when the extent comes from the pointer's own `__ESBMC_is_fresh` clause, which is
what an element-wise array postcondition needs
([#7255](https://github.com/esbmc/esbmc/pull/7255)). An index in an `assigns`
target is read in the pre-state, so the ring-buffer idiom complies and the
off-by-one write is caught — both verdicts were inverted before
([#7206](https://github.com/esbmc/esbmc/pull/7206)). A replace-side
`__ESBMC_is_fresh` is decided by which object the argument names rather than how
the call site spells it, and the extent check is no longer skipped for stack
storage ([#7465](https://github.com/esbmc/esbmc/pull/7465)).

**Diagnostics.**
A run that proved a property with `--no-unwinding-assertions` after cutting a
loop short now says the result holds only up to the bound
([#7338](https://github.com/esbmc/esbmc/pull/7338)). A SIGSEGV or SIGBUS inside
ESBMC is reported as an internal error without needing `--segfault-handler`,
which still selects the fuller backtrace and memory map in its place
([#7404](https://github.com/esbmc/esbmc/pull/7404)). A witness step from a header
or an operational model carries an `originfile` instead of placing model line
numbers on the user's source ([#7457](https://github.com/esbmc/esbmc/pull/7457)),
uncaught-exception properties are anchored on the raise site rather than printed
at line 0 ([#7445](https://github.com/esbmc/esbmc/pull/7445)), and each log line
is written in one call, so a `--k-induction-parallel` child can no longer splice
itself into the middle of a verdict
([#7384](https://github.com/esbmc/esbmc/pull/7384)).

**Memory model.**
Every allocation path is capped at `PTRDIFF_MAX`, as glibc does, since above it
an object's offset reads negative in the bounds checks and the pointer
comparator ([#7306](https://github.com/esbmc/esbmc/pull/7306)). The
reachable-leak walk follows an object's contents as well as its type, so an
allocation that was never cast at the site no longer reads as forgotten memory
([#7506](https://github.com/esbmc/esbmc/pull/7506)). Two false alarms went with
them: a pointer written through one union arm is now found when read through
another ([#7369](https://github.com/esbmc/esbmc/pull/7369)), and a `memcpy` into
a one-pointer struct keeps the pointer instead of dropping it to invalid
([#7380](https://github.com/esbmc/esbmc/pull/7380)).

**CBMC goto binaries.**
`__CPROVER_OBJECT_SIZE` no longer crashes ESBMC during SMT encoding, and both it
and `__builtin_object_size` size the object addressed rather than `sizeof(*p)` —
a scalar reached through a `void *` or the `signed char *` CPROVER's write-set
checks cast to used to report 0 or 1
([#7410](https://github.com/esbmc/esbmc/pull/7410),
[#7417](https://github.com/esbmc/esbmc/pull/7417)). A `__CPROVER_is_fresh`
precondition allocates the object it promises rather than leaving the parameter
unconstrained ([#7405](https://github.com/esbmc/esbmc/pull/7405)), and
`__CPROVER_rounding_mode` and `fesetround` are connected to the model that reads
them ([#7428](https://github.com/esbmc/esbmc/pull/7428),
[#7456](https://github.com/esbmc/esbmc/pull/7456)).

**Solver.**
Bitvector `%` is encoded as `a - (a / b) * b`, its SMT-LIB defining identity, so
a formula containing both `/` and `%` shares one division circuit and proving a
`%` implementation against the C99 6.5.5 identity collapses from a timeout to
0.001 s ([#7449](https://github.com/esbmc/esbmc/pull/7449)). Under `--ir-ieee` a
float symbol is constrained to a representable magnitude, which had let a
counterexample report `r = 0.000000` while denying `r == 0`
([#7336](https://github.com/esbmc/esbmc/pull/7336)). Separately, the
simplifier's equivalence checker now proves every per-node rewrite rather than
only the outermost one, which turned up two unsound folds
([#7327](https://github.com/esbmc/esbmc/pull/7327),
[#7351](https://github.com/esbmc/esbmc/pull/7351)).

The website documentation has been updated to match. As always, the full list is
in the [commit history](https://github.com/esbmc/esbmc/commits/master/) — and if
you hit a bug or a missing feature, we would love an
[issue report](https://github.com/esbmc/esbmc/issues).
