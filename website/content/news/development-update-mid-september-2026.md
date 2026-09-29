---
title: "Development Update: Mid-September 2026"
date: 2026-09-20T10:00:00+01:00
draft: false
tags:
  - ESBMC
  - FormalVerification
  - ModelChecking
  - OpenSource
---

Following the [early-September update](/news/development-update-early-september-2026),
231 commits landed on `master` between 4 and 20 September. Here are the changes
a user will notice.

**The C++ standard-library headers ESBMC could not include.** Because ESBMC
compiles C++ with `-nostdinc++`, a header with no operational model is a
`PARSING ERROR` on the `#include` alone, whether or not the program uses
anything from it. Twenty headers closed that way this fortnight:
`<numbers>` ([#7638](https://github.com/esbmc/esbmc/pull/7638)),
`<concepts>` ([#7639](https://github.com/esbmc/esbmc/pull/7639)),
`<version>` ([#7647](https://github.com/esbmc/esbmc/pull/7647)),
`<charconv>` ([#7649](https://github.com/esbmc/esbmc/pull/7649)),
`<cfenv>`, `<cinttypes>`, `<cwctype>` and six more C wrappers
([#7799](https://github.com/esbmc/esbmc/pull/7799)),
`<ratio>` ([#7804](https://github.com/esbmc/esbmc/pull/7804)),
`<execution>`, `<codecvt>` and `<scoped_allocator>`
([#7909](https://github.com/esbmc/esbmc/pull/7909)),
`<latch>`, `<semaphore>` and `<stop_token>`
([#7912](https://github.com/esbmc/esbmc/pull/7912)),
`<flat_map>`, `<flat_set>` and `<syncstream>`
([#7914](https://github.com/esbmc/esbmc/pull/7914)), and
`<regex>` and `<mdspan>` ([#7915](https://github.com/esbmc/esbmc/pull/7915)).
`<ranges>`, `<format>`, `<print>`, `<coroutine>`, `<generator>`, `<barrier>`
and `<stdfloat>` are what remain.

Two of these are worth reading the small print on, because being *includable* is
not the same as being *modelled*. `<regex>` models no matching at all: a regular
expression engine unrolled into SSA is not something a solver can carry, and a
partial engine would give wrong verdicts on the patterns it mishandles, so every
function that reports a match answers nondeterministically and a property that
depends on a particular pattern matching is reported `FAILED` rather than
proved. `<syncstream>` likewise transfers no characters, and `<charconv>` omits
the floating-point overloads for the same reason — an approximate
shortest-round-trip formatter returns wrong digits silently, where an absent
overload stays the compile error it already was. The
[C++ Support](/docs/c-cpp/supported-features) page now says which is which per
header.

**Invariant synthesis for affine loops.** k-induction cannot prove a property
needing a relation between a loop counter and an accumulator: the interval
domain is non-relational, so at the loop head it knows the counter's range and
nothing tying the accumulator to it. `--synthesise-loop-invariants` recognises
affine counter/accumulator loops and emits the closed form as a
`LOOP_INVARIANT` ([#7479](https://github.com/esbmc/esbmc/pull/7479)), which the
existing havoc schema *asserts before it assumes* — so a wrong candidate fails a
claim rather than producing an unsound proof. It trades bug-finding for proving:
over `regression/{esbmc,esbmc-unix,k-induction,loop-invariants}` it fires on 87
of 2716 files, of which 9 lose a bug that bounded BMC finds, and no file gains a
false proof or a false alarm. Opt-in, off by default, and documented under
[Loop Invariants](/docs/loop-invariants#synthesising-invariants-for-affine-loops).

**`--loop-invariant-check` can refute again.** The `UNKNOWN` downgrade added in
#7491 was decided by a claim's position — everything downstream of the havoc —
which is right while the abstraction admits the claim holding and wrong when the
invariant pins the counterexample outright, the case the mode exists to serve.
ESBMC now asks the solver whether the claim can hold at all on a feasible
abstract path before downgrading; UNSAT means no abstract state satisfies it, and
since the concrete states are a subset, the violation is real
([#7626](https://github.com/esbmc/esbmc/pull/7626)). Two further fixes in the
same area: a `do`-`while` head is the first instruction of the body rather than
a guard, so the combined pass was copying an empty body and discharging the
inductive step against no iteration at all
([#7497](https://github.com/esbmc/esbmc/pull/7497)); and storage written through
a dereference has no symbol to havoc, so the pointee is now havocked through the
pointer and resolved against symex's own value set
([#7518](https://github.com/esbmc/esbmc/pull/7518)).

**A verdict that said `UNKNOWN` after finding a bug.** Under `--multi-property`
— explicit, or implied by `--parallel-solving` or `--all-witnesses` —
`--k-induction` and `--incremental-bmc` keep going past a violation, but the
exhausted-`k` exit ignored what they had recorded: a program whose loop cannot
be unwound printed its counterexamples and then ended `VERIFICATION UNKNOWN`
with exit status 0. That exit now reports the recorded violation
([#7913](https://github.com/esbmc/esbmc/pull/7913)). Relatedly, the C library
idiom `(void)((c) || (assert(0), 0))` was folded into `ASSERT c` by matching any
two-statement branch and erasing the second statement unchecked; under
`--multi-property`, which keeps checking past a failed assertion, that lost
real code ([#7917](https://github.com/esbmc/esbmc/pull/7917)).

**An undefined conversion that verified clean.** C11 6.3.1.4p1 makes a
floating-point to integer conversion undefined when the integral part is not
representable in the destination type. ESBMC's existing cast check was gated on
`--int-encoding`, so the default bitvector mode had none and reported
`VERIFICATION SUCCESSFUL` on `(long long)1e300`. `--overflow-check` now covers it
([#7622](https://github.com/esbmc/esbmc/pull/7622)); `--no-fp-conversion-check`
turns just that check off, as SV-COMP needs, whose no-overflow property is about
signed-integer arithmetic only.

**Concurrency: two schedules that were never explored.** Partial-order
reduction keyed only static and heap-typed objects, so a mutex or datum in
`main`'s frame handed to a worker through a pointer had no key in either thread:
MPOR pruned the only schedule reaching the bug, and a write to such a local was
not even a context-switch point, so `--no-por` missed it too. Every address-taken
local is now keyed as a global is
([#7826](https://github.com/esbmc/esbmc/pull/7826)). Separately, race
instrumentation put a call *inside* the atomic block guarding its own accesses
whenever the result was stored in shared memory or the call went through a
global function pointer, so races inside the callee were never reported and racy
programs verified `SUCCESSFUL`
([#7768](https://github.com/esbmc/esbmc/issues/7768)).

**Memory model.** A write or `free` through a pointer the value set cannot
resolve exhaustively now fails when the pointer is none of the recorded targets
— freed globals, stack objects and heap interiors were previously missed
([#7773](https://github.com/esbmc/esbmc/pull/7773)). The alignment check reads
the object's base alignment rather than assuming the base carries the access
width, so a pointer laundered out of a packed struct no longer reads as aligned
([#7721](https://github.com/esbmc/esbmc/pull/7721)). A flexible array member has
size zero as C17 6.7.2.1p18 requires, rather than the storage of a one-element
array ([#7774](https://github.com/esbmc/esbmc/pull/7774)). And
`::operator new(n)` with a non-constant `n` allocates `n` bytes instead of one,
which had made every in-bounds access through the returned pointer an
out-of-bounds report ([#7651](https://github.com/esbmc/esbmc/pull/7651)).

**GNU vector extensions.** Four defects, each of which aborted or silently
dropped a cast: a vector comparison was typed `bool`, so `v4i m = a == b;`
assigned a one-bit value to a 128-bit vector and the solver aborted on the width
mismatch ([#7904](https://github.com/esbmc/esbmc/pull/7904)); dereferencing a
pointer to a vector aborted in the pointer analysis, where the read and write
sides disagreed about what `index2t` admits
([#7920](https://github.com/esbmc/esbmc/pull/7920)); the last statement of a GNU
statement expression was decayed to `&x[0]` although C never decays a vector
([#7921](https://github.com/esbmc/esbmc/pull/7921)); and a cast between vector
types was dropped entirely, so `(v4u)c` verified as `c`
([#7922](https://github.com/esbmc/esbmc/pull/7922)).

**C front end.** `__builtin_{add,sub,mul}_overflow` and the carry builtins
arrive unlowered, unlike the typed family clang expands itself, and with no body
returned nondeterministic results behind a warning
([#7586](https://github.com/esbmc/esbmc/pull/7586)).
`__atomic_test_and_set` and `__atomic_clear` were the only two GCC atomic
builtins the front end did not name
([#7658](https://github.com/esbmc/esbmc/pull/7658)). `__bf16` and `__mfp8` are
modelled, so `#include <immintrin.h>` no longer fails under clang 22
([#7894](https://github.com/esbmc/esbmc/pull/7894)); `_Complex int z = {1, 2}`
converts ([#7801](https://github.com/esbmc/esbmc/pull/7801)); and an integer
sentinel pointer — `0xffffffffffffffffUL` into a `void *`, as preprocessed
kernel and CIL sources write it — is accepted outside `--sv-comp` as well, since
GCC accepts it and refusing was a false rejection of input a mainstream
toolchain compiles ([#7741](https://github.com/esbmc/esbmc/pull/7741)).

**Diagnostics instead of aborts.** An uncaught exception is reported at the
raise it came from, one property per raise site, rather than at `line 0` under
`global` once a second disagreeing site cleared the anchor
([#7770](https://github.com/esbmc/esbmc/pull/7770)). An initializer list the
front end cannot model reports its type, arity and location instead of tripping
a bare `assert` ([#7702](https://github.com/esbmc/esbmc/pull/7702)). A target
clang cannot map — `--ppc-macos` — is reported as `PARSING ERROR` rather than a
SIGSEGV ([#7759](https://github.com/esbmc/esbmc/pull/7759)). A zero-width
dereference fails loudly in release builds instead of reading `bytes[-1]` inside
ESBMC ([#7737](https://github.com/esbmc/esbmc/pull/7737)). And ESBMC warns when
an option value names another option, which had silently made `--show-loops` a
witness filename ([#7538](https://github.com/esbmc/esbmc/pull/7538)).

**Python runs faster.** The operational models are precompiled to a GOTO binary
at build time, the way `c2goto` builds the C library, instead of being
re-converted from AST JSON and re-lowered on every run
([#7747](https://github.com/esbmc/esbmc/pull/7747),
[#7778](https://github.com/esbmc/esbmc/pull/7778)): a trivial run drops from
3.26 s to 1.80 s and the Python regression suite from 2952 s to 1932 s. A
module's AST is parsed on first lookup rather than at startup
([#7777](https://github.com/esbmc/esbmc/pull/7777)), model functions the program
cannot reach are dropped before GOTO conversion
([#7780](https://github.com/esbmc/esbmc/pull/7780)), and the advisory mypy check
nobody asked for is now opt-in behind `--python-typecheck`.

**Python correctness.** A union mixing a scalar with a container was narrowed to
its leftmost member, which folded a comparison against the returned list to
false and its negation to true — a proof of a false property
([#7877](https://github.com/esbmc/esbmc/pull/7877),
[#7880](https://github.com/esbmc/esbmc/pull/7880)). A module CPython imports but
ESBMC has no AST for — `sys`, or a stdlib file the resolver filters out — no
longer aborts the run, and `sys` has a data-only model
([#7677](https://github.com/esbmc/esbmc/pull/7677)); a module that fails to
compile or whose body raises is reported for what it is rather than escaping as
a CPython traceback with no verdict
([#7681](https://github.com/esbmc/esbmc/pull/7681)). An inherited
`@staticmethod` called through an instance binds its arguments to the right
slots ([#7781](https://github.com/esbmc/esbmc/pull/7781)). `random.choice` and
`random.sample` dispatch on the sequence type instead of running the int-list
model over a `str` or a tuple and reporting a dereference failure
([#7676](https://github.com/esbmc/esbmc/pull/7676)). And a tagged
(dynamically-typed) scalar can now be passed as a function argument, stored in a
list, and compound-assigned or negated
([#7573](https://github.com/esbmc/esbmc/pull/7573),
[#7708](https://github.com/esbmc/esbmc/pull/7708),
[#7834](https://github.com/esbmc/esbmc/pull/7834)).

**NumPy.** A 2-D array parameter keeps its full shape through the C-ABI
row-pointer decay, so `.shape`, `.ndim`, `.size` and
`numpy.transpose` / `.T` / `.transpose()` read it rather than the decayed 1-D
type — this was the one NumPy shape that produced a silently wrong array value
rather than an explicit rejection. Sorting and searching gained row and column
views and 2-D arrays with an `axis` argument
([#7722](https://github.com/esbmc/esbmc/pull/7722)).

**CUDA.** The model's device list was never populated, so every `cudaSetDevice`
failed, no current device was tracked, and a kernel launched on one GPU with
memory from another verified. The model now keeps a per-host-thread current
device, tags each `cudaMalloc` with its device, and asserts at every launch that
each pointer argument is host memory, the current device's, or an enabled peer's
([#7772](https://github.com/esbmc/esbmc/pull/7772)).

**Solver.** The Bitwuzla backend moved to Bitwuzla's C++ API, where `Term` and
`Sort` release themselves, after a first pass fixed the leaks that needed no API
change ([#7507](https://github.com/esbmc/esbmc/pull/7507),
[#7508](https://github.com/esbmc/esbmc/pull/7508)). `--overflow-check` on
`(__int128)a * (__int128)b` for two `long long`s generated a VCC whose encoding
doubles the destination width to 256 bits; a same-signedness widened multiply
cannot overflow when the destination is at least `w1 + w2` bits, so the check
folds without reaching the solver, where it had OOM'd under Bitwuzla and timed
out under Z3 and Boolector
([#7843](https://github.com/esbmc/esbmc/pull/7843)). A function's address can no
longer coincide with `SIG_DFL`, `SIG_ERR` or `SIG_IGN`, which C11 7.14p3
requires ([#7800](https://github.com/esbmc/esbmc/pull/7800)).

**Two SV-COMP fixes.** Flattening a pointer into an untyped byte object and
reading it back went through the integer-to-pointer path, which rebuilds a
pointer from an address alone — and an address does not identify one, so a key
an `aws-c-common` harness stored came back as a different pointer and ESBMC
reported a false alarm on a task whose expected verdict is true
([#7895](https://github.com/esbmc/esbmc/pull/7895)). And a violation witness
named every local but not the value that made the property fail when that value
was read through a pointer into an object the program never wrote, so a
validator had nothing to replay — which turns a correct `false` into no points
([#7893](https://github.com/esbmc/esbmc/pull/7893)).

The website documentation has been updated to match. As always, the full list is
in the [commit history](https://github.com/esbmc/esbmc/commits/master/) — and if
you hit a bug or a missing feature, we would love an
[issue report](https://github.com/esbmc/esbmc/issues).
