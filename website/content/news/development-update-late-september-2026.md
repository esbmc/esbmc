---
title: "Development Update: Late September 2026"
date: 2026-10-01T10:00:00+01:00
draft: false
tags:
  - ESBMC
  - FormalVerification
  - ModelChecking
  - OpenSource
---

Following the [mid-September update](/news/development-update-mid-september-2026),
145 commits landed on `master` between 20 September and 1 October. Most of the
user-visible ones fix a wrong verdict, so this update starts there; the Python,
C/C++ and Ladder Diagram front ends follow.

## Wrong verdicts in the verification engine

**False proofs in the default configuration.** Three fixes this fortnight
concern runs with no options at all.
Interval-based guard pruning, which symbolic execution performs by default
(`--no-interval-symex-guard` turns it off), tracked only assignments, assumes,
declarations and `DEAD`; the writes symex performs itself — binding a call's
result or a callee's parameters, returning into the caller's left-hand side, a
body-less call storing through a pointer, a recursive callee overwriting its
caller's locals — left stale intervals behind, and a loop guard was pruned as
false. The issue's program verified:

```c
int f(void) { int r = nondet_int(); return r; }
int main() {
  int n = 0, i = 0;
  n = f();               /* the domain still believed n == 0 */
  while (i < n) { i++; if (i > 3) break; }
  __ESBMC_assert(i == 0, "loop not entered");
}
```

Each of those writes now havocs what it overwrites
([#8061](https://github.com/esbmc/esbmc/pull/8061)). A multi-dimensional VLA's
flattened size was multiplied in the type of its second level, so in
`int a[2][3][m]` with `3 * m == 2^32 + 2` the 32-bit stride wrapped,
`a[1][0][0]` aliased `a[0][0][2]`, and an assertion that the two are distinct
objects was proved; mismatched level widths (`int a[2][m][3]` indexed
symbolically) aborted the solver instead. The product is now taken in `size_t`
([#8018](https://github.com/esbmc/esbmc/pull/8018)). And the pointer round trip
added in [#7895](https://github.com/esbmc/esbmc/pull/7895) tied every flattened
pointer to every pointer rebuilt at the same address; when the stored pointer was
NULL-based, the tie left the formula with no model, so a reachable `assert(0)`
after a pointer was stored in `malloc`'d memory and read back through `void *`
verified, and an SV-COMP counterexample was lost. Only flattened pointers are
tied now; a rebuilt one is defined by the flattens that reach it by data flow
([#7926](https://github.com/esbmc/esbmc/pull/7926)). This one affected `master`
between 19 and 27 September, not a release.

**False proofs under k-induction.** Under `--add-symex-value-sets`, which every
`--k-induction` run enables, the inductive step bound each pointer havoc to the
pointer's loop-entry value and its loop-entry points-to set, so it checked only
the loop-entry pointer state and proved loops whose bug lies beyond k:

```c
int a[2] = {1, 2};
int *p = a, s = 0;
for (int i = 0; i < 3; i++) { s += *p; p++; }   /* reads a[2] */
```

Both restrictions are gone ([#7972](https://github.com/esbmc/esbmc/pull/7972)).
This costs proofs: a safe list walk that was proved only through the restriction
now ends `UNKNOWN`, and SV-COMP list proofs may be lost the same way. Under
`--k-induction --interval-analysis`, a `while (1)` loop headed by an `assert`
aborted, and where the instruction before the loop head was a jump past the
loop, the havoc never reached the loop and the inductive step proved a missed
bug; the havoc now always goes at the head
([#8067](https://github.com/esbmc/esbmc/pull/8067)).

**False proofs under non-default options.** `--interval-symex-assert` joined a
parked path's interval snapshot back only for variables a phi function
assigned, so an `__ESBMC_assume` on one branch survived the merge and pruned
assertions the other branch violates
([#8051](https://github.com/esbmc/esbmc/pull/8051)). Under `--no-slice`, a VLA
declared on an untaken path with a negative bound sign-extended to a size the
address space cannot hold, the formula was UNSAT on every path, and
`assert(n != -1)` passed; the exact layout is now required only up to the
largest object size symex admits
([#8087](https://github.com/esbmc/esbmc/pull/8087)). Under `--big-endian`, the
SMT layer laid arrays and structs out with element 0 in the low bits while the
byte operations read address 0 from the high bits, so
`union { short a[4]; short b[4]; }` stored `a[1]` at `b[2]` and
`assert(u.b[1] != 5)` verified; short union members, symbolic-offset struct byte
reads and union counterexample values were wrong for the same reason
([#8084](https://github.com/esbmc/esbmc/pull/8084)).

**`--gcse`, three ways.** The common-subexpression pass (off by default) had
three independent soundness defects, each a false proof. An assignment target
equal to an available expression — such as the `(signed int)b` that `b ^= 1`
lowers to — was replaced by the cached symbol, dropping the store; stores
through arrays, aliases and shared heap objects left stale expressions
available; and a cached symbol was reused after a kill when a guard or call
made its expression available again
([#7999](https://github.com/esbmc/esbmc/pull/7999)). The points-to analysis
behind it read the pointer in `*e`, `&e[i]` and `e->f` as a fresh node with no
constraints, which it took to mean "points to nothing"
([#8074](https://github.com/esbmc/esbmc/pull/8074)). And the analysis is
sequential, so a value another thread wrote in between was reused; `--gcse` is
now skipped with a warning on a program that may create threads
([#8072](https://github.com/esbmc/esbmc/pull/8072)).

**A false alarm under `--no-propagation`.** `assume(x == c)` was lifted into an
assignment that bumped `x`'s SSA version without emitting a step, so later reads
of `x` were unconstrained and produced spurious counterexamples
([#7975](https://github.com/esbmc/esbmc/pull/7975)).

## Multi-property results under the k-step strategies

**`--multi-property` under the k-step strategies.** This finishes the reporting
work tracked in [discussion #7900](https://github.com/esbmc/esbmc/discussions/7900).
`--k-induction` and `--incremental-bmc` ran a separate property check at every
k, renumbered the claims each time, and reported a base case that came back
UNSAT — a bounded result — as `PASSED` under an interim
`VERIFICATION SUCCESSFUL`. One table now covers the whole run, printed once
where it concludes, with ids that stay fixed across k; a base case records only
violations, and the forward condition or inductive step settles the rest
([#7941](https://github.com/esbmc/esbmc/pull/7941)). Clearing claims that a
round had settled was keyed on position and guard text, which dropped
`--loop-invariant` body copies and the assertions a NULL-pointer check was
raised in without solving them, and the forward condition then reported them
`PASSED`; completeness is now judged per round
([#7963](https://github.com/esbmc/esbmc/pull/7963)). Two checks sharing a
description and position, such as the bound checks of `a[i] + a[j]`, shared one
row, so once one failed the other was skipped as already verified; each now has
its own row, ending with its condition where the rows would otherwise read the
same ([#7923](https://github.com/esbmc/esbmc/pull/7923)). Interval analysis gave
a literally false `ASSERT` no successor, so under
`--k-induction --interval-analysis --multi-property` the code after a violated
assertion was treated as unreachable and the inductive step proved an assertion
there; the
same pruning made `--all-witnesses` delete code after a failed assertion
([#7916](https://github.com/esbmc/esbmc/pull/7916)). The loop-free program of
[#1902](https://github.com/esbmc/esbmc/issues/1902), which kept incrementing k,
now stops at k = 2 and is pinned by a test
([#7970](https://github.com/esbmc/esbmc/pull/7970)), and a new
`multi-property-matrix` suite pins every row, summary and verdict under each
strategy; 14 of its 22 cells fail on v8.5
([#7973](https://github.com/esbmc/esbmc/pull/7973)). The
[usage page](/docs/usage#under-the-verification-strategies) lists what each
strategy reports per property.

## Python

**`is None` on an `Optional` value.** `r: Optional[str] = f(nondet_int())`
followed by `assert r is not None` verified `SUCCESSFUL` even when `f` could
return `None`: symex folded `is None` to false for a string pointer, and an
unannotated caller of an `Optional[T]` function got a placeholder type that
could not hold `None`. A NULL string pointer is now `None`, and `T if c else None`
yields `T*` rather than an `Optional` struct
([#8014](https://github.com/esbmc/esbmc/pull/8014)); `Optional[List]` and
`Optional[Set]` use the container's pointer type in the same way
([#8020](https://github.com/esbmc/esbmc/pull/8020)). `Optional[int]` was an
`int*` holding the value, so a stored `0` could not be told from `None`;
`Optional[int|float|bool]` now uses the `Optional<T>` struct that `int | None`
already had, across variables, returns, parameters, fields and dict values, and
a `None` returned from a function declared `-> int` is reported rather than read
as 0 ([#8050](https://github.com/esbmc/esbmc/pull/8050)).
`Optional[Dict]` and `Optional[Tuple]` remain open
([#8017](https://github.com/esbmc/esbmc/issues/8017)).

**String rendering that proved what CPython refutes.** The model behind
`str()`, f-strings and `print` of a runtime float truncated to six fractional
digits, had no scientific notation and printed `-0.0` as `0.0`, so
`assert str(v) != "1e+16"` with `v = 1e16` verified. It now renders only strings
that are provably CPython's shortest repr — integral values below 2**53, and
values in [1e-4, 2**32) whose 6-digit decimal reads back exactly — and leaves
every other value an unconstrained string
([#7953](https://github.com/esbmc/esbmc/pull/7953)). The constant fold had the
same class of bug (`10000000000000002.0` where CPython prints
`1.0000000000000002e+16`) and now takes the shortest round-trip digits with
CPython's layout ([#7959](https://github.com/esbmc/esbmc/pull/7959)). `repr()`,
`ascii()` and f-string `!r`/`!a`/`=` are modelled for runtime ints, bools,
floats and ASCII strings, where they previously gave a nondeterministic string or
an unsupported call; a format spec on a `!r` field, which was folded to the
unpadded value (`f"{42!r:>6}"` as `'42'`, a false proof), now yields an
unconstrained string ([#7952](https://github.com/esbmc/esbmc/pull/7952),
[#7959](https://github.com/esbmc/esbmc/pull/7959)). `repr('a')` no longer fails
to terminate, and `f'{None!r}'` folds to `'None'`
([#8022](https://github.com/esbmc/esbmc/pull/8022)). Containers and non-ASCII
strings under `!r` stay unconstrained.

**Code points and byte order.** Runtime `ord()` read the first UTF-8 byte
as a signed char, so `assert ord(chr(200)) < 0` verified; a new model decodes the
first code point ([#7950](https://github.com/esbmc/esbmc/pull/7950)), and
`len(chr(200))` counts one code point rather than two bytes
([#8023](https://github.com/esbmc/esbmc/pull/8023)). Strings built at run time
and indexing still work on bytes. `int.from_bytes` turned every byteorder other
than the literal `"big"` into little-endian, so a value folded on a loop's first
iteration was flipped on the second — a false proof
([#7944](https://github.com/esbmc/esbmc/pull/7944)). The byteorder is now folded
where the callee is resolved, so an alias (`f = int.from_bytes`) honours it and
`int.from_bytes(b)` defaults to big-endian as in CPython rather than raising
`TypeError`; a byteorder that is not a literal is refused with a located error
instead of guessed ([#7948](https://github.com/esbmc/esbmc/pull/7948)).

**Classes.** A method called through a derived class's name bound the
first base, depth-first, whose method symbol already existed, which proved false
results for forward references, `async def` overrides and class attributes, and
picked the wrong method in a diamond. Where every class in the hierarchy is bound
once in the file, the call now follows Python's C3 method resolution order, and
is refused rather than bound to another class's method when the deciding class
has no converted symbol yet ([#8058](https://github.com/esbmc/esbmc/pull/8058)).
`len(xs[i])` over a list of class instances never called `__len__`; it does now
when the program defines a single class, and stops with an error otherwise
([#7977](https://github.com/esbmc/esbmc/pull/7977)). `isinstance("int", type)`
was folded to true because a type object is modelled as a char array
([#7989](https://github.com/esbmc/esbmc/pull/7989)). A module-level class and a
function- or method-scope class of the same name shared one symbol and the later
constructor was silently dropped; that is now refused, listing the conflicting
lines ([#7942](https://github.com/esbmc/esbmc/pull/7942)). In the other
direction, a program class that shares its name with a class in an imported
module is no longer refused: the imported class is renamed and `mod.Name`,
`pkg.mod.Name` and `from mod import Name` are rewritten, where the scoping is
plain enough to do so ([#8026](https://github.com/esbmc/esbmc/pull/8026)).
`cls(...)` inside a `@classmethod` constructs the class the method runs on,
including through a subclass, where that class is known statically
([#7958](https://github.com/esbmc/esbmc/pull/7958)).

**Conversions and unary operators.** `bool()` and the numeric
constructors relabelled most arguments instead of converting them, so
`assert bool(g(k)) + 1 == 6` verified where CPython fails it, and
`bool(~k)` aborted in the solver
([#7961](https://github.com/esbmc/esbmc/pull/7961)). `bool()` of a list or dict
relabelled the container pointer: `bool(xs)` of an empty list parameter was
proved true, and `bool([1, 2])` aborted; it now tests emptiness as `not x` does
([#7969](https://github.com/esbmc/esbmc/pull/7969)). An `if` or `while` test of
the form `~h(0)` or `-h(1)` was evaluated as `not h(...)`, which missed bugs, and
`assert -f() == -2.5` crashed in SMT encoding
([#7957](https://github.com/esbmc/esbmc/pull/7957)). `x = 1.5; x = True` aborted
with a `binary2integer` assertion, and `b = True; x = 1.5; x = b` proved
`not isinstance(b, bool)`; a bool assigned to a numeric variable is now converted
rather than retyped ([#7951](https://github.com/esbmc/esbmc/pull/7951)).

**Lambdas, sorting and lists.** An unannotated lambda's parameters and
result were `double`, so an int above 2**53 was rounded and
`g = lambda n: n + 0; x: int = 2**53 + 1; assert g(x) == 2**53` verified; parameters now take
the scalar type every call site agrees on, and the return type follows an
integral body ([#7949](https://github.com/esbmc/esbmc/pull/7949)). A lambda over
a class instance (`lambda c: c.speed` applied to `cars[0]`) aborted the solver
with a sort mismatch ([#7934](https://github.com/esbmc/esbmc/pull/7934)), as did
`sorted`/`min`/`max(cars, key=lambda c: c.speed)`; the sorted scan also applied
the key O(n²) times, which gave a false proof for an impure key, and now applies
it once per element as CPython does
([#7955](https://github.com/esbmc/esbmc/pull/7955)). `a = stack.pop()` was
typed as the list rather than an element, which produced a false
`VERIFICATION FAILED` when popped values were passed to a helper
([#8053](https://github.com/esbmc/esbmc/pull/8053)). An element of a list mixing
strings and numbers, such as `[3.0, "+", 2.0]`, is read as a tagged value, so
`isinstance(token, float)` answers per element
([#8052](https://github.com/esbmc/esbmc/pull/8052)). `d, n = heappop(h)` on a
heap of tuples now unpacks, and a class object used as a value (`A = int`) no
longer crashes symex under `--smt-symex-guard`
([#8059](https://github.com/esbmc/esbmc/pull/8059)). Two crashes on dynamically
typed values are fixed: reading a branch-divergent value back out of a dict
([#7943](https://github.com/esbmc/esbmc/pull/7943)), and calling
`bit_length()`, `bit_count()`, `conjugate()` or `is_integer()` on one inside an
`isinstance` guard ([#8085](https://github.com/esbmc/esbmc/pull/8085)).

**Bytes.** A function returning `-> bytes` now keeps its length and can be indexed inline, `bytes +
bytes` is concatenation rather than unmodelled NumPy broadcasting, and a bare
`bytes` parameter takes its size from the call site. `nondet_bytes(n)` gives `n`
independent nondeterministic bytes for a constant `n`
([#7902](https://github.com/esbmc/esbmc/pull/7902)).

**Located preprocessor errors.** A rejection raised while preprocessing
— `ERROR: Decimal() with non-constant arguments is not supported` — reached the
user with no position, because only one of the raise sites attached one. Every
such rejection now carries the `file:line:col` of the narrowest node that could
not be handled ([#7940](https://github.com/esbmc/esbmc/pull/7940)).

**NumPy.** A method chained directly on a constructor, such as
`np.eye(3).transpose()`, produced a nondeterministic result instead of the value;
those methods are now evaluated
([#7976](https://github.com/esbmc/esbmc/pull/7976)), and the ones that were
refused in the interim (`reshape`, `prod`, `std`, `var`, `argmin`, `argmax`,
`diagonal`) followed ([#8011](https://github.com/esbmc/esbmc/pull/8011)). A
function that builds an array in a local and returns it keeps the array's type,
a pinned `KNOWNBUG` until now; `searchsorted` takes a vector of values and a
`sorter=` argument; `sort`/`argsort` accept `kind='stable'`/`'mergesort'`; and
a parameter with a symbolic shape gets a named diagnostic rather than an
`AttributeError` ([#7925](https://github.com/esbmc/esbmc/pull/7925)). A `dtype=`
keyword on a constructor no longer makes later `transpose`, `sort` or reducer
calls decline, and `*_like` constructors accept a `dtype=` override
([#7976](https://github.com/esbmc/esbmc/pull/7976)). `.size` works on a
constructor call (`np.full((2, 4), 7).size`) and `searchsorted` on symbolic
search values ([#8011](https://github.com/esbmc/esbmc/pull/8011)).

## C and C++

**C++ objects built in the wrong place.** Several ways of creating an object
ran the right constructor on the wrong storage, or none at all, and most of them
gave a false proof:

- `new C[2]{C(1), C(2)}` ran `C(1)` on every element, so `p[1].v == 1` verified;
  elements past the end of a braced list were zeroed rather than built by clang's
  array filler, so `C c[3]{C(1)}` never called `C()` and `S s[2]{S{1, 2}}`
  ignored `S`'s default member initialisers
  ([#8069](https://github.com/esbmc/esbmc/pull/8069)). For scalars the list was
  dropped altogether: `new int[2]{1, 2}` then `assert(p[1] == 2)` was a false
  alarm, and with a user-replaced `operator new[]` a false proof
  ([#8038](https://github.com/esbmc/esbmc/pull/8038)).
- Placement `new` at an address with a side effect — any call, such as
  `new (std::addressof(buf)) int(42)` — was treated as an allocating `new`: the
  object went to fresh memory and the buffer kept its old bytes. The address is
  now evaluated once, before the initialiser, as [expr.new]/19 requires
  ([#8028](https://github.com/esbmc/esbmc/pull/8028)).
- A class member initialised from a prvalue of its own type, `C impl_ =
  C::make();`, is that prvalue's result object ([dcl.init]/17.6.1); ESBMC copied
  a temporary into it and destroyed the temporary, or for `C impl_ = C{&x};` ran
  the constructor on no object at all. The member is now built in place
  ([#8025](https://github.com/esbmc/esbmc/pull/8025)).
- Before C++17, clang elides the copy in `C c = C::make();` and `return C(x);`,
  but ESBMC ran the copy constructor and destroyed a second object. Under
  `--std c++14`, Apple clang's default, a destructor count that aborts natively
  verified, and the true count was a false alarm
  ([#8041](https://github.com/esbmc/esbmc/pull/8041)).
- A function-local static with a dynamic initialiser was initialised before
  `main`, so `void n() { static int c = bump(); }` ran `bump()` even when `n`
  was never called. It is now initialised on the first pass through its
  declaration, guarded, as [stmt.dcl]/3 requires
  ([#7984](https://github.com/esbmc/esbmc/pull/7984)).
- In C and C++ alike, `char a[4] = {"ab"}` stored the literal's address cast to
  `char` into `a[0]` and zeroed the rest, so `a[1] == 0` verified. A braced
  string literal now initialises the array as the bare literal does (C11
  6.7.9p14) ([#8070](https://github.com/esbmc/esbmc/pull/8070)).

**Two declarations that shared one symbol.** ESBMC names a declaration by
clang's USR, and in four places the USR did not tell two different entities
apart, so the second was verified through the first. When several C files are
verified together, each file's copy of a `static` function a header defines —
and its static locals — was one symbol, so a counter called once from each of
two files appeared to count to 2
([#7998](https://github.com/esbmc/esbmc/pull/7998)). Two same-named locals
declared by one macro expansion shared storage, so a value stored through an
`int s` was read back through an `unsigned char s`
([#7986](https://github.com/esbmc/esbmc/pull/7986)). `struct S` defined inside
two different functions was one type, read through whichever layout came first
([#7982](https://github.com/esbmc/esbmc/pull/7982)). And `f<int A::*>` and
`f<long B::*>`, or overloads differing only in a member-pointer parameter, shared
one body ([#7979](https://github.com/esbmc/esbmc/pull/7979)). All four gave false
proofs. The fix in #7986 briefly gave a block-scope `extern` declared in a macro
its own symbol instead of the global it names; every MatIEC-generated PLC
program, which binds located variables that way, then failed a dereference check
before reaching its properties. Declarations with external storage keep the
global's identity again ([#8037](https://github.com/esbmc/esbmc/pull/8037)).

Two related aborts are gone: a C++20 class-type template argument, `g<S{1}>()`,
ended the run with "Unable to generate the USR"
([#7987](https://github.com/esbmc/esbmc/pull/7987)), and
`std::hash<std::thread::id>{}` was a `CONVERSION ERROR` because the friend
declaration in `<thread>` left the record registered as incomplete
([#7938](https://github.com/esbmc/esbmc/pull/7938)).

**GNU vector extensions.** Following the four fixes in the previous update, a
vector can now be read *and written* through a pointer to it, at its own
alignment, so the vectorised loop of
[#1030](https://github.com/esbmc/esbmc/issues/1030) —
`*(v4si *)(result + i) = a * b;` — verifies instead of aborting with
"Unrecognized dest type during dereference". A vector struct member now sits at
its alignment rather than straight after the preceding field
([#7919](https://github.com/esbmc/esbmc/pull/7919)). Three solver aborts are
fixed: a store into an array of vectors, `v4i a[1]; a[0][0] = c;`
([#8015](https://github.com/esbmc/esbmc/pull/8015)); arithmetic between a
constant vector and one built lane by lane, which paired each lane with the whole
other vector ([#8077](https://github.com/esbmc/esbmc/pull/8077)); and reading a
union containing vectors through its bytes, or writing an array of structs with
a vector member at a symbolic index
([#8078](https://github.com/esbmc/esbmc/pull/8078)).

**`_Float16` has the right format.** `_Float16` and `__fp16` were built with 4
exponent bits and 11 fraction bits; IEEE 754 binary16 has 5 and 10. The largest
finite value was therefore about 256, so `(_Float16)1000.0f` folded to `+inf`
and a correct program reported `FAILED`, and since Bitwuzla does not support
that format, any `_Float16` reaching the solver aborted the run. ESBMC now agrees
with clang on the maximum finite value, rounding, overflow, the smallest normal
and subnormal values, and `sizeof`
([#7932](https://github.com/esbmc/esbmc/pull/7932)).

**Library models.** `is >> std::ws`, `std::u16string` / `std::u32string`, and
`shared_ptr<void>` / `unique_ptr<void, D>` compile with libc++ but were
`PARSING ERROR`s; all three now parse
([#7980](https://github.com/esbmc/esbmc/pull/7980)). POSIX `getline` is declared
in `<stdio.h>` — and so in `<cstdio>`, where its absence had been a
`PARSING ERROR` — and modelled: it returns a fresh buffer holding
nondeterministic bytes from the stream, so a program cannot prove the line it
read back is the one it had before
([#7933](https://github.com/esbmc/esbmc/pull/7933)).

## Ladder Diagram

**Variables written outside the program.** A PLC program's
`VAR_IN_OUT` and `%M` marker variables can be written between scans by a
caller, an HMI or a fieldbus, but the LD front end re-sampled only inputs, so a
violation caused by an outside write was missed, and `VAR_EXTERNAL` was not read
at all ("undeclared variable"). These variables now take any value at the start
of each scan cycle, and the run lists them
(`LD: sampling each scan, as writable outside the program: ...`).
`--ld-closed-world` asserts that only the program writes them. One scan loop
cannot model a faster task preempting the program mid-scan, or several program
instances, so a configuration with more than one task or more than one program
instance is now refused with `UnsupportedConstruct`
([#8065](https://github.com/esbmc/esbmc/pull/8065)).

**False proofs from scoping and unmodelled wiring.** Every POU's
interface variables went into one document-wide list, so a function block's
variable and a program variable of the same name became one symbol: a program
variable declared `TRUE` could start `FALSE` and hide a violation. Program
variables are now declared only by the program POU; a function block with an
LD body, whose rungs ran in the program's scope, and a document with more than
one program POU, whose bodies were merged into one scan, are refused
([#8063](https://github.com/esbmc/esbmc/pull/8063)). In graphical rungs, an
operator block or a timer/counter pin wired to an `outVariable` left the
variable unassigned, a block wired to a `PT` or `PV` pin was read as 0, and an
unparsable literal or initial value was read as 0 or as its numeric prefix
(`"1.5"` as 1) behind a warning; each gave `VERIFICATION SUCCESSFUL` on programs
that violate the property, and each is now refused, while an integral decimal
such as `"2.0"` is still read as an integer
([#8064](https://github.com/esbmc/esbmc/pull/8064)). And a user function block
whose Structured Text body failed to translate was skipped, so its outputs kept
stale values and a property over them could be proved although the real body
violates it; each output and in/out pin of such a block now takes a
nondeterministic value every scan. `--ld-sound-mode` therefore no longer turns
an untranslatable body into a no-op
([#8079](https://github.com/esbmc/esbmc/pull/8079)).

**Function-block inputs and Structured Text.** Every input of a
user function-block instance was re-sampled nondeterministically each scan,
ignoring its wiring, which produced false alarms on correct programs. An input
now takes its wire's value: a program variable, another block's output pin, or a
`BOOL` or integer literal. An unwired input, or one fed by something not
modelled, stays nondeterministic with a warning
([#8034](https://github.com/esbmc/esbmc/pull/8034)). The Structured Text
translator for function-block bodies had no Boolean level: `Q := A AND B;` was
modelled as `Q := A`, and a parenthesised Boolean expression dropped the whole
body. `AND`/`&`, `OR`, `XOR` and `NOT` now parse at IEC 61131-3 precedence on
`BOOL` operands, and `VAR_TEMP`, `VAR_IN_OUT` and outputs after the first keep
their declared type instead of becoming 32-bit integers
([#8035](https://github.com/esbmc/esbmc/pull/8035)).

**Rung order and evaluation follow Beremiz.** When the right
power rail listed no coils, the order of graphical rungs came from hash-map
iteration, and a contact reading a variable that a coil wrote saw a scan-start
snapshot, so a rung saw an earlier rung's write one scan late. Coils now run in
Beremiz's layout order (coils less than 10 units apart vertically share a row
and are ordered left to right; rows top to bottom), whether or not the rail
lists them, and evaluation is sequential: each coil re-reads its contacts and a
block steps once per scan ([#8036](https://github.com/esbmc/esbmc/pull/8036)).
Because the last powered coil wins, ESBMC now warns when one variable has both
set and reset coils, naming the winning coil where it can
([#8080](https://github.com/esbmc/esbmc/pull/8080)).

**Counters.** `CTU` kept counting past `PV` and `CTD` went
negative, saturating only at the 32-bit bounds, where MATIEC counts up only
while `CV < PV` and down only while `CV > 0`. `Q` was unaffected, but logic that
reads `CV` saw values the runtime never produces, a source of false alarms; the
bounds now match MATIEC ([#8066](https://github.com/esbmc/esbmc/pull/8066)). A
graphical `CTD` never read its `LD` pin, so the counter could not be reloaded;
`CTD` now loads `CV := PV` before counting down, accepts CODESYS's `LOAD` as
well as `LD`, and `CTU`'s `R` and `CTD`'s `LD` take the power flow of the
contact or block `Q` wired to them, with other sources refused
([#8033](https://github.com/esbmc/esbmc/pull/8033)).

## Crashes, counterexamples, contracts and a new option

**Crashes and hangs.** Release builds for Linux x86_64 are built with the
goto-contractor, whose interval library left the FPU rounding toward +inf after
its static initialisation; CaDiCaL's local-search setup then never terminated,
and Bitwuzla and cvc5 reported `Out of memory` on bit-vector formulas of a few
dozen gates. `main` now restores round-to-nearest before any thread starts
([#8045](https://github.com/esbmc/esbmc/pull/8045)). The simplifier folded
`(a + k) - a` to `k` in the offset's width rather than `ptrdiff_t`'s, and both
Bitwuzla and Z3 aborted on code as ordinary as
`ptrdiff_t n = k; if (c) n = (a + 3) - a;`
([#8040](https://github.com/esbmc/esbmc/pull/8040)). A byte-wise walk bounded by
another member's address, `for (char *p = (char *)&s.a[0]; p != (char *)&s.a[1]; ++p)`,
never left the loop under default flags because the guard did not fold through
the casts ([#8009](https://github.com/esbmc/esbmc/pull/8009)); the normalisation
that fixed it then rewrote the anchor of an array of byte arrays
(`unsigned char pool[8][32]`) without end and crashed with SIGBUS
([#8029](https://github.com/esbmc/esbmc/pull/8029)). `new D[2]()` for a
polymorphic class `D` aborted in the pointer analysis, which matched the zero
fill to the base subobject rather than the whole element
([#8075](https://github.com/esbmc/esbmc/pull/8075)).

**Counterexamples.** Symbolic execution lowers `s.f = v` into a whole-object update, so the
trace builder read and printed all of `s` for a step that shows `s.f`. It now
reads the model for the component the step prints: on the PLC benchmark from
InduByte/esbmc-evaluation#6 the run goes from 32.7 s to 22.4 s and the
counterexample from 28.1 MB to 0.98 MB, with 1668 values now concrete; violation
witnesses carry the assumptions that component writes contribute
([#7936](https://github.com/esbmc/esbmc/pull/7936)). The SMT-LIB backend read a
signed bit-vector model value as a natural number, so `a = -4` printed as
`4294967292` and `f = (a < 0)` as `f = 0`; the verdict was unaffected
([#7931](https://github.com/esbmc/esbmc/pull/7931)). The NeuroSym backend now
serves counterexamples from its own model output instead of a second solve, and
no longer hangs on a wide shift amount or aborts when `--neurosym-model-prog` is
missing ([#7928](https://github.com/esbmc/esbmc/pull/7928)).

**Function contracts.** Under `--enforce-contract`, struct pointer parameters
are no longer backed by one implicit element: like any other pointer they get a
nondet extent, so `s->field` needs
`__ESBMC_requires(__ESBMC_is_fresh(s, sizeof(*s)))`. Only a C++ `this` keeps one
object, which the language guarantees
([#8042](https://github.com/esbmc/esbmc/pull/8042)). This closes the
contract-soundness umbrella [#6485](https://github.com/esbmc/esbmc/issues/6485).

**Transition-system extraction (experimental).** `--ts-check` extracts the main
loop as a transition system — initial state, one-iteration step relation,
inputs and bad states — prints whether extraction succeeded, and stops with
`VERIFICATION UNKNOWN`; `--ts-dump` prints the system. No engine consumes it
yet ([#8081](https://github.com/esbmc/esbmc/pull/8081);
[usage](/docs/usage#extracting-a-transition-system-experimental)).

The website documentation has been updated to match. As always, the full list is
in the [commit history](https://github.com/esbmc/esbmc/commits/master/) — and if
you hit a bug or a missing feature, we would love an
[issue report](https://github.com/esbmc/esbmc/issues).
