---
title: "Development Update: Early October 2026"
date: 2026-10-13T10:00:00+01:00
draft: false
tags:
  - ESBMC
  - FormalVerification
  - ModelChecking
  - OpenSource
---

Following the [late-September update](/news/development-update-late-september-2026),
222 commits (219 pull requests) landed on `master` between 1 and 13 October,
ending with the v8.6 release. As before, most of the user-visible ones fix a
wrong verdict, so this update starts there; the Python, C/C++, operational
model and Ladder Diagram changes follow.

## Wrong verdicts in the verification engine

**False proofs under `--k-induction`.** The inductive step havocked only the
named variables a loop assigns, so storage the loop writes through a pointer —
`*p`, `p->f`, a callee, `memcpy` — kept its pre-loop value, and the step proved
properties that fail later in the loop:

```c
int main() {
  int x = 0;
  int *p = &x;
  for (;;) {
    *p = *p + 1;          /* x is written only through p */
    __ESBMC_assert(x < 10, "x stays below 10");
  }
}
```

The step now havocs that storage as well, naming its targets with the Andersen
points-to analysis, and turns itself off when it cannot name them. To keep it on
for LDV drivers, Andersen now models body-less calls, varargs, `realloc`,
allocation wrappers and pointers laundered through integers as symex does. The
SV-COMP wrapper now uses k-induction for `valid-memsafety` and
`valid-memcleanup` too ([#8221](https://github.com/esbmc/esbmc/pull/8221)).
Under `--loop-invariant-check`, a loop writing a pointee wider than 1024 bits or
calling a function that writes through a global pointer left the written object
unchanged; such loops are now left to the unwinder
([#8167](https://github.com/esbmc/esbmc/pull/8167),
[#8162](https://github.com/esbmc/esbmc/pull/8162)).

**Interval analysis.** The integer interval domain computed over unbounded
integers. `--interval-analysis` evaluated `-1U` as `[-1, -1]`, so for an
`unsigned x` equal to `UINT_MAX` it proved `x != -1U`
([#8101](https://github.com/esbmc/esbmc/pull/8101)), and
`--interval-analysis-arithmetic` turned `UINT_MAX + 1U` into `2^32` and proved
`x + 1U == 0` unreachable ([#8111](https://github.com/esbmc/esbmc/pull/8111)).
Results are now wrapped into the result type; the same arithmetic backs the
default symex guard. Division sampled only the divisor's endpoints, so with
`x` in `[-2, 2]` and nonzero, `assert(100 / x <= 50)` was proved under
`--interval-analysis --interval-analysis-arithmetic`, although `x == 1` gives
100; a divisor range containing 0 now gives no bound
([#8126](https://github.com/esbmc/esbmc/pull/8126)). Under
`--interval-analysis-wrapped`, the meet of two arcs overlapping at both ends
kept one piece and turned a reachable assume into `ASSUME 0`
([#8208](https://github.com/esbmc/esbmc/pull/8208)). Neither analysis saw what
`scanf`, ESBMC's intrinsics or `__builtin_*_overflow` write through their
arguments: with `--interval-analysis`, `scanf("%d", &x)` left `x == 5`, and by
default symex's own domain kept `n` across `memcpy(&n, ...)` and pruned a
reachable loop iteration. Such calls now havoc what their arguments point into
([#8092](https://github.com/esbmc/esbmc/pull/8092)).

**The memory model.** Every no-overlap constraint between objects was made
conditional on the object's allocation bit, which symex sets only for heap
objects, so once a program called `malloc` and `free`, two live locals could
share an address and
`assert((unsigned long)a != (unsigned long)b)` was a false alarm. Stack and
static objects are now always disjoint
([#8193](https://github.com/esbmc/esbmc/pull/8193)). A floor keeping every
object off the first page, added for Kani's `NonNull::dangling()`
([#8207](https://github.com/esbmc/esbmc/pull/8207)), made every successful
`ldv_malloc` path in SV-COMP's LDV harnesses infeasible, since they accept an
allocation only at or below 2012, and two tasks became wrong `true` answers. It
is now opt-in through `--reserve-first-page`, which Kani and verify-rust-std
runs should pass ([#8288](https://github.com/esbmc/esbmc/pull/8288)).

**Variable-length arrays.** C fixes a VLA's size where its declarator is reached
(C11 6.7.6.2p5), but ESBMC read the size variable wherever the type was used:

```c
int n = 3;
int a[2][n];
a[1][0] = 7;
int (*p)[n] = a;
n = 5;
assert(p[1][0] != 7);     /* p's rows still hold 3 ints */
```

verified, as did the same through a `typedef`, and a VLA redeclared in a loop
took the next iteration's size. Each size is now bound to a local before its
declarator ([#8019](https://github.com/esbmc/esbmc/pull/8019)). Pointer
arithmetic over `int (*p)[m]` scaled by `m` as a free symbol and gave false
bounds violations ([#8109](https://github.com/esbmc/esbmc/pull/8109)).

**The memory builtins.** Symex's `memset`, `memcpy`, `memmove`, `memcmp` and
`memchr` dropped an invalid pointer target without a claim, so
`p = (char *)0x1000; if (c) p = a; memset(p, 0, 4);` verified, and `memcmp` did
not claim its operands non-NULL
([#8154](https://github.com/esbmc/esbmc/pull/8154),
[#8134](https://github.com/esbmc/esbmc/pull/8134)). `realloc` copied only the
first object the old pointer could address
([#8160](https://github.com/esbmc/esbmc/pull/8160)). Under `--big-endian`,
`memset` and `memcpy` placed bytes in little-endian order
([#8135](https://github.com/esbmc/esbmc/pull/8135)), and clang still predefined
the target's `__BYTE_ORDER__` ([#8089](https://github.com/esbmc/esbmc/pull/8089)).
The size behind `__builtin_object_size`, `__CPROVER_OBJECT_SIZE` and `r_ok` was
that of the first object a pointer may address; each candidate now contributes
its own ([#8227](https://github.com/esbmc/esbmc/pull/8227),
[#8236](https://github.com/esbmc/esbmc/pull/8236)).

**`sprintf` and `scanf`.** `sprintf` and `snprintf` never wrote their
destination; symex only computed the return value, so the buffer kept its old
contents and an overflowing write went unchecked:

```c
char b[8] = "abcdefg";
sprintf(b, "x");
assert(b[1] == 'b');      /* b[1] is now '\0' */
```

A constant output is now stored byte by byte through checked dereferences
([#8285](https://github.com/esbmc/esbmc/pull/8285)). For `%d`/`%i`/`%u` of a
nondet integer the exact output is stored, so `char d[2]; sprintf(d, "%d", x)`
with `x` in `[10, 99]` reports the overflow
([#8310](https://github.com/esbmc/esbmc/pull/8310)); where only a bound on the
length is known, nondet characters are written up to it
([#8308](https://github.com/esbmc/esbmc/pull/8308)). The formatter now applies
flags, precision and width, so `printf("%+d", 5)` returns 2
([#8303](https://github.com/esbmc/esbmc/pull/8303)). `scanf`, `sscanf` and
`fscanf` always returned the number of conversions, so
`sscanf("abc", "%d", &x) == 1` verified; the count now ranges from `EOF` to the
number of items, and only the items it covers are stored
([#8294](https://github.com/esbmc/esbmc/pull/8294)).

**`va_list`.** Every `va_list` in a frame shared one cursor
([#8177](https://github.com/esbmc/esbmc/pull/8177)); a `va_list` passed to a
helper, the `vprintf` pattern, was read against the helper's frame, and so was a
`va_copy` made there ([#8174](https://github.com/esbmc/esbmc/pull/8174),
[#8292](https://github.com/esbmc/esbmc/pull/8292)); and a `va_arg` on one
branch advanced the cursor for the path that skipped it
([#8323](https://github.com/esbmc/esbmc/pull/8323)). Each read the wrong
argument, mostly as a false proof.

**What runs after `main`, and `longjmp`.** Symex ended the path at `main`'s
return unless `--memory-leak-check` or `--termination` was set, so `atexit`
handlers were never checked, and the C++ front end never destroyed objects with
static storage duration: a failing assertion in either was `SUCCESSFUL`, and a
global freeing its memory in its destructor was a false leak. Both now run after
`main` returns ([#8339](https://github.com/esbmc/esbmc/pull/8339)). The
`longjmp` model had an empty body, so execution continued after the call:

```c
jmp_buf b;
int g = 0;
void f(void) { g = 1; longjmp(b, 2); }
int main() {
  if (setjmp(b) == 0)
    f();
  else
    assert(g == 0);       /* fails natively */
}
```

verified. A new pass resumes after the matching `setjmp`; it does nothing in a
program that never calls `longjmp`
([#8312](https://github.com/esbmc/esbmc/pull/8312)).

**Concurrency.** Under `--data-races-check`, a lock release followed by a join
or a return gave the releasing thread no context switch, so real races were
missed, including three SV-COMP `ctcs-race-*` tasks. Unlocks now yield under
race checking; other runs are unchanged
([#8191](https://github.com/esbmc/esbmc/pull/8191)). The atomic op-then-fetch,
compare-and-swap, exchange and lock builtins had empty bodies that left the
object unchanged ([#8122](https://github.com/esbmc/esbmc/pull/8122),
[#8124](https://github.com/esbmc/esbmc/pull/8124)), and `pthread_once` never
ran its routine ([#8289](https://github.com/esbmc/esbmc/pull/8289)).

**Smaller cases.** `int i; i /= 3.5;` aborted on a sort error; a compound
assignment now runs in floating point when either side is floating (C11
6.5.16.2p3) ([#8173](https://github.com/esbmc/esbmc/pull/8173)). A compound
literal was built once ahead of an unbraced loop body
([#8128](https://github.com/esbmc/esbmc/pull/8128)). `abs(INT_MIN)` had no
overflow claim under `--overflow-check`
([#8265](https://github.com/esbmc/esbmc/pull/8265)), and under `--gcse`,
`*p + 1` was reused after `free(p)`
([#8071](https://github.com/esbmc/esbmc/pull/8071)).

## Python

**Exceptions CPython raises.** Several operations that raise in CPython were
modelled as succeeding. A local read on a path that never bound it took an
arbitrary value; it now raises `UnboundLocalError`:

```python
def f(c: bool) -> int:
    if c:
        k = 1
    return k

f(nondet_bool())
```

([#8345](https://github.com/esbmc/esbmc/pull/8345)). A module-level call to a
function defined later raises `NameError`
([#8453](https://github.com/esbmc/esbmc/pull/8453)), and a method no longer
reads a class attribute by its bare name, since a class body is not an
enclosing scope ([#8231](https://github.com/esbmc/esbmc/pull/8231)).
Arithmetic on a `None` `Optional[T]` raises `TypeError`
([#8344](https://github.com/esbmc/esbmc/pull/8344)), a method call on `None`
raises `AttributeError` ([#8346](https://github.com/esbmc/esbmc/pull/8346)),
and `int()`, `math.floor`/`ceil`/`trunc` and `round()` of an infinity or NaN
raise `OverflowError` or `ValueError` instead of truncating
([#8272](https://github.com/esbmc/esbmc/pull/8272),
[#8295](https://github.com/esbmc/esbmc/pull/8295)).

**An `assert` ends the path.** `assert` and `raise AssertionError` were lowered
to a plain assertion, so a failing assertion did not stop the path, and an
exception CPython could never reach was reported after it:

```python
def main() -> None:
    l = [0, 1]
    i = nondet_int()
    assert 0 <= i and i < 2
    x = l[i]            # IndexError only where the assert has already failed
    assert x >= 0

main()
```

reported both the assertion and an `IndexError`; only the assertion fails now.
`raise AssertionError` throws like any other exception, so its message no longer
reaches the property comment ([#8270](https://github.com/esbmc/esbmc/pull/8270)).
On top of this, the new `--python-ignore-assertions` drops only the asserts a
Python `assert` produces, keeping the assume after each, so one exception family
can be checked at a time. `--no-assertions` also drops the operational model's
checks: with `r` either `1` or `"s"`, `r / 2` verifies under `--no-assertions`
and reports the `TypeError` under `--python-ignore-assertions`
([#8287](https://github.com/esbmc/esbmc/pull/8287)).

**Evaluation order.** Exception guards and hoisted calls of a later `and`/`or`
operand were planted ahead of the whole statement, so they ran when Python
short-circuits the operand; they now run only when the operand is evaluated
([#8304](https://github.com/esbmc/esbmc/pull/8304),
[#8311](https://github.com/esbmc/esbmc/pull/8311),
[#8300](https://github.com/esbmc/esbmc/pull/8300)). `0 < g() < 5` called `g`
twice ([#8293](https://github.com/esbmc/esbmc/pull/8293)), as did `f() % m`
([#8430](https://github.com/esbmc/esbmc/pull/8430)); a call in an f-string never
ran ([#8309](https://github.com/esbmc/esbmc/pull/8309)).

**Lists and strings.** The list `==` fast path compared lists from where they
were built, so after appends through a parameter `[1, 1] == [1, 2]` was proved
([#8127](https://github.com/esbmc/esbmc/pull/8127)); a store that changes an
element's type (`x[0] = ""` into an int slot) wrote through the old type
([#8349](https://github.com/esbmc/esbmc/pull/8349),
[#8350](https://github.com/esbmc/esbmc/pull/8350)). `["y"] == ["y"]` gave no
verdict without `--unwind` ([#8099](https://github.com/esbmc/esbmc/pull/8099)).
`"-".join([a, "b"])` with a parameter `a` became `"-b"`
([#8424](https://github.com/esbmc/esbmc/pull/8424)), and `str.replace` on an
attribute receiver folded to `""`
([#8446](https://github.com/esbmc/esbmc/pull/8446)). `list(range(n))` with a
non-constant bound had no elements, a false `IndexError`
([#8195](https://github.com/esbmc/esbmc/pull/8195)).

**Types.** `Optional[Tuple]` and `Optional[Dict]` get the `Optional<T>` struct
scalars use; a function returning one lost both its returns
([#8235](https://github.com/esbmc/esbmc/pull/8235)). `isinstance()` on a value
whose type was lost answered `False` and is now refused
([#8315](https://github.com/esbmc/esbmc/pull/8315)); `isinstance` and
`issubclass` accept `X | Y` ([#8397](https://github.com/esbmc/esbmc/pull/8397));
and `type(a) is A` compared against `"pointer"`
([#8420](https://github.com/esbmc/esbmc/pull/8420)). `return x + y` was typed
from a global `x` rather than the parameter, truncating a float result
([#8317](https://github.com/esbmc/esbmc/pull/8317)).

**Newly modelled.** A class nested in a class body
([#8278](https://github.com/esbmc/esbmc/pull/8278)); a `for` over a user
iterator through `__iter__`/`__next__`
([#8332](https://github.com/esbmc/esbmc/pull/8332)); `cls = A; cls()`
([#8393](https://github.com/esbmc/esbmc/pull/8393)); `getattr` with a literal
name ([#8327](https://github.com/esbmc/esbmc/pull/8327)); call-site `*xs`
unpacking ([#8335](https://github.com/esbmc/esbmc/pull/8335),
[#8301](https://github.com/esbmc/esbmc/pull/8301)); a walrus in a `while`
condition ([#8456](https://github.com/esbmc/esbmc/pull/8456)); `str %` with
non-constant operands ([#8245](https://github.com/esbmc/esbmc/pull/8245)); and
the `re` module's flags ([#8264](https://github.com/esbmc/esbmc/pull/8264)). A
`@staticmethod` called on an instance received the receiver
([#8431](https://github.com/esbmc/esbmc/pull/8431)), and a returned lambda's
free variables were unconstrained
([#8443](https://github.com/esbmc/esbmc/pull/8443)). A function the program
never refers to no longer stops conversion when its body cannot be converted
([#8452](https://github.com/esbmc/esbmc/pull/8452)).

**`--python` takes a path.** The value was only looked up in `$PATH`, so an
absolute path was ignored, and an unresolvable name silently fell back to
`python3`. A path is now honoured and an unresolvable value is an error
([#8269](https://github.com/esbmc/esbmc/pull/8269)).

**NumPy views.** Every view is now a pointer into its source's storage with a
per-axis shape and stride, so views of rank 3 or more and views with symbolic
bounds, steps or indices are modelled rather than rejected or verified as copies
([#8155](https://github.com/esbmc/esbmc/pull/8155)). Views keep their aliasing
when passed to or returned from a function or stored in a container
([#8185](https://github.com/esbmc/esbmc/pull/8185),
[#8280](https://github.com/esbmc/esbmc/pull/8280)). Rebinding the source name
detached each view into its own snapshot, so a write through one view was not
seen through another — a false proof
([#8201](https://github.com/esbmc/esbmc/pull/8201)).

## C and C++

**When temporaries are destroyed.** [class.temporary]/4 destroys a temporary at
the end of its full-expression. ESBMC destroyed it at the end of the enclosing
block, or never, in most of the places a temporary can occur:

```cpp
int live = 0;
struct C {
  int v;
  C(int x) : v(x) { ++live; }
  ~C() { --live; }
};
int get(const C &c) { return c.v; }
int main() {
  get(C(1));            // ~C runs at the end of this statement
  assert(live == 1);    // verified; now fails
}
```

Fixed for expression statements ([#8110](https://github.com/esbmc/esbmc/pull/8110)),
`return` statements ([#8108](https://github.com/esbmc/esbmc/pull/8108),
[#8151](https://github.com/esbmc/esbmc/pull/8151)), conditions
([#8125](https://github.com/esbmc/esbmc/pull/8125),
[#8140](https://github.com/esbmc/esbmc/pull/8140),
[#8299](https://github.com/esbmc/esbmc/pull/8299)), declarations
([#8161](https://github.com/esbmc/esbmc/pull/8161)) and comma operands
([#8322](https://github.com/esbmc/esbmc/pull/8322)). Temporaries in an arm of
`?:`, `&&` or `||` that did not run were destroyed anyway
([#8118](https://github.com/esbmc/esbmc/pull/8118)), an aggregate destroyed an
element's initialising temporary as well as the element
([#8113](https://github.com/esbmc/esbmc/pull/8113)), and a temporary bound to a
reference member died before the aggregate
([#8159](https://github.com/esbmc/esbmc/pull/8159)). Most of these were false
proofs on destructor counts; some were false use-after-free reports.

**Copy elision before C++17.** Clang elides these copies before C++17, but ESBMC
ran the copy constructor and destroyed a second object. Under `--std c++14`,
Apple clang's default, a conditional's branches
([#8120](https://github.com/esbmc/esbmc/pull/8120)), a thrown temporary, where
the extra destruction gave a false use-after-free in the handler
([#8117](https://github.com/esbmc/esbmc/pull/8117)), a by-value argument
([#8141](https://github.com/esbmc/esbmc/pull/8141)), and nested copies such as
`C c = C(C(1));` ([#8137](https://github.com/esbmc/esbmc/pull/8137),
[#8138](https://github.com/esbmc/esbmc/pull/8138)) now follow the C++17
lowering.

**Exceptions.** An exception thrown in a callee jumped straight to the
caller's handler, skipping every destructor on the way:

```cpp
int dtors = 0;
struct Guard { ~Guard() { ++dtors; } };
void thrower() { throw 1; }
void f() { Guard g; thrower(); }
int main() {
  try { f(); } catch (int) {}
  assert(dtors == 0);   // verified; now fails
}
```

Each call now unwinds its frame's locals on its exceptional edge
([#8121](https://github.com/esbmc/esbmc/pull/8121)). Per [except.ctor], a
throwing constructor destroys the subobjects already built
([#8139](https://github.com/esbmc/esbmc/pull/8139)), a throwing element in
`new C[n]` destroys the elements already built
([#8142](https://github.com/esbmc/esbmc/pull/8142)), and a delegating
constructor's throwing body destroys the object
([#8336](https://github.com/esbmc/esbmc/pull/8336)).

**`new` and `delete`.** `new S[2][2]` aborted for any struct
([#8076](https://github.com/esbmc/esbmc/pull/8076)); `new char[4]{"ab"}` stored
nothing ([#8105](https://github.com/esbmc/esbmc/pull/8105)); array placement
`new (buf) T[n]` built the elements in fresh memory
([#8107](https://github.com/esbmc/esbmc/pull/8107)); and a program's placement
or aligned `operator new` was never called
([#8146](https://github.com/esbmc/esbmc/pull/8146),
[#8297](https://github.com/esbmc/esbmc/pull/8297)). `delete[]` now destroys
elements in decreasing order of address
([#8153](https://github.com/esbmc/esbmc/pull/8153)).

**Other C++ fixes.** A function-local static array with a dynamic initialiser
was initialised before `main` ([#8115](https://github.com/esbmc/esbmc/pull/8115),
[#8168](https://github.com/esbmc/esbmc/pull/8168)). An override with unnamed
parameters forwarded its last argument for all of them
([#8093](https://github.com/esbmc/esbmc/pull/8093)). Same-named local classes in
different functions shared one `type_info`
([#8334](https://github.com/esbmc/esbmc/pull/8334)). And ESBMC builds against
LLVM 18 again, the minimum `CMakeLists.txt` declares
([#8143](https://github.com/esbmc/esbmc/pull/8143)).

## Operational models and builtins

A body-less library function or builtin returns a nondet value and writes
nothing, so a program relying on its result gets a false alarm, and one relying
on its reads or writes can get a false proof. Now modelled:

- **Math.** `cbrt`, `cbrtf` and `cbrtl`
  ([#8273](https://github.com/esbmc/esbmc/pull/8273),
  [#8298](https://github.com/esbmc/esbmc/pull/8298)); `ilogb` and `logb`
  ([#8268](https://github.com/esbmc/esbmc/pull/8268)); and the `__builtin_`
  spellings of `<math.h>` functions, which now reach the models
  ([#8176](https://github.com/esbmc/esbmc/pull/8176)). `log10` came from a
  truncated series, so `assert(log10(1000.0) != 3.0)` verified; it is now
  musl's ([#8324](https://github.com/esbmc/esbmc/pull/8324)). `ldexpl` and
  `frexpl` read a 128-bit `long double` with the x87 mantissa width
  ([#8290](https://github.com/esbmc/esbmc/pull/8290)).
- **C library.** `strnlen`, `strndup`, `strcasecmp` and `strncasecmp`, so their
  over-reads are now reported
  ([#8296](https://github.com/esbmc/esbmc/pull/8296),
  [#8320](https://github.com/esbmc/esbmc/pull/8320),
  [#8328](https://github.com/esbmc/esbmc/pull/8328)); `qsort` and `bsearch`
  ([#8279](https://github.com/esbmc/esbmc/pull/8279)).
- **Bit builtins.** `__builtin_clrsb`, the rotate and bit-reverse builtins, and
  `__builtin_abs`/`labs`/`llabs`
  ([#8169](https://github.com/esbmc/esbmc/pull/8169),
  [#8166](https://github.com/esbmc/esbmc/pull/8166),
  [#8266](https://github.com/esbmc/esbmc/pull/8266),
  [#8170](https://github.com/esbmc/esbmc/pull/8170)).

## CBMC and Kani goto binaries

Kani tags informational assertions with the `reachability_check` and `cover`
property classes, which CBMC reports as failures and Kani discounts; the new
repeatable `--ignore-property-class <class>` skips them
([#8157](https://github.com/esbmc/esbmc/pull/8157)). Under `--function`, a CBMC
binary's globals started nondet, because Kani's binaries carry no
`__CPROVER_initialize`; they are now initialised
([#8197](https://github.com/esbmc/esbmc/pull/8197),
[#8229](https://github.com/esbmc/esbmc/pull/8229)). The first-object size fixed
by [#8227](https://github.com/esbmc/esbmc/pull/8227) above had failed 45 of
verify-rust-std's 53 `align_to` contract harnesses.

## Ladder Diagram

**Scan semantics that match MATIEC.** `TON` fired and `TOF` released one scan
early ([#8179](https://github.com/esbmc/esbmc/pull/8179)). IEC `INT` was a
32-bit integer, so a sum above 32767 did not wrap as in the MATIEC reference; it
is now 16-bit, and `UINT`, `SINT`, `LINT` and `WORD`, which have no model, are
refused ([#8198](https://github.com/esbmc/esbmc/pull/8198)). A timer's `Q` read
by a contact was inferred to be an input and started each scan nondeterministic
([#8445](https://github.com/esbmc/esbmc/pull/8445)). The `mutual_exclusion`
monitor checked only that not every listed variable holds at once; it now
checks every pair ([#8214](https://github.com/esbmc/esbmc/pull/8214)).

**Refusals instead of wrong proofs.** The front end now refuses what it used to
verify as something else: negated and edge coils, read as plain coils, and
`negated="1"`, read as not negated
([#8216](https://github.com/esbmc/esbmc/pull/8216)); an unrecognised type such
as `BYTE`, read as `BOOL` ([#8217](https://github.com/esbmc/esbmc/pull/8217));
a wire to a pin the block lacks, such as `PTT` for `PT`
([#8186](https://github.com/esbmc/esbmc/pull/8186)); a cyclic wire graph
([#8215](https://github.com/esbmc/esbmc/pull/8215)); a comparison block beside a
live predecessor, so `y = a OR (n > 3)` no longer verifies as `y = a`
([#8435](https://github.com/esbmc/esbmc/pull/8435)); aliased timer and counter
pins and coils on inputs ([#8218](https://github.com/esbmc/esbmc/pull/8218));
and wired `EN`/`ENO` pins and repeated declarations
([#8521](https://github.com/esbmc/esbmc/pull/8521)).

## SV-COMP wrapper

The wrapper supports SV-COMP's new Python track, whose properties read
`CHECK( init(main()), ! uncaught(<ExceptionType>))` and were all answered
`Unsupported Property`; the exception family a run falsified is read from the
violated property rather than assumed
([#8190](https://github.com/esbmc/esbmc/pull/8190),
[#8348](https://github.com/esbmc/esbmc/pull/8348)). SV-COMP no longer assumes
allocation succeeds, so the wrapper drops `--force-malloc-success` and
`--force-realloc-success` ([#8129](https://github.com/esbmc/esbmc/pull/8129)).
Pthread benchmarks now run with `--context-bound 3`, which saves timeouts at a
cost: a run that hits the bound still reports `VERIFICATION SUCCESSFUL`, so a
bug that needs more than three switches becomes a wrong `true`
([#8104](https://github.com/esbmc/esbmc/pull/8104)).

## ESBMC v8.6

The version bump ([#8491](https://github.com/esbmc/esbmc/pull/8491)) closes this
period, and these changes ship in
[ESBMC v8.6](https://github.com/esbmc/esbmc/releases/tag/v8.6), with binaries
for Linux (x86_64 and ARMv8), macOS and Windows. The release also contains the
work covered in the September updates, including the multi-property reporting
under the k-step strategies, the NumPy additions and the new C++
standard-library headers.

The website documentation has been updated to match. As always, the full list is
in the [commit history](https://github.com/esbmc/esbmc/commits/master/) — and if
you hit a bug or a missing feature, we would love an
[issue report](https://github.com/esbmc/esbmc/issues).
