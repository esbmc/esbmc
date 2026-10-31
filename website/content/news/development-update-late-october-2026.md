---
title: "Development Update: Late October 2026"
date: 2026-10-27T10:00:00+00:00
draft: false
tags:
  - ESBMC
  - FormalVerification
  - ModelChecking
  - OpenSource
---

The [early-October update](/news/development-update-early-october-2026) ended at
the v8.6 release. Since then, 515 commits from 509 pull requests landed on
`master` between 13 and 27 October; about three in five fix the Python front
end. This update starts with the wrong verdicts in the verification engine, then
covers Python, NumPy, C and C++, the operational models and the smaller front
ends, and ends with a table of the new options.

## Wrong verdicts in the verification engine

**False proofs in the default configuration.** A GNU case range,
`case lo ... hi:`, matched only `lo`, so a value inside the range went to
`default`. ESBMC missed bugs inside the range and raised false alarms on
fallthrough. This program verified:

```c
int v = nondet_int(), r = 0;
switch (v)
{
case 1 ... 9: r = 1; break;
default:      r = 2;
}
if (v == 5)
  __ESBMC_assert(r == 2, "5 is in the range");   /* now FAILED */
```

The arm is now guarded by `lo <= v && v <= hi`, in C and C++
([#8717](https://github.com/esbmc/esbmc/pull/8717)). A jump into a block — a
`goto`, a sideways `goto`, or a `switch` case in a nested block — emitted no
declaration for the block's locals, so symex read their values from the
previous lifetime:

```c
for (int pass = 0; pass < 2; ++pass)
{
  if (pass == 1)
    goto WAIT;
  {
    int i = 5;
    continue;
  WAIT:
    r = i;     /* i is indeterminate here (C11 6.2.4p6) */
  }
}
assert(r == 5);   /* now FAILED with --unwind 3 */
```

The jump now declares the locals it skips
([#9194](https://github.com/esbmc/esbmc/pull/9194)). An integer-to-pointer cast
matched only the objects registered when the cast was converted, so with the
slicer on, `p != &a` could be proved for `p = (int *)n` when `&a` first appeared
later. Static variables and functions are now registered up front
([#9266](https://github.com/esbmc/esbmc/pull/9266)); an automatic local whose
address is taken before the cast is still open
([#1539](https://github.com/esbmc/esbmc/issues/1539)). A pointer built from an
integer, or read back from stored bits, can now also be an object registered
later, which closes [#7855](https://github.com/esbmc/esbmc/issues/7855)
([#8788](https://github.com/esbmc/esbmc/pull/8788),
[#9294](https://github.com/esbmc/esbmc/pull/9294)). The SV-COMP task
`aws_hash_table_create_harness` moves from a wrong `false` to a timeout, not to
a correct `true`.

**False alarms from the simplifier.** A write to a narrower member of a
constant union zeroed the rest of the wider member:

```c
union { unsigned char c; unsigned int i; } u = {0};
u.i = 0x1200;
u.c = 5;
__ESBMC_assert(u.i == 0x1205, "");   /* was FAILED; now SUCCESSFUL */
```

([#9061](https://github.com/esbmc/esbmc/pull/9061)). Folding `c ? 1 : 0` to
`(T)c` also fired when the zero arm was `-0.0`, which turned it into `+0.0`: a
missed bug on `c ? -0.0 : 1.0` and a false alarm on `signbit(fmin(-0.0, 1.0))`
([#8736](https://github.com/esbmc/esbmc/pull/8736)).

**Overflow checks.** The shared bit-vector overflow encodings, used by Z3 and
the other backends without a native operation, missed `INT_MIN + INT_MIN`, which
wraps to 0 ([#8728](https://github.com/esbmc/esbmc/pull/8728)), and
`0 - INT_MIN` ([#8741](https://github.com/esbmc/esbmc/pull/8741)):

```c
int a = nondet_int();
__ESBMC_assume(a == INT_MIN);
int r = a + a;    /* --overflow-check --z3: now FAILED, overflow on add */
```

In the other direction, Bitwuzla reported `0x80000000u / 0xffffffffu` as an
overflow under `--unsigned-overflow-check`
([#8742](https://github.com/esbmc/esbmc/pull/8742)), and `c++` on a
`signed char` holding 127 was reported although C11 6.5.2.4p2 promotes the
operand to `int` ([#8744](https://github.com/esbmc/esbmc/pull/8744)). Two
`_BitInt(40)` operands were widened to `long`, so their 40-bit overflow was
never tested ([#8837](https://github.com/esbmc/esbmc/pull/8837)). Under `--ir`,
SMT-LIB's Euclidean division made `-7 / 2` equal -4; C's `/` and `%` and
Python's `//` and `%` are now correct there
([#8715](https://github.com/esbmc/esbmc/pull/8715)), and a cast to a signed type
is checked against the signed range, so `(long)-1` is no longer an overflow
([#8778](https://github.com/esbmc/esbmc/pull/8778),
[#8781](https://github.com/esbmc/esbmc/pull/8781)).

**Function contracts and loop invariants.** In contract mode every
`__ESBMC_requires` and `__ESBMC_ensures` becomes an assumption, and only
enforced or replaced functions had them removed. In any other function, an
`ensures` pruned every path through it:

```c
int width(int c)                 /* neither replaced nor enforced */
{
  __ESBMC_ensures(__ESBMC_return_value == 1);
  return c <= 0xFFFF ? 1 : 2;
}
/* main calls set(x), which has its own contract, then: */
__ESBMC_assert(width(x) == 1, "width can be 2");
/* --replace-call-with-contract set: now FAILED */
```

The clauses are now stripped from every function after the passes run
([#9182](https://github.com/esbmc/esbmc/pull/9182)). An assigns clause naming
`c->p` accepted a write through `c->p`
([#9187](https://github.com/esbmc/esbmc/pull/9187)). Under
`--loop-invariant-check`, a base case whose only counterexample runs through the
invariant's havoc came back UNSAT, so `--k-induction` and `--incremental-bmc`
proved programs with a real bug, and `--k-induction-parallel` reported a correct
program as `FAILED`. Such a base case is now unknown
([#8869](https://github.com/esbmc/esbmc/pull/8869)). This costs proofs: the
issue's program now ends `VERIFICATION UNKNOWN`.

**Interval analysis and `--error-label`.** The wrapped interval join dropped
values (`[2,3] ⊔ [3,0]` gave `[3,0]`), so
`--interval-analysis --interval-analysis-wrapped` proved `x != 2` for an `x`
that can be 2 ([#8814](https://github.com/esbmc/esbmc/pull/8814)). With
`--error-label` on the first statement of a loop body, the loop body was removed
and symex checked a later assertion on a path the program does not have
([#9062](https://github.com/esbmc/esbmc/pull/9062)).

**Memory and the C library.** Releasing `alloca` storage ran `free()`'s checks,
so every `alloca` raised a false "invalid pointer freed"
([#8723](https://github.com/esbmc/esbmc/pull/8723)), and `__builtin_alloca`,
which `<alloca.h>` uses, was never released, so a use after return verified
([#8727](https://github.com/esbmc/esbmc/pull/8727)):

```c
#include <alloca.h>
int *f(void) { int *p = alloca(sizeof(int)); *p = 1; return p; }
int main(void) { int *p = f(); return *p; }
/* now FAILED: dereference failure: invalidated dynamic object */
```

A failed `realloc` was reported as leaking the object it never created
([#8566](https://github.com/esbmc/esbmc/pull/8566)). The `printf` family's
return value was only bounded by the formatter's range, so
`assert(printf("%d", x) == 3)` failed for `x` in `[100, 999]`; it is now the
length written ([#8573](https://github.com/esbmc/esbmc/pull/8573)). `sprintf`
measures `%s`, field widths and `%o`/`%x`/`%X` exactly, which removes false
out-of-bounds reports on small buffers
([#8563](https://github.com/esbmc/esbmc/pull/8563),
[#9186](https://github.com/esbmc/esbmc/pull/9186)), and `asprintf`'s buffer
holds `retval + 1` bytes instead of one
([#9167](https://github.com/esbmc/esbmc/pull/9167)).
`__builtin_object_size` ignored the pointer's offset
([#8775](https://github.com/esbmc/esbmc/pull/8775)), and `__CPROVER_r_ok`
counted elements rather than bytes, so `!r_ok(a, 8)` was proved for an
`int[4]` ([#8817](https://github.com/esbmc/esbmc/pull/8817)). In a program that
can spawn threads, a counter loop over a shared global was
collapsed to its final value, hiding the values another thread can read
([#8786](https://github.com/esbmc/esbmc/pull/8786)).

**Layout that disagreed with clang.** Under `--32`, `long long` and `double`
struct members were aligned to 8 where clang uses 4
([#8758](https://github.com/esbmc/esbmc/pull/8758)); `_Complex` members had
alignment 1 ([#8759](https://github.com/esbmc/esbmc/pull/8759),
[#8762](https://github.com/esbmc/esbmc/pull/8762)); and `_BitInt(65)` took 16
where x86-64 caps it at 8 ([#8773](https://github.com/esbmc/esbmc/pull/8773)).
Each made a struct `memcpy` lose or overread part of a member. On AArch64 Darwin
and Windows, `long double` is now modelled as the IEEE double it is there
([#8863](https://github.com/esbmc/esbmc/pull/8863)).

## Speed

Symex and SMT encoding time grew quadratically in `--unwind`; they now scale
linearly ([#9169](https://github.com/esbmc/esbmc/pull/9169)); a follow-up
keeps the member value sets of a dead local, which a later read still needs
([#9253](https://github.com/esbmc/esbmc/pull/9253)). Flattened array rows are
padded to power-of-two strides, so a symbolic index costs a shift rather than a
multiply: Bitwuzla's solve time on the PR's benchmark drops from 0.76–0.91 s to
0.27–0.31 s, at the price of more memory for padded constant tables
([#9235](https://github.com/esbmc/esbmc/pull/9235)). Z3's multiplication
overflow check uses Z3's own predicates, 9.2 s to 1.1 s on the issue's program
([#9279](https://github.com/esbmc/esbmc/pull/9279)).

## Python

**List and string folds that proved what CPython refutes.** The preprocessor
folds `l[i]` to an element of the literal `l` was bound to, and after an
`if`/`else` that was one arm's literal:

```python
a = nondet_int()
if a > 0:
    l = [1]
else:
    l = [2]
assert l[0] == 2      # was SUCCESSFUL; now FAILED
```

A literal bound in a branch is now forgotten at the join
([#8625](https://github.com/esbmc/esbmc/pull/8625)), as one a loop changes is
([#8574](https://github.com/esbmc/esbmc/pull/8574)). A str name had the same
defect ([#8427](https://github.com/esbmc/esbmc/pull/8427)), and `sep.join(xs)`
was folded even when `xs` changed through an alias or a callee
([#8552](https://github.com/esbmc/esbmc/pull/8552)). `list(xs)` returned `xs`
itself:

```python
xs = [1, 2]
ys = list(xs)
ys.append(3)
assert len(xs) == 3   # was SUCCESSFUL; now FAILED
```

It is now a shallow copy ([#8851](https://github.com/esbmc/esbmc/pull/8851),
[#8856](https://github.com/esbmc/esbmc/pull/8856)). Other false proofs of this
kind: `d = m[:]; d[0] = [5]` also changed `m[0]`
([#8746](https://github.com/esbmc/esbmc/pull/8746),
[#8747](https://github.com/esbmc/esbmc/pull/8747)); `b[0] = 1` over a str
element kept the str type ([#8605](https://github.com/esbmc/esbmc/pull/8605));
`d, n = xs.pop()` popped twice ([#8772](https://github.com/esbmc/esbmc/pull/8772));
`return a + [1]` was typed as a float
([#9051](https://github.com/esbmc/esbmc/pull/9051)); `data[0]["c"]` was read as a
str ([#9016](https://github.com/esbmc/esbmc/pull/9016)); and `f"{s[1]}!"`
rendered the character as its code
([#8725](https://github.com/esbmc/esbmc/pull/8725)).

**Classes.** An inherited method was found depth-first, not in C3 order:

```python
class A:
    def func(self) -> int: return 0
class B(A): pass
class C(A):
    def func(self) -> int: return 1
class D(B, C): pass
assert D().func() == 0   # was SUCCESSFUL; now FAILED (C.func runs)
```

([#8987](https://github.com/esbmc/esbmc/pull/8987)). A base-class method calling
`self.g()` ran the base's `g` on a subclass that overrides it
([#9058](https://github.com/esbmc/esbmc/pull/9058)), and `super(C, self)`
ignored `C` ([#8634](https://github.com/esbmc/esbmc/pull/8634)). A decorator the
program defines was not applied, so `@double def g` verified `g(3) == 3`;
decorated functions are now rebound as in Python, and decorator calls such as
`@times(3)` are refused ([#8618](https://github.com/esbmc/esbmc/pull/8618)).
`isinstance(True, int)` was `False`
([#8584](https://github.com/esbmc/esbmc/pull/8584)), `isinstance(A, str)` for a
class `A` was `True` ([#8586](https://github.com/esbmc/esbmc/pull/8586)), and two
classes with the same layout matched each other
([#8400](https://github.com/esbmc/esbmc/pull/8400)).

A name holds one C type, the class it is first bound to. Where it holds
instances of different classes on different paths, method calls, attribute
reads, copies, container stores and `isinstance` used that class on every path;
these are now refused ([#8623](https://github.com/esbmc/esbmc/pull/8623),
[#8799](https://github.com/esbmc/esbmc/pull/8799),
[#9056](https://github.com/esbmc/esbmc/pull/9056)), while `a = A(); a = C()` on
straight-line code is accepted ([#9259](https://github.com/esbmc/esbmc/pull/9259)).
Attribute reads that the struct layout cannot answer are refused too: one added
on some paths only, one the class may own instead, and one a skipped base
`__init__` sets ([#8643](https://github.com/esbmc/esbmc/pull/8643),
[#8637](https://github.com/esbmc/esbmc/pull/8637),
[#8852](https://github.com/esbmc/esbmc/pull/8852)). A method call on a receiver
of unknown class was dropped, with its writes
([#9042](https://github.com/esbmc/esbmc/pull/9042)).

**Ints and floats.** Python does not enforce `-> int`, so
`def foo() -> int: x = 1.114; return x` returns 1.114, but ESBMC truncated it
and proved `foo() == 1`. A float returned from a function annotated `-> int` is
now checked to be finite and integral, the only case the int slot holds exactly
([#8849](https://github.com/esbmc/esbmc/pull/8849)), and a float passed to an
`int` parameter is refused rather than cast
([#9189](https://github.com/esbmc/esbmc/pull/9189)). Folding `float()`, `abs()`,
`round()` or `str()` of an int literal above 2**31 wrapped it
([#8738](https://github.com/esbmc/esbmc/pull/8738)). An int-or-float name is one
C double, so an int above 2**53 was rounded; such a read is now refused
([#8761](https://github.com/esbmc/esbmc/pull/8761)), and a function called with
both kinds is specialised per kind
([#8639](https://github.com/esbmc/esbmc/pull/8639),
[#8955](https://github.com/esbmc/esbmc/pull/8955)). Python ints are unbounded,
but the default bit-vector encoding still wraps them at 64 bits
([#8892](https://github.com/esbmc/esbmc/issues/8892)).

**Transcendental functions.** `math.exp(x)`, `np.exp(x)` and the like on a
run-time value used ESBMC's C math models, whose values are not the C
library's:

```python
import math
x: float = nondet_float()
if x == 1.0:
    assert math.exp(x) != 2.718281828459045   # was SUCCESSFUL; now FAILED
```

`exp`, `log`, `log2`, `log10`, `sqrt`, the trigonometric and hyperbolic
functions and their inverses now return any double within 8 ulps of the C
library's value, clamped to the function's range, with exact results for NaN,
infinities and signed zeros; `math.*` raises `ValueError` and `OverflowError`
where CPython does ([#9198](https://github.com/esbmc/esbmc/pull/9198)). This
costs proofs that relied on an exact value: `assert np.exp(x) >= np.e` at
`x == 1.0` now fails, and is proved again with `--python-math-ulps 0`, which
assumes a correctly rounded library and is unsound where NumPy uses SVML.

**None and exceptions.** An `Optional[int]` index holding `None` read `l[0]`;
list and str indexing now raise `TypeError` on that path
([#9215](https://github.com/esbmc/esbmc/pull/9215)). Only the root of an
attribute chain was checked for `None`, so `a.nxt.val` with `a.nxt` left `None`
was proved safe ([#9120](https://github.com/esbmc/esbmc/pull/9120)); a read
through `None` is now an `AttributeError` rather than a C NULL dereference after
which the path went on ([#8756](https://github.com/esbmc/esbmc/pull/8756)).
`int(None)`, `float(None)` and `x in s` for an `Optional[str]` `s` holding
`None` verified
([#9099](https://github.com/esbmc/esbmc/pull/9099),
[#9100](https://github.com/esbmc/esbmc/pull/9100),
[#9101](https://github.com/esbmc/esbmc/pull/9101)). The `else` clause of a `try`
ran inside the `try`, so its own handlers caught what it raised:

```python
def f() -> int:
    try:
        x = 1
    except ValueError:
        return -1
    else:
        raise ValueError("raised in else")
assert f() == -1   # was SUCCESSFUL; now FAILED: uncaught exception: ValueError
```

([#9102](https://github.com/esbmc/esbmc/pull/9102)). A `TypeError` on a
dynamically typed operand and an `IndexError` from `pop()` were model asserts
that `except` could not catch; they are now raised
([#8636](https://github.com/esbmc/esbmc/pull/8636)). Programs that CPython
rejects with `TypeError` verified, among them str arithmetic beyond `+`
([#8562](https://github.com/esbmc/esbmc/pull/8562)) and `list + tuple`
([#9097](https://github.com/esbmc/esbmc/pull/9097)), and so did `int("1 2")`
([#8740](https://github.com/esbmc/esbmc/pull/8740)). A call whose arguments do
not bind now raises `TypeError` when it runs, instead of stopping conversion
even when it is never reached ([#8619](https://github.com/esbmc/esbmc/pull/8619)).

**Asking about one exception family.** Without `--multi-property`, a Python run
stops at the first exception that can escape, so a reachable
`ZeroDivisionError` left "can an `IndexError` escape?" unanswered.
`--python-check-exceptions IndexError,KeyError` keeps as properties only the
listed types and their subclasses; any other uncaught exception ends its path
unreported, and asserts become assumptions unless `AssertionError` is covered
([#8522](https://github.com/esbmc/esbmc/pull/8522)). The SV-COMP wrapper uses it
to scope each Python property ([#8585](https://github.com/esbmc/esbmc/pull/8585)),
and confirms a Python violation by re-running with `--ir --z3`, keeping `false`
only when the re-run reports the same property at the same line
([#9171](https://github.com/esbmc/esbmc/pull/9171)).

**Function values.** A call through a value that may hold functions with
different return types got one return type, so `isinstance` and comparisons
disagreed with CPython on every path but one. Such calls are now refused
([#8657](https://github.com/esbmc/esbmc/pull/8657)), and the common shapes,
such as `h = f if c else g; h()` and `def call(h): return h()`, are rewritten so
each path sees its own result type
([#9039](https://github.com/esbmc/esbmc/pull/9039),
[#9149](https://github.com/esbmc/esbmc/pull/9149)). Calling a function held in
an attribute, a tuple or a dict, or returned by another call, now converts
([#8436](https://github.com/esbmc/esbmc/pull/8436),
[#8599](https://github.com/esbmc/esbmc/pull/8599),
[#9006](https://github.com/esbmc/esbmc/pull/9006)).

**Values whose type differs by path.** A name or return holding an int on one
path and a str on another is a tagged value. More shapes now produce one, such
as a return that mixes a str with a parameter
([#8536](https://github.com/esbmc/esbmc/pull/8536)) or a rebind in an `if`
without `else` ([#9107](https://github.com/esbmc/esbmc/pull/9107)), and
`type()` and truthiness of a tagged value are decided per path
([#9145](https://github.com/esbmc/esbmc/pull/9145),
[#9270](https://github.com/esbmc/esbmc/pull/9270)). Several dozen further PRs
type an unannotated parameter, attribute or result that was left `void *` or
`double`, mostly turning a false alarm or a crash into CPython's verdict. Two
also removed false proofs: a parameter typed in the callee's scope rather than
the caller's ([#8785](https://github.com/esbmc/esbmc/pull/8785)), and an
imported function's float parameter, which was truncated
([#8582](https://github.com/esbmc/esbmc/pull/8582)).

**Newly modelled constructs.** These were refused or stopped conversion before:

- `match` with literal, value, capture, wildcard, `as` and `|` patterns, lowered
  to `if`/`elif`; sequence, mapping and class patterns are refused by name
  ([#8591](https://github.com/esbmc/esbmc/pull/8591)).
- A module-level `async def` awaited directly or passed to `asyncio.run`
  ([#9072](https://github.com/esbmc/esbmc/pull/9072)).
- `**kwargs` ([#9002](https://github.com/esbmc/esbmc/pull/9002)), forwarded
  `*args` ([#8870](https://github.com/esbmc/esbmc/pull/8870)) and `nonlocal`
  ([#8729](https://github.com/esbmc/esbmc/pull/8729)).
- `yield from` ([#9213](https://github.com/esbmc/esbmc/pull/9213)), and `iter()`
  and `next()` ([#8883](https://github.com/esbmc/esbmc/pull/8883)).
- `collections.namedtuple` ([#8824](https://github.com/esbmc/esbmc/pull/8824)),
  `abc.ABC` ([#8550](https://github.com/esbmc/esbmc/pull/8550)), and
  `ArithmeticError` and `LookupError`
  ([#8570](https://github.com/esbmc/esbmc/pull/8570)).
- Dict merge with `**`, `|` and `|=` ([#8602](https://github.com/esbmc/esbmc/pull/8602)),
  `functools.reduce` ([#8656](https://github.com/esbmc/esbmc/pull/8656)), and a
  drained `itertools.chain` or `repeat`
  ([#8829](https://github.com/esbmc/esbmc/pull/8829)).
- `io.StringIO` ([#8621](https://github.com/esbmc/esbmc/pull/8621)),
  `subprocess.run` ([#9172](https://github.com/esbmc/esbmc/pull/9172)) and six
  `os` and `os.path` functions ([#9244](https://github.com/esbmc/esbmc/pull/9244)).
- `unittest.TestCase` assertions, which compared lists by address
  ([#9276](https://github.com/esbmc/esbmc/pull/9276)), and `%f` of a run-time
  float, rounded half to even ([#8791](https://github.com/esbmc/esbmc/pull/8791)).
- `len(l)`, `l[k]` and `__ESBMC_assigns` in Python contracts, so
  `--replace-call-with-contract` no longer havocs every global
  ([#8767](https://github.com/esbmc/esbmc/pull/8767),
  [#8753](https://github.com/esbmc/esbmc/pull/8753)).

**Imports and termination.** `from pkg import mod` for a submodule
([#8607](https://github.com/esbmc/esbmc/pull/8607)) and `import pkg.sub`
([#8992](https://github.com/esbmc/esbmc/pull/8992)) now resolve, and with
`import one; import two`, `two.f()` no longer runs `one.f`
([#9059](https://github.com/esbmc/esbmc/pull/9059)). `if "":` ran `strlen`
without end ([#8720](https://github.com/esbmc/esbmc/pull/8720)), and `2 ** n`
for a symbolic `n` ran out of memory in the floating-point model
([#8880](https://github.com/esbmc/esbmc/pull/8880)).

## NumPy

- **Views** passed to user functions cover subviews taken in the callee, 2-D
  row slices, run-time bounds and several layouts per callee
  ([#8538](https://github.com/esbmc/esbmc/pull/8538)), and can be passed to
  builtins, methods, constructors and lambdas
  ([#8641](https://github.com/esbmc/esbmc/pull/8641)).
- **`dtype=` and constructors** share one normaliser; `object` and structured
  dtypes are refused. String and complex arrays are modelled, where the
  imaginary part used to be dropped, and `linspace(endpoint=False)` is no longer
  ignored ([#8795](https://github.com/esbmc/esbmc/pull/8795)).
- **`np.random`** distributions and `choice` are nondeterministic values in each
  distribution's support; `seed`, `default_rng` and `choice(p=)` are refused.
  The same PR fixes wrong successes: `int_array * 2.5` truncated the scalar,
  `v[1].sum()` on a view row folded to 0, and rebinding an array to another
  shape retyped earlier reads ([#8902](https://github.com/esbmc/esbmc/pull/8902)).
- **`np.nditer`** gains its flags, multiple broadcast operands and the iteration
  protocol. `np.linalg.det`, `inv` and `solve` are bounded around LAPACK's
  rounding, so `det([[1., 2.], [3., 4.]]) == -2.0` now fails, as under CPython
  ([#9066](https://github.com/esbmc/esbmc/pull/9066)).
- **Transcendental functions** of run-time arguments are bounded as described
  above ([#9198](https://github.com/esbmc/esbmc/pull/9198)), and `np.array()`
  of a list that is not a literal is refused instead of typed as a
  nondeterministic rank-1 array ([#9141](https://github.com/esbmc/esbmc/pull/9141)).

## C and C++

**C.** `__builtin_popcountg` had no model, so its result was nondeterministic
([#8718](https://github.com/esbmc/esbmc/pull/8718)). An `unsigned char`
enumerator of 200 read as -56 ([#8748](https://github.com/esbmc/esbmc/pull/8748)).
When several files are verified together, a weak definition beat a later strong
one ([#8842](https://github.com/esbmc/esbmc/pull/8842),
[#8836](https://github.com/esbmc/esbmc/pull/8836)). A label in front of
`int x, y;` ended the variables' lifetime right after declaring them
([#9178](https://github.com/esbmc/esbmc/pull/9178)), and
`__builtin_constant_p(h())` for a `void` call crashed symex
([#9188](https://github.com/esbmc/esbmc/pull/9188)).

**C++.** `throw "abc"` stored the array, so `catch (const char *)` read the
bytes as a pointer ([#8749](https://github.com/esbmc/esbmc/pull/8749)).
`T t; throw t.v;` threw the value `~T` left behind
([#8787](https://github.com/esbmc/esbmc/pull/8787)). A destructor throwing during
unwinding now calls `std::terminate`
([#8867](https://github.com/esbmc/esbmc/pull/8867)). `catch (B *)` of a `C *`
with `C : A, B` saw the `A` subobject
([#8793](https://github.com/esbmc/esbmc/pull/8793)), and `catch (E e)` of a
thrown `F` dispatched to `F` ([#8800](https://github.com/esbmc/esbmc/pull/8800)).
`(p->*pmf)()` through a pointer to a virtual member ran the static class's
method ([#8834](https://github.com/esbmc/esbmc/pull/8834)); a covariant override
returned an unadjusted pointer ([#8792](https://github.com/esbmc/esbmc/pull/8792));
`dynamic_cast` used the wrong base offset
([#8802](https://github.com/esbmc/esbmc/pull/8802)) and now fails for an
ambiguous base, as [expr.dynamic.cast]/8 requires
([#8803](https://github.com/esbmc/esbmc/pull/8803)). C++20 concepts and
`requires` expressions are converted
([#8432](https://github.com/esbmc/esbmc/pull/8432)).

## Operational models

`strtod` added `digit * 0.1`, `digit * 0.01`, ... rounding at each step, so
`strtod("0.3")` was 0.30000000000000004, and `atof` had no model. When the digits
are exact in the result type the result is now correctly rounded (C11 F.5), and
`atof` is `strtod(nptr, NULL)` ([#8798](https://github.com/esbmc/esbmc/pull/8798)):

```c
assert(strtod("0.3", NULL) == 0.3);   /* now SUCCESSFUL */
assert(atof("2.5") == 2.5);
```

`strtol` and its siblings accepted 8 and 9 below base 10, misplaced `endptr`,
and wrote `errno` where the program does not read it
([#8733](https://github.com/esbmc/esbmc/pull/8733)). `fesetround` accepted an
unknown mode as round-to-nearest; it now returns nonzero and leaves the mode
unchanged, per C11 7.6.3.2 ([#8730](https://github.com/esbmc/esbmc/pull/8730)).
`atan2` lost the sign of zero and mishandled NaN and infinities
([#8779](https://github.com/esbmc/esbmc/pull/8779)), `hypot` overflowed well
inside the double range ([#8782](https://github.com/esbmc/esbmc/pull/8782)), and
`ldexp` truncated subnormal results
([#8780](https://github.com/esbmc/esbmc/pull/8780)). In the C++ library,
`std::map::at` inserted a missing key and `unordered_map::at` returned a default
value; both now throw `std::out_of_range`
([#8750](https://github.com/esbmc/esbmc/pull/8750)), as do `string::substr`,
`string::erase`, `deque::at`, `string_view::at` and `array::at`, which asserted
or did not check ([#8754](https://github.com/esbmc/esbmc/pull/8754)).

## Solidity

`--overflow-check` reported overflow on correct contracts for a `<<`, which
Solidity truncates ([#8763](https://github.com/esbmc/esbmc/pull/8763)), for a
negative `int8` cast result while missing `127 + 1`
([#8781](https://github.com/esbmc/esbmc/pull/8781)), and for `**` and
`type(int).min` on values that fit
([#8876](https://github.com/esbmc/esbmc/pull/8876)). A shift count was
truncated, so `uint8(1) << 256` gave 1
([#8845](https://github.com/esbmc/esbmc/pull/8845)).

## Ladder Diagram, LTL, transition systems, contracts and other

**Ladder Diagram.** An edge contact compared its operand with a value latched at
the end of the previous scan, so it missed an edge a later rung wrote; it now
senses the change between its own evaluations, as IEC 61131-3 Table 75 and
Beremiz do ([#8907](https://github.com/esbmc/esbmc/pull/8907)). `REAL` is now
binary32 rather than a C `double` ([#8910](https://github.com/esbmc/esbmc/pull/8910)).
A wired `EN` or `ENO` on a user function block, and five inputs the front end
translated wrongly, are refused ([#8859](https://github.com/esbmc/esbmc/pull/8859),
[#8858](https://github.com/esbmc/esbmc/pull/8858)).

**LTL.** An `--ltl` run gave no verdict when a monitored assignment sat in a
loop body of `main` ([#8939](https://github.com/esbmc/esbmc/pull/8939)). `--ltl`
steps the monitor after every update of a global a proposition reads, which for
code generated from Simulink, Lustre or SCADE reports bad prefixes at mid-step
states. `--ltl-sample-points` steps it only at `__ESBMC_ltl_sample()` calls,
which the harness places after each `model_step()`; without `--ltl` the call is
a no-op ([#8951](https://github.com/esbmc/esbmc/pull/8951)).

**Transition systems (experimental).** `--ts-bmc` runs bounded model checking
over the extracted transition system up to `--max-k-step`, and falls back to
ordinary BMC when the program is not one. On a loop whose assertion fails on the
fifth pass it prints
`Transition-system counterexample at step 4 (standard BMC: --unwind 5)`.
`--ts-no-fallback`, `--ts-no-live-filter`, `--ts-fresh-solver`,
`--ts-push-pop` and `--bitwuzla-opt` came in the same PR, a step towards
k-induction and PDR over the transition system
([#8655](https://github.com/esbmc/esbmc/pull/8655)).

**Contracts.** `__ESBMC_stack_used` and `__ESBMC_frame_size` let a contract
bound recursion depth (CWE-674) without unwinding the recursion: the engine
charges each function's frame size to the ghost on entry, and nothing is
injected unless a clause reads it ([#8556](https://github.com/esbmc/esbmc/pull/8556)).
Under `--function`, a CBMC binary's constructors were never called
([#8857](https://github.com/esbmc/esbmc/pull/8857)).

**Output and crashes.** With `--multi-property` under a k-step strategy,
`VERIFICATION FAILED` was printed more than once
([#8822](https://github.com/esbmc/esbmc/pull/8822)). An unbuilt solver or an
undefined `__ESBMC_*` call exits with 6 instead of `abort()`
([#8839](https://github.com/esbmc/esbmc/pull/8839)). A data race made
`--parallel-solving` abort about one run in 30
([#8567](https://github.com/esbmc/esbmc/pull/8567)), and
`--no-reachable-memory-leak` threw on a global with an anonymous member, such as
a `pthread_mutex_t`, in 132 SV-COMP tasks
([#9023](https://github.com/esbmc/esbmc/pull/9023)).

## New options

| Option | What it does | PR |
|---|---|---|
| `--python-check-exceptions types` | Report only uncaught exceptions of the listed types or their subclasses | [#8522](https://github.com/esbmc/esbmc/pull/8522) |
| `--python-math-ulps n` | How many ulps a math or NumPy transcendental result may lie from the C library's value (default 8) | [#9198](https://github.com/esbmc/esbmc/pull/9198) |
| `--report-unreached-claims` | List each claim no path reaches as `PASSED (never reached)` | [#8875](https://github.com/esbmc/esbmc/pull/8875) |
| `--ts-bmc` | Bounded model checking over the extracted transition system | [#8655](https://github.com/esbmc/esbmc/pull/8655) |
| `--ts-no-fallback` | Report `UNKNOWN` instead of falling back when the program is not a transition system | [#8655](https://github.com/esbmc/esbmc/pull/8655) |
| `--ts-no-live-filter` | Keep every havocked variable as state rather than slicing to the properties' cone of influence | [#8655](https://github.com/esbmc/esbmc/pull/8655) |
| `--ts-fresh-solver` | With `--ts-bmc`, solve each bound as one formula in a new solver | [#8655](https://github.com/esbmc/esbmc/pull/8655) |
| `--ts-push-pop` | Use push/pop instead of assumptions in `--ts-bmc` (cross-check) | [#8655](https://github.com/esbmc/esbmc/pull/8655) |
| `--bitwuzla-opt name=value` | Set a Bitwuzla option, e.g. `rewrite-level=1` (repeatable) | [#8655](https://github.com/esbmc/esbmc/pull/8655) |
| `--ltl-sample-points` | Step the LTL monitor only at `__ESBMC_ltl_sample()` calls | [#8951](https://github.com/esbmc/esbmc/pull/8951) |

A claim symex never reaches was left out of the property table, which is how
two earlier reports read as proofs. `--report-unreached-claims` is opt-in, so
default output is unchanged:

```c
int x = 0;
if (x)
  assert(0);
assert(x == 0);
/* PASSED [main.assertion.1] line 6 assertion 0 (never reached)
   ** 0 of 2 properties failed, 2 passed (1 never reached) */
```

The website documentation has been updated to match. As always, the full list is
in the [commit history](https://github.com/esbmc/esbmc/commits/master/) — and if
you hit a bug or a missing feature, we would love an
[issue report](https://github.com/esbmc/esbmc/issues).
