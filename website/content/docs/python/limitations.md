---
title: Limitations
weight: 4
---

> **Note**: The following limitations apply to the current version of ESBMC-Python. Many are actively being addressed. Check the [issue tracker](https://github.com/esbmc/esbmc/issues) for the latest status.

## Control Flow and Loops

- `for` loops support direct iteration over `range()`, lists, strings, tuples, and generators (functions using `yield` and generator expressions).
- List, set, and generator comprehensions are supported. Dictionary comprehensions are accepted by the parser but produce a dictionary whose subsequent key lookups raise `KeyError`.
- Iteration over dictionaries via `d.keys()`, `d.values()`, and `d.items()` is supported inside `for` loops (see [Supported Features — Dictionaries](./supported-features#dictionaries)).

## Lists

- `list.sort()` does not support the `key` keyword argument; `reverse` is supported.
- `sorted()` does not support the `key` keyword argument; `reverse` is supported.

## Sets

- Set methods `.issubset()`, `.issuperset()`, and `.symmetric_difference()` are not supported; use the equivalent binary operators (`<=`, `>=`, `^`) instead.

## Dictionaries

- Supported operations are: literals, subscript access/assignment, `del`, `in`/`not in`, equality, iteration over `keys()`/`values()`/`items()`, `update()`, `get()`, `setdefault()`, `pop()`, and `popitem()`. Other methods (e.g., `copy()`) are not yet implemented.

## Tuples

- Tuple repetition (`*`) is not yet supported and currently aborts the frontend.

## Complex Numbers

- The `complex()` constructor accepts string arguments only when the string is a compile-time constant (e.g., `complex("1+2j")` is folded by the frontend). Constructing from a runtime string is rejected with the error `complex() does not support non-literal string arguments`.
- `cmath.polar()` and `cmath.rect()` rely on the `atan2` model; results may differ from CPython in edge cases involving signed zeros and NaN.

## Built-in Functions

- `min()` and `max()` support two-argument form and single-list form only; the `key` keyword argument is not supported (`default` is supported).
- `any()` and `all()` currently support only list literals as arguments. `any()` rejects other iterables with a parse-time error; `all()` may trigger a dereference failure on non-list iterables.
- `sum()` supports `int` and `float` element types only.
- `sorted()` supports `int`, `float`, and `str` element types only; the `key` keyword argument is not supported (`reverse` is supported).
- `input()` is modelled as a nondeterministic string with a maximum length of 256 characters (under-approximation).
- `print()` evaluates all arguments for side effects but does not produce actual output during verification.
- `enumerate()` supports standard usage patterns but may have limitations with complex nested iterables or advanced parameter combinations.

## Lambda Expressions

- Return type inference is naive and defaults to `double`.
- Parameter types are assumed to be `double` for simplicity.

## F-Strings

- Complex expressions inside f-strings may have limited support.
- Custom format specifications for user-defined types are not supported.

## Union and Any Types

- Union types are resolved to the widest type among their members (`float > int > bool`) at verification time; true union semantics are not maintained.
- Union types containing types beyond basic primitives (`int`, `float`, `bool`) may default to pointer types.
- Type narrowing based on runtime type checks within Union-typed functions is not tracked.
- `Any` type inference only supports primitive return types (`int`, `float`, `bool`) and expressions evaluating to those types; string return values are not supported and will produce an error.
- Other return types (`objects`, `arrays`, `null`) are not supported for `Any`-typed functions; inference defaults to `double` when no type can be determined.

## Regular Expressions (`re` module)

- Only `re.match()`, `re.search()`, and `re.fullmatch()` are supported.
- Match objects do not expose group-capture methods (`.group()`, `.groups()`, `.span()`).
- Complex patterns beyond the explicitly supported constructs exhibit nondeterministic behavior.
- Not supported: lookahead/lookbehind assertions, backreferences, named groups, conditional patterns, Unicode property escapes.

## Random Module

- `random.choice()`, `random.shuffle()`, `random.sample()`, `random.seed()`, and other functions beyond `random()`, `uniform()`, `randint()`, `getrandbits()`, and `randrange()` are not yet supported.

## Collections Module

- `defaultdict`: Only basic subscript access/assignment is supported. The `__missing__` hook, type-factory calls (e.g., `defaultdict(list)`), and methods beyond `__getitem__`/`__setitem__` are not supported.
- `Counter`: Only `__getitem__`, `__setitem__`, `values()`, and truthiness are supported. `most_common()`, `elements()`, `subtract()`, `update()`, and arithmetic operators on `Counter` are not supported.
- `OrderedDict`, `deque`, `namedtuple`, `ChainMap`, and other `collections` types are not supported.

## Datetime Module

- Only `datetime.datetime(year, month, day)` is supported; `date`, `time`, and `timedelta` classes are not.
- Date arithmetic, string formatting (`strftime`), and parsing (`strptime`, `fromisoformat`) are not supported.

## Decimal Module

- `Decimal()` supports construction from strings (e.g., `Decimal("10.5")`), integers, and no arguments; other forms may not be handled.
- `quantize()`, rounding modes, and decimal context operations are not supported.

## Heapq Module

- `heapify()` is modelled as a no-op; the heap invariant is not enforced structurally.
- `nlargest()`, `nsmallest()`, and `merge()` are not supported.

## Time Module

- `time.time()` is modelled as a monotonically increasing counter (increments by 1.0 per call), not real wall-clock time.
- Other functions (`monotonic()`, `perf_counter()`, `strftime()`, `gmtime()`, `localtime()`, etc.) are not supported.

## NumPy Module

- Arrays are modelled as plain Python lists; array shapes, dtypes, multi-dimensional indexing (`a[i, j]`), and broadcasting are not supported.
- Most NumPy functions beyond those listed in [Supported Features — NumPy](./supported-features#numpy-module-numpy) are not available.
- Several math stub functions (e.g., `np.sin`, `np.sqrt`) return constant placeholder values rather than computing the real result; these are suitable only for type-inference testing, not numerical verification.
- `numpy.linalg.det` is a 2-scalar stub; general matrix operations are not supported.

## Exception Handling

- Core built-in exception types are supported, but not all Python standard library exceptions; custom exception hierarchies with complex inheritance patterns may not be fully handled.

## Class Attributes

- Type inference for class attributes requires values with clear, determinable types; complex expressions may require explicit type annotations.

## Missing Return Detection

- Does not analyze return statements inside lambda expressions within the main function body.

## Concurrency

- **`Lock` model is invisible to `--deadlock-check`** ([#4581](https://github.com/esbmc/esbmc/issues/4581)). `threading.Lock.acquire` lowers to `__ESBMC_atomic_begin / __ESBMC_assume / __ESBMC_atomic_end`, mirroring `pthread_mutex_lock_noassert`. The deadlock checker only inspects the pthread mutex wait graph, so reverse-order lock acquisition between two Python threads is *not* reported as a deadlock — ESBMC explores all interleavings and reports `VERIFICATION SUCCESSFUL`.
- **`Thread(args=(instance,))` value-copies object arguments** ([#4583](https://github.com/esbmc/esbmc/issues/4583)). When a `Thread` target receives a class instance with non-trivial attributes (e.g. a `threading.Lock`), the args-capture struct copies the descriptor by value and breaks attribute dereference inside the trampoline body. Workaround: share state via module-level globals instead of instance attributes passed through `args=`.
- **Symex does not interleave at Python module-global accesses** ([#4584](https://github.com/esbmc/esbmc/issues/4584)). `--data-races-check` correctly flags W/W races on a module global, but symex's per-statement scheduler does not insert interleaving points at function-internal reads/writes of these globals. A classic split read-modify-write race (`tmp = counter; counter = tmp + 1` from two threads) reports `VERIFICATION SUCCESSFUL` instead of finding the schedule where both threads read `counter == 0` before either writes. The C equivalent of the same program is correctly reported as `VERIFICATION FAILED`.
- **Thread shapes refused at parse time** with explicit errors:
  - `Thread` subclassing with `run` override
  - Lambda or runtime-variable `target=`
  - Positional argument forms (`Thread(f, (a, b))`)
  - `args=` bound to a variable instead of a tuple literal (`Thread(target=f, args=payload)`)
  - `daemon=`, `name=`, `kwargs=`, `group=` keyword arguments
  - `Thread` construction inside loops or comprehensions
  - `Thread` reassignment within the same scope
  - `Thread` as a class attribute (`class C: t = Thread(...)`)
  - `target` defined after the caller in source order
  - `from threading import *`
- **Other `threading` primitives are not supported**: `RLock`, `Semaphore`, `Condition`, `Event`, `Barrier`, `Timer` are refused at parse time. `queue.Queue` is also unsupported and blocks the existing `regression/python/concurrency_fail` example.
- **The CPython Global Interpreter Lock (GIL) is not modelled** ([#4579](https://github.com/esbmc/esbmc/issues/4579)). Translated programs execute under sequentially-consistent POSIX semantics rather than GIL-serialised bytecode execution, so the analysis over-approximates the set of feasible interleavings compared to actual CPython execution. This preserves safety but may produce spurious concurrency counterexamples.

## Module System

- Built-in variable support is limited to `__name__`; `__file__`, `__doc__`, `__package__`, and other built-ins are not yet supported.
