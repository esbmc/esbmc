---
title: Limitations
weight: 4
---

> **Note**: The following limitations apply to the current version of ESBMC-Python. Many are actively being addressed. Check the [issue tracker](https://github.com/esbmc/esbmc/issues) for the latest status.

## Control Flow and Loops

- `for` loops support direct iteration over `range()`, lists, strings (including the result of a `str(...)` call, e.g. `for digit in str(n)`), tuples, and generators (functions using `yield` and generator expressions).
- `for ... else` and `while ... else` are supported: the `else` clause is lowered into a did-not-break flag, so it runs only when the loop completes without `break` (a `break` inside a nested loop stays bound to that inner loop).
- List, set, dictionary, and generator comprehensions are supported. Dictionary comprehensions populate a real dict (see [Supported Features — Dictionaries](./supported-features#dictionaries)); the iterable must be a `range(...)`, a list of tuples, or a `d.items()` view (with an optional `if` filter). Comprehensions over other iterables (e.g. another dict comprehension or an arbitrary generator) may not be handled.
- Iteration over dictionaries via `d.keys()`, `d.values()`, and `d.items()` is supported inside `for` loops (see [Supported Features — Dictionaries](./supported-features#dictionaries)). The destructuring form `for u, v in d:` over a dict with tuple keys works for **local dict literals** and for **unannotated parameter dicts** with scalar or integer-tuple keys (recovered from the call sites); so do the deferred form `for edge in d:` followed by `u, v = edge`, and iteration over `sorted(d)`. Passing a custom `key=` to `sorted` disables that path, and string-tuple-keyed parameter dicts are still not handled ([#5571](https://github.com/esbmc/esbmc/issues/5571)).

## Lists

- `list.sort()` supports `reverse`. `xs.sort(key=...)` is rewritten to
  `xs = sorted(xs, key=...)`, so it carries `sorted()`'s restrictions — and the
  rewrite needs a bare name as the receiver and no positional argument.
- `sorted()` supports `reverse`, and applies `key=` only where the iterable's
  shape is known at conversion time — see [Built-in Functions](#built-in-functions).

## Sets

- The supported set methods are `.issubset()`, `.issuperset()`, `.symmetric_difference()`, `.update()`, `.union()`, `.intersection()`, and `.difference()` (see [Supported Features — Sets](./supported-features#sets)). The `.union()`/`.intersection()`/`.difference()` methods take exactly one argument (the zero-arg and variadic forms produce a clean error). Other named methods (`.add()`, `.remove()`, `.discard()`, `.isdisjoint()`, etc.) are not supported; use the equivalent binary operators (`-`, `&`, `|`, `^`) where one exists.

## Dictionaries

- Supported operations are: literals, subscript access/assignment, `del`, `in`/`not in`, equality, iteration over `keys()`/`values()`/`items()`, `update()`, `get()`, `setdefault()`, `pop()`, `popitem()`, and `clear()`. Other methods (e.g., `copy()`) are not yet implemented.

## Complex Numbers

- The `complex()` constructor accepts literal strings and a limited set of frontend-folded string expressions (for example, conditionals between literal complex strings). Arbitrary runtime strings are still rejected with the error `complex() does not support non-literal string arguments`.


## Built-in Functions

- `min()` and `max()` support two-argument form and single-list form only (`default` is supported). `key=` is folded over **constant** lists for the `lambda x: x[K]`, `key=abs` and `key=len` forms, and otherwise lowered to a linear scan that really applies the key, including over a list literal whose elements are symbolic *scalars*. Ties keep the first occurrence (as CPython does), and an empty iterable raises `IndexError` where CPython raises `ValueError`. A shape the scan cannot lower — a list literal containing tuples, a dict view call (`d.keys()`/`.values()`/`.items()`), a bound method as the key, an element the key subscripts arriving as a subscripted parameter — is refused with a named error rather than answered with the key dropped.
- `any()` and `all()` currently support only list literals as arguments. `any()` rejects other iterables with a parse-time error; `all()` may trigger a dereference failure on non-list iterables.
- `sum()` supports `int` and `float` element types only.
- `sorted()` supports `int`, `float`, and `str` element types, plus a homogeneous list of tuples (the element types are carried through, so `for u, v in sorted(pairs)` unpacks). `reverse=` is supported. `key=` is applied over a list literal whose elements are symbolic scalars, and over a constant list or dict literal — the latter read through `d.__getitem__` — with a lambda or an undecorated, never-rebound module-level `def` as the key, including when the call is a `for` loop's iterable. A list *of tuples* has only the constant-fold path: with symbolic tuple elements the scan declines it. A dict with a symbolic value, a bound method other than `__getitem__` (`key=d.get`), or any other shape the preprocessor cannot fold is refused with `sorted() with key= is only supported over a constant iterable` rather than sorted in natural order.
- `input()` is modelled as a nondeterministic string with a maximum length of 256 characters (under-approximation).
- `print()` evaluates each argument expression once (so safety checks and call side effects reach the GOTO program) but produces no actual output during verification.
- `enumerate()` supports the iterable + `start` keyword forms; nested or unusually-shaped iterables are not exercised by the regression suite and may surface edge cases.

## Walrus Operator

- The walrus operator `:=` is supported only where the target is evaluated exactly once: `if`/`elif` conditions, standalone assignment expressions, and comprehension filters (see [Supported Features](./supported-features#basic-constructs)).
- Use inside a boolean (`and`/`or`) operand is refused: `ERROR: Walrus operator ':=' in a boolean (and/or) operand is not supported`.
- Use in a `while`-loop condition is refused: `ERROR: Walrus operator ':=' in a while-loop condition is not supported`.

## Lambda Expressions

- Return type inference is naive and defaults to `float`.
- A parameter's type is recovered from the calls made through the bound name
  when every call agrees on it, and when a subscripted parameter's argument is
  bound to a list literal. Anything less certain keeps the `float` default, so a
  lambda whose parameter is neither annotated nor pinned by its call sites is
  still assumed to be a float.

## F-Strings

- Complex expressions inside f-strings may have limited support.
- Custom format specifications for user-defined types are not supported.

## Strings

- Most `str.*()` methods now degrade to a sound nondeterministic over-approximation when the receiver is not a compile-time constant (see [Supported Features — Strings](./supported-features#strings)). A growing set have precise runtime operational models: the case transforms `swapcase`, `upper`, `lower`, `capitalize`, `title` (which cap the receiver at ~255 characters, asserting on longer input — `upper` truncates instead); the predicates `isupper`, `islower`, `isalpha`, `isdigit`, `isalnum`, `isspace`; `count`; and `find`/`rfind`. `str.join` likewise has a precise model (bounded to a 511-character result) when its iterable is a variable whose initialiser cannot be folded (e.g. a `List[str]` parameter), but falls back to a nondet `char *` when the iterable is a non-foldable expression such as `sorted(...)`, a comprehension, or a function-call result. Other methods (`casefold`, `isnumeric`, `isidentifier`, `removeprefix`, `removesuffix`, `center`, `ljust`, `rjust`, `zfill`, `expandtabs`, `partition`, `format`, `format_map`, `splitlines`, etc.) return a nondet value of the appropriate shape, so assertions on their specific functional result will report `VERIFICATION FAILED` on symbolic input.
- `partition()` on a non-constant receiver returns `("", "", "")` — the same shape Python uses when the separator is not found.
- `splitlines()` on a non-constant receiver returns an empty list.

## Dynamic Typing

A variable whose type diverges across an `if`/`else` is carried as a tagged
value (see [Supported Features — Dynamic Typing](./supported-features#dynamic-typing)),
within these bounds:

- A tag holds one of `bool`, `int`, `float` or `str`. `isinstance` against an aggregate or a user class is therefore answered `False`, not consulted.
- Arithmetic (`+`, `-`, `*`, `/`) is supported against a **literal** operand, and `+`, `-` and `/` between two tagged operands; `+` additionally concatenates strings. A non-numeric operand raises `TypeError`.
- Ordered comparisons (`<`, `<=`, `>`, `>=`) work against a literal and between two tagged operands, raising `TypeError` on a type mismatch. `==` treats `bool` and `int` as the same type, so `True == 1`.
- Divergence is detected across an `if`/`elif`/`else` chain only when every branch assigns the name; a chain with a branch that leaves it unassigned is not tagged.
- `x is None` is folded only against a literal `None`. A computed operand is not folded, since that would drop its side effects.
- Rebinding a tagged variable to a list, tuple or class instance is refused inside a loop or a conditional body, where the join of the retyped aliases is not modelled.

## Union and Any Types

- Union types are resolved to the widest type among their members (`float > int > bool`) at verification time; true union semantics are not maintained.
- Union types containing types beyond basic primitives (`int`, `float`, `bool`) may default to pointer types.
- Type narrowing based on runtime type checks within Union-typed functions is not tracked.
- `Any` type inference only supports primitive return types (`int`, `float`, `bool`) and expressions evaluating to those types; string return values are not supported and will produce an error.
- Other return types (`objects`, `arrays`, `null`) are not supported for `Any`-typed functions; inference defaults to `double` when no type can be determined.

## Regular Expressions (`re` module)

- Only `re.match()`, `re.search()`, and `re.fullmatch()` are supported.
- Group-capture methods (`.group()`, `.groups()`, `.span()`) are rewritten by the parser into direct calls to internal helpers, and only the `(\d+)` pattern is recognised precisely; everything else returns a nondeterministic value.
- The result of `re.match` / `re.search` / `re.fullmatch` is a `bool`, not an `Optional[Match]`: `if m:` works, `if m is None:` does not.
- Complex patterns beyond the explicitly supported constructs exhibit nondeterministic behavior.
- Not supported: lookahead/lookbehind assertions, backreferences, named groups, conditional patterns, Unicode property escapes.

## Random Module

- Functions beyond `random()`, `uniform()`, `randint()`, `getrandbits()`, `randrange()`, `choice()`, `shuffle()`, `sample()`, and `seed()` are not yet supported.
- `random.shuffle(lst)` is an under-approximation that leaves the list untouched.
- `random.sample(population, k)` is an under-approximation that returns the first `k` elements of `population` rather than `k` distinct nondeterministic indices.
- `random.seed(a)` is a no-op; the model is stateless, so seeding cannot make subsequent calls deterministic.

## Collections Module

- `defaultdict`: subscript access/assignment and the common type-factory forms are supported — `defaultdict(list)` (with `.append()` on the materialised list), the built-in scalar factories `defaultdict(int)` / `float` / `bool` / `str`, and nullary `lambda` factories whose body is a constant or built-in constructor (e.g. `defaultdict(lambda: float('inf'))`). On an unannotated dict the value type is also inferred from a constant literal subscript assignment (`d[k] = 5`). The `__missing__` hook and other methods are not.
- `Counter`: only `__getitem__`, `__setitem__`, `values()`, and truthiness are supported. `most_common()` accepts the call but its result is unusable in any subsequent expression — comparisons trip a frontend "Unsupported comparison" error ([#4665](https://github.com/esbmc/esbmc/issues/4665)). `elements()`, `subtract()`, and arithmetic operators are not supported.
- `Counter.update(...)` / `dict.update(...)` accept only the single-positional-argument form; the keyword-argument form (`c.update(a=1)`) is rejected at parse time even though it is valid CPython.
- `OrderedDict` supports construction and basic indexing / append / `__setitem__`. `deque` adds the FIFO-front methods `popleft()` and `appendleft()` on top of construction / indexing / `append` / `__setitem__`; other `deque` methods (`extend`, `rotate`, `maxlen`, etc.) are not supported. `namedtuple`, `ChainMap`, and other `collections` types are not supported.

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

- Arrays are modelled with a restricted subset: `.shape` is available for modelled arrays, tuple indexing is lowered through chained indexing, and direct scalar broadcasting still covers simple binary operators such as `a + n` and `a * n`. Higher-dimensional arrays are rejected explicitly; full NumPy dtype semantics and unrestricted N-dimensional indexing remain unsupported.
- Element-wise `np.add`/`np.subtract`/`np.multiply`/`np.divide`/`np.power` support literal list-backed 1D/2D inputs with NumPy-style broadcasting. Runtime-constructed inputs and higher-dimensional inputs are rejected with deterministic frontend errors rather than falling through to the SMT backend.
- Only the NumPy functions listed in [Supported Features — NumPy](./supported-features#numpy-module-numpy) have executable support.
- The reductions (`sum`/`prod`/`min`/`max`/`mean`/`argmin`/`argmax`), comparison/logical ufuncs (`greater`/`less`/`equal`/`logical_*`/`where`), and constructors (`arange`/`full`/`eye`/`identity`/`linspace`) are constant-folded over list-backed (1D/2D) inputs and constant shapes; runtime-constructed inputs and higher-rank shapes are rejected with deterministic frontend errors.
- `np.arange()` materialises its result at conversion time, so its arguments must be constant — a name bound to a literal is resolved first, but a function parameter is rejected with `TypeError: numpy.arange() currently supports constant numeric inputs only` rather than routed through the operational model's while loop, which did not terminate in practice. A range past 10000 elements is declined for the same reason, and `step=0` raises `ValueError`.
- A returned array keeps its metadata only for the shapes listed under [Supported Features — NumPy](./supported-features#numpy-module-numpy). An unannotated **2-D array parameter** is typed as flat 1-D inside the callee's own body, so `numpy.transpose`'s "1-D is a no-op" fallback can return it unchanged where caller-side inlining does not mask it — the one shape here that yields a wrong value rather than an explicit rejection. An unannotated function that builds an array through a local before returning it, and a captured list mutated without a `global` declaration, are pinned as `KNOWNBUG`.
- A view onto the base array needs literal bounds and a fixed-shape 1-D or 2-D source: 1-D slices (any step, including reversed), 2-D row and column views, `np.diagonal`, `np.ravel` and `a.flat[i]` alias the buffer; a symbolic bound or index, or a 3-D source, still produces an independent copy. `np.diagonal` is read-only, and a diagonal used inline (`np.diagonal(a)[i]`) rather than bound to a name is declined. `np.fill_diagonal` requires a value whose length matches the diagonal exactly.
- `np.arccos`, `np.fmod`, `np.transpose`, `np.dot`, and `np.matmul` now lower to executable models (they were previously type-inference-only stubs), each under a stated restriction: `np.arccos` rejects runtime 2D arrays; `np.fmod` rejects `np.array(...)`-wrapped operands (`Unsupported operation: numpy.fmod on array operands`); `np.transpose` is limited to 2D and rejects higher rank; `np.dot`/`np.matmul` cover 1D/2D integer and float inputs.
- `numpy.linalg.det` supports constant numeric 2x2 and 3x3 matrices. Other `numpy.linalg` operations, complex determinants, runtime-constructed matrices, and larger matrix sizes are not supported.

## Exception Handling

- Core built-in exception types are supported, but not all Python standard library exceptions; custom exception hierarchies with complex inheritance patterns may not be fully handled.
- `try`/`finally` is supported (including bare `try`/`finally`), and a `return`/`break`/`continue` escaping the `try`, a handler, or the `finally` runs the `finally` first. Three shapes are still refused at parse time rather than lowered unsoundly: a non-empty `else` clause on the `try` (a pre-existing gap — `orelse` is silently dropped today), a `finally` that itself escapes, and an escape nested under another `try` or `with`, where the two cleanups would have to run innermost-first.

## Methods Without an Operational Model

- A method call whose receiver class cannot be resolved — most commonly a method invoked directly on a **container literal**, e.g. `{1}.isdisjoint({2})` or `[1].foobar()` — evaluates to a **nondeterministic value**, so neither the assertion nor its negation can be discharged and both report `VERIFICATION FAILED`. This is deliberate: the previous fallback returned a null (falsy) value, which *proved* the negation of any such call. Binding the receiver to a name first (`s = {1}` … `s.isdisjoint({2})`) gets the modelled semantics.
- An attribute assigned from a method whose return type is not the enclosing class, then used as a receiver (`self.pub = self.make_publisher()` followed by `self.pub.publish(...)`), degrades to `Unsupported function 'publish' is reached` / `VERIFICATION FAILED` rather than resolving the call.
- `self.attr = self.method()` types `attr` by the **enclosing** class and does not perform virtual dispatch, so a subclass override is ignored and a valid polymorphic program can be reported as a false `VERIFICATION FAILED` (pinned as a KNOWNBUG in `regression/python/github_6242_override`).

## Class Attributes

- Type inference for class attributes requires values with clear, determinable types; complex expressions may require explicit type annotations.
- Recovering a self-referential attribute's type from constructor arguments (the linked-list / tree pattern, e.g. `self.successor = successor` set via `Node(2, a)`) works both within a module and across the module boundary for an imported class (`from node import Node`). It relies on unifying against module-level `ClassName(...)` instantiations: if the class is never instantiated at module scope with the relevant positional argument, the attribute type cannot be recovered and an explicit annotation is required.

## Callable Attributes

- A callable member's signature is recovered from an explicit `Callable[...]` annotation or from the parameter an unannotated `self.fn = fn` names. A callable chosen at runtime (the assigned value varies by path) and a container of callables such as `List[Callable]` are not supported.

## Missing Return Detection

- Does not analyze return statements inside lambda expressions within the main function body.

## Concurrency

- **`Lock` model is invisible to `--deadlock-check`** ([#4581](https://github.com/esbmc/esbmc/issues/4581)). `threading.Lock.acquire` lowers to `__ESBMC_atomic_begin / __ESBMC_assume / __ESBMC_atomic_end`, mirroring `pthread_mutex_lock_noassert`. The deadlock checker only inspects the pthread mutex wait graph, so reverse-order lock acquisition between two Python threads is *not* reported as a deadlock — ESBMC explores all interleavings and reports `VERIFICATION SUCCESSFUL`.
- **`Thread(args=(instance,))` value-copies object arguments** ([#4583](https://github.com/esbmc/esbmc/issues/4583)). When a `Thread` target receives a class instance with non-trivial attributes (e.g. a `threading.Lock`), the args-capture struct copies the descriptor by value and breaks attribute dereference inside the trampoline body. Workaround: share state via module-level globals instead of instance attributes passed through `args=`.
- **Symex does not interleave at Python module-global accesses** ([#4584](https://github.com/esbmc/esbmc/issues/4584)). `--data-races-check` correctly flags W/W races on a module global, but symex's per-statement scheduler does not insert interleaving points at function-internal reads/writes of these globals. A classic split read-modify-write race (`tmp = counter; counter = tmp + 1` from two threads) reports `VERIFICATION SUCCESSFUL` instead of finding the schedule where both threads read `counter == 0` before either writes. The C equivalent of the same program is correctly reported as `VERIFICATION FAILED`.
- **Thread shapes refused at parse time** with explicit errors:
  - Lambda or runtime-variable `target=`
  - Positional argument forms (`Thread(f, (a, b))`)
  - `args=` bound to a variable instead of a tuple literal (`Thread(target=f, args=payload)`)
  - `daemon=`, `name=`, `kwargs=`, `group=` keyword arguments
  - `Thread` construction inside loops or comprehensions
  - `Thread` reassignment within the same scope
  - `Thread` as a class attribute (`class C: t = Thread(...)`)
  - `target` defined after the caller in source order
  - `from threading import *`
- **`Thread` subclassing is supported** (see [Supported Features](/docs/python/supported-features#thread-subclassing)), with these shapes refused at parse time: multiple inheritance, a class below module scope, a missing `run`, an overridden `start`, a non-bare `super().__init__()`, a class defined after its constructing function, instance reassignment, binding by anything other than a simple assignment, construction inside a loop, and assignment to a `global`/`nonlocal` name from a function.
- **Other `threading` primitives are not supported**: `RLock`, `Semaphore`, `Condition`, `Event`, `Barrier`, `Timer` are refused at parse time. The `queue` module now has a single-threaded model (`queue.Queue`/`LifoQueue`; see [Supported Features — Queue](./supported-features#queue-module-queue)), but its blocking `put()`/`get()` semantics are not modelled, so it does not provide thread synchronisation.
- **The CPython Global Interpreter Lock (GIL) is not modelled** ([#4579](https://github.com/esbmc/esbmc/issues/4579)). Translated programs execute under sequentially-consistent POSIX semantics rather than GIL-serialised bytecode execution, so the analysis over-approximates the set of feasible interleavings compared to actual CPython execution. This preserves safety but may produce spurious concurrency counterexamples.

## Unittest Module

- The model covers the assertion vocabulary listed under
  [Supported Features — Unittest](./supported-features#unittest-module-unittest);
  `assertRaises`, `assertAlmostEqual`, the `subTest` context manager and the
  class-level `setUpClass` / `tearDownClass` hooks are not modelled.
- `unittest.main()` runs the tests *discovered in the file being verified*.
  `test.support` and `sys.path`-aware module resolution — what a CPython
  regression test relies on — are not implemented
  ([#6745](https://github.com/esbmc/esbmc/issues/6745)).

## Module System

- Built-in variable support is limited to `__name__` and `__file__`; `__doc__`, `__package__`, and other built-ins are not yet supported.
