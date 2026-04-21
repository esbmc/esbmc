---
title: Limitations
weight: 4
---

> **Note**: The following limitations apply to the current version of ESBMC-Python. Many are actively being addressed. Check the [issue tracker](https://github.com/esbmc/esbmc/issues) for the latest status.

## Control Flow and Loops

- `for` loops support direct iteration over `range()`, lists, and strings. Iteration over tuples and generators is not yet supported.
- Comprehensions (list, dict, set, generator) are not supported.
- Iteration over dictionaries via `d.keys()`, `d.values()`, and `d.items()` is supported inside `for` loops (see [Supported Features — Dictionaries](./supported-features#dictionaries)).

## Lists

- String slicing does not support a step value (e.g., `s[::2]` is not supported).
- `list.sort()` does not support the `key` or `reverse` keyword arguments.
- `sorted()` does not support the `key` or `reverse` keyword arguments.

## Sets

- Set methods (`.add()`, `.discard()`, `.remove()`, `.update()`, `.issubset()`, `.issuperset()`, `.symmetric_difference()`) are not supported; use binary operators (`-`, `&`, `|`) instead.
- Set ordering operators (`<`, `<=`, `>`, `>=` for subset/superset relations) are not supported.
- `frozenset` is not supported.

## Dictionaries

- Supported operations are: literals, subscript access/assignment, `del`, `in`/`not in`, equality, iteration over `keys()`/`values()`/`items()`, `update()`, `get()`, `setdefault()`, `pop()`, and `popitem()`. Other methods (e.g., `copy()`) are not yet implemented.

## Tuples

- Tuple indexing requires constant indices; variable indices (e.g., `t[i]` where `i` is a variable) are not supported.
- Tuple iteration (`for item in my_tuple:`) is not yet supported.
- Tuple methods `.count()` and `.index()` are not yet supported.
- Tuple concatenation (`+`) and repetition (`*`) are not yet supported.
- Tuple slicing is not yet supported.

## Complex Numbers

- The `complex()` constructor does not support string arguments (e.g., `complex("1+2j")` is not supported).
- `cmath.polar()` and `cmath.rect()` rely on the `atan2` model; results may differ from CPython in edge cases involving signed zeros and NaN.

## Built-in Functions

- `min()` and `max()` support two-argument form and single-list form only; the `key` and `default` keyword arguments are not supported.
- `any()` and `all()` currently support only list literals as arguments; other iterable types are not supported.
- `sum()` supports `int` and `float` element types only; the optional `start` argument is not supported.
- `sorted()` supports `int`, `float`, and `str` element types only; `key` and `reverse` keyword arguments are not supported.
- `input()` is modelled as a nondeterministic string with a maximum length of 256 characters (under-approximation).
- `print()` evaluates all arguments for side effects but does not produce actual output during verification.
- `enumerate()` supports standard usage patterns but may have limitations with complex nested iterables or advanced parameter combinations.

## Lambda Expressions

- Return type inference is naive and defaults to `double`.
- Higher-order and nested lambda expressions are not supported.
- Parameter types are assumed to be `double` for simplicity.

## F-Strings

- Complex expressions inside f-strings may have limited support.
- Only basic integer (`:d`, `:i`) and float (`:.Nf`) format specs are supported; advanced format specs (e.g., string alignment `:>10`, `:<5`) are not.
- Nested f-strings are not supported.
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

## Module System

- Built-in variable support is limited to `__name__`; `__file__`, `__doc__`, `__package__`, and other built-ins are not yet supported.
