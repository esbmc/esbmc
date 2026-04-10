---
title: Limitations
weight: 4
---

> **Note**: The following limitations apply to the current version of ESBMC-Python. Many are actively being addressed. Check the [issue tracker](https://github.com/esbmc/esbmc/issues) for the latest status.

## Control Flow and Loops

- Only `for` loops using the `range()` function are supported; direct iteration over lists (`for x in my_list:`) is not — use index-based access with `range(len(my_list))` instead.
- Iteration over dictionaries via `d.keys()`, `d.values()`, and `d.items()` is supported (see [Supported Features — Dictionaries](./supported-features#dictionaries)).

## Lists and Strings

- String slicing does not support a step value (e.g., `s[::2]` is not supported).
- `any()` currently supports only list literals as arguments; other iterable types are not supported.

## Dictionaries

- Supported operations are: literals, subscript access/assignment, `del`, `in`/`not in`, equality, iteration over `keys()`/`values()`/`items()`, `update()`, `get()`, `setdefault()`, `pop()`, and `popitem()`. Other methods (e.g., `copy()`) are not yet implemented.

## Tuples

- Tuple indexing requires constant indices; variable indices (e.g., `t[i]` where `i` is a variable) are not supported.
- Tuple iteration (`for item in my_tuple:`) is not yet supported.
- Tuple methods `.count()` and `.index()` are not yet supported.
- Tuple concatenation (`+`) and repetition (`*`) are not yet supported.
- Tuple slicing is not yet supported.

## Built-in Functions

- `min()` and `max()` support only two arguments; iterables and the `key`/`default` parameters are not handled.
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

- `random.randrange()` with a single argument (e.g., `randrange(10)`) is not supported; two arguments are required.
- Other `random` functions (`choice`, `shuffle`, `sample`, `seed`, etc.) are not yet supported.

## Exception Handling

- Core built-in exception types are supported, but not all Python standard library exceptions; custom exception hierarchies with complex inheritance patterns may not be fully handled.

## Class Attributes

- Type inference for class attributes requires values with clear, determinable types; complex expressions may require explicit type annotations.

## Missing Return Detection

- Does not analyze return statements inside lambda expressions within the main function body.

## Module System

- Built-in variable support is limited to `__name__`; `__file__`, `__doc__`, `__package__`, and other built-ins are not yet supported.
