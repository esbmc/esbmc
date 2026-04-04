---
title: Supported Features
weight: 3
---

This page is a reference of all Python language constructs, data structures, and standard library modules currently supported by ESBMC-Python.

## Basic Constructs

- **Control flow**: `if`/`elif`/`else`, `for` (with `range()`), `while`
- **Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **Logical operations**: `and`, `or`, `not`
- **Identity comparisons**: `is`, `is not` (including `x is None`, `x is not None`)
- **None handling**: Proper type distinction from `int`, `bool`, `str`, etc.; correctly falsy in boolean contexts (`None and True` → `None`, `None or 1` → `1`)
- **Global variables**: The `global` keyword for accessing and modifying global scope from within functions

## Functions and Methods

- **Function definitions**: Parameters, return values, and calls
- **Variadic parameters**: `*args` syntax for variable-length positional argument lists
- **Type annotations**: Basic types (`int`, `float`, `bool`, `str`), `Any`, `Union[T1, T2]`, PEP 604 `T1 | T2` syntax
- **Union types**: Both `Union[int, bool]` and `int | bool` syntax are supported, including chained unions with multiple members
- **Type widening**: ESBMC selects the widest type from Union members using the hierarchy `float > int > bool`
- **Any type**: When a function is annotated `-> Any`, ESBMC infers the actual return type by analyzing return statements in the function body; supports `int`, `float`, `bool`, and expressions evaluating to those types
- **Variable inference**: Variables annotated with `Any` that are assigned from function calls inherit the function's inferred return type
- **Lambda expressions**: Single-expression lambdas with multiple parameters; converted to regular functions and stored as function pointers; can be assigned to variables and called indirectly

## Object-Oriented Programming

- **Classes**: Definitions, methods, and attributes
- **Class attributes**: Class-level variables shared across all instances; supports both explicit type annotations and automatic type inference from assigned values; accessible via both `instance.attr` and `ClassName.attr`
- **Instance variables**: Attributes defined in `__init__`
- **Inheritance**: Single and multi-level inheritance; verification of scenarios involving overridden methods
- **`super()` calls**: `super().__init__(...)` and other `super().method(...)` calls, enabling verification of polymorphic behavior and parent-constructor side effects

## String Formatting and Literals

- Basic variable interpolation: `f"Hello {name}!"`; multiple variables in one f-string
- Built-in variable access in f-strings: `f"Running as: {__name__}"`
- Integer format specs: `f"{num:d}"`, `f"{num:i}"`
- Float format specs: `f"{val:.2f}"`, `f"{price:.1f}"`
- Boolean formatting: automatic conversion to `True`/`False` strings
- Empty and literal f-strings: `f""`, `f"Just a string"`
- F-string concatenation with other strings
- IEEE 754–compliant 32-bit and 64-bit float-to-string conversion

## Data Structures

### Lists

- `append(x)`: Add element to end
- `clear()`: Remove all elements
- `pop([i])`: Remove and return element at index (default: last)
- `remove(x)`: Remove first occurrence of value
- `copy()`: Return a shallow copy
- `extend(iterable)`: Append all elements from an iterable
- `reverse()`: Reverse in place
- `insert(i, x)`: Insert at position; handles index at/beyond end, within bounds, and empty lists
- `in` operator: Membership testing (`2 in [1, 2, 3]`)
- `+` operator: List concatenation (`[1,2] + [3,4]`)
- **Repetition**: `lst * n` with both literal and variable lists
- **Nested lists**: Method calls on subscripted elements (e.g., `nested[i].append(v)`)
- **Typed instance attributes**: `self.attr: List[T]` instance attributes with full method support

### Strings

**Predicates**: `startswith()`, `endswith()`, `isspace()`, `isalpha()`, `isdigit()`, `islower()`, `isupper()`, `isalnum()`, `isnumeric()`, `isidentifier()`

**Case conversion**: `lower()`, `upper()`, `capitalize()`, `title()`, `swapcase()`, `casefold()`

**Search**: `find()`, `rfind()`, `index()`, `count()` (with optional range arguments)

**Modification**: `replace()`, `strip()`, `lstrip()`, `rstrip()`, `removeprefix()`, `removesuffix()`

**Splitting/joining**: `split()`, `splitlines()`, `partition()`, `join()`

**Padding**: `center()`, `ljust()`, `rjust()`, `zfill()`, `expandtabs()`

**Formatting**: `format()` with `{}`, `{0}`, `{name}` placeholders; `format_map()` with constant dicts

**Slicing**: `s[start:end]`, omitted bounds (`s[:end]`, `s[start:]`), negative indices (`s[-3:]`), empty slices

**Operators**: `in` (substring test), `*` (repetition: `"a" * 3`, `3 * "a"`, boolean multipliers)

### Tuples

- Literals: `(1, 2, 3)`, `()`, `(5,)` (single-element)
- Mixed types: `(42, "hello", 3.14)`
- Nested tuples: `((1, 2), (3, 4))`
- Constant-index access with bounds checking: `t[0]`, `t[2]`
- Generic annotation (`t: tuple`) and parameterized annotation (`-> tuple[int, int]`)
- Equality comparison: `t1 == (1, 2, 3)`
- `len()` built-in
- `isinstance(obj, tuple)` type checking

### Dictionaries

- **Literals**: `{"a": 1, "b": 2}`
- **Subscript access**: `d["a"]`; raises `KeyError` if absent
- **Subscript assignment**: `d["c"] = 3`
- **Membership**: `"a" in d`, `"a" not in d`
- **Deletion**: `del d["a"]`; raises `KeyError` if absent
- **Equality**: `d1 == d2` (order-independent)
- **Iteration**: `for` loops over `d.keys()`, `d.values()`, `d.items()`
- **`update(other)`**: Merge another dict
- **`get(key[, default])`**: Return value or default; returns `Optional[T]` when no default is provided
- **`setdefault(key[, default])`**: Insert key with default if absent, then return value; supports `int`, `float`, `bool`, `str`
- **`pop(key[, default])`**: Remove and return value; raises `KeyError` if absent and no default
- **`popitem()`**: Remove and return last inserted `(key, value)` pair; raises `KeyError` if empty
- **Nested dicts**: `dict[int, dict[int, int]]`
- **`Optional[T]` values**: `dict[str, Optional[T]]` storage and retrieval

## Bytes and Integers

Supports byte and integer operations including type conversions and bit-length queries.

## Error Handling

- **`try`/`except`** blocks with multiple handlers and `except ExceptionType as var` binding
- **`raise`** statements with exception instantiation and custom messages
- **`assert`** statements for property verification
- **`__ESBMC_assume`** for constraining non-deterministic inputs
- **`ImportError` guards**: Imports inside `try/except ImportError` are handled statically

**Built-in exception hierarchy**:
- `BaseException` → `Exception` → `AssertionError`, `ValueError`, `TypeError`, `IndexError`, `KeyError`, `ZeroDivisionError`, `ImportError`
- `OSError` → `FileNotFoundError`, `FileExistsError`, `PermissionError`

Exception instances expose a message attribute and support `__str__()`.

## Cover Properties and Reachability

`__ESBMC_cover(cond)` checks whether a condition is satisfiable at a given program point (inverted assertion semantics: a counterexample means the condition *is* reachable).

Use with `--multi-property` to report all reachable/unreachable points without stopping at the first result.

```python
x: int = __VERIFIER_nondet_int()
if x > 0:
    __ESBMC_cover(x > 100)  # Is this branch reachable?
```

## Strict Type Checking

The `--strict-types` flag enables type compatibility validation for function arguments at verification time. When a type mismatch is detected, a `TypeError` is generated with a descriptive message. Instance methods, class methods, and static methods are all handled with appropriate implicit-parameter awareness.

## Code Quality Analysis

- **Missing return statement detection**: Statically detects non-void functions lacking return statements on all paths; reports as verification failures; excludes `__init__` automatically

## Module System

- **`__name__`**: Set to `"__main__"` when run directly; set to the module name when imported. Enables `if __name__ == "__main__":` idioms.
- **Imports**: Standard `import` and `from ... import ...` styles validated at verification time

## Built-in Functions

| Function | Notes |
|---|---|
| `abs`, `divmod` | Standard arithmetic |
| `int`, `float`, `chr`, `str`, `hex`, `oct` | Type conversions |
| `len` | Works on lists, strings, tuples |
| `range` | Used in `for` loops |
| `min(a, b)`, `max(a, b)` | Two-argument form only; promotes `int` to `float` |
| `any([...])` | List literals only; short-circuit OR logic |
| `enumerate(iterable, start=0)` | Tuple unpacking and single-variable forms; optional `start` |
| `isinstance(obj, type)` | Runtime type checking |
| `float("nan")`, `float("inf")` | Special values (case-insensitive, whitespace-tolerant) |
| `input()` | Modelled as nondeterministic string, max 256 chars |
| `print(...)` | Arguments evaluated for side effects; no output produced |

## Math Module (`math`)

**Constants**: `pi`, `e`, `inf`, `tau`, `nan`

**Rounding**: `floor(x)`, `ceil(x)`, `trunc(x)`, `fabs(x)`, `modf(x)`

**Power/log**: `exp(x)`, `expm1(x)`, `exp2(x)`, `log(x)`, `log1p(x)`, `log2(x)`, `log10(x)`, `pow(x, y)`

**Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`

**Hyperbolic**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Integer helpers**: `factorial(n)`, `gcd(a, b)`, `lcm(a, b)`, `isqrt(n)`, `perm(n, k)`, `comb(n, k)`, `prod(lst)` (expects `list[int]`)

**Geometry**: `hypot(x, y)`, `dist(p, q)` (expects `list[float]`)

**Utilities**: `fmod(x, y)`, `copysign(x, y)`, `degrees(x)`, `radians(x)`, `isclose(a, b)`, `isfinite(x)`, `isnan(x)`, `isinf(x)`

## Regular Expression Module (`re`)

**Functions**: `re.match(pattern, string)`, `re.search(pattern, string)`, `re.fullmatch(pattern, string)`

**Supported pattern features**:
- Universal match: `.*`
- Empty patterns and literal strings (no metacharacters)
- Character class ranges with quantifiers: `[a-z]+`, `[A-Z]+`, `[0-9]*`
- Digit sequences: `\d+`, `\d*`
- Alternation: `(x|y)z*`
- Prefix-with-wildcard: patterns ending with `.*`

**Match results**: Usable for Boolean/`None` testing (`if re.match(...)`, `re.search(...) is not None`). Group-capture methods (`.group()`, `.groups()`, `.span()`) are not supported.

**Type validation**: Both arguments must be string or bytes-like; invalid types raise `TypeError`.

**Verification approach**: Uses operational models combining direct pattern recognition, literal string matching, and nondeterministic behavior for complex patterns.

## Random Module (`random`)

All functions are modelled using nondeterministic values with appropriate constraints via `__ESBMC_assume`, allowing ESBMC to explore all possible values within the specified ranges.

- `random.random()` → nondeterministic `float` in `[0.0, 1.0)`
- `random.uniform(a, b)` → nondeterministic `float` N where `min(a,b) ≤ N ≤ max(a,b)`
- `random.getrandbits(k)` → nondeterministic integer with k random bits
- `random.randrange(start, stop, step=1)` → randomly selected integer from the specified range

See also: [Random Operational Model](./Random-Operational-Model-in-ESBMC)

## OS Module (`os`)

All `os` functions use nondeterministic modelling to verify both success and failure paths.

**Path**: `os.path.exists(path)`, `os.path.basename(path)`

**Directory operations**:
- `os.makedirs(path, exist_ok=False)`: Creates directory tree; supports `exist_ok`
- `os.mkdir(path)`: Creates single directory; may raise `FileExistsError`
- `os.rmdir(path)`: Removes empty directory; may raise `OSError`
- `os.listdir(path)`: Lists directory contents

**File operations**:
- `os.remove(path)`: Removes file; may raise `FileNotFoundError`
- `os.popen(cmd)`: Opens a pipe (modelled for verification)
