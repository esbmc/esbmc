---
title: Supported Features
weight: 3
---

This page is a reference of all Python language constructs, data structures, and standard library modules currently supported by ESBMC-Python.

## Basic Constructs

- **Control flow**: `if`/`elif`/`else`, `for` (with `range()`), `while`, `for ... else` and `while ... else` (the `else` clause runs when the loop completes without `break`; both are lowered via a did-not-break flag)
- **Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **Logical operations**: `and`, `or`, `not`
- **Identity comparisons**: `is`, `is not` (including `x is None`, `x is not None`)
- **Tuple-unpacking assignment**: `a, b = b, a` and cross-binding forms like `a, b = b, a % b` evaluate the entire right-hand side before binding any target (Python's parallel-assignment semantics), so swaps and idioms such as the Euclidean-GCD loop `while b: a, b = b, a % b` are lowered correctly. The simple non-cross-binding shape (`x, y = 1, 2`) uses direct assignment. Unpacking targets may be subscripts or attributes (`a[i], b.x = ...`), and may be **nested** (`(a, b), c = ((1, 2), 3)`, including over a runtime right-hand side such as a for-loop element or function return: `for (u, v), w in items:`).
- **Walrus operator** (PEP 572 `:=`): assignment expressions in the contexts where the target is evaluated exactly once — an `if`/`elif` condition (`if (n := len(data)) > 2:`), a standalone assignment expression (`x = (y := 5)`), and a comprehension filter (`[d for v in data if (d := v * 2) > 4]`). The expression evaluates to the bound value. Use inside `and`/`or` operands and `while`-loop conditions is refused with a clear diagnostic (see [Limitations](./limitations#walrus-operator)).
- **None handling**: Proper type distinction from `int`, `bool`, `str`, etc.; correctly falsy in boolean contexts (`None and True` → `None`, `None or 1` → `1`)
- **Global variables**: The `global` keyword for accessing and modifying global scope from within functions
- **Context managers**: `with` and `async with` statements via preprocessor desugaring into explicit `__enter__`/`__exit__` calls:
  - `with EXPR as VAR: BODY` — binds the return value of `__enter__()` to `VAR`
  - `with EXPR: BODY` — context manager used without variable binding
  - `with A as a, B as b: BODY` — multiple context managers in one statement; expanded left-to-right, `__exit__` called in reverse order
  - `async with` is handled identically to `with`
  - Exception suppression: when the body raises, `__exit__(type, value, traceback)` is called and the exception is re-raised iff the return value is falsy, matching CPython semantics. This covers static `return True`, conditional returns (`return self.flag`, `return et is ValueError`), implicit-`None` bodies (propagate by default), and single-level base-class inheritance of `__exit__`.

## Functions and Methods

- **Function definitions**: Parameters, return values, and calls
- **Variadic parameters**: `*args` syntax for variable-length positional argument lists
- **Type annotations**: Basic types (`int`, `float`, `bool`, `str`), `Any`, `Union[T1, T2]`, PEP 604 `T1 | T2` syntax (including `T | None` on class attributes)
- **Union types**: Both `Union[int, bool]` and `int | bool` syntax are supported, including chained unions with multiple members
- **Type widening**: ESBMC selects the widest type from Union members using the hierarchy `float > int > bool`
- **Any type**: When a function is annotated `-> Any`, ESBMC infers the actual return type by analyzing return statements in the function body; supports `int`, `float`, `bool`, and expressions evaluating to those types
- **Variable inference**: Variables annotated with `Any` that are assigned from function calls inherit the function's inferred return type
- **`Optional[T]` equality**: Equality (`==`, `!=`, `is`, `is not`) between an `Optional[T]` value and a matching primitive succeeds after an `is not None` round-trip — the primitive side is implicitly cast to the pointer-backed representation. Ordered comparisons (`<`, `>`, `<=`, `>=`) on `Optional[T]` are deliberately disabled (they would compare addresses, not values).
- **`Optional[T]` return type**: A function annotated `-> Optional[T]` (the `typing.Optional` subscript form) lowers to a `T*` pointer with `None` encoded as `NULL`, so a body that returns `None` on some path verifies correctly. `None` comparisons (`is`, `is not`, `==`, `!=`) applied directly to such a call — `f() is None`, `f() == None` — are evaluated against the function's return type rather than collapsed to a constant. (The dedicated `Optional<T>` struct representation is used for the PEP 604 `T | None` annotation with primitive `T`, not for the `Optional[T]` subscript form.)
- **Lambda expressions**: Single-expression lambdas with multiple parameters; converted to regular functions and stored as function pointers; can be assigned to variables and called indirectly

## Object-Oriented Programming

- **Classes**: Definitions, methods, and attributes
- **Class attributes**: Class-level variables shared across all instances; supports both explicit type annotations and automatic type inference from assigned values; accessible via both `instance.attr` and `ClassName.attr`
- **PEP 604 attribute annotations**: `self.x: T | None` (and other `T1 | T2` `BinOp` annotations) are recognised and mapped to the same pointer-to-`T` encoding used for `Optional[T]`
- **Instance variables**: Attributes defined in `__init__`
- **Object reference semantics**: When an instance attribute is assigned an aliased class-instance reference (e.g. `self.head = head` from a constructor parameter), the field is stored as a reference, so mutating the object through one binding is visible through the attribute (and vice versa). This makes linked-list, queue, and tree patterns that reassign such attributes through chained references (`curr = q.head; curr = curr.nxt; q.head = curr`) verify correctly. A fresh-constructor RHS (`self.a: A = A()`) is still constructed in place by value.
- **Return-by-reference for class instances**: A function whose return annotation resolves to a user-defined class returns a `Cls*` reference to the heap-allocated object rather than a value copy, so the returned object survives the callee frame and preserves identity/aliasing across the call boundary (`y = f(x); y.v = 1` is observed through `x` when `f` returns its argument). This matches CPython and the pointer representation already used for locals, parameters, and `self`; `return self` / `return param` and `return ClassName(...)` are all handled.
- **Self-referential instance attributes**: When an attribute set in `__init__` from a `param=None`-defaulted parameter (e.g. `self.successor = successor`) is populated at construction time (`Node(2, a)`), its field type is recovered by unifying the matching positional constructor argument across module-level `ClassName(...)` calls — enabling linked-list / tree patterns and multi-level attribute chains such as `c.successor.successor`. This also works when the class is imported from another module (`from node import Node`): the attribute types are inferred across the module boundary, so a nested read like `node.successor.value` on an imported-class instance resolves.
- **Inheritance**: Single and multi-level inheritance; verification of scenarios involving overridden methods
- **`super()` calls**: `super().__init__(...)` and other `super().method(...)` calls, enabling verification of polymorphic behavior and parent-constructor side effects
- **Explicit base-class `__init__`**: unbound parent-constructor calls of the form `Base.__init__(self, ...)` (the pre-`super()` idiom) are dispatched to the base constructor with `self` bound correctly
- **`@property` getters**: reading a `@property`-decorated attribute (`obj.area`) invokes the decorated getter method rather than looking up a struct field; inherited properties resolve through the base class
- **Classes as first-class values**: a class name passed as a bare value (`register(Twist)`) — the class object itself, not an instance — is modelled as an opaque placeholder for inert uses (storing/forwarding the class), while normal construction through the name (`Twist()`) is unaffected
- **Class-method defaults**: `Name` defaults referencing `ESBMC_default_*` helpers are hoisted past the enclosing `ClassDef` so they remain visible at call sites

## String Formatting and Literals

- Basic variable interpolation: `f"Hello {name}!"`; multiple variables in one f-string
- Built-in variable access in f-strings: `f"Running as: {__name__}"`
- Integer format specs: `f"{num:d}"`, `f"{num:i}"`
- Float format specs: `f"{val:.2f}"`, `f"{price:.1f}"`
- Full format specs applied to f-string fields: fill/alignment (`<`, `>`, `^`, and sign-aware `=`), width, zero-padding, and the `+`/`-`/space sign flags — e.g. `f"{x:03d}"` → `"007"`, `f"{n:05d}"` → `"-0042"`, `f"{s:>5}"` → `"   ab"`, `f"{v:07.2f}"` → `"0001.50"`
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
- `extend(iterable)`: Append all elements from an iterable (list, string, tuple, or function-call result)
- `reverse()`: Reverse in place
- `sort()`: Sort in place (no arguments; key/reverse parameters not supported)
- `insert(i, x)`: Insert at position; handles index at/beyond end, within bounds, and empty lists
- `count(x)`: Return the number of occurrences of a value
- `index(x[, start[, end]])`: Return the position of the first occurrence of a value; the optional `start`/`end` bounds search the slice `l[start:end]` and return the absolute index (CPython slice-clamping semantics), raising `ValueError` if not found
- `in` operator: Membership testing (`2 in [1, 2, 3]`), including membership of a user-class instance by identity (`obj in [obj]`)
- `del l[i]`: Remove the element at a constant index; `del l[i:j]` removes a slice
- **Slice assignment**: `l[i:j] = src` and the extended form `l[i:j:k] = src`, including grow/shrink replacement (step 1), step > 1 (CPython requires matching lengths), and negative step (`l[::-1] = src`)
- `+` operator: List concatenation (`[1,2] + [3,4]`)
- **Repetition**: `lst * n` with both literal and variable lists
- **Nested lists**: Method calls on subscripted elements (e.g., `nested[i].append(v)`)
- **Typed instance attributes**: `self.attr: List[T]` instance attributes with full method support

### Strings

**Predicates**: `startswith()`, `endswith()`, `isspace()`, `isalpha()`, `isdigit()`, `islower()`, `isupper()`, `isalnum()`, `isnumeric()`, `isidentifier()`, `istitle()`, `isascii()`, `isdecimal()`, `isprintable()`

`startswith()`/`endswith()` also accept a **tuple of affixes** (`s.startswith(("ab", "x"))` is true if `s` matches any element) and the optional **position arguments** `s.startswith(prefix, start[, end])`, evaluated as `s[start:end].startswith(prefix)` with Python slice clamping over constant receivers.

**Case conversion**: `lower()`, `upper()`, `capitalize()`, `title()`, `swapcase()`, `casefold()`

**Search**: `find()`, `rfind()`, `index()`, `rindex()`, `count()` (with optional range arguments). `index()`/`rindex()` are the raising forms of `find()`/`rfind()` — they return the first/last position and raise `ValueError` when the substring is absent (rather than returning `-1`).

**Modification**: `replace()`, `strip()`, `lstrip()`, `rstrip()`, `removeprefix()`, `removesuffix()`, `translate(str.maketrans(...))` (constant-folded over ASCII operands; the two- and three-argument `maketrans` forms map and delete characters, matching CPython)

**Splitting/joining**: `split()`, `rsplit()`, `splitlines()`, `partition()`, `rpartition()`, `join()`

**Padding**: `center()`, `ljust()`, `rjust()`, `zfill()`, `expandtabs()`

**Formatting**: `format()` with `{}`, `{0}`, `{name}` placeholders, including **format specs** on any positional, indexed, or keyword field (`"{:.2f}"`, `"{:>5}"`, `"{:^6}"`, `"{:05d}"`, `"{:+d}"`, `"{:08.2f}"` — fill/alignment, width, zero-pad, sign, and precision, applied to the original constant value); `format_map()` with constant dicts

**Printf-style `%` operator**: bare conversions (`"%s=%d" % ("x", 5)`, `"%x" % 255`), flags/width/precision (`"%.2f" % 3.14159`, `"%05d" % 7`, `"%-5d" % 42`, `"%+d" % 5`, `"%8.2f"`, `"%e"`, sign-aware zero padding `"%07.2f" % -3.1`), integer precision as a minimum digit count (`"%.3d" % 5` → `"005"`), string width/precision truncation (`"%10s"`, `"%.3s"`), and the `%(name)s` mapping form against a right-hand dict (`"%(n)d" % {"n": 5}`, repeated keys, `%%` literal percent)

**Slicing**: `s[start:end]`, omitted bounds (`s[:end]`, `s[start:]`), negative indices (`s[-3:]`), empty slices

**Operators**: `in` (substring test), `*` (repetition: `"a" * 3`, `3 * "a"`, boolean multipliers)

**Comparison**: `==`, `!=`, and the ordered comparisons `<`, `<=`, `>`, `>=` over constant strings, evaluated lexicographically by code point (characters compared as unsigned bytes, so high-bit characters order as in CPython)

**Non-constant receivers**: Calls with a non-constant string receiver no longer abort GOTO conversion. Three layers cooperate to give a sound result:

1. **Constant folding (symex layer).** When the receiver and arguments are compile-time constants — either as literals or after AST-level const-propagation of a single `Assign`/`AnnAssign` in the enclosing function — the result folds to an exact value. Folded methods include `swapcase`, `upper`, `lower`, `casefold`, `capitalize`, `title`, `isalpha`, `isdigit`, `isalnum`, `isspace`, `isupper`, `islower`, `startswith`, `endswith`, `count`, `find`, `rfind`, `index`, `strip`, `lstrip`, and `rstrip`.
2. **Runtime operational models.** A growing set of `str` methods lower to bounded operational models in `src/c2goto/library/python/string.c` rather than a bare nondet. These include the case transforms `lower()`, `upper()`, `swapcase()`, `capitalize()`, `title()` (`__python_str_lower`/`_upper`/`_swapcase`/`_capitalize`/`_title`), the predicates `isupper()`, `islower()`, `isalpha()`, `isdigit()`, `isalnum()`, `isspace()`, the counter `count(sub)`, and the searches `find()`/`rfind()`. Concrete arguments fold via symex's constant propagation; symbolic receivers get a real symbolic count, predicate, or returned string rather than an unconstrained nondet. The string-returning models cap the receiver at ~255 characters (a 256-byte buffer): an over-length receiver trips an explicit assertion (e.g. `String too long for swapcase() - exceeds 255 characters`), except `upper()`, which truncates at the buffer bound. `str.join(iterable)` also lowers to a runtime model (`__python_str_join`, 511-character result bound) when its iterable is a variable whose initialiser cannot be folded (e.g. a `List[str]` parameter): an empty list yields `""` and a non-empty list is concatenated element-by-element with the separator.
3. **Nondet fallback.** For all other string methods, a non-constant receiver yields a sound symbolic value (nondet `char *`, `bool`, or `int`) instead of aborting. `partition()` falls back to the 3-tuple `("", "", "")` so `len(t) == 3` holds; `splitlines()` falls back to an empty list. `format()` and `format_map()` follow the same shape when their format string or arguments are non-constant. `join()` falls back to a nondet `char *` when its iterable argument is not a literal list that can be folded at conversion time (e.g. `sorted(...)`, a list comprehension, or a function-call result), so such calls convert instead of aborting. The result is a sound over-approximation: specific functional values are not preserved, but safety checks downstream remain meaningful.

### Sets

- **Literals**: `{1, 2, 3}`
- **Empty set**: `set()` (note: `{}` creates an empty dict, not a set)
- **From iterable**: `set(list)`, `set(str)`, `set(d.keys())`, `set(d.values())`
- **Operators**: `-` (difference), `&` (intersection), `|` (union), `^` (symmetric difference, equivalent to `.symmetric_difference()`); the augmented assignment `^=` is supported too
- **Methods**: `issubset(other)`, `issuperset(other)`, `symmetric_difference(other)`, `update(other)`, and the variadic method forms `union(*others)`, `intersection(*others)`, `difference(*others)` (non-mutating; they accept any number of operands, route to the same builders as the `|`/`&`/`-` operators, and return a fresh set). Subset/superset relations are evaluated directly over the operand lists (a set-materialization bypass), so `set(iterable).issuperset(...)` works without first building the set.
- **Membership**: `x in s`, `x not in s`
- **Equality**: `s1 == s2`, `s1 != s2` (order-independent)
- **`len()`** built-in

### Tuples

- Literals: `(1, 2, 3)`, `()`, `(5,)` (single-element)
- Mixed types: `(42, "hello", 3.14)`
- Nested tuples: `((1, 2), (3, 4))`
- Constant-index access with bounds checking: `t[0]`, `t[2]`
- Generic annotation (`t: tuple`) and parameterized annotation (`-> tuple[int, int]`)
- Equality comparison: `t1 == (1, 2, 3)`
- Ordering comparison (`<`, `<=`, `>`, `>=`): element-wise **lexicographic**, matching CPython (`(1, 2) < (1, 3)`)
- `tuple(...)` constructor: `tuple(t)` returns the tuple unchanged; `tuple(list)` (a literal, variable, or list-returning call such as `sorted(...)`) builds a shallow copy of the list; `tuple("ab")` over a constant string yields a tuple of single-character strings (`('a', 'b')`). `tuple(bytes)` is rejected with a clean error (CPython would produce a tuple of ints).
- `len()` built-in
- `isinstance(obj, tuple)` type checking

### Dictionaries

- **Literals**: `{"a": 1, "b": 2}`
- **Constructor**: `dict()` (empty), `dict(a=1, b=2)` (keyword form), and `dict([("k", 9)])` (positional iterable of key/value pairs)
- **Subscript access**: `d["a"]`; raises `KeyError` if absent
- **Subscript assignment**: `d["c"] = 3`
- **Membership**: `"a" in d`, `"a" not in d`
- **Deletion**: `del d["a"]`; raises `KeyError` if absent
- **Equality**: `d1 == d2` (order-independent)
- **Iteration**: `for` loops over `d.keys()`, `d.values()`, `d.items()`, and directly over the dict (`for k in d:`). For a **local dict literal** with tuple keys, the destructuring form `for u, v in d:` is also supported — each key is unrolled as a tuple literal so it unpacks correctly. Iteration, dict comprehensions, and `.items()` unpacking over an **unannotated parameter dict** are sound for **scalar keys** and **integer-tuple keys** — the concrete `dict[K, V]` is recovered (scope-aware) from the call sites, and an ambiguous shape stays a clean error rather than a wrong guess. String-tuple-keyed parameter dicts remain a known gap ([#5571](https://github.com/esbmc/esbmc/issues/5571)).
- **`update(other)`**: Merge another dict; also accepts keyword arguments (`d.update(a=1, b=2)`)
- **`get(key[, default])`**: Return value or default; returns `Optional[T]` when no default is provided
- **`setdefault(key[, default])`**: Insert key with default if absent, then return value; supports `int`, `float`, `bool`, `str`
- **`pop(key[, default])`**: Remove and return value; raises `KeyError` if absent and no default
- **`popitem()`**: Remove and return last inserted `(key, value)` pair; raises `KeyError` if empty
- **`clear()`**: Remove all entries in place; the dict stays usable afterwards (`len(d) == 0`)
- **Nested dicts**: `dict[int, dict[int, int]]`
- **`Optional[T]` values**: `dict[str, Optional[T]]` storage and retrieval
- **Dict comprehensions**: `{k: v for ...}` is lowered to an empty dict plus a population loop. Supported iterables include `range(...)` (constant or symbolic bound), a list of tuples, and `d.items()` with a `(key, value)` tuple target (`{k: v + 1 for k, v in d.items()}`), with optional `if` filters. Subsequent key lookups return the populated values rather than raising `KeyError`.

## Complex Numbers

- **Literals**: `3+4j`, `1j`, `0j`
- **Constructor**: `complex(real, imag)` with `int`, `float`, or `bool` arguments
- **Attributes**: `.real`, `.imag` (read-only `float`)
- **Methods**: `.conjugate()` — returns complex conjugate
- **Arithmetic**: `+`, `-`, `*`, `/`, `**`; augmented assignment (`+=`, `-=`, `*=`, `/=`)
- **Promotion**: `int`, `float`, and `bool` operands are automatically promoted to `complex`
- **`abs(z)`**: Returns the magnitude as a `float` (IEEE-754 hypot)
- **Boolean context**: `bool(z)` is `False` only when both `.real` and `.imag` are `0.0` (signed-zero aware)
- **Equality**: `==`, `!=`; ordering operators (`<`, `<=`, `>`, `>=`) and `//`, `%` raise `TypeError`
- **Annotations**: `z: complex`; `Optional[complex]`, `Union[complex, float]`

## Enum Module (`enum`)

- **`Enum`** base class: members store `value: int` and `name: str`
- **Comparison**: `==`, `!=`
- **Hash**: `__hash__()` returns `value`
- **Representation**: `__str__()` and `__repr__()` return `name`

## Bytes and Integers

Byte sequences and integer class methods:

- **`bytes(...)` constructor** — `bytes(iterable-of-ints)` (e.g. `bytes([1, 2, 3])`) and `bytes(n)` (`n` zero bytes) build a real byte array, like a `b"..."` literal, so `len()` and indexing work; byte literals (`b"abc"`) are also supported
- **`int.from_bytes(bytes_data, byteorder, *, signed=False)`** — converts a byte sequence to an integer; supports big- and little-endian, signed and unsigned. Endianness may be given positionally (`int.from_bytes(b, "big")`) or as the `byteorder=` keyword (`int.from_bytes(b, byteorder="big")`)
- **`int.to_bytes(length=1, byteorder='big', *, signed=False)`** — converts an integer to a byte sequence. `length` and `byteorder` may each be passed positionally or by keyword, and both default (CPython 3.11+: `(5).to_bytes()` → one big-endian byte). A non-constant `byteorder` is rejected with a clean error rather than silently defaulting to big-endian
- **`bytes.hex([sep[, bytes_per_sep]])`** — constant-folds a literal `bytes` object to its hex string. With the optional one-character `sep` (and optional `bytes_per_sep` group size) it reproduces CPython grouping exactly: `bytes([1, 2, 3]).hex("-")` → `"01-02-03"`, `bytes([0xb9, 0x01, 0x9e, 0xf3]).hex("_", 2)` → `"b901_9ef3"` (positive group size counts from the right, negative from the left)
- **`bytes.fromhex(s)`** — constant-folds a hex string to a `bytes` object (the inverse of `.hex()`); accepts upper/lowercase digits and ASCII whitespace *between* byte pairs, and raises CPython's `ValueError` on odd-length or non-hex input
- **`bytes.startswith`/`endswith`/`find`/`rfind` over literal operands** — folded directly over the byte-array representation when the receiver (and affix/sub argument) are literal `bytes([...])` constructors, so `bytes([1,2,3]).endswith(bytes([2,3]))` is `True` and `bytes([1,2,3]).find(bytes([2,3]))` is `1`. `find`/`rfind` also accept a single integer byte. Non-literal receivers, `b"..."` literals, and the position-argument forms fall through to the existing dispatch unchanged
- **`bytes.index`/`rindex` over literal operands** — the raising counterparts of `find`/`rfind`, folded over the byte-array representation: `bytes([1,2,3]).index(bytes([2,3]))` is `1`, `rindex` returns the last occurrence, a single integer byte is accepted (`bytes([1,2,3]).index(2)`), NUL bytes are treated as ordinary data, and an absent subsequence raises a catchable `ValueError` (rather than `find`'s `-1`)
- **`int.bit_length(n)`** — returns the number of bits required to represent `n` in binary. The operational model bounds the loop length by `512`, which covers narrow 64-bit `IntWide` and 512-bit `--ir` bignum receivers and guarantees termination on symbolic `n` without an explicit `--unwind`.
- **`int.conjugate()`** — returns the integer unchanged (the conjugate of a real integer is itself; part of the numeric-tower API)
- **Numeric-tower properties** — `int.numerator` / `int.denominator` (an `int` is the ratio `n/1`), `int.real` / `int.imag`, and `float.real` / `float.imag`. `float.numerator` / `float.denominator` and these properties on a `bool` deliberately raise a clean `AttributeError` (CPython's `float` is not a `Rational`).
- **`float.is_integer()`** — constant-folds on a literal float receiver (e.g. `(2.0).is_integer()` → `True`), evaluated as `isfinite(d) && d == trunc(d)` to match CPython
- **`float.as_integer_ratio()` / `int.as_integer_ratio()`** — folds a numeric literal to its exact `(numerator, denominator)` pair in lowest terms (`(2.5).as_integer_ratio()` → `(5, 2)`, `(5).as_integer_ratio()` → `(5, 1)`, `(-2.5)` → `(-5, 2)`), including the exact dyadic ratio of a non-representable decimal (`(0.1).as_integer_ratio()`); the folded tuple unpacks (`n, d = (1.5).as_integer_ratio()`)
- **`str.encode()` / `bytes.decode()`** — standalone constant-folded conversions over **ASCII** data (`s.encode()` → byte array of ordinals, `b.decode()` → string of byte values); a non-ASCII / multi-byte character falls through to a clean error, matching CPython's `UnicodeDecodeError`. The round-trip form `s.encode().decode()` is also supported.

## Error Handling

- **`try`/`except`** blocks with multiple handlers and `except ExceptionType as var` binding
- **`try`/`finally`** blocks: the `finally` body runs on normal completion, after a caught exception, and when an exception propagates uncaught (run `finally`, then re-raise). Bare `try`/`finally` (no `except`) is supported. Shapes that cannot be lowered soundly are refused with a clean diagnostic (see [Limitations](./limitations#exception-handling))
- **`raise`** statements with exception instantiation and custom messages; bare `raise` re-raises the active exception inside an `except` handler
- **`assert`** statements for property verification
- **`__ESBMC_assume`** for constraining non-deterministic inputs
- **`ImportError` guards**: Imports inside `try/except ImportError` are handled statically

**Built-in exception hierarchy**:
- `BaseException` → `Exception` → `AssertionError`, `ValueError`, `TypeError`, `IndexError`, `KeyError`, `ZeroDivisionError`, `ImportError`
- `OSError` → `FileNotFoundError`, `FileExistsError`, `PermissionError`

Exception instances expose a message attribute and support `__str__()`.

## Concurrency (`threading` module)

ESBMC-Python lowers `threading` primitives onto ESBMC's existing pthread operational model in `src/c2goto/library/pthread_lib.c`, so symex's interleaving exploration, deadlock detection, and data-race detection apply.

### `threading.Lock`

`Lock.acquire()` and `Lock.release()` mirror `pthread_mutex_lock_noassert` via `__ESBMC_atomic_begin / __ESBMC_assume / __ESBMC_atomic_end`. Supported usage:

- `lock = threading.Lock()` at module scope or as a class instance attribute
- `lock.acquire()` / `lock.release()`
- Multiple `Lock` instances co-existing on the same object (e.g. paired `mutex` and `lock` fields)
- `from threading import Lock` aliasing

### `threading.Thread`

For each `Thread(target=f, args=(...))` construction site, the frontend synthesises a per-call-site trampoline `__pythread_trampoline_<N>` plus three helpers in `pthread_lib.c` (`__pyt_init_tid`, `__pyt_terminate`, `__pyt_join`) that mirror `pthread_create` / `pthread_join` bookkeeping. The resulting GOTO program is a direct call (not a function pointer), so symex preserves precision.

Supported `Thread` shapes:

- `target=f` — `f` must be a function statically resolvable at the construction site (a `Name` or attribute chain). Lambdas, runtime-callable values, and `Thread`-subclassing's `run` override are out of scope.
- `args=(...)` — must be a tuple literal whose elements are expressions evaluable at the construction site. Passing simple values (ints, floats, bools, strings) works end-to-end.
- `t.start()` and `t.join()` — lower to `pthread_create` and `pthread_join` semantics; `join` establishes happens-before.
- Multiple construction sites per program, with independent trampolines.

### Data-race detection

Python module-level globals (declared with `x: T = …` or `x = …` at module scope) are flagged in the symbol table so they are visible to ESBMC's race-assertion pass. Two threads writing to the same global without synchronisation are detected under `--data-races-check`:

```python
import threading

shared: int = 0

def writer_a() -> None:
    global shared
    shared = 1

def writer_b() -> None:
    global shared
    shared = 2

t1 = threading.Thread(target=writer_a)
t2 = threading.Thread(target=writer_b)
t1.start(); t2.start(); t1.join(); t2.join()
```

`esbmc race.py --incremental-bmc --data-races-check` reports `W/W data race on py:race.py@shared` and `VERIFICATION FAILED`.

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
- **Local bindings shadow imported modules**: When a name is both an imported module and a local binding (e.g. a parameter `node` while `from node import Node` is in scope), attribute access such as `node.value` resolves to the local binding, following Python's LEGB rule, rather than to a module member.
- **Selective imports preserve module-level constants**: `from M import f, C` retains plain `Assign` bindings such as `INT_BOUND = 1024` in addition to `AnnAssign` ones. Tuple-unpacking targets are treated atomically.
- **Parser package layout**: The Python parser ships as a package under `src/python-frontend/parser/` (entrypoint `parser/__main__.py`, public facade `parser/__init__.py`, import resolution in `parser/import_resolver.py`). The resolver emits deterministic, review-friendly diagnostics for missing modules, cyclic imports, and relative-import rewrites.

## Built-in Functions

| Function | Notes |
|---|---|
| `abs`, `divmod` | Standard arithmetic |
| `int`, `float`, `bool`, `chr`, `ord`, `str`, `repr`, `hex`, `oct`, `bin`, `ascii` | Type conversions and representations. `bin`, `hex`, and `oct` accept non-literal integer arguments: a compile-time-foldable expression (e.g. `bin(round(3.0))`) folds to the exact literal, while a genuinely symbolic operand (a function parameter or variable) lowers to a runtime operational model (`__python_int_to_{bin,hex,oct}`) producing the correctly prefixed string (`0b`/`0x`/`0o`, a leading `-` for negatives, lowercase hex digits); a non-integer argument still raises `TypeError`. `bin` is `LLONG_MIN`-safe; `ascii` emits `\xNN`/`\uNNNN`/`\UNNNNNNNN` escapes for non-ASCII codepoints. |
| `pow(b, e)` | Shares the `**` operator lowering (integer, float, bool operands) |
| `pow(b, e, m)` | 3-argument modular exponentiation: exact `BigInt` for constant integer operands; symbolic operands raise an unsupported diagnostic rather than emit unsound floating-point modulo |
| `format(value[, spec])` | Builtin formatting (distinct from the `str.format()` method). Constant-folds a literal integer with a bare presentation-type spec (`'d'`/`'x'`/`'X'`/`'o'`/`'b'` or empty) — `format(255, "x")` → `"ff"` (no `0x`/`0o`/`0b` prefix, leading `-` for negatives, `LLONG_MIN`-safe) — and a constant string with the default spec to itself. Width/alignment/precision specs, float values, and variable arguments raise a clean error rather than a wrong fold |
| `callable(obj)`, `issubclass(cls, base)` | Resolved at compile time from the symbol table and AST class hierarchy |
| `len` | Works on lists, sets, strings, tuples |
| `range` | Used in `for` loops |
| `min(a, b)`, `max(a, b)` | Two-argument form only; promotes `int` to `float` |
| `min([...])`, `max([...])` | Single-list form; supports `int`, `float`, and `str` element types. The `key=` argument is honoured over **constant** lists for the `lambda x: x[K]`, `key=abs`, and `key=len` forms (the winning element is returned; ties break toward the first occurrence) |
| `sum([...])` | Sum of list elements; supports `int` and `float` |
| `sum(range(EXPR))` | Single-arg `sum` of a single-arg `range` is rewritten to the Gauss closed form `EXPR * (EXPR - 1) // 2 if EXPR > 0 else 0`, yielding an exact value (and `0` for `EXPR <= 0`) instead of a nondet result |
| `sorted(iterable)` | Returns a new sorted list; supports `int`, `float`, and `str` elements |
| `any([...])` | List literals only; short-circuit OR logic |
| `all([...])` | List literals only; short-circuit AND logic |
| `enumerate(iterable, start=0)` | Tuple unpacking and single-variable forms; optional `start` |
| `zip(a, b, ...)` | Lowered to an index-based `while` loop in `for` form, mirroring `enumerate` |
| `reversed(iter)` | Lowered to an index-based `while` loop in `for` form; `reversed(range(...))` is rewritten to an equivalent forward `range(...)` |
| `filter(pred, iter)` | Lowered to an index-based `while` loop guarded by `pred` in `for` form |
| `list()` | Zero-arg constructor lowers to an empty list literal. `list(iterable)` over a list, `range`, or tuple (`list((2, 3))`, `list(t)`) builds a real list; `list("abc")` over a constant string yields a list of single-character strings. `list(bytes)` is rejected with a clean diagnostic (CPython would produce a list of ints). |
| `isinstance(obj, type)` | Runtime type checking |
| `float("nan")`, `float("inf")` | Special values (case-insensitive, whitespace-tolerant) |
| `input()` | Modelled as nondeterministic string, max 256 chars |
| `print(...)` | Arguments evaluated for side effects; no output produced |

## Complex Math Module (`cmath`)

All functions accept `complex`, `float`, `int`, or `bool` arguments. Real inputs are promoted to complex automatically.

**Constants**: `pi`, `e`, `tau`, `inf`, `nan`, `infj`, `nanj`

**Conversion**: `phase(z)`, `polar(z)` → `(r, φ)`, `rect(r, φ)` → `complex`

**Power/log**: `exp(z)`, `log(z[, base])`, `log10(z)`, `sqrt(z)`

**Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`

**Hyperbolic**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Utilities**: `isnan(z)`, `isinf(z)`, `isfinite(z)`, `isclose(a, b[, rel_tol, abs_tol])`

## Math Module (`math`)

**Constants**: `pi`, `e`, `inf`, `tau`, `nan`

**Rounding**: `floor(x)`, `ceil(x)`, `trunc(x)`, `fabs(x)`, `modf(x)`

**Power/log**: `exp(x)`, `expm1(x)`, `exp2(x)`, `log(x)`, `log1p(x)`, `log2(x)`, `log10(x)`, `pow(x, y)`

**Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`

**Hyperbolic**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Integer helpers**: `factorial(n)`, `gcd(*ints)`, `lcm(*ints)` (variadic — any number of integer arguments, including the 0- and 1-argument forms, folded via the binary model and working on symbolic operands), `isqrt(n)`, `perm(n[, k])`, `comb(n, k)`, `prod(lst[, start])` (expects `list[int]`)

**Geometry**: `hypot(x, y)`, `dist(p, q)` (expects `list[float]`)

**Utilities**: `fmod(x, y)`, `remainder(x, y)`, `copysign(x, y)`, `degrees(x)`, `radians(x)`, `isclose(a, b)`, `isfinite(x)`, `isnan(x)`, `isinf(x)`

**Advanced**: `cbrt(x)`, `erf(x)`, `erfc(x)`, `gamma(x)`, `lgamma(x)`, `frexp(x)` → `(mantissa, exponent)`, `ldexp(x, i)`, `nextafter(x, y)`, `ulp(x)`, `sumprod(a, b)` (expects `list[float]`), `fsum(values)` (expects `list[float]`)

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
- `random.randint(a, b)` → nondeterministic `int` N where `a ≤ N ≤ b`
- `random.getrandbits(k)` → nondeterministic non-negative `int` in `[0, 2**k − 1]`; raises `ValueError` if `k < 0`; returns `0` when `k == 0`
- `random.randrange(start[, stop[, step]])` → randomly selected integer from the specified range; single-argument form (`randrange(stop)`) is also supported
- `random.choice(seq)` → nondeterministic element `seq[i]` for a constrained index; raises `IndexError` on an empty sequence
- `random.sample(population, k)` → under-approximation that returns the first `k` elements of `population`; raises `ValueError` if `k < 0` or `k > len(population)`
- `random.shuffle(lst)` → under-approximation; leaves the list untouched
- `random.seed(a=0)` → no-op; nondet outputs already cover any seed-dependent outcome

See also: [Random Operational Model](./random-operational-model)

## Collections Module (`collections`)

- **`defaultdict(default_factory)`**: Dict subclass that returns a default value for missing keys; modelled as a plain `dict` with a nondeterministic default. When the dict has no value annotation, the value type is inferred from the factory or from a subscript assignment in the enclosing function: built-in type factories (`defaultdict(int)`, `float`, `bool`, `str`), nullary `lambda` factories whose body is a constant or a built-in constructor call (`defaultdict(lambda: float('inf'))`), and constant literal assignments (`d[k] = 5`, `0.0`, `True`, `"x"`) all map to the matching value type, so `min`/`max`/comparisons over `d[k]` no longer fall back to `char *`
- **`Counter`**: Mapping of elements to integer counts; supports `__getitem__`, `__setitem__`, `values()`, and boolean truthiness
- **`deque`**: List-backed double-ended queue; supports construction, indexing, `__setitem__`, `append()`, and the FIFO-front methods `popleft()` (front pop) and `appendleft()` (front insert), enabling FIFO/BFS patterns. Aliased imports such as `from collections import deque as Queue` resolve correctly
- **`OrderedDict`**: Supports construction and basic indexing / `append` / `__setitem__`

## Queue Module (`queue`)

A single-threaded verification model: `queue.Queue` is backed by a plain list (FIFO) and `queue.LifoQueue` by a list-backed stack (LIFO). Both the qualified form (`queue.Queue()`) and `from queue import LifoQueue` work.

- **`Queue`** (FIFO): `put(item)` → append, `get()` → pop front, in insertion order
- **`LifoQueue`** (LIFO / stack): `put(item)` → append, `get()` → pop back
- **Shared methods**: `qsize()`, `empty()`, `full()`, `put_nowait()`, `get_nowait()`; `task_done()` and `join()` are accepted no-ops
- **`maxsize`**: tracked by `full()` (`Queue(2)`); `put()` does not block on it

The blocking semantics of `put()`/`get()` (the `block`/`timeout` arguments) are not modelled — there is nothing to block on under sequential symbolic execution. An unguarded `get()` on an empty queue pops from an empty list, reported as an `IndexError`; guard with `empty()`/`qsize()` first.

## Datetime Module (`datetime`)

- **`datetime.datetime(year, month, day)`**: Constructs a datetime object with fields `year`, `month`, `day`, `hour` (0), `minute` (0), `second` (0), `microsecond` (0)

## Decimal Module (`decimal`)

- **`Decimal`** class with full arithmetic:
  - **Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=`
  - **Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`
  - **Unary**: `-d` (negation), `+d` (unary plus), `abs(d)`
  - **Query methods**: `is_nan()`, `is_snan()`, `is_qnan()`, `is_infinite()`, `is_finite()`, `is_zero()`, `is_signed()`, `is_normal()`, `is_subnormal()`
  - **Copy methods**: `copy_abs()`, `copy_negate()`, `copy_sign(other)`
  - **Comparison / selection methods**: `compare(other)` (returns a `Decimal` `-1`/`0`/`1`), `max(other)`, `min(other)`, `adjusted()` (adjusted exponent)
  - Special values: infinity and NaN (via `is_special` flag; propagated through all operations)

## Heapq Module (`heapq`)

All functions operate on plain Python lists used as min-heaps.

- `heapify(heap)`: No-op in the model (heap invariant assumed)
- `heappush(heap, item)`: Appends item to the heap list
- `heappop(heap)`: Removes and returns the minimum element
- `heappushpop(heap, item)`: Pushes then pops the minimum

## Time Module (`time`)

Functions use a monotonic counter model.

- `time.time()`: Returns a monotonically increasing `float` (increments by 1.0 each call)
- `time.sleep(seconds)`: Validates `seconds >= 0`; no actual delay

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

## NumPy Module (`numpy`)

Partial executable support for list-backed arrays, element-wise arithmetic, selected math functions, and small determinants, now covering 1-D, 2-D, and 3-D (n-D) shapes. Some APIs remain stubs for type inference only.

**Array construction**: `np.array(l)`, `np.zeros(shape)`, `np.ones(shape)` for 1-D, 2-D, and 3-D shapes, including explicit constructor `dtype` coercion for literal `bool`, `int`, and `float` inputs; `np.arange(n)`, `np.full(shape, value)`, `np.eye(N[, M])`, `np.identity(n)`, and `np.linspace(start, stop, num)`. **Symbolic shapes** are handled: a nondeterministic dimension such as `np.zeros(n)` with a constrained `n` yields an array of the corresponding length (`len(a) == 0` when `n == 0`)

**Indexing**: n-D tuple indexing `a[i, j, k]` on 2-D and 3-D arrays (`a[0, 0, 0]`, `a[1, 1, 1]`); supplying more indices than the array has dimensions is rejected. **Boolean-mask selection** `a[mask]` returns the elements where a same-length boolean array is `True` (an all-`False` mask yields an empty array); a non-boolean or symbolic mask is rejected

**Slicing**: bounded 1-D slicing `a[i:j]` on a list-backed array returns a new `list[T]` (bounded, open-ended `a[i:]`/`a[:j]`, and full-copy `a[:]` forms), with the slice typed as the element type rather than collapsing to a scalar. **2-D slicing** selects whole rows (`a[i, :]`) and whole columns (`a[:, j]`)

**Shape manipulation**: `np.reshape(a, shape)` (2-D and 3-D targets; an incompatible element count is rejected), `np.flatten(a)` / `np.ravel(a)` (row-major 1-D view), `np.squeeze(a)` (drop unit-length axes; a non-unit axis is rejected), `np.stack([a, b, ...])` (join arrays along a new leading axis, e.g. 1-D → 2-D), `np.concatenate([a, b, ...])` (join along the existing axis), and `a.astype(dtype)` (dtype conversion; `astype` to a complex dtype is rejected)

**Reductions**: `np.sum(a)`, `np.prod(a)`, `np.min(a)`, `np.max(a)`, `np.mean(a)`, `np.argmin(a)`, `np.argmax(a)` over list-backed arrays

**Comparison and logical ufuncs**: `np.greater`, `np.greater_equal`, `np.less`, `np.less_equal`, `np.equal`, `np.not_equal`, `np.logical_and`, `np.logical_or`, `np.logical_not`, and `np.where(cond, a, b)` (element-wise select; also the scalar-condition form `np.where(False, 1, 2)`)

**Element-wise arithmetic**: `np.add(a, b)`, `np.subtract(a, b)`, `np.multiply(a, b)`, `np.divide(a, b)`, `np.power(a, b)` on literal list-backed inputs, with NumPy-style broadcasting for 1D/2D shapes

**Complex elements**: element-wise complex arithmetic (`add`/`subtract`/`multiply`/`divide`) on complex scalars and arrays, plus `np.conjugate(z)` and `.real`/`.imag` on complex results; division by zero is reported. Complex determinants are rejected (see [Limitations](./limitations#numpy-module))

**Math**: `np.ceil(x)`, `np.floor(x)`, `np.fabs(x)`, `np.sqrt(x)`, `np.trunc(x)`, `np.round(x)`, `np.rint(x)`, `np.copysign(x, y)`, `np.fmin(x, y)`, `np.fmax(x, y)`, `np.remainder(x, y)`, `np.nextafter(x, y)`, `np.sin(x)`, `np.cos(x)`, `np.tan(x)`, `np.arcsin(x)`, `np.arctan(x)`, `np.arccos(x)`, `np.sinh(x)`, `np.cosh(x)`, `np.tanh(x)`, `np.exp(x)`, `np.log(x)`, `np.log2(x)`, `np.log10(x)`, `np.isclose(a, b)` on scalar or literal list-backed 1D/2D inputs. `np.arccos` additionally lowers a runtime 1D array through the libm operational model; a runtime 2D `arccos` is still rejected. The two-output helpers `np.modf(x)` → `(frac, int)` and `np.frexp(x)` → `(mantissa, exponent)` are also supported.

**Modulo**: `np.fmod(x, y)` on scalars and on literal list-backed 1D/2D inputs with NumPy-style broadcasting. Operands wrapped in `np.array(...)` are rejected with `Unsupported operation: numpy.fmod on array operands` rather than mis-folded to a scalar.

**Element-wise float ufuncs**: `np.add`/`np.subtract`/`np.multiply`/`np.divide` on float arrays dispatch to a typed `*_double` operational model instead of reinterpreting IEEE-754 payloads as `int64`. The `--python-no-fold` flag suppresses the frontend's constant-folding paths and forces SMT encoding (useful for differential testing of the folder against the encoder).

**Linear algebra**:

- `np.dot(a, b)`, `np.matmul(a, b)`: 1D/2D inputs with both integer and float backends (via `linalg.c`), including symbolic elements. The integer path carries the operand dtype width and asserts each result element fits the dtype range, so narrow dtypes (`int16`/`int32`) flag accumulation overflow (trivially satisfied for the default 64-bit `int`; combine with `--overflow-check` for int64-level overflow)
- `np.transpose(a)`: 2D arrays, including runtime array variables (1D is the identity); higher-rank transpose is rejected
- `np.linalg.det(a)`: constant numeric 2x2 and 3x3 matrices (complex-valued matrices are rejected)
- `np.linalg.inv(a)`: matrix inverse of a constant numeric 2x2 matrix
- `np.linalg.norm(a)`: Euclidean (L2) norm of a 1-D array (`np.linalg.norm(np.array([3.0, 4.0]))` → `5.0`)
- `np.linalg.eig(a)`: eigenvalues of a constant numeric 2x2 matrix (returned as a list)
- `np.linalg.svd(a)`: singular values of a constant numeric 2x2 matrix (returned as a list)
