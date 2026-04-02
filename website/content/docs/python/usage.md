---
title: Usage
weight: 2
---

## Prerequisites

ESBMC-Python requires the following Python packages to be installed before running verification:

```bash
pip install ast2json
```

Python 3.8 or later is recommended.

## Basic Invocation

Pass a `.py` file directly to `esbmc`:

```bash
esbmc main.py
```

ESBMC will parse the file, convert it to a GOTO program, and run bounded model checking. By default the loop unwind bound is 1; use `--unwind N` to increase it:

```bash
esbmc main.py --unwind 10
```

## Useful Flags

| Flag | Description |
|---|---|
| `--unwind N` | Set the global loop unwind bound to N iterations |
| `--multi-property` | Continue verification after the first failure, reporting all violated properties |
| `--strict-types` | Enable strict type checking for function arguments at verification time |
| `--generate-pytest-testcase` | Generate pytest test cases from counterexamples (see [Pytest Test Generation](./pytest_testgen)) |

## Writing Verification Harnesses

Use `__ESBMC_assume` to constrain inputs and `assert` to express properties:

```python
def divide(a: int, b: int) -> int:
    return a // b

b: int = __VERIFIER_nondet_int()
__ESBMC_assume(b != 0)
assert divide(10, b) >= 0
```

Use `__ESBMC_cover()` with `--multi-property` to check reachability of specific program points:

```python
x: int = __VERIFIER_nondet_int()
if x > 0:
    __ESBMC_cover(x > 100)   # Is this branch reachable?
```

---

## Examples

### Example 1: Division by Zero

The following Python program runs without error in a standard interpreter, but ESBMC reveals a hidden division-by-zero bug.

```python
import random as rand

def div1(cond: int, x: int) -> int:
    if (not cond):
        return 42 // x
    else:
       return x // 10

cond: int = rand.random()
x: int = rand.random()

assert div1(cond, x) != 1
```

```bash
esbmc main.py
```

```
[Counterexample]

State 1 file main.py line 12 column 8 function random thread 0
----------------------------------------------------
  value = 2.619487e-10

State 3 file main.py line 12 column 8 function random thread 0
----------------------------------------------------
  value = 3.454678e-77

State 5 file main.py line 5 column 8 function div1 thread 0
----------------------------------------------------
Violated property:
  file main.py line 5 column 8 function div1
  division by zero
  x != 0

VERIFICATION FAILED
```

ESBMC finds a path where the randomly generated `x` is cast to integer zero, causing division by zero. The counterexample values can be used to construct a concrete failing test:

```python
def div1(cond: int, x: int) -> int:
    if not cond:
        return 42 // x
    else:
        return x // 10

cond = int(2.619487e-10)  # → 0
x = int(3.454678e-77)     # → 0
print(div1(cond, x))      # ZeroDivisionError
```

---

### Example 2: Lambda Expression Verification

```python
def test_lambda_expressions():
    add_ten = lambda x: x + 10
    result1: int = add_ten(5)
    assert result1 == 15

    calculate_volume = lambda length, width, height: length * width * height
    volume: float = calculate_volume(2.0, 3.0, 4.0)
    assert volume == 24.0

    absolute_diff = lambda a, b: a - b if a > b else b - a
    diff1: int = absolute_diff(10, 3)
    diff2: int = absolute_diff(3, 10)
    assert diff1 == 7
    assert diff2 == 7

    is_in_range: bool = lambda x, lower, upper: lower <= x <= upper
    assert is_in_range(5, 1, 10) == True
    assert is_in_range(15, 1, 10) == False

test_lambda_expressions()
```

```bash
esbmc main.py
```

```
VERIFICATION SUCCESSFUL
```

---

### Example 3: Exception Handling

ESBMC can verify that exception-handling paths behave correctly.

```python
def safe_divide(a: int, b: int) -> int:
    try:
        result = a // b
        return result
    except ZeroDivisionError as e:
        return -1

def test_exception_handling() -> None:
    assert safe_divide(10, 2) == 5
    assert safe_divide(10, 0) == -1
    assert safe_divide(8, 0) == 0   # Bug: should be -1

test_exception_handling()
```

```bash
esbmc main.py --multi-property
```

```
Violated property:
  file main.py line 16 column 4 function test_exception_handling
  assertion
  return_value$_safe_divide$3 == 0

VERIFICATION FAILED
```

---

### Example 4: List Bounds Checking

```python
def access_list_element(index: int) -> int:
    my_list = [10, 20, 30, 40, 50]
    return my_list[index]

def test_bounds() -> None:
    assert access_list_element(2) == 30
    result = access_list_element(10)  # Out-of-bounds access
    assert result == 0

test_bounds()
```

```bash
esbmc main.py --multi-property
```

```
Violated property:
  file main.py line 3 column 4 function access_list_element
  array bounds violated: array `my_list' upper bound
  index < 5

VERIFICATION FAILED
```

---

### Example 5: Missing Return Statement Detection

```python
def calculate_grade(score: int) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    # Missing return for score < 60!

def safe_divide(a: int, b: int) -> int:
    if b == 0:
        return 0
    else:
        return a // b

score = 50
result = calculate_grade(score)
```

```bash
esbmc main.py --multi-property
```

```
Type checking warning:
main.py:1: error: Missing return statement

Violated property:
  Missing return statement detected in function 'calculate_grade'

VERIFICATION FAILED
```

> **Note**: Missing return detection only applies to functions with explicit non-void return type annotations. Constructors (`__init__`) are automatically excluded.
