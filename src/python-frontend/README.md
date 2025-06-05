# ESBMC Python Front-end

## Overview

The Python frontend handles the conversion of Python code into an internal representation, which is then translated into the GOTO language. This process includes three key steps:

1. Generating an Abstract Syntax Tree (AST) in JSON format.
2. Annotating the AST with type information.
3. Translating Python statements into a set of symbols in the Intermediate Representation (IRep) format.

The ESBMC backend finalizes the conversion by performing symbolic execution on the GOTO program, producing instructions in Single Static Assignment (SSA) form.
Following symbolic execution, we generate a first-order logic formula, which an SMT solver discharges. </br></br>

<p align="center">
  <img src="./images/arch.png" alt="ESBMC Architecture" width="65%" />
</p>

<p align="center"><em>Python Front-end Architecture</em></p>


## AST Generation

The translation of Python code starts by parsing .py files into an AST. This is achieved using the [ast](https://docs.python.org/3/library/ast.html) and [ast2json](https://pypi.org/project/ast2json/) modules, which generate the AST in JSON format. The process runs alongside the Python interpreter, producing a JSON file for each Python file, including imported modules.

This approach's main advantage is that it utilizes a native Python module, ensuring adherence to the language.

## Type Annotation

After generating the AST, we add JSON nodes with type information. [PEP 484](https://peps.python.org/pep-0484/) introduced an optional type system, allowing developers to annotate variables using the format **`var-name:type`**.

Our method involves traversing the AST and replacing assignments with their corresponding type-annotated nodes. The figure below shows the representation for <code>x:int = 10</code>.

```json
{
    "_type": "AnnAssign",
    "annotation": {
        "_type": "Name",
        "col_offset": 2,
        "ctx": {
            "_type": "Load"
        },
        "end_col_offset": 5,
        "end_lineno": 1,
        "id": "int",
        "lineno": 1
    },
    "target": {
        "_type": "Name",
        "col_offset": 0,
        "ctx": {
            "_type": "Store"
        },
        "end_col_offset": 1,
        "end_lineno": 1,
        "id": "x",
        "lineno": 1
    },
    "value": {
        "_type": "Constant",
        "col_offset": 8,
        "end_col_offset": 10,
        "end_lineno": 1,
        "kind": null,
        "lineno": 1,
        "n": 10,
        "s": 10,
        "value": 10
    }
}
```

We can infer type from constants, variables with inferred or pre-annotated types, binary expressions, and class instances.


## Symbol Table Generation
The final step in the frontend involves converting the annotated JSON AST into a symbol table using our C++ IRep API. This API enables the creation of a control-flow graph (CFG) from the program, allowing us to model constructs such as assignments, expressions, conditionals, loops, functions, and classes. The resulting information is stored in a context structure, which serves as the input for the GOTO conversion process.

## Features Supported by ESBMC-Python

Below is an overview of ESBMC-Python's key capabilities:

### Basic Constructs
- **Control Structures**: Supports conditional statements (`if-else`) and loops (`for-range`, `while`).
- **Arithmetic**: Includes standard arithmetic operations (e.g., addition, subtraction, multiplication, division).
- **Logical Operations**: Supports logical constructs (e.g., `AND`, `OR`, `NOT`).
- **Identity Comparisons**: Supports `is` and `is not` operators for identity-based comparisons, including `x is None`, `x is y`, or `x is not None`.

### Functions and Methods
- **Function Handling**: This allows for defining, calling, and verifying functions, including parameter passing and return values.
- **Annotations**: Supports type annotations.
### Object-Oriented Programming
- **Classes**: Supports class definitions, methods, and attributes.
- **Inheritance**: Handles inheritance and verifies scenarios involving inheritance issues.
### Data Types and Structures
- **Dynamic Typing**: Accommodates Python's dynamic typing in variable assignments.
- **Data Structures**: Supports operations on Python's built-in data structures, such as lists and strings, including concatenation and bounds checks.
- **Bytes and Integers**: Supports bytes and integer operations, such as conversions and bit length.
### Error Handling and Assertions
- **Assertions**: Supports `assert` statements for program verification.
- **Assumptions**: Supports `assume` statements for specifying assumptions for verification.

### Additional Capabilities
- **Nondeterministic Variables**: Models nondeterminism to explore multiple execution paths.
- **Recursion**: Supports and verifies recursive functions.
- **Imports**: Handles import styles and validates their usage.
- **Numeric Types**: Supports manipulation of numeric types (e.g., bytes, integers, floats).
- **Built-in Functions**: Supports Python's built-in functions, such as `abs`, `int`, `float`, `chr`, `str`, `hex`, `oct`, `len`, and `range`.
- **Verification properties**: Division-by-zero, indexing errors, arithmetic overflow, and user-defined assertions.

### Limitations

The current version of ESBMC-Python has the following limitations:

- Only `for` loops using the `range()` function are supported.
- List and String support are partial and limited in functionality.
- Dictionaries are not supported at all.

### Example: Division by Zero in Python

The following Python program executes without issues in standard Python 3. However, when analyzed using ESBMC, it reveals a hidden bug: a possible division by zero.

```python
import random as rand

def div1(cond: int, x: int) -> int:
    if (not cond):
        return 42 // x
    else:
       return x // 10

cond:int = rand.random()
x:int = rand.random()

assert div1(cond, x) != 1
```

**Command:**

```bash
$ esbmc main.py
```

**ESBMC Output:**

```
[Counterexample]


State 1 file main.py line 12 column 8 function random thread 0
----------------------------------------------------
  value = 2.619487e-10 (00111101 11110010 00000000 01000000 00000010 00000000 00010000 00001000)

State 3 file main.py line 12 column 8 function random thread 0
----------------------------------------------------
  value = 3.454678e-77 (00110000 00010000 00000000 01000000 00000010 00000000 00010000 00000000)

State 5 file main.py line 5 column 8 function div1 thread 0
----------------------------------------------------
Violated property:
  file main.py line 5 column 8 function div1
  division by zero
  x != 0


VERIFICATION FAILED
```

ESBMC successfully identifies a path where the randomly generated variable x evaluates to zero (or very close to zero, causing integer division by zero). This triggers a property violation, and ESBMC generates a counterexample showing the precise values of `x` and `cond` that lead to the failure. An executable test case can be created from this counterexample to expose this implementation error as follows:

````python
def div1(cond: int, x: int) -> int:
    if not cond:
        return 42 // x
    else:
        return x // 10

# Constructing values that become 0 when cast to int
cond = int(2.619487e-10)  # → 0
x = int(3.454678e-77)     # → 0

print(f"cond: {cond}, x: {x}")
print(div1(cond, x))  # Triggers division by zero
````

```bash
$ python3 main.py
```

````
cond: 0, x: 0
Traceback (most recent call last):
  File "/home/lucas/examples/site/div-test.py", line 12, in <module>
    print(div1(cond, x))  # Triggers division by zero
  File "/home/lucas/examples/site/div-test.py", line 3, in div1
    return 42 // x
ZeroDivisionError: integer division or modulo by zero
````

This example highlights how symbolic model checking can uncover subtle bugs that may not be triggered during regular testing.

# Numpy Formal Verification with ESBMC

## What We Are Trying to Verify

### Targeted Numpy Features

This verification focuses on common numerical operations provided by Numpy, particularly:

- N-dimensional array computations  
- Broadcasting behavior  
- Mathematical functions (e.g., `np.add`, `np.multiply`, `np.power`)  
- Precision-sensitive operations (e.g., `np.exp`, `np.sin`, `np.arccos`)  

### Why It Matters

While Python and Numpy silently handle overflows or undefined behavior at runtime, model checkers such as **ESBMC** can expose hidden issues that go undetected during normal test execution.

As highlighted by **Harzevili et al., 2023**, common issues in ML-related libraries include:

- Integer overflows and underflows  
- Division by zero  
- Precision errors due to rounding or limited bit-width  
- Out-of-bounds access in arrays  

## Verifying Numpy Programs with ESBMC

### Black-Box Verification with ESBMC

This approach treats Numpy as a black box by analyzing **assertions written by the developer**.

#### Example: Detecting Integer Overflow

```python
import numpy as np

x = np.add(2147483647, 1, dtype=np.int32)
```

**Python3 Runtime Output:**

No error — NumPy silently wraps on overflow for fixed-width dtypes (like int32).

**ESBMC Output:**

```
[Counterexample]

State 1 file main.py line 3 column 0 thread 0
----------------------------------------------------
Violated property:
  file main.py line 3 column 0
  arithmetic overflow on add
  !overflow("+", 2147483647, 1)


VERIFICATION FAILED
```

An executable test case can be created from this counterexample to expose this implementation error as follows:

````python
import numpy as np

x = np.add(2147483647, 1, dtype=np.int32)

print("Result:", x)         # Expected: -2147483648 due to overflow
print("Type:", type(x))     # <class 'numpy.int32'>
print("Correctly overflowed:", x == -2147483648)

# Optional assertion to expose unexpected behavior
assert x == -2147483648, "Overflow did not wrap around correctly"
````

```bash
$ python3 main.py
```

````
Result: -2147483648
Type: <class 'numpy.int32'>
Correctly overflowed: True
````

**Explanation:**  

ESBMC performs bit-precise analysis and treats signed overflow as undefined or erroneous, unlike NumPy’s permissive semantics.

- np.int32 represents 32-bit signed integers: range is −2,147,483,648 to 2,147,483,647.
- The expression 2147483647 + 1 equals 2147483648, which exceeds the upper bound.
- In np.int32, this overflows and wraps around to −2,147,483,648.

While NumPy permits this silent overflow, ESBMC correctly identifies it as a violation of safe arithmetic.

#### Matrix Determinant (`np.linalg.det`)

You can also verify the correctness of determinant computations for 2D NumPy arrays:

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
x = np.linalg.det(a)
assert x == -2
````

ESBMC symbolically executes the closed-form expression for small matrices, enabling the detection of singular matrices, ill-conditioned operations, or incorrect expectations.


### White-Box Verification

For deeper analysis, symbolically execute individual functions using **non-determinism** to verify all possible input paths.

#### Example:

```python
def integer_squareroot(n: uint64) -> uint64:
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
```

**Command:**

```bash
$ esbmc main.py --function integer_squareroot --incremental-bmc
```

**ESBMC Output:**

```
[Counterexample]


State 1 file square.py line 2 column 4 function integer_squareroot thread 0
----------------------------------------------------
  x = 0xFFFFFFFFFFFFFFFF (11111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111)

State 2 file square.py line 3 column 4 function integer_squareroot thread 0
----------------------------------------------------
  y = 0 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000)

State 3 file square.py line 5 column 8 function integer_squareroot thread 0
----------------------------------------------------
  x = 0 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000)

State 4 file square.py line 6 column 8 function integer_squareroot thread 0
----------------------------------------------------
Violated property:
  file square.py line 6 column 8 function integer_squareroot
  division by zero
  x != 0


VERIFICATION FAILED

Bug found (k = 1)
```

**Explanation:**  
This highlights a potential bug: `n // x` is unsafe if `x == 0`.

---

## Why ESBMC Matters for Numpy

| Feature                  | Python Behavior        | ESBMC Behavior                  |
|--------------------------|------------------------|----------------------------------|
| Integer overflow         | Silently wraps         | Detects and reports violations   |
| Float precision loss     | Tolerated silently     | Symbolically tracked             |
| Division by zero         | Raises at runtime      | Verified statically              |
| Unsafe dtype conversions | May truncate silently  | Triggers verification errors     |

## ESBMC – NumPy Math Library Mapping

Here, we document the mapping between ESBMC's math library implementations and their NumPy equivalents. 

These mappings help test and verify floating-point behavior consistently across C and Python environments.

Reference: https://numpy.org/doc/stable/reference/routines.math.html

### Mathematical & Trigonometric Functions

| ESBMC File | NumPy Equivalent       | Category      |
|------------|------------------------|---------------|
| `acos.c`   | `np.arccos`, `np.acos` | Inverse trig  |
| `atan.c`   | `np.arctan`, `np.atan` | Inverse trig  |
| `cos.c`    | `np.cos`               | Trig          |
| `sin.c`    | `np.sin`               | Trig          |

### Rounding & Remainders

| ESBMC File     | NumPy Equivalent              | Category             |
|----------------|-------------------------------|----------------------|
| `ceil.c`       | `np.ceil`                     | Rounding             |
| `floor.c`      | `np.floor`                    | Rounding             |
| `round.c`      | `np.round`, `np.around`       | Rounding             |
| `rint.c`       | `np.rint`                     | Rounding             |
| `trunc.c`      | `np.trunc`, `np.fix`          | Rounding             |
| `fmod.c`       | `np.fmod`                     | Remainder            |
| `remainder.c`  | `np.remainder`                | Remainder            |
| `remquo.c`     | `divmod` + sign logic         | Remainder + Quotient |

### Floating Point Properties

| ESBMC File    | NumPy Equivalent                    | Category             |
|---------------|-------------------------------------|----------------------|
| `copysign.c`  | `np.copysign`                       | Floating point ops   |
| `frexp.c`     | `np.frexp`                          | Float decomposition  |
| `modf.c`      | `np.modf`                           | Float decomposition  |
| `fpclassify.c`| `np.isnan`, `np.isinf`, `np.isfinite`| Classification       |

### Comparisons, Extrema

| ESBMC File | NumPy Equivalent                    | Category             |
|------------|-------------------------------------|----------------------|
| `fmin.c`   | `np.fmin`                           | Min function         |
| `fmax.c`   | `np.fmax`                           | Max function         |
| `fdim.c`   | `np.maximum(x - y, 0)` (approx.)    | Difference           |

### Exponents and Powers

| ESBMC File | NumPy Equivalent | Category     |
|------------|------------------|--------------|
| `exp.c`    | `np.exp`         | Exponential  |
| `pow.c`    | `np.power`       | Power        |

### Miscellaneous

| ESBMC File     | NumPy Equivalent         | Category              |
|----------------|--------------------------|-----------------------|
| `fabs.c`       | `np.fabs`, `np.absolute` | Absolute value        |
| `sqrt.c`       | `np.sqrt`                | Square root           |
| `nextafter.c`  | `np.nextafter`           | Floating-point step   |


## References

For more information about our frontend, please refer to our ISSTA 2024 [tool paper](https://dl.acm.org/doi/abs/10.1145/3650212.3685304).

Harzevili et al. (2023).  
*Characterizing and Understanding Software Security Vulnerabilities in Machine Learning Libraries.*  
[arXiv:2303.06502](https://arxiv.org/abs/2303.06502)

*NumPy Mathematical functions*
[Documentation](https://numpy.org/doc/stable/reference/routines.math.html)

---
