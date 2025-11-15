# ESBMC Python Front-end

## Overview

The Python frontend converts Python code into an internal representation, which is then translated into the GOTO language. This process includes three key steps:

1. Generating an Abstract Syntax Tree (AST) in JSON format.
2. Annotating the AST with type information.
3. Translating Python statements into a set of symbols in the Intermediate Representation (IRep) format of ESBMC.

The ESBMC backend finalizes the conversion by performing symbolic execution on the GOTO program, producing instructions in Single Static Assignment (SSA) form.
Following symbolic execution, we generate a subset of first-order logical formulas, which an SMT solver discharges. </br></br>

<p align="center">
  <img src="./images/arch.png" alt="ESBMC Architecture" width="65%" />
</p>

<p align="center"><em>Python Front-end Architecture</em></p>


## AST Generation

Python code translation starts by parsing `.py` files into an AST. This is achieved using the [ast](https://docs.python.org/3/library/ast.html) and [ast2json](https://pypi.org/project/ast2json/) modules, which convert the AST to JSON. The process runs alongside the Python interpreter, producing a JSON file for each Python file, including imported modules.

This approach's main advantage is that it utilizes a native Python module, ensuring adherence to the language.

## Type Annotation

After generating the AST, we add JSON nodes with type information. [PEP 484](https://peps.python.org/pep-0484/) introduced an optional type system, allowing developers to annotate variables using the format **`var-name:type`**.

Our method traverses the AST and replaces assignments with their corresponding type-annotated nodes. The figure below shows the representation for <code>x:int = 10</code>.

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
The final step in the frontend is to convert the annotated JSON AST into a symbol table using our C++ IRep API. This API enables the creation of a control-flow graph (CFG) from the program, allowing us to model constructs such as assignments, expressions, conditionals, loops, functions, and classes. The resulting information is stored in a context structure, which serves as the input for the GOTO conversion process.

## Features Supported by ESBMC-Python

Below is an overview of ESBMC-Python's key capabilities:

### Basic Constructs
- **Control Structures**: Supports conditional statements (`if-else`) and loops (`for-range`, `while`).
- **Arithmetic**: Includes standard arithmetic operations (e.g., addition, subtraction, multiplication, division).
- **Logical Operations**: Supports logical constructs (e.g., `AND`, `OR`, `NOT`).
- **Identity Comparisons**: Supports `is` and `is not` operators for identity-based comparisons, including `x is None`, `x is y`, or `x is not None`.
  - **None Type Handling**: Comprehensive support for Python's None value with proper type distinction from other types (`int`, `bool`, `str`, etc.).
  - **None Comparisons**: Correctly evaluates comparisons and identity checks involving None (e.g., `x is None`, `x == None`, `x != 0`).
  - **None in Logical Expressions**: Properly handles `None` in boolean contexts where it is falsy (e.g., `None and True` returns `None`, `None or 1` returns `1`).
- **Global Variables:** Recognizes the `global` keyword for accessing and modifying variables in the global scope from within functions.

### String Formatting and Literals

- **F-String Support**: Comprehensive support for Python's f-string (formatted string literal) syntax, including:
  - **Basic Variable Interpolation**: f"Hello {name}!" with support for multiple variables in a single f-string
  - **Built-in Variable Access**: Supports built-in variables such as __name__ within f-strings: f"Running as: {__name__}"
  - **Format Specifications**:
    - **Integer formatting**: f"{num:d}" and f"{num:i}"
    - **Float formatting with precision**: f"{val:.2f}", f"{price:.1f}"
    - **Mixed format specifications**: f"Items: {count:d}, Price: {price:.1f}"
  - **Boolean Formatting**: Automatic conversion of boolean values to strings (True/False)
  - **Empty and Literal F-strings**: Support for f"" (empty) and f"Just a string" (literal-only)
  - **String Concatenation**: F-string results can be concatenated with other strings
  - **IEEE 754 Compliance**: Proper handling of 32-bit and 64-bit floating point numbers with accurate string conversion

### Functions and Methods
- **Function Handling**: This allows for defining, calling, and verifying functions, including parameter passing and return values.
- **Variadic Parameters**: Supports the `*args` syntax for functions with variable-length argument lists, allowing functions to accept an arbitrary number of positional arguments (e.g., `def func(*args):`).
- **Annotations**: Supports type annotations, including:
  - **Basic Type Annotations**: Standard Python types (int, float, bool, str, etc.).
  - **Any Type**: Supports `Any` from the typing module for functions with dynamic return types.
    - **Automatic Type Inference**: When `Any` is used as a return type, ESBMC automatically infers the actual type by analyzing return statements in the function body.
    - **Type Hierarchy**: Uses the same type widening hierarchy as Union types (`float > int > bool`).
    - **Supported Return Types**: `Any` type functions can return `int`, `float`, `bool`, or expressions (`BinOp`, `UnaryOp`) that evaluate to these types.
    - **Variable Type Inference**: Variables annotated with `Any` that are assigned from function calls inherit the function's inferred return type.
  - **Union Types**: Supports both `Union[Type1, Type2, ...]` from the typing module and PEP 604 syntax (`Type1 | Type2`) for functions that can return multiple types.
    - **Union Syntax Support**: Both `Union[int, bool]` and `int | bool` syntaxes are supported.
    - **Chained Unions**: Supports chained union types with multiple members (e.g., `int | bool | float`).
    - **Type Widening**: Automatically selects the widest type from Union members based on type hierarchy (`float > int > bool`).
    - **Return Type Inference**: When Union types are used, ESBMC analyzes return statements to determine the appropriate type.
    - **Mixed Type Verification**: Enables verification of functions that conditionally return different types based on runtime conditions.
- **Lambda Expressions**: Supports basic lambda expressions that are converted to regular functions and stored as function pointers. Lambda functions can be assigned to variables and called indirectly. Supports single-expression lambdas with multiple parameters.

### Object-Oriented Programming
- **Classes**: Supports class definitions, methods, and attributes.
- **Class Attributes**: Supports class-level attributes (variables shared across all instances):
  - **Automatic Type Inference**: Unannotated class attributes are automatically type-inferred from their assigned values (e.g., species = "Homo sapiens" is inferred as str).
  - **Explicit Type Annotations**: Also supports explicit type annotations for class attributes when needed.
  - **Access Patterns**: Supports both instance-based and class-based attribute access (e.g., `instance.attr` and `ClassName.attr`).
- **Instance Variables**: Supports instance-specific attributes defined in `__init__` methods.
- **Inheritance**: Handles inheritance and verifies scenarios involving inheritance issues.
- **super() calls**: Supports the `super()` function to call methods from a superclass. This allows verifying behaviors in which a derived class explicitly invokes base-class methods, enabling the analysis of polymorphic behavior and the proper propagation of assertions or side effects.

### Data Types and Structures
- **Dynamic Typing**: Accommodates Python's dynamic typing in variable assignments.
- **Data Structures**: Supports operations on Python's built-in data structures, including lists, strings, and tuples, with features such as concatenation and bounds checks.
  - **List Operations**:
    - **append()**: Add elements to the end of a list.
    - **extend()**: Extends a list by appending all elements from an iterable (e.g., `list1.extend(list2)` or `list1.extend([3, 4, 5])`).
    - **insert()**: Insert elements at a specific index position.
      - When the index equals the list length, the element is appended to the end.
      - When the index exceeds the list length, the element is appended to the end.
      - When the index is within bounds, existing elements are shifted right.
      - Supports insertion into empty lists at index 0.
    - **Membership Testing (`in` operator)**: Supports Python's `in` operator for list membership testing (e.g., `2 in [1, 2, 3]` returns `True`).
    - **Concatenation (+ operator)**: Fully supports the list + list operation (e.g., `[1,2] + [3,4] → [1,2,3,4]`), producing a new list containing all elements of both operands in order.
  - **String Operations**:
    - **Membership Testing (in operator)**: Supports Python's `in` operator for substring testing (e.g., `"o" in "foo"` returns `True`).
    - **startswith() method**: Supports prefix checking for strings (e.g., `"foo".startswith("f")` returns True).
    - **endswith() method**: Supports suffix checking for strings (e.g., `"foo".endswith("oo")` returns True).
    - **lstrip() method**: Removes leading whitespace characters from strings (e.g., `"  hello".lstrip()` returns `"hello"`).
    - **isspace() method**: Returns `True` if all characters in the string are whitespace characters and the string is non-empty, `False` otherwise.
    - **String Slicing**: Comprehensive support for Python's slice notation on strings:
      - **Basic Slicing**: `string[start:end]` returns a substring from index `start` to `end-1`.
      - **Omitted Bounds**: Supports slices with omitted start (`string[:end]`) or end (`string[start:]`) indices.
      - **Negative Indices**: Full support for negative indexing (e.g., `string[-3:]` returns the last three characters).
      - **Empty Slices**: Correctly handles edge cases such as `string[0:0]` (returns empty string).
  - **Tuple Support**: Support for Python's tuple data structure:
    - **Tuple Literals**: Supports tuple creation using parentheses syntax (e.g., `(1, 2, 3)`, `(42, "hello", 3.14)`).
    - **Empty Tuples**: Supports empty tuple literals `()`.
    - **Single Element Tuples**: Handles single-element tuples with trailing comma (e.g., `(5,)`).
    - **Mixed Type Tuples**: Supports tuples containing elements of different types (e.g., integers, strings, floats, booleans).
    - **Nested Tuples**: Support for tuples containing other tuples (e.g., `((1, 2), (3, 4))`).
    - **Tuple Indexing**: Supports accessing tuple elements using subscript notation with constant indices (e.g., `t[0]`, `t[2]`).
      - Includes bounds checking to detect out-of-range access.
    - **Tuple Type Annotations**: Supports both generic tuple annotations and parameterized tuple type hints:
      - **Generic Annotation**: `t: tuple = (1, 2, 3)` - type inferred from value.
      - **Parameterized Annotation**: `def foo() -> tuple[int, int]`: - explicitly specifies element types.
    - **Tuple Equality**: Supports equality comparisons between tuples (e.g., `t1 == (1, 2, 3)`).
    - **len() Function**: The built-in `len()` function works with tuples to return the number of elements.
    - **isinstance() Type Checking**: Supports runtime type checking with `isinstance(obj, tuple)`.
- **Bytes and Integers**: Supports byte and integer operations, such as conversions and bit length.

### Error Handling and Assertions
- **Assertions**: Supports `assert` statements for program verification.
- **Assumptions**: Supports `assume` statements for specifying assumptions for verification.

### Cover Properties and Reachability Analysis
- **Cover Properties**: Supports `__ESBMC_cover()` function for checking if conditions are satisfiable at specific program points.
  - **Reachability Semantics**: Cover properties use inverted assertion semantics, where `cover(cond)` behaves as `assert(!cond)`:
    - **Counterexample (failed assertion)**: condition is satisfiable (reachable).
    - **Proof (passed assertion)**: condition is unsatisfiable (unreachable).
  - **Use Cases**:
    - **Path Reachability**: Verify that specific code branches can be executed.
    - **Value Coverage**: Check if variables can achieve certain values or ranges.
    - **Dead Code Detection**: Identify unreachable code paths.
    - **Test Adequacy**: Ensure test scenarios exercise interesting program states.
  - **Multi-Property Mode**: Cover properties work best with the `--multi-property` flag, allowing ESBMC to continue verification after satisfiable covers are found.

### Module System and Built-in Variables

- **Module Imports**: Handles import styles and validates their usage.
- **name Variable**: Supports Python's built-in __name__ variable that contains the name of the current module:
  - Set to "__main__" when the module is run directly as the main program.
  - Set to the module name when the module is imported.
  - Enables verification of the common Python idiom if __name__ == "__main__":
  - Supports proper distinction between main module execution and imported module behavior.

### Additional Capabilities
- **Nondeterministic Variables**: Models nondeterminism to explore multiple execution paths.
- **Recursion**: Supports and verifies recursive functions.
- **Imports**: Handles import styles and validates their usage.
- **Numeric Types**: Supports manipulation of numeric types (e.g., bytes, integers, floats).
- **Built-in Functions**: 
  - **Arithmetic and conversions**: Supports Python's built-in functions, such as `abs`, `divmod`, `int`, `float`, `chr`, `str`, `hex`, `oct`, `len`, and `range`.
  - **Enhanced float() constructor**: Supports conversion from strings including special values such as `nan`, `inf`, `-inf`, `infinity`, and `+infinity` (case-insensitive with whitespace handling).
  - **Min/Max**: Supports `min(a, b)` and `max(a, b)` with type promotion (int-to-float). Currently limited to two arguments.
  - **any()**: Supports Python's `any()` built-in function with the following behavior:
    - **List Literals Only**: Currently supports `any()` only with list literals as arguments (e.g., `any([x, True, 0]`)).
    - **Truthiness Evaluation**: Correctly evaluates truthiness according to Python semantics:
      - `None` is always falsy.
      - `bool` values are used directly.
      - Integers and floats are truthy if non-zero.
      - Pointers are truthy if not NULL.
    - **Short-Circuit OR Logic**: Returns `True` if any element in the list is truthy, `False` if all elements are falsy or the list is empty.
    - **Type Handling**: Handles mixed-type lists with support for nested containers and complex structures containing `None`, integers, floats, and booleans.
  - **Input**: Models `input()` as a non-deterministic string of up to 256 characters. This enables verifying programs that rely on user input.
  - **Print**: Supports `print()` statements for output. All arguments are evaluated to ensure proper side-effect handling during verification, though the actual output is not produced.
  - **Enumerate**: Supports `enumerate(iterable, start=0)` for iterating over sequences with automatic indexing. Handles both tuple unpacking `(for i, x in enumerate(...))` and single variable assignment `(for item in enumerate(...))`. Supports an optional `start` parameter and works with lists, strings, and other iterables.
- **Verification properties**: Division-by-zero, indexing errors, arithmetic overflow, and user-defined assertions.

### Math Module Support
- **math.comb(n, k)**: Calculates binomial coefficients `C(n, k) = n! / (k! * (n-k)!)`.
  - Supports verification of combinatorial properties such as symmetry: `C(n, k) = C(n, n-k)`.
  - Includes built-in type checking and input validation (assertion failures for negative inputs or non-integer types).
- **math.floor(x)**: Returns the largest integer less than or equal to x.
- **math.ceil(x)**: Returns the smallest integer greater than or equal to x.
  - Both functions `math.floor(x)` and `math.ceil(x)` include built-in assertions to reject infinity and NaN inputs.
  - Supports verification of edge cases, including very small values, large values (e.g., 1e12), and boundary conditions.

### Regular Expression (re) Module Support
ESBMC-Python provides operational modeling and verification capabilities for Python's re module, enabling pattern-matching verification:
- **Pattern Matching Functions**:
  - **re.match(pattern, string)**: Attempts to match a pattern at the beginning of a string. Returns a match object (truthy) on success or None (falsy) on failure.
  - **re.search(pattern, string)**: Searches for a pattern anywhere within a string. Returns a match object (truthy) if found or None (falsy) if not found.
  - **re.fullmatch(pattern, string)**: Matches a pattern against the entire string. Returns a match object (truthy) if the whole string matches or None (falsy) otherwise.
- **Supported Pattern Features**:
  - **Universal match**: .* pattern (matches any string).
  - **Empty patterns**: Empty string patterns.
  - **Literal strings**: Patterns without metacharacters (e.g., "abc", "hello").
  - **Character class ranges**: [a-z]+, [A-Z]+, [0-9]* patterns with + and * quantifiers
  - **Digit sequences**: \d+ and \d* patterns (with raw string literals like r"\d+")
  - **Alternation**: (x|y)z* patterns
  - **Prefix matching with wildcard**: Patterns ending with .* (e.g., "a.*")
- **Type Validation**: Built-in runtime type checking ensures both pattern and string arguments are string or bytes-like objects. Invalid types raise TypeError exceptions that can be verified.
- **Verification Approach**: Uses operational models that combine:
  - Direct pattern recognition for supported regex constructs.
  - Literal string matching for patterns without metacharacters.
  - Nondeterministic behavior modeling for complex patterns, allowing ESBMC to explore both match and non-match scenarios.
- **Match Object Handling**: Match results can be tested for truthiness (e.g., if re.match(...)) or compared with None using identity operators (e.g., re.search(...) is not None)

### Random Module Support
ESBMC-Python supports modeling and verification of the random module functions using nondeterministic values with appropriate constraints:
- **random.random()**: Returns a nondeterministic floating-point number in the range [0.0, 1.0).
- **random.uniform(a, b)**: Returns a nondeterministic floating-point number N such that:
  - If a ≤ b: a ≤ N ≤ b
  - If a > b: b ≤ N ≤ a
- **random.getrandbits(k)**: Returns a nondeterministic integer with k random bits.
- **random.randrange(start, stop=None, step=1)**: Returns a randomly selected integer from the specified range.
These random functions use nondeterministic modeling with appropriate constraints (`__ESBMC_assume`), allowing ESBMC to explore all possible values within the specified ranges during verification. This enables thorough testing of code that depends on random values.

### OS Module Support
ESBMC-Python provides modeling and verification capabilities for Python's os module, enabling verification of file system operations with nondeterministic behavior modeling:
- **Path Operations**:
  - **os.path.exists(path)**: Checks if a path exists (file or directory).
  - **os.path.basename(path)**: Returns the base name of a pathname.
- **Directory Operations**:
  - **os.makedirs(path, exist_ok=False)**: Creates a directory and any necessary parent directories.
    - Supports the `exist_ok` parameter to control behavior when the directory already exists.
  - **os.mkdir(path)**: Creates a single directory.
    - Models nondeterministic behavior: may raise `FileExistsError` if the directory already exists.
    - Enables verification of error handling in directory creation scenarios.
  - **os.rmdir(path)**: Removes an empty directory.
    - Models nondeterministic behavior: may raise `OSError` if the directory is not empty.
    - Enables verification of proper cleanup and error handling in directory removal operations.
  - **os.listdir(path)**: Lists directory contents.
- **File Operations**:
  - **os.remove(path)**: Removes a file.
    - Models nondeterministic behavior: may raise `FileNotFoundError` if the file does not exist.
    - Enables verification of error handling in file deletion scenarios.
  - **os.popen(cmd)**: Opens a pipe to or from a command (modeled for verification).
- **Nondeterministic Modeling**:
  - The `os` module functions use nondeterministic modeling to explore different execution paths during verification:
  - File and directory existence is modeled nondeterministically, allowing ESBMC to verify both success and failure scenarios.
  - This enables thorough verification of exception handling code that deals with file system operations.

### Special Value Detection:
- **math.isnan(x)**: Returns True if x is NaN (Not a Number).
- **math.isinf(x)**: Returns True if x is positive or negative infinity.
- Both functions use ESBMC's internal operations for accurate verification according to the IEEE-754 standard.

### Exception Handling

- **Try-Except Blocks**: Supports comprehensive exception handling with try-except syntax for controlling program flow and verifying error conditions.
- **Multiple Exception Handlers**: Supports multiple except clauses to handle different exception types.
- **Exception Catching**: Supports catching exceptions with variable binding using except ExceptionType as variable syntax.
  - **Improved Variable Scope Handling**: Exception variables are declared and scoped within their catch blocks, ensuring correct symbol table management during verification.
- **Exception Hierarchy**: Implements Python's exception hierarchy where all exceptions inherit from BaseException.
- **Built-in Exception Classes**:
  - **BaseException**: Base class for all exceptions.
  - **Exception**: Base class for all built-in exceptions (inherits from BaseException).
  - **AssertionError**: Raised when an assert statement fails.
  - **ValueError**: Raised for inappropriate argument values.
  - **TypeError**: Raised for inappropriate argument types.
  - **IndexError**: Raised for sequence index out of range.
  - **KeyError**: Raised for missing dictionary keys.
  - **ZeroDivisionError**: Raised for division by zero operations.
  - **OSError**: Base class for I/O related errors.
  - **FileNotFoundError**: Raised when a file or directory is not found (inherits from OSError).
  - **FileExistsError**: Raised when trying to create a file or directory that already exists (inherits from OSError).
  - **PermissionError**: Raised when an operation lacks the necessary permissions (inherits from OSError).
- **Exception Objects**: Exception instances contain message attributes and support string representation via __str__() method.
- **Exception Raising**: Supports raise statements with exception instantiation and custom error messages.

### Strict Type Checking
ESBMC-Python provides an optional strict type-checking mode that enforces type compatibility for function arguments at verification time. 
- **Command-Line Option**:
  - **--strict-types**: Enables strict type checking for function arguments during verification. When this flag is provided, ESBMC validates that all arguments passed to functions match their declared parameter types.
- **Behavior**:
  - **Type Validation**: Checks that each argument's type matches the corresponding parameter's type annotation.
  - **Instance Methods**: Automatically accounts for the implicit self parameter in instance methods.
  - **Class Methods**: Automatically accounts for the implicit cls parameter in class methods.
  - **Static Methods**: Validates all parameters without implicit parameter handling.
  - **Type Errors**: When a type mismatch is detected, it generates a `TypeError` exception with a descriptive message indicating which argument has an incompatible type.

### Code Quality and Control Flow Analysis

- **Missing Return Statement Detection**: ESBMC-Python performs static analysis to detect functions with missing return statements:
  - **Type-Aware Analysis**: Only flags functions with non-void return type annotations that lack proper return statements.
  - **Control Flow Analysis**: Analyzes all execution paths through conditional statements (if-else structures) to ensure comprehensive return coverage.
  - **Constructor Exception**: Automatically excludes class constructors (__init__ methods) from missing return analysis since they don't require explicit return statements.
  - **Descriptive Error Messages**: Provides clear error messages indicating which function has missing return paths.
  - **Verification Integration**: Missing return statements are detected as verification failures, enabling early detection of potential runtime errors.

### Limitations

The current version of ESBMC-Python has the following limitations:

- Only `for` loops using the `range()` function are supported.
- List and String support are partial and limited in functionality. Currently supported list methods include `append()` and `insert()`.
- String slicing does not support step values (e.g., string[::2] for every second character is not supported).
- Dictionaries are not supported at all.
- `min()` and `max()` currently support only two arguments and do not handle iterables or the key/default parameters.
- `any()` currently supports only list literals as arguments and does not support other iterable types.
- `input()` is modeled as a nondeterministic string with a maximum length of 256 characters (under-approximation).
- `print()` evaluates all arguments for side effects but does not produce actual output during verification.
- `enumerate()` supports standard usage patterns but may have limitations with complex nested iterables or advanced parameter combinations.
- Exception handling supports the core built-in exception types but may not cover all Python standard library exceptions or custom exception hierarchies with complex inheritance patterns.
- Built-in variables support is limited to __name__; other Python built-ins such as __file__, __doc__, __package__ are not yet supported.
- Lambda expressions have the following limitations:
  - Return type inference is currently naive (defaults to double type)
  - Higher-order and nested lambda expressions are not supported
  - Parameter types are assumed to be double for simplicity
- F-String Limitations:
  - Complex expressions within f-strings may have limited support
  - Advanced format specifications beyond basic integer `(:d, :i)` and float `(:.Nf)` formatting may not be fully supported
  - Nested f-strings are not supported
  - String alignment and padding format specifications (e.g., `:>10`, `:<5`) are not supported
  - Custom format specifications for user-defined types are not supported
- Missing Return Statement Detection Limitations:
  - Does not analyze return statements inside lambda expressions within the main function body.
- Random Module Limitations:
  - `random.randrange()` with a single argument (e.g., randrange(10)) is not supported.
  - Other random module functions (e.g., choice, shuffle, sample, seed) are not yet supported.
- Class Attribute Limitations:
  - Type inference for class attributes requires values that have clear, determinable types. Complex expressions may require explicit type annotations.
- Regular Expression (re) Module Limitations:
  - Only `re.match()`, `re.search()`, and `re.fullmatch()` are supported.
  - Match objects do not expose group capture methods (e.g., .group(), .groups(), .span()). Match results are only usable for Boolean/None testing.
  - Limited pattern syntax support compared to full Python regex. Complex patterns beyond the explicitly supported constructs may exhibit nondeterministic behavior.
  - Advanced regex features are not supported: lookahead/lookbehind assertions, backreferences, named groups, conditional patterns, and Unicode property escapes.
  - Union Type Limitations:
    - Union types are resolved to the widest type among their members (`float > int > bool`) at verification time rather than maintaining true union semantics.
    - Union types containing types beyond basic primitives (`int, float, bool`) may default to pointer types.
    - Type narrowing based on runtime type checks within Union-typed functions is not explicitly tracked.
- Any Type Limitations:
  - Any type inference only supports return values of primitive types: `int`, `float`, `bool`, and expressions that evaluate to these types.
  - String return values are explicitly not supported and will cause a verification error with the message "Unsupported return type 'string' detected".
  - Other types (`objects`, `arrays`, `null`) are not supported as return values for Any-typed functions.
  - Type inference defaults to `double (float)` when no specific type can be determined from return statements.
- Tuple Limitations:
  - Tuple indexing requires constant indices; variable indices are not supported (e.g., `t[i]` where `i` is a variable will fail).
  - Tuple iteration (e.g., `for item in my_tuple:`) is not yet supported.
  - Tuple methods such as `.count()` and `.index()` are not yet supported.
  - Tuple concatenation with `+` operator is not yet supported.
  - Tuple repetition with `*` operator is not yet supported.
  - Tuple slicing is not yet supported.

### Example 1: Division by Zero in Python

The following Python program executes without issues in standard Python 3. However, when analyzed with ESBMC, it reveals a hidden bug: a possible division-by-zero.

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

ESBMC successfully identifies a path in which the randomly generated variable x evaluates to zero (or very close to zero), leading to an integer division by zero. This triggers a property violation, and ESBMC generates a counterexample showing the precise values of `x` and `cond` that lead to the failure. An executable test case can be created from this counterexample to expose this implementation error as follows:

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

This example illustrates how symbolic model checking can reveal subtle bugs that may not be detected during regular testing.

### Example 2: Lambda Expression Verification

This example demonstrates ESBMC-Python's support for lambda expressions:

````Python
def test_lambda_expressions():
    # Basic arithmetic lambda
    add_ten = lambda x: x + 10
    result1:int = add_ten(5)
    assert result1 == 15

    # Multi-parameter lambda
    calculate_volume = lambda length, width, height: length * width * height
    volume:float = calculate_volume(2.0, 3.0, 4.0)
    assert volume == 24.0

    # Lambda with conditional logic
    absolute_diff = lambda a, b: a - b if a > b else b - a
    diff1:int = absolute_diff(10, 3)
    diff2:int = absolute_diff(3, 10)
    assert diff1 == 7
    assert diff2 == 7

    # Lambda for boolean operations
    is_in_range:bool = lambda x, lower, upper: lower <= x <= upper
    assert is_in_range(5, 1, 10) == True
    assert is_in_range(15, 1, 10) == False

test_lambda_expressions()
````

**Command:**

```bash
$ esbmc main.py
```

**ESBMC Output:**

```
Parsing main.py
Converting
Generating GOTO Program
GOTO program creation time: 0.637s
GOTO program processing time: 0.015s
Starting Bounded Model Checking
Symex completed in: 0.007s (36 assignments)
Slicing time: 0.002s (removed 26 assignments)
Generated 12 VCC(s), 6 remaining after simplification (10 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.005s
Solving with solver Boolector 3.2.2
Runtime decision procedure: 0.000s
BMC program time: 0.014s

VERIFICATION SUCCESSFUL
````

### Example 3: Exception Handling Verification

This example shows how ESBMC can verify exception handling behavior:

````python
def safe_divide(a: int, b: int) -> int:
    try:
        result = a // b
        return result
    except ZeroDivisionError as e:
        return -1

def test_exception_handling() -> None:
    # Normal case
    assert safe_divide(10, 2) == 5
    
    # Division by zero case
    assert safe_divide(10, 0) == -1
    
    # This assertion will fail - demonstrating ESBMC can verify exception paths
    assert safe_divide(8, 0) == 0  # Should be -1, not 0

test_exception_handling()
````

**Command:**

```bash
$ esbmc main.py --multi-property
```

**ESBMC Output:**

```
[Counterexample]


State 1 file main.py line 3 column 8 function safe_divide thread 0
----------------------------------------------------
  result = 8 / 0

State 2 file main.py line 16 column 4 function test_exception_handling thread 0
----------------------------------------------------
Violated property:
  file main.py line 16 column 4 function test_exception_handling
  assertion
  return_value$_safe_divide$3 == 0

[Counterexample]


State 1 file main.py line 3 column 8 function safe_divide thread 0
----------------------------------------------------
Violated property:
  file main.py line 3 column 8 function safe_divide
  division by zero
  b != 0
````

### Example 4: List Bounds Checking

This example demonstrates ESBMC's ability to detect array/list bounds violations:

````python
def access_list_element(index: int) -> int:
    my_list = [10, 20, 30, 40, 50]
    return my_list[index]

def test_bounds() -> None:
    # Valid access
    assert access_list_element(2) == 30
    
    # This will trigger a bounds violation
    result = access_list_element(10)  # Index out of bounds
    assert result == 0

test_bounds()
````

**Command:**

```bash
$ esbmc main.py --multi-property
```

**ESBMC Output:**

```
[Counterexample]


State 1 file main.py line 11 column 4 function test_bounds thread 0
----------------------------------------------------
Violated property:
  file main.py line 11 column 4 function test_bounds
  assertion
  result == 0

[Counterexample]


State 1 file main.py line 3 column 4 function access_list_element thread 0
----------------------------------------------------
Violated property:
  file main.py line 3 column 4 function access_list_element
  array bounds violated: array `my_list' upper bound
  index < 5
````

### Example 5: Math Module Verification

This example showcases ESBMC's support for math module functions:

````python
import math

def test_math_functions() -> None:
    # Test floor and ceil functions
    assert math.floor(3.7) == 3
    assert math.ceil(3.2) == 4
    
    # Test combinatorial function
    assert math.comb(5, 2) == 10  # C(5,2) = 5!/(2!*3!) = 10
    
    # Test symmetry property of combinations
    n = 6
    k = 2
    assert math.comb(n, k) == math.comb(n, n - k)
    
    # Test special value detection
    nan_value = float('nan')
    inf_value = float('inf')
    
    assert math.isnan(nan_value) == True
    assert math.isinf(inf_value) == True
    assert math.isnan(5.0) == False

test_math_functions()
````

**Command:**

```bash
$ esbmc main.py
```

**ESBMC Output:**

```
Parsing main.py
Converting
Generating GOTO Program
GOTO program creation time: 0.896s
GOTO program processing time: 0.014s
Starting Bounded Model Checking
Unwinding loop 22 iteration 1   file /tmp/esbmc-python-astgen-72d0-61bd-f6b1/models/math.py line 42 column 4 function comb
Unwinding loop 22 iteration 2   file /tmp/esbmc-python-astgen-72d0-61bd-f6b1/models/math.py line 42 column 4 function comb
Unwinding loop 22 iteration 1   file /tmp/esbmc-python-astgen-72d0-61bd-f6b1/models/math.py line 42 column 4 function comb
Unwinding loop 22 iteration 2   file /tmp/esbmc-python-astgen-72d0-61bd-f6b1/models/math.py line 42 column 4 function comb
Unwinding loop 22 iteration 1   file /tmp/esbmc-python-astgen-72d0-61bd-f6b1/models/math.py line 42 column 4 function comb
Unwinding loop 22 iteration 2   file /tmp/esbmc-python-astgen-72d0-61bd-f6b1/models/math.py line 42 column 4 function comb
Symex completed in: 0.014s (51 assignments)
Slicing time: 0.000s (removed 47 assignments)
Generated 23 VCC(s), 2 remaining after simplification (4 assignments)
No solver specified; defaulting to Boolector
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.003s
Solving with solver Boolector 3.2.2
Runtime decision procedure: 0.000s
BMC program time: 0.017s

VERIFICATION SUCCESSFUL
````

### Example 6: Missing Return Statement Detection

This example demonstrates ESBMC-Python's ability to detect missing return statements in functions with return type annotations:

````python
def calculate_grade(score: int) -> str:
    """Function with missing return statement for some execution paths"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    # Missing return statement for score < 60!

def process_number(x: int) -> int:
    """Another function with missing return for positive numbers"""
    if x == 0:
        return 0
    if x < 0:
        return -x
    # Missing return for positive numbers - just has expression without return
    x * 2

def safe_divide(a: int, b: int) -> int:
    """Correctly implemented function with all paths covered"""
    if b == 0:
        return 0
    else:
        return a // b

# Test the functions
score = 50
result = calculate_grade(score)  # This will trigger missing return detection

value = process_number(5)  # This will also trigger missing return detection

safe_result = safe_divide(10, 2)  # This should work fine
````

**Command:**

```bash
$ esbmc main.py --multi-property
```

**ESBMC Output:**

```
Parsing main.py

Type checking warning:
main.py:1: error: Missing return statement
main.py:13: error: Missing return statement
Found 2 errors in 1 file (checked 1 source file)

Converting
Generating GOTO Program
GOTO program creation time: 1.156s
GOTO program processing time: 0.016s
Starting Bounded Model Checking
Symex completed in: 0.004s (17 assignments)
Slicing time: 0.000s (removed 15 assignments)
Generated 3 VCC(s), 2 remaining after simplification (2 assignments)
No solver specified; defaulting to Boolector
Slicing time: 0.000s (removed 0 assignments)
No solver specified; defaulting to Boolector
Solving claim 'Missing return statement detected in function 'process_number' at file main.py line 13 column 0 function process_number' with solver Boolector 3.2.2
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.000s
Solving with solver Boolector 3.2.2
Runtime decision procedure: 0.007s
✗ FAILED: 'Missing return statement detected in function 'process_number' at file main.py line 13 column 0 function process_number'

[Counterexample]


State 1 file main.py line 13 column 0 function process_number thread 0
----------------------------------------------------
Violated property:
  file main.py line 13 column 0 function process_number
  Missing return statement detected in function 'process_number'
  0

Slicing time: 0.000s (removed 0 assignments)
No solver specified; defaulting to Boolector
Solving claim 'Missing return statement detected in function 'calculate_grade' at file main.py line 1 column 0 function calculate_grade' with solver Boolector 3.2.2
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.000s
Solving with solver Boolector 3.2.2
Runtime decision procedure: 0.000s
✗ FAILED: 'Missing return statement detected in function 'calculate_grade' at file main.py line 1 column 0 function calculate_grade'

[Counterexample]


State 1 file main.py line 1 column 0 function calculate_grade thread 0
----------------------------------------------------
Violated property:
  file main.py line 1 column 0 function calculate_grade
  Missing return statement detected in function 'calculate_grade'
  0

Properties: 2 verified, ✗ 2 failed
Solver: Boolector 3.2.2 • Decision procedure total time: 0.008s • Avg: 0.004s/property

VERIFICATION FAILED
````

In this example, ESBMC successfully identifies two functions with missing return statements:
- `calculate_grade`: Missing a return statement for scores below 60.
- `process_number`: Missing a return statement for positive numbers (line 19 has an expression x * 2 but no return keyword).

The `safe_divide` function passes verification because it properly handles all execution paths with appropriate return statements.

Key Benefits of Missing Return Detection:
- **Early Bug Detection**: Catches potential None return values before runtime.
- **Type Safety**: Ensures functions with return type annotations actually return values on all paths.
- **Code Quality**: Improves code reliability by enforcing complete return coverage.
- **Constructor Awareness**: Automatically excludes __init__ methods which don't need explicit returns.

This analysis helps prevent `TypeError` exceptions that would occur at runtime when the missing return path is executed, as Python implicitly returns None for functions without explicit return statements.

### Example 7: Strict Type Checking

This example demonstrates ESBMC-Python's strict type checking feature, which enforces type compatibility for function arguments at verification time:

```python
def greet(name: str) -> str:
    return name

def add(a: int, b: int) -> int:
    return a + b

def process_data(value: int, factor: float, label: str) -> float:
    return value * factor

result1 = add(5, 10)  # Correct - both arguments are int
assert result1 == 15

result3 = add(5, "10")  # TypeError: second argument is str, expected int

result4: str = greet(42)  # TypeError: argument is int, expected str

result5 = process_data("wrong", 2.5, "label")  # TypeError: first argument is str, expected int
```

**Command (without strict type checking):**
```bash
$ esbmc main.py
```

**ESBMC Output:**
```
Converting
Generating GOTO Program
GOTO program creation time: 1.345s
GOTO program processing time: 0.026s
Starting Bounded Model Checking
Symex completed in: 0.006s (13 assignments)
Caching time: 0.000s (removed 0 assertions)
Slicing time: 0.002s (removed 13 assignments)
Generated 1 VCC(s), 0 remaining after simplification (0 assignments)
BMC program time: 0.007s

VERIFICATION SUCCESSFUL
```

**Command (with strict type checking):**
```bash
$ esbmc main.py --strict-types --multi-property
```

**ESBMC Output:**
```
Converting
Generating GOTO Program
GOTO program creation time: 1.433s
GOTO program processing time: 0.029s
Starting Bounded Model Checking
ERROR: Exception thrown of type TypeError at file main.py line 13
ERROR: Exception thrown of type TypeError at file main.py line 15
ERROR: Exception thrown of type TypeError at file main.py line 17
Symex completed in: 0.002s (10 assignments)
Caching time: 0.000s (removed 2 assertions)
Slicing time: 0.000s (removed 7 assignments)
Generated 4 VCC(s), 1 remaining after simplification (1 assignments)
No solver specified; defaulting to Boolector
Slicing time: 0.000s (removed 0 assignments)
No solver specified; defaulting to Boolector
Solving claim 'Throwing an exception of type TypeError but there is not catch for it. at file main.py line 17 column 0' with solver Boolector 3.2.2
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.000s
Solving with solver Boolector 3.2.2
Runtime decision procedure: 0.002s
✗ FAILED: 'Throwing an exception of type TypeError but there is not catch for it. at file main.py line 17 column 0'

[Counterexample]


State 1 file main.py line 17 column 0 thread 0
----------------------------------------------------
Violated property:
  file main.py line 17 column 0
  Throwing an exception of type TypeError but there is not catch for it.

Slicing time: 0.000s (removed 0 assignments)
No solver specified; defaulting to Boolector
Solving claim 'Throwing an exception of type TypeError but there is not catch for it. at file main.py line 15 column 0' with solver Boolector 3.2.2
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.000s
Solving with solver Boolector 3.2.2
Runtime decision procedure: 0.000s
✗ FAILED: 'Throwing an exception of type TypeError but there is not catch for it. at file main.py line 15 column 0'

[Counterexample]


State 1 file main.py line 15 column 0 thread 0
----------------------------------------------------
Violated property:
  file main.py line 15 column 0
  Throwing an exception of type TypeError but there is not catch for it.

Slicing time: 0.000s (removed 0 assignments)
No solver specified; defaulting to Boolector
Solving claim 'Throwing an exception of type TypeError but there is not catch for it. at file main.py line 13 column 0' with solver Boolector 3.2.2
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 0.000s
Solving with solver Boolector 3.2.2
Runtime decision procedure: 0.000s
✗ FAILED: 'Throwing an exception of type TypeError but there is not catch for it. at file main.py line 13 column 0'

[Counterexample]


State 1 file main.py line 13 column 0 thread 0
----------------------------------------------------
Violated property:
  file main.py line 13 column 0
  Throwing an exception of type TypeError but there is not catch for it.

Properties: 3 verified, ✗ 3 failed
Solver: Boolector 3.2.2 • Decision procedure total time: 0.002s • Avg: 0.000s/property

VERIFICATION FAILED
```

In this example, ESBMC with `--strict-types` successfully detects all type mismatches during the parsing phase and generates corresponding verification failures:
- **Line 13**: `add(5, "10")` - Argument 2 has incompatible type "str"; expected "int"
- **Line 15**: `greet(42)` - Argument 1 has incompatible type "int"; expected "str"
- **Line 17**: `process_data("wrong", 2.5, "label")` - Argument 1 has incompatible type "str"; expected "int"

**Key Benefits of Strict Type Checking:**
- **Early Detection**: Type errors are identified during the parsing phase with clear diagnostic messages.
- **Type Safety Enforcement**: Catches type mismatches at verification time rather than runtime.
- **Exception Modeling**: Type errors are modeled as uncaught `TypeError` exceptions, triggering verification failures.
- **Descriptive Error Messages**: Provides a clear indication of which argument has a type error and what types were expected vs. actual.
- **Optional Feature**: Disabled by default to allow flexible verification; enable only when type safety is critical.

This feature helps prevent `TypeError` exceptions that would occur at runtime when incompatible types are passed to functions, improving code reliability through static type verification.

### Example 8: Cover Properties for Reachability Analysis

This example demonstrates how cover properties can be used to analyze program reachability and verify that specific conditions are possible:

````Python
def analyze_paths(x: int) -> str:
    """Function with multiple execution paths"""
    if x > 100:
        return "high"
    elif x > 50:
        return "medium"
    elif x > 0:
        return "low"
    else:
        return "zero_or_negative"

def test_reachability() -> None:
    x: int = nondet_int()
    
    # Cover: Can x be greater than 100? (Expected: FAIL - satisfiable)
    __ESBMC_cover(x > 100)
    
    # Cover: Can x be in the medium range? (Expected: FAIL - satisfiable)
    __ESBMC_cover(50 < x <= 100)
    
    # Add constraint: x must be positive
    __ESBMC_assume(x > 0)
    
    # Cover: Can x still be negative? (Expected: SUCCESS - not satisfiable)
    __ESBMC_cover(x < 0)
    
    # Cover: Can x be positive? (Expected: FAIL - satisfiable)
    __ESBMC_cover(x > 0)
    
    result = analyze_paths(x)
    
    # Cover: Can we reach the "high" path? (Expected: FAIL - satisfiable)
    __ESBMC_cover(result == "high")
    
    # Cover: Can we reach the "zero_or_negative" path? (Expected: SUCCESS - not satisfiable after assume)
    __ESBMC_cover(result == "zero_or_negative")

def test_dead_code_detection() -> None:
    """Using cover to identify unreachable code"""
    value: int = nondet_int()
    __ESBMC_assume(value >= 0)
    
    if value < 0:
        # This is dead code - the cover should succeed (prove unreachability)
        __ESBMC_cover(True)
        print("This branch is unreachable")
    
    if value >= 0:
        # This is reachable code - the cover should fail (show reachability)
        __ESBMC_cover(True)
        print("This branch is reachable")

test_reachability()
test_dead_code_detection()
````

**Command:**

```bash
$ esbmc main.py --unwind 10 --multi-property
```

**ESBMC Output:**
```
✗ FAILED: 'assertion at file main.py line 16 column 4 function test_reachability'

[Counterexample]


State 1 file main.py line 13 column 4 function test_reachability thread 0
----------------------------------------------------
  x = 9223372036854767699 (01111111 11111111 11111111 11111111 11111111 11111111 11100000 01010011)

State 2 file main.py line 16 column 4 function test_reachability thread 0
----------------------------------------------------
Violated property:
  file main.py line 16 column 4 function test_reachability
  assertion
  !(x > 100)

✗ FAILED: 'assertion at file main.py line 19 column 4 function test_reachability'

[Counterexample]


State 1 file main.py line 13 column 4 function test_reachability thread 0
----------------------------------------------------
  x = 79 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 01001111)

State 2 file main.py line 19 column 4 function test_reachability thread 0
----------------------------------------------------
Violated property:
  file main.py line 19 column 4 function test_reachability
  assertion
  !(50 < x && x <= 100)

✗ FAILED: 'assertion at file main.py line 28 column 4 function test_reachability'

[Counterexample]


State 1 file main.py line 13 column 4 function test_reachability thread 0
----------------------------------------------------
  x = 2047 (00000000 00000000 00000000 00000000 00000000 00000000 00000111 11111111)

State 3 file main.py line 28 column 4 function test_reachability thread 0
----------------------------------------------------
Violated property:
  file main.py line 28 column 4 function test_reachability
  assertion
  !(x > 0)

✗ FAILED: 'assertion at file main.py line 33 column 4 function test_reachability'

[Counterexample]


State 1 file main.py line 13 column 4 function test_reachability thread 0
----------------------------------------------------
  x = 103 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 01100111)

State 3 file main.py line 33 column 4 function test_reachability thread 0
----------------------------------------------------
Violated property:
  file main.py line 33 column 4 function test_reachability
  assertion
  !(return_value$_strcmp$1 == 0)

✗ FAILED: 'assertion at file main.py line 50 column 8 function test_dead_code_detection'

[Counterexample]


State 1 file main.py line 13 column 4 function test_reachability thread 0
----------------------------------------------------
  x = 2047 (00000000 00000000 00000000 00000000 00000000 00000000 00000111 11111111)

State 3 file main.py line 40 column 4 function test_dead_code_detection thread 0
----------------------------------------------------
  value = 0 (00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000)

State 5 file main.py line 50 column 8 function test_dead_code_detection thread 0
----------------------------------------------------
Violated property:
  file main.py line 50 column 8 function test_dead_code_detection
  assertion
  !1
````

## Understanding Cover Property Results:

- **FAILED (counterexample found)** = The condition is satisfiable/reachable.
  - This is the expected outcome for conditions you want to verify as possible.
  - ESBMC provides concrete values that satisfy the condition.
- **PASSED (proof found)** = The condition is unsatisfiable/unreachable.
  - This indicates the condition cannot be satisfied given the constraints.
  - Useful for verifying dead code or proving certain states are unreachable.
- **Key Insights from this Example**:
  - **Lines 16, 19, 28, 33, 50**: These cover properties fail (have counterexamples), confirming that various values and code paths are reachable.
  - **Lines 25, 36**: These cover properties pass (have proofs), demonstrating that after the `__ESBMC_assume(x > 0)`, negative values and the "zero_or_negative" path are provably unreachable.
  - **Dead Code Detection**: The cover property at line 50 fails (counterexample), confirming that the branch is reachable. The cover at line 45 (in the if value < 0 branch) would pass (proof), proving that the branch is dead code.

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
