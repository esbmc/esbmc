# HTML Report Generation

ESBMC can generate interactive HTML reports that visualize counterexample traces, making it easier to understand and debug verification failures.

## Usage

To generate an HTML report when verification fails, use the `--generate-html-report` flag as follows:

```bash
esbmc example.c --generate-html-report
```

When verification fails, and a counterexample is found, ESBMC will create an HTML file named `report-<n>.html` in the current directory.

### Example Output

```
$ esbmc test.c --generate-html-report --k-induction
...
Generating HTML report for trace: 1
...
VERIFICATION FAILED
```

This creates `report-1.html`, which can be opened in any web browser.

## Report Features

### Bug Summary

The report header displays:
- **File**: The absolute path to the source file containing the violation.
- **Violation**: The exact location (function, line, column) and type of the error.

### Annotated Source Code

The main section shows your source code with:
- **Line numbers** for easy reference.
- **Syntax highlighting** for keywords (supports C/C++ and Python).
- **Interactive trace markers** showing the execution path leading to the violation.

### Trace Navigation

Each step in the counterexample trace is displayed as a numbered marker:
- **Yellow markers (Event)** indicate assignments, function arguments, and the final violation.
- **Gray markers (Control)** indicate assumptions and assertion checks.
- **Navigation arrows** (← →) allow jumping between trace steps.

### Trace Information

Each trace step displays context-specific information:

- **Assumption**: "Assumption restriction".
- **Assertion**: The assertion condition.
- **Assignment**: Variable and value.
- **Function Call**: Argument binding.

## Keyboard Shortcuts

The HTML report supports keyboard navigation:

| Key | Action |
|-----|--------|
| `j` | Navigate to next trace event |
| `k` | Navigate to previous trace event |
| `Shift+S` | Toggle showing only relevant lines |
| `?` | Toggle keyboard shortcuts help |

## Filtering

Use the "Show only relevant lines" checkbox to hide source lines that are not part of the counterexample trace. This helps focus on the code path that leads to the violation.

## Supported Languages

HTML reports support syntax highlighting for:

### C/C++
Keywords highlighted: `auto`, `break`, `case`, `char`, `const`, `continue`, `default`, `do`, `double`, `else`, `enum`, `extern`, `float`, `for`, `goto`, `if`, `int`, `long`, `register`, `return`, `short`, `signed`, `sizeof`, `static`, `struct`, `switch`, `typedef`, `union`, `unsigned`, `void`, `volatile`, `while`

### Python
Keywords highlighted: `False`, `None`, `True`, `and`, `as`, `assert`, `async`, `await`, `break`, `class`, `continue`, `def`, `del`, `elif`, `else`, `except`, `finally`, `for`, `from`, `global`, `if`, `import`, `in`, `is`, `lambda`, `nonlocal`, `not`, `or`, `pass`, `raise`, `return`, `try`, `while`, `with`, `yield`

## Examples

### C Program

```c
// example.c
#include <stdlib.h>

int main() {
    int *p = NULL;
    *p = 42; 
    return 0;
}
```

```bash
esbmc example.c --generate-html-report
```

### Python Program

```python
# example.py
def factorial(n:int) -> int:
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

n:int = nondet_int()
__ESBMC_assume(n > 0);
__ESBMC_assume(n < 6);

result:int = factorial(n)
assert(result != 120)
```

```bash
esbmc main.py --incremental-bmc --generate-html-report
```

## Technical Notes

- Reports are self-contained HTML files with no external dependencies
- Multiple reports can be generated in the same directory (numbered sequentially)
- View ESBMC command options via "Show analyzer invocation" in the report
