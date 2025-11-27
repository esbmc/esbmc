# Coverage Analysis

ESBMC supports coverage analysis for branch, condition, and assertion coverage. This helps identify which code paths have been tested in your program.

## Usage

ESBMC provides three coverage analysis modes:

```bash
# Branch coverage
esbmc example.c --branch-coverage

# Condition coverage
esbmc example.c --condition-coverage

# Assertion coverage
esbmc example.c --assertion-coverage
```

## Coverage Modes

### Branch Coverage

Branch coverage verifies that all branches of conditional statements are executed. For each `if`, `else`, `while`, or `for` statement, both the true and false branches must be covered.

**Example:**

```c
int check_sign(int n) {
    if (n > 0)
        return 1;
    else
        return -1;
}

int main() {
    check_sign(5);   // Only covers true branch
    return 0;
}
```

```bash
$ esbmc example.c --branch-coverage
...
[Coverage]
Branches : 2
Reached : 1
Branch Coverage: 50%

```

The **Branch Coverage: 50%** shows incomplete coverage. The output indicates that `n <= 0` is needed to cover the else branch.

**Full coverage example:**

```c
int main() {
    check_sign(5);    // Covers true branch
    check_sign(-3);   // Covers false branch
    return 0;
}
```

```bash
$ esbmc example.c --branch-coverage
...
[Coverage]
Branches : 2
Reached : 2
Branch Coverage: 100%
```

The **Branch Coverage: 100%** indicates all branches have been tested.

### Condition Coverage

Condition coverage verifies that each boolean sub-expression in conditional statements evaluates to both true and false at least once. Unlike branch coverage which tests branches, condition coverage tests individual conditions within complex expressions.

**Example:**

```c
int validate(int x, int y) {
    if (x > 0 && y > 0)  // Complex condition
        return 1;
    else
        return 0;
}

int main() {
    validate(5, 3);   
    return 0;
}
```

```bash
$ esbmc example.c --condition-coverage
...
[Coverage]

Reached Conditions:  4
Short Circuited Conditions:  0
Total Conditions:  4

Condition Properties - SATISFIED:  2
Condition Properties - UNSATISFIED:  2

Condition Coverage: 50%
```

The **Condition Coverage: 50%** shows that only the true cases of both conditions were tested. For full condition coverage, each condition must evaluate to both true and false at least once:
- `x > 0` must be true (✓ tested) and false (✗ not tested)
- `y > 0` must be true (✓ tested) and false (✗ not tested)

**Full coverage example:**

```c
int main() {
    validate(5, 3);    // x>0: T, y>0: T
    validate(5, -1);   // x>0: T, y>0: F
    validate(-1, -1);  // x>0: F (y>0 short-circuited)
    return 0;
}
```
Note: Due to short-circuit evaluation in `&&`, when `x>0` is false, `y>0` is never evaluated. To test `y>0` as both true and false, `x>0` must be true in those tests.

### Assertion Coverage

Assertion coverage verifies that all assertions in the code are reached and tested. This ensures that all validation checks are exercised.

**Example:**

```c
int process(int n) {
    if (n > 10) {
        assert(n < 100);  
        return n * 2;
    }
    else {
        assert(n < 10);   
        return n + 2;
    }
    return n;
}

int main() {
    process(5);    // Covers second assertion
    return 0;
}
```

```bash
$ esbmc example.c --assertion-coverage
...
[Coverage]
Total Asserts: 2
Total Assertion Instances: 2
Reached Assertion Instances: 1
Assertion Instances Coverage: 50%
```

**Full coverage example:**

```c
int main() {
    process(50);   // Covers first assertion
    process(5);    // Covers second assertion
    return 0;
}
```

```bash
$ esbmc example.c --assertion-coverage
...
[Coverage]
Total Asserts: 2
Total Assertion Instances: 2
Reached Assertion Instances: 2
Assertion Instances Coverage: 100%
```

## Supported Languages

Coverage analysis is supported for:

- **C** - All C features supported by ESBMC
- **C++** - All C++ features supported by ESBMC
- **Python** - Full Python frontend support
- **Solidity** - Full Solidity frontend support

## Interpreting Coverage Results

The key output of coverage analysis is the **coverage percentage** shown in the `[Coverage]` section. This indicates what portion of your code paths have been exercised.

### Coverage Statistics

ESBMC reports:
- **Total elements**: Number of branches/conditions/assertions in the code
- **Reached elements**: How many were covered by the test inputs
- **Coverage percentage**: (Reached / Total) × 100%

### Understanding the Output

Coverage analysis uses verification internally to determine reachability. You may see `VERIFICATION SUCCESSFUL` or `VERIFICATION FAILED` in the output, but these are intermediate messages during coverage computation, not indicators of program correctness.

**What matters:** The coverage percentage (e.g., "Branch Coverage: 75%")
**Less relevant:** VERIFICATION status (just shows coverage tool operation)

## Python Examples

### Branch Coverage

```python
def is_positive(n: int) -> int:
    if n > 0:
        return 1
    else:
        return 0

# Only covers positive branch
is_positive(10)
```

```bash
$ esbmc example.py --branch-coverage
...
[Coverage]
Branches : 2
Reached : 1
Branch Coverage: 50%
```

The **Branch Coverage: 50%** indicates incomplete coverage.

### Condition Coverage

```python
def check_range(x: int, y: int) -> int:
    if x > 0 and y > 0:
        return 1
    else:
        return 0

# Full condition coverage - accounting for short-circuit evaluation
check_range(5, 3)    # x>0: T, y>0: T
check_range(5, -1)   # x>0: T, y>0: F
check_range(-1, -1)  # x>0: F (y>0 short-circuited)
```

```bash
$ esbmc example.py --condition-coverage
...
[Coverage]

Reached Conditions:  4
Short Circuited Conditions:  0
Total Conditions:  4

Condition Properties - SATISFIED:  4
Condition Properties - UNSATISFIED:  0

Condition Coverage: 100%
```

The **Condition Coverage: 100%** shows each condition was tested as both true and false.

### Assertion Coverage

```python
def validate_positive(n: int) -> int:
    assert n > 0, "n must be positive"
    return n * 2

validate_positive(5)
```

```bash
$ esbmc example.py --assertion-coverage
...
[Coverage]

Total Asserts: 1
Total Assertion Instances: 1
Reached Assertion Instances: 1
Assertion Instances Coverage: 100%
```

## Technical Notes

- Coverage analysis in ESBMC uses **symbolic execution** and **SMT solving**
- Unlike traditional testing tools, ESBMC **determines path reachability** and calculates exact coverage percentages
- All paths are explored **automatically** without manual test case writing
- Coverage analysis adds false assertions to test each branch/condition
- Verification time increases with code complexity

### Combining with Other Flags

Coverage analysis can be combined with other ESBMC options:

```bash
# With bounded model checking
esbmc example.c --branch-coverage --unwind 10

# With k-induction
esbmc example.c --branch-coverage --k-induction
```