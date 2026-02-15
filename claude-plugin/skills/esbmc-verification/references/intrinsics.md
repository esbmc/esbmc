# ESBMC Intrinsics Reference

## Overview

ESBMC intrinsics are special functions that communicate with the verification engine. They enable:
- Creating symbolic (non-deterministic) values
- Adding assumptions to constrain inputs
- Specifying properties to verify
- Controlling verification behavior

## Non-Deterministic Value Creation

Non-deterministic values represent any possible value of a type. Use them for symbolic inputs.

### C/C++ Intrinsics

```c
// Integer types
int x = __ESBMC_nondet_int();
unsigned int u = __ESBMC_nondet_uint();
short s = __ESBMC_nondet_short();
unsigned short us = __ESBMC_nondet_ushort();
long l = __ESBMC_nondet_long();
unsigned long ul = __ESBMC_nondet_ulong();
long long ll = __ESBMC_nondet_longlong();
unsigned long long ull = __ESBMC_nondet_ulonglong();

// Character types
char c = __ESBMC_nondet_char();
unsigned char uc = __ESBMC_nondet_uchar();
signed char sc = __ESBMC_nondet_schar();

// Boolean
_Bool b = __ESBMC_nondet_bool();

// Floating point
float f = __ESBMC_nondet_float();
double d = __ESBMC_nondet_double();

// Pointer (creates symbolic pointer)
void *p = __ESBMC_nondet_ptr();

// Size type
size_t sz = __ESBMC_nondet_size_t();
```

### Typed Non-Determinism

```c
// Macro for any type
#define nondet(type) __ESBMC_nondet_##type()

// Usage
int x = nondet(int);
float f = nondet(float);
```

### Python Intrinsics

```python
from esbmc import (
    nondet_int,
    nondet_float,
    nondet_bool,
    nondet_str
)

x: int = nondet_int()
f: float = nondet_float()
b: bool = nondet_bool()
s: str = nondet_str()  # respects --nondet-str-length
```

## Assumptions

Assumptions constrain the possible values of symbolic variables. They guide verification toward meaningful inputs.

### C/C++ Syntax

```c
// Basic assumption
__ESBMC_assume(condition);

// Examples
int x = __ESBMC_nondet_int();
__ESBMC_assume(x > 0);           // x is positive
__ESBMC_assume(x < 100);         // x is less than 100
__ESBMC_assume(x % 2 == 0);      // x is even

// Array bounds assumption
int arr[10];
int idx = __ESBMC_nondet_int();
__ESBMC_assume(idx >= 0 && idx < 10);
arr[idx] = 42;  // Safe access

// Multiple constraints
int a = __ESBMC_nondet_int();
int b = __ESBMC_nondet_int();
__ESBMC_assume(a > 0 && a < 50);
__ESBMC_assume(b > a);           // b > a > 0
```

### Python Syntax

```python
from esbmc import assume, nondet_int

x: int = nondet_int()
assume(x > 0)
assume(x < 100)
```

### Assumption Best Practices

1. **Constrain early**: Add assumptions immediately after creating nondet values
2. **Avoid over-constraining**: Too many assumptions may make verification trivial
3. **Document assumptions**: Comment why constraints are needed
4. **Check for contradictions**: Contradictory assumptions make all properties pass vacuously

```c
// Good: Clear, documented constraints
int age = __ESBMC_nondet_int();
__ESBMC_assume(age >= 0 && age <= 150);  // Valid human age range

// Bad: Over-constrained (only one value possible)
int x = __ESBMC_nondet_int();
__ESBMC_assume(x == 42);  // Not actually symbolic anymore
```

## Assertions

Assertions specify properties that must hold. ESBMC verifies these automatically.

### C/C++ Syntax

```c
// Standard assert (recommended for portability)
#include <assert.h>
assert(condition);

// ESBMC-specific with message
__ESBMC_assert(condition, "Description of property");

// Examples
int result = compute(x);
assert(result >= 0);
__ESBMC_assert(result <= MAX_VALUE, "Result within bounds");
```

### Python Syntax

```python
from esbmc import esbmc_assert

# Standard Python assert
assert result >= 0

# ESBMC assert with message
esbmc_assert(result <= MAX_VALUE, "Result within bounds")
```

### Assertion Patterns

```c
// Precondition checking
void process(int *ptr, int size) {
    __ESBMC_assert(ptr != NULL, "Null pointer");
    __ESBMC_assert(size > 0, "Invalid size");
    // ...
}

// Postcondition checking
int abs_value(int x) {
    int result = (x < 0) ? -x : x;
    __ESBMC_assert(result >= 0, "Absolute value is non-negative");
    return result;
}

// Loop invariant
int sum = 0;
for (int i = 0; i < n; i++) {
    sum += arr[i];
    __ESBMC_assert(sum >= 0, "Running sum invariant");  // If arr contains non-negative
}

// Relationship assertions
__ESBMC_assert(output == expected, "Correctness");
__ESBMC_assert(sorted(arr, n), "Array is sorted");
```

## Unreachability

Mark code locations that should never be reached.

### C/C++ Syntax

```c
// Mark unreachable code (requires --enable-unreachability-intrinsic)
__ESBMC_unreachable();

// Example: exhaustive switch
switch (state) {
    case STATE_A: handle_a(); break;
    case STATE_B: handle_b(); break;
    case STATE_C: handle_c(); break;
    default:
        __ESBMC_unreachable();  // Should never happen
}

// Example: dead code detection
if (x > 0) {
    // process
} else if (x < 0) {
    // process
} else if (x == 0) {
    // process
} else {
    __ESBMC_unreachable();  // Mathematically impossible
}
```

### Command Line

```bash
esbmc file.c --enable-unreachability-intrinsic
```

## Atomic Operations

For concurrent program verification.

### C/C++ Syntax

```c
// Atomic begin/end blocks
__ESBMC_atomic_begin();
// Critical section - executed atomically
shared_var = compute();
flag = true;
__ESBMC_atomic_end();

// Example: lock-free counter
__ESBMC_atomic_begin();
int old = counter;
counter = old + 1;
__ESBMC_atomic_end();
```

## Memory Safety Intrinsics

### Dynamic Memory

```c
// Check if memory is valid
__ESBMC_assert(__ESBMC_is_valid(ptr), "Pointer is valid");

// Check buffer bounds
__ESBMC_assert(__ESBMC_buffer_size(ptr) >= required, "Buffer large enough");
```

### Array Bounds

```c
// Explicit bounds checking
int arr[10];
int idx = __ESBMC_nondet_int();
if (__ESBMC_is_valid_index(arr, idx)) {
    arr[idx] = value;
}
```

## Verification Control

### Property Selection

```c
// Mark specific claims
__ESBMC_assert(cond1, "Claim 1");  // Claim #1
__ESBMC_assert(cond2, "Claim 2");  // Claim #2
__ESBMC_assert(cond3, "Claim 3");  // Claim #3
```

```bash
# Verify only claim 2
esbmc file.c --claim 2
```

### Verification Hints

```c
// Suggest loop invariant (for k-induction)
__ESBMC_loop_invariant(invariant_condition);

// Example
int i = 0;
while (i < n) {
    __ESBMC_loop_invariant(i >= 0 && i <= n);
    // loop body
    i++;
}
```

## Complete Example

```c
#include <stdlib.h>

// Verified binary search
int binary_search(int *arr, int size, int target) {
    // Preconditions
    __ESBMC_assert(arr != NULL, "Array not null");
    __ESBMC_assert(size > 0, "Positive size");

    int left = 0;
    int right = size - 1;

    while (left <= right) {
        // Loop invariants
        __ESBMC_assert(left >= 0, "Left bound valid");
        __ESBMC_assert(right < size, "Right bound valid");
        __ESBMC_assert(left <= right + 1, "Bounds ordered");

        int mid = left + (right - left) / 2;  // Overflow-safe
        __ESBMC_assert(mid >= left && mid <= right, "Mid in range");

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;  // Not found
}

int main() {
    int size = __ESBMC_nondet_int();
    __ESBMC_assume(size > 0 && size <= 100);

    int *arr = malloc(size * sizeof(int));
    __ESBMC_assume(arr != NULL);

    // Initialize sorted array symbolically
    for (int i = 0; i < size; i++) {
        arr[i] = __ESBMC_nondet_int();
        if (i > 0) {
            __ESBMC_assume(arr[i] >= arr[i-1]);  // Sorted
        }
    }

    int target = __ESBMC_nondet_int();
    int result = binary_search(arr, size, target);

    // Postcondition
    if (result >= 0) {
        __ESBMC_assert(result < size, "Result in bounds");
        __ESBMC_assert(arr[result] == target, "Found correct element");
    }

    free(arr);
    return 0;
}
```

## Python Complete Example

```python
from esbmc import nondet_int, assume, esbmc_assert

def binary_search(arr: list[int], target: int) -> int:
    """Verified binary search implementation."""
    esbmc_assert(len(arr) > 0, "Array not empty")

    left = 0
    right = len(arr) - 1

    while left <= right:
        esbmc_assert(left >= 0, "Left bound valid")
        esbmc_assert(right < len(arr), "Right bound valid")

        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def main():
    # Symbolic sorted array
    size: int = nondet_int()
    assume(size > 0 and size <= 10)

    arr: list[int] = []
    prev: int = nondet_int()
    assume(prev > -1000)

    for i in range(size):
        val: int = nondet_int()
        assume(val >= prev)
        arr.append(val)
        prev = val

    target: int = nondet_int()
    result = binary_search(arr, target)

    if result >= 0:
        esbmc_assert(result < len(arr), "Result in bounds")
        esbmc_assert(arr[result] == target, "Found correct element")

if __name__ == "__main__":
    main()
```

## Quick Reference

| Intrinsic | Purpose |
|-----------|---------|
| `__ESBMC_nondet_<type>()` | Create symbolic value |
| `__ESBMC_assume(cond)` | Add constraint |
| `__ESBMC_assert(cond, msg)` | Verify property |
| `__ESBMC_unreachable()` | Mark unreachable code |
| `__ESBMC_atomic_begin/end()` | Atomic section |
| `__ESBMC_loop_invariant(cond)` | Loop invariant hint |
