# Adding Verification Intrinsics to Code

## Process

### Step 1: Analyze the Code

Read the code and identify:
- **Function inputs**: Parameters that should become symbolic
- **External inputs**: User input, file reads, network data
- **Array accesses**: Indices that need bounds assumptions
- **Pointer operations**: Allocations, dereferences, frees
- **Arithmetic operations**: Potential overflow points
- **Critical properties**: What should always be true (postconditions)
- **Implicit assumptions**: What the code assumes about inputs

### Step 2: Add Symbolic Inputs (for test harnesses)

Replace concrete inputs with non-deterministic values:

```c
// Before: concrete test
int test_sort() {
    int arr[5] = {3, 1, 4, 1, 5};
    sort(arr, 5);
}

// After: symbolic verification
int test_sort() {
    int arr[5];
    for (int i = 0; i < 5; i++) {
        arr[i] = __ESBMC_nondet_int();
    }
    sort(arr, 5);
}
```

### Step 3: Add Preconditions (Assumptions)

Constrain inputs to valid ranges:

```c
void process(int *arr, int size, int index) {
    __ESBMC_assume(arr != NULL);
    __ESBMC_assume(size > 0 && size <= MAX_SIZE);
    __ESBMC_assume(index >= 0 && index < size);
    // Original code follows...
}
```

### Step 4: Add Postconditions (Assertions)

Specify what must be true after operations:

```c
int abs_value(int x) {
    int result = (x < 0) ? -x : x;
    __ESBMC_assert(result >= 0, "Absolute value is non-negative");
    return result;
}
```

### Step 5: Add Safety Assertions

Insert checks at critical points:

```c
// Before array access
__ESBMC_assert(idx >= 0 && idx < size, "Array index in bounds");
arr[idx] = value;

// Before pointer dereference
__ESBMC_assert(ptr != NULL, "Pointer not null");
*ptr = value;

// After allocation
ptr = malloc(size);
__ESBMC_assert(ptr != NULL, "Allocation succeeded");

// After arithmetic (if overflow checking needed)
int sum = a + b;
__ESBMC_assert(sum >= a, "No overflow in addition");  // For positive values
```

### Step 6: Add Loop Invariants (for k-induction)

For unbounded verification, add invariants:

```c
int sum = 0;
for (int i = 0; i < n; i++) {
    __ESBMC_assert(i >= 0 && i <= n, "Loop index invariant");
    __ESBMC_assert(sum >= 0, "Running sum invariant");
    sum += arr[i];
}
```

## Guidelines

1. **Be conservative with assumptions**: Only assume what is truly necessary
2. **Be liberal with assertions**: Assert expected properties at critical points
3. **Document constraints**: Add comments explaining why assumptions exist
4. **Constrain symbolic values realistically**: Use domain knowledge for bounds
5. **Don't over-constrain**: Too many assumptions make verification trivial
6. **Preserve original behavior**: Intrinsics should not change program semantics

## Common Verification Patterns

### Safe Array Access
```c
#define SIZE 100
int arr[SIZE];
int idx = __ESBMC_nondet_int();
__ESBMC_assume(idx >= 0 && idx < SIZE);
arr[idx] = value;  // Verified safe
```

### Safe Pointer Handling
```c
int *ptr = malloc(sizeof(int));
if (ptr != NULL) {
    *ptr = value;
    // ... use ptr
    free(ptr);
    ptr = NULL;  // Prevent use-after-free
}
```

### Safe Integer Arithmetic
```c
#include <limits.h>
int a = __ESBMC_nondet_int();
int b = __ESBMC_nondet_int();
__ESBMC_assume(a >= 0 && a <= 10000);
__ESBMC_assume(b >= 0 && b <= 10000);
int sum = a + b;  // Cannot overflow
```

### Safe Division
```c
int a = __ESBMC_nondet_int();
int b = __ESBMC_nondet_int();
__ESBMC_assume(b != 0);  // Prevent division by zero
int result = a / b;
```

## Python Patterns

### Symbolic Inputs
```python
from esbmc import nondet_int, assume, esbmc_assert

x: int = nondet_int()
assume(x > 0 and x < 100)
```

### Postcondition Checking
```python
result = compute(x)
esbmc_assert(result >= 0, "Result non-negative")
```

### List Verification
```python
size: int = nondet_int()
assume(size > 0 and size <= 10)
lst: list[int] = []
for _ in range(size):
    val: int = nondet_int()
    assume(val > -100 and val < 100)
    lst.append(val)
```

## C++ Patterns

### Class Invariant Verification
```cpp
class BoundedCounter {
    int value;
    int max_val;
public:
    BoundedCounter(int max) : value(0), max_val(max) {
        __ESBMC_assert(max > 0, "Max must be positive");
    }
    void increment() {
        if (value < max_val) value++;
        __ESBMC_assert(value >= 0 && value <= max_val, "Invariant");
    }
};
```

### STL Container Verification
```cpp
#include <vector>
std::vector<int> v;
int n = __ESBMC_nondet_int();
__ESBMC_assume(n > 0 && n <= 10);
for (int i = 0; i < n; i++) {
    v.push_back(__ESBMC_nondet_int());
}
assert(v.size() == (size_t)n);
```
