# ESBMC Plugin Tutorial

This tutorial walks through using the ESBMC Claude Code plugin to verify C, C++, and Python programs. By the end, you will know how to run verification, interpret results, add verification annotations, and use the plugin commands.

## Prerequisites

1. **ESBMC** installed and available in your PATH
   - Download from [ESBMC releases](https://github.com/esbmc/esbmc/releases) or build from source
   - Verify: `esbmc --version`

2. **Claude Code** CLI installed

3. **Plugin installed** (see [README.md](README.md) for installation)

## Part 1: Verifying a C Program

### Step 1: Write a simple C program

Create a file `example.c`:

```c
#include <stdlib.h>
#include <assert.h>

int safe_divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

int main() {
    int *arr = malloc(5 * sizeof(int));
    if (arr == NULL) return 1;

    for (int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }

    assert(arr[2] == 20);
    assert(safe_divide(10, 2) == 5);

    free(arr);
    return 0;
}
```

### Step 2: Run basic verification

In Claude Code, type:

```
/verify example.c
```

This runs ESBMC with default checks (array bounds, null pointers, division by zero, assertions) and reports the result.

### Step 3: Run with additional checks

```
/verify example.c all
```

This enables memory leak checking, overflow checking, and concurrency checking in addition to defaults.

### Step 4: Run a full audit

```
/audit example.c
```

This performs six verification passes with increasing depth:
1. Quick scan
2. Memory safety
3. Integer safety
4. Concurrency (if detected)
5. Deep verification with higher bounds
6. K-induction proof attempt

You receive a structured report with findings per category.

## Part 2: Finding Bugs in C Code

### Step 1: Write code with a bug

Create `buggy.c`:

```c
#include <stdlib.h>

int main() {
    int arr[10];
    int i = 11;  // Out of bounds!
    arr[i] = 42;
    return 0;
}
```

### Step 2: Verify

```
/verify buggy.c
```

ESBMC reports `VERIFICATION FAILED` with a counterexample trace showing the out-of-bounds access at `arr[11]`.

### Step 3: Understand the counterexample

The trace shows:
- The value of `i` (11)
- The array bounds (0-9)
- The exact line where the violation occurs

### Step 4: Fix and re-verify

Fix the code:

```c
int i = 5;  // Within bounds
arr[i] = 42;
```

Run `/verify buggy.c` again — it now reports `VERIFICATION SUCCESSFUL`.

## Part 3: Adding Verification Annotations

Annotations let you verify properties beyond default safety checks. Ask Claude Code to help:

> "Add verification intrinsics to my binary search function"

Claude will use the ESBMC skill to:
1. Identify symbolic inputs
2. Add preconditions (`__ESBMC_assume`)
3. Add postconditions (`__ESBMC_assert`)
4. Add loop invariants

### Example: Annotated Binary Search

```c
#include <stdlib.h>

int __ESBMC_nondet_int(void);
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);

int binary_search(int *arr, int size, int target) {
    __ESBMC_assert(arr != NULL, "Array not null");
    __ESBMC_assert(size > 0, "Positive size");

    int left = 0, right = size - 1;

    while (left <= right) {
        __ESBMC_assert(left >= 0 && right < size, "Bounds valid");
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

int main() {
    int size = __ESBMC_nondet_int();
    __ESBMC_assume(size > 0 && size <= 20);

    int *arr = malloc(size * sizeof(int));
    __ESBMC_assume(arr != NULL);

    // Fill with sorted symbolic values
    for (int i = 0; i < size; i++) {
        arr[i] = __ESBMC_nondet_int();
        if (i > 0) __ESBMC_assume(arr[i] >= arr[i-1]);
    }

    int target = __ESBMC_nondet_int();
    int result = binary_search(arr, size, target);

    if (result >= 0) {
        __ESBMC_assert(result < size, "Result in bounds");
        __ESBMC_assert(arr[result] == target, "Found correct element");
    }

    free(arr);
    return 0;
}
```

Verify with:

```
/verify binary_search.c memory
```

## Part 4: Verifying C++ Code

ESBMC supports C++11 through C++20.

### Step 1: Write a C++ program

Create `stack_test.cpp`:

```cpp
#include <vector>
#include <cassert>

int __ESBMC_nondet_int(void);
void __ESBMC_assume(bool);
void __ESBMC_assert(bool, const char *);

class SafeStack {
    std::vector<int> data;
    int max_size;
public:
    SafeStack(int max) : max_size(max) {
        __ESBMC_assert(max > 0, "Max must be positive");
    }

    bool push(int val) {
        if ((int)data.size() >= max_size) return false;
        data.push_back(val);
        return true;
    }

    int pop() {
        __ESBMC_assert(!data.empty(), "Cannot pop empty stack");
        int val = data.back();
        data.pop_back();
        return val;
    }

    bool empty() const { return data.empty(); }
    int size() const { return (int)data.size(); }
};

int main() {
    int cap = __ESBMC_nondet_int();
    __ESBMC_assume(cap > 0 && cap <= 5);

    SafeStack s(cap);
    assert(s.empty());

    // Push elements
    for (int i = 0; i < cap; i++) {
        bool ok = s.push(i);
        assert(ok);
    }

    assert(s.size() == cap);
    assert(!s.push(999));  // Full, should fail

    // Pop all
    for (int i = 0; i < cap; i++) {
        s.pop();
    }
    assert(s.empty());

    return 0;
}
```

### Step 2: Verify

```
/verify stack_test.cpp
```

ESBMC verifies the class invariants, bounds checks, and assertions.

### C++ Tips
- Use `--std c++17` (or c++20) if you need specific standard features
- Use `--full-inlining` for template-heavy code
- STL containers work through ESBMC's operational models

## Part 5: Verifying Python Code

ESBMC can verify Python programs. All functions must have type annotations.

### Step 1: Write a Python program

Create `verify_sort.py`:

```python
from esbmc import nondet_int, assume, esbmc_assert

def is_sorted(arr: list[int]) -> bool:
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True

def insertion_sort(arr: list[int]) -> list[int]:
    result: list[int] = arr.copy()
    for i in range(1, len(result)):
        key: int = result[i]
        j: int = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result

def main() -> None:
    size: int = nondet_int()
    assume(size >= 0 and size <= 4)

    arr: list[int] = []
    for _ in range(size):
        val: int = nondet_int()
        assume(val > -50 and val < 50)
        arr.append(val)

    sorted_arr = insertion_sort(arr)

    esbmc_assert(is_sorted(sorted_arr), "Result is sorted")
    esbmc_assert(len(sorted_arr) == len(arr), "Length preserved")

if __name__ == "__main__":
    main()
```

### Step 2: Verify

```
/verify verify_sort.py
```

### Python Tips
- Type annotations are **required** on all function parameters and return types
- Import intrinsics from `esbmc`: `nondet_int`, `nondet_float`, `nondet_bool`, `assume`, `esbmc_assert`
- Use `--generate-pytest-testcase` to get a pytest from a counterexample
- Use `--nondet-str-length N` to control maximum symbolic string length
- Keep list sizes small (<=5) for feasible verification times

## Part 6: Interpreting Results

### VERIFICATION SUCCESSFUL
All properties hold within bounds. For bounded model checking, this means no bugs found up to the unwind limit. For k-induction, this is a full proof.

### VERIFICATION FAILED
A bug was found. The counterexample trace shows:
- Variable values at each step
- The execution path leading to the violation
- Which property failed and where

Ask Claude to help interpret: "Explain this ESBMC counterexample and suggest a fix."

### VERIFICATION UNKNOWN
Verification could not complete. Try:
- Reducing `--unwind` value
- Adding more `__ESBMC_assume` constraints
- Increasing `--timeout`
- Using `--incremental-bmc` instead of fixed bounds

## Part 7: Verification Strategies

Different verification goals call for different strategies:

| Goal | What to Do |
|------|-----------|
| Quick bug check | `/verify file.c` |
| Thorough audit | `/audit file.c` |
| Prove correctness | Ask: "Prove this function correct with k-induction" |
| Memory safety | `/verify file.c memory` |
| Concurrency safety | `/verify threaded.c concurrent` |
| All checks | `/verify file.c all` |

## Part 8: Working with Claude Code

The plugin enhances Claude Code's ability to help with verification tasks. Some useful prompts:

- "Verify this C file for memory safety"
- "Add ESBMC intrinsics to this function"
- "Run a security audit on src/parser.c"
- "Find potential buffer overflows in this code"
- "Prove this sorting algorithm correct"
- "Check this concurrent code for data races"
- "What does this ESBMC counterexample mean?"
- "Make this function verifiable with ESBMC"

Claude will automatically use the ESBMC skill when these topics come up, providing guidance on commands, intrinsics, and result interpretation.

## Quick Reference

### Commands
| Command | Purpose |
|---------|---------|
| `/verify <file>` | Quick verification |
| `/verify <file> memory` | With memory leak checks |
| `/verify <file> overflow` | With overflow checks |
| `/verify <file> concurrent` | With concurrency checks |
| `/verify <file> all` | All safety checks |
| `/audit <file>` | Full security audit |

### Common ESBMC Flags
| Flag | Purpose |
|------|---------|
| `--unwind N` | Set loop bound |
| `--timeout Ns` | Set time limit |
| `--memory-leak-check` | Check for leaks |
| `--overflow-check` | Check signed overflow |
| `--k-induction` | Prove unbounded |
| `--incremental-bmc` | Auto-increase bounds |
| `--multi-property` | Check all assertions |

### Intrinsics Cheat Sheet

**C/C++:**
```c
int x = __ESBMC_nondet_int();       // Symbolic input
__ESBMC_assume(x > 0 && x < 100);   // Constrain
__ESBMC_assert(result >= 0, "msg");  // Verify
```

**Python:**
```python
from esbmc import nondet_int, assume, esbmc_assert
x: int = nondet_int()
assume(x > 0 and x < 100)
esbmc_assert(result >= 0, "msg")
```
