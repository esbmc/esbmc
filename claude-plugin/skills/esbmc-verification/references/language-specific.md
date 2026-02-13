# Language-Specific ESBMC Features

## C Verification

### Standard Selection

```bash
# C89/C90
esbmc file.c --std c89

# C99
esbmc file.c --std c99

# C11 (default)
esbmc file.c --std c11

# C17/C18
esbmc file.c --std c17
```

### Include Paths and Defines

```bash
# Add include path
esbmc file.c -I /path/to/includes

# Define macro
esbmc file.c -D DEBUG -D VERSION=2

# System root
esbmc file.c --sysroot /custom/sysroot
```

### Architecture Options

```bash
# 32-bit verification
esbmc file.c --32

# 64-bit verification (default)
esbmc file.c --64

# Endianness
esbmc file.c --little-endian
esbmc file.c --big-endian

# Unsigned char
esbmc file.c --funsigned-char
```

### C-Specific Checks

```bash
# Full memory safety
esbmc file.c --memory-leak-check

# VLA size checks
esbmc file.c --vla-size-check

# Printf format checks
esbmc file.c --printf-check

# Struct field bounds
esbmc file.c --struct-fields-check
```

## C++ Verification

### Standard Selection

```bash
# C++11
esbmc file.cpp --std c++11

# C++14
esbmc file.cpp --std c++14

# C++17
esbmc file.cpp --std c++17

# C++20
esbmc file.cpp --std c++20
```

### C++ Specific Options

```bash
# Full function inlining (helpful for templates)
esbmc file.cpp --full-inlining

# Disable abstracted C++ includes
esbmc file.cpp --no-abstracted-cpp-includes
```

### C++ Features Supported

- Classes and inheritance
- Templates (with limitations)
- STL containers (with operational models)
- Exceptions (partial support)
- Lambda expressions
- Smart pointers
- RAII patterns

### C++ Verification Example

```cpp
#include <vector>
#include <cassert>

int main() {
    std::vector<int> v;
    v.push_back(1);
    v.push_back(2);

    assert(v.size() == 2);
    assert(v[0] == 1);

    return 0;
}
```

```bash
esbmc example.cpp --unwind 10
```

## Python Verification

### Requirements

- Python 3.10+ installed
- Type annotations required for verification
- CPython parser used

### Basic Usage

```bash
# Verify Python file (auto-detected by .py extension)
esbmc script.py

# Specify Python interpreter (default: python from $PATH)
esbmc script.py --python /usr/bin/python3.11
```

### Python-Specific Options

```bash
# Strict type checking
esbmc script.py --strict-types

# Override return annotation with inferred type
esbmc script.py --override-return-annotation

# Set max length for nondet strings
esbmc script.py --nondet-str-length 32

# Generate pytest from counterexample
esbmc script.py --generate-pytest-testcase
```

### Python Type Annotations

Type annotations are essential for Python verification:

```python
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def find_max(lst: list[int]) -> int:
    assert len(lst) > 0
    max_val = lst[0]
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val
```

### Python Intrinsics

```python
from esbmc import nondet_int, assume, esbmc_assert

def verify_abs():
    x: int = nondet_int()
    assume(x >= -100 and x <= 100)

    result = abs(x)
    esbmc_assert(result >= 0, "abs returns non-negative")
```

### Python Verification Example

```python
# verify_search.py
def binary_search(arr: list[int], target: int) -> int:
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def main():
    arr = [1, 3, 5, 7, 9]
    idx = binary_search(arr, 5)
    assert idx == 2
```

```bash
esbmc verify_search.py --unwind 10
```

## Solidity Smart Contract Verification

### Basic Usage

```bash
# Verify Solidity contract
esbmc --sol contract.sol --contract MyContract

# With JSON AST
esbmc --sol contract.sol contract.solast --contract MyContract
```

### Solidity-Specific Options

```bash
# Check reentrancy vulnerabilities
esbmc --sol contract.sol --contract MyContract --reentry-check

# Model external calls as arbitrary (unknown behavior)
esbmc --sol contract.sol --contract MyContract --unbound

# Model inter-contract calls within bounded system
esbmc --sol contract.sol --contract MyContract --bound

# Verify internal/private functions too
esbmc --sol contract.sol --contract MyContract --no-visibility

# Negate property for specific function
esbmc --sol contract.sol --contract MyContract --negating-property transfer
```

### Solidity Verification Example

```solidity
// SimpleToken.sol
pragma solidity ^0.8.0;

contract SimpleToken {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;

    constructor(uint256 _initial) {
        balances[msg.sender] = _initial;
        totalSupply = _initial;
    }

    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;

        // Invariant: total supply unchanged
        assert(balances[msg.sender] + balances[to] <= totalSupply);
    }
}
```

```bash
esbmc --sol SimpleToken.sol --contract SimpleToken --overflow-check
```

### Common Solidity Checks

- Integer overflow/underflow
- Reentrancy vulnerabilities
- Access control violations
- Assertion failures
- Balance invariants

## Java/Kotlin Verification (Jimple)

### Requirements

- JDK 11+ installed
- Soot framework for Jimple conversion

### Basic Usage

```bash
# Verify Jimple file (auto-detected by .jimple extension)
# First convert .class to .jimple using Soot
esbmc MyClass.jimple

# Verify with classpath
esbmc MyClass.jimple -I /path/to/deps
```

### Java Verification Example

```java
// ArraySum.java
public class ArraySum {
    public static int sum(int[] arr) {
        int total = 0;
        for (int i = 0; i < arr.length; i++) {
            total += arr[i];
        }
        return total;
    }

    public static void main(String[] args) {
        int[] test = {1, 2, 3, 4, 5};
        int result = sum(test);
        assert result == 15;
    }
}
```

```bash
javac ArraySum.java
# Convert to Jimple using Soot, then verify
esbmc ArraySum.jimple --unwind 10
```

## CUDA Verification

### Basic Usage

```bash
# Verify CUDA file
esbmc kernel.cu --32
```

### CUDA-Specific Features

- Thread block modeling
- Shared memory verification
- Race condition detection
- Barrier synchronization

### CUDA Example

```cuda
// vector_add.cu
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

```bash
esbmc vector_add.cu --32 --unwind 5
```

## CHERI-C Verification

### What is CHERI

CHERI (Capability Hardware Enhanced RISC Instructions) provides hardware-enforced memory safety through capabilities.

### Basic Usage

```bash
# Hybrid mode (pointers and capabilities)
esbmc file.c --cheri hybrid

# Pure capability mode
esbmc file.c --cheri purecap

# Uncompressed capabilities
esbmc file.c --cheri purecap --cheri-uncompressed
```

### CHERI-Specific Checks

- Capability bounds violations
- Capability permission violations
- Capability provenance tracking

## Cross-Language Tips

### Choosing Loop Bounds

| Language | Typical Bounds | Notes |
|----------|----------------|-------|
| C/C++ | 10-50 | Depends on loop complexity |
| Python | 5-20 | Lists/iteration overhead |
| Solidity | 5-15 | Gas limits typically bound loops |
| Java | 10-30 | Similar to C++ |

### Memory Model Considerations

| Language | Memory Model |
|----------|--------------|
| C | Explicit allocation/deallocation |
| C++ | RAII, smart pointers |
| Python | Garbage collected |
| Solidity | Storage vs memory |
| Java | Garbage collected |

### Common Verification Patterns

```bash
# Quick check (all languages)
esbmc <file> --unwind 5 --timeout 30s

# Thorough check
esbmc <file> --unwind 20 --overflow-check --memory-leak-check

# Production verification
esbmc <file> --k-induction --multi-property --timeout 5m
```
