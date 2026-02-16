# ESBMC Plugin for Claude Code

A comprehensive Claude Code plugin for ESBMC (Efficient SMT-based Context-Bounded Model Checker) integration. This plugin enables formal verification of C, C++, Python, Solidity, Java, and Kotlin programs directly within your Claude Code workflow.

**New to the plugin?** See the [Tutorial](TUTORIAL.md) for a step-by-step guide covering C, C++, and Python verification.

## What is ESBMC?

ESBMC is a software model checker that can detect bugs or prove their absence in programs. It works by:

1. Parsing source code into an internal representation
2. Converting to a GOTO intermediate program
3. Performing symbolic execution
4. Encoding the verification conditions as SMT formulas
5. Using SMT solvers to check satisfiability

If the formula is satisfiable, a bug exists (with a counterexample); if unsatisfiable, the property holds within the verification bounds.

## Installation

### Prerequisites

1. **ESBMC** must be installed on your system
   - Download from [ESBMC releases](https://github.com/esbmc/esbmc/releases)
   - Or build from source: https://github.com/esbmc/esbmc

2. **Claude Code** CLI installed

### Plugin Installation

From within Claude Code, add the marketplace and install the plugin:

```
/plugin marketplace add esbmc/esbmc
/plugin install esbmc-plugin@esbmc-marketplace
```

## Features

### 1. Verification Skill (`esbmc-verification`)

The core skill provides comprehensive guidance for using ESBMC. It triggers automatically when you mention:

- "verify code", "check for bugs", "find memory leaks"
- "detect buffer overflow", "check assertions"
- "verify C/C++ code", "verify Python code"
- "run ESBMC", "model check", "prove correctness"
- "find undefined behavior", "check for race conditions"
- "add verification intrinsics", "add ESBMC intrinsics"
- "annotate code for verification", "make code verifiable"
- "add preconditions", "add postconditions", "add symbolic inputs"

### 2. Commands

#### `/verify <file> [checks]`

Quick verification of a source file.

```bash
# Basic verification
/verify src/parser.c

# With memory leak checking
/verify src/memory.c memory

# With overflow checking
/verify src/math.c overflow

# With concurrency checking
/verify src/threaded.c concurrent

# With all checks
/verify src/critical.c all

# Solidity contract
/verify contracts/Token.sol --contract Token
```

#### `/audit <file>`

Comprehensive security audit with multiple verification passes:

- Quick scan for obvious bugs
- Memory safety analysis
- Integer overflow detection
- Concurrency safety (if applicable)
- Deep verification with higher bounds
- *k*-Induction proof attempts

```bash
/audit src/critical_module.c
```

### 3. Utility Scripts

Located in `skills/esbmc-verification/scripts/`:

#### `quick-verify.sh`

Fast verification wrapper with sensible defaults.

```bash
./quick-verify.sh program.c              # Basic verification
./quick-verify.sh program.c -m -o        # Memory + overflow checks
./quick-verify.sh program.c -a -t 5m     # All checks, 5min timeout
./quick-verify.sh contract.sol --contract MyContract
```

#### `full-audit.sh`

Comprehensive security audit script.

```bash
./full-audit.sh program.c                # Full audit
./full-audit.sh program.c -r report.txt  # Save report to file
./full-audit.sh program.c -t 5m          #5-minute timeout per check
```

## Supported Languages

| Language | Extension | Usage |
|----------|-----------|-------|
| C | `.c` | `esbmc file.c` |
| C++ | `.cpp`, `.cc` | `esbmc file.cpp` |
| Python | `.py` | `esbmc file.py` |
| Solidity | `.sol` | `esbmc --sol file.sol --contract Name` |
| CUDA | `.cu` | `esbmc file.cu` |
| Java/Kotlin | `.jimple` | `esbmc file.jimple` |

## Safety Properties

### User-Defined Safety Properties
Users can specify custom safety properties using `assert` statements:

- `assert(expression)`: The expression must always evaluate to true
- If an assertion can fail, ESBMC reports a verification failure and provides a counterexample

These allow users to verify functional correctness requirements, invariants, and application-specific safety conditions.

### Default Checks (Always On)
- Array bounds violations
- Division by zero
- Null pointer dereference
- User assertion failures

### Optional Checks
- `--overflow-check` - Signed integer overflow
- `--unsigned-overflow-check` - Unsigned overflow
- `--memory-leak-check` - Memory leaks
- `--deadlock-check` - Deadlock detection
- `--data-races-check` - Data race detection
- `--nan-check` - Floating-point NaN
- `--ub-shift-check` - Undefined shift behavior

## Verification Strategies

### Bounded Model Checking (Default)
```bash
esbmc file.c --unwind 10
```
Fast bug finding with fixed loop bounds.

### *k*-Induction (Unbounded Proofs)
```bash
esbmc file.c --k-induction
```
Proves properties hold for ALL executions.

### Incremental BMC
```bash
esbmc file.c --incremental-bmc
```
Iteratively increases bounds until a bug is found.

### Multi-Property
```bash
esbmc file.c --multi-property
```
Verifies all assertions in one run.

## ESBMC Intrinsics

Use these in your source code for verification:

### C/C++
```c
// Symbolic values
int x = __ESBMC_nondet_int();

// Assumptions (constrain inputs)
__ESBMC_assume(x > 0 && x < 100);

// Assertions (properties to verify)
__ESBMC_assert(result >= 0, "Result non-negative");

// Atomic sections
__ESBMC_atomic_begin();
// critical section
__ESBMC_atomic_end();
```

### Python
```python
x: int = nondet_int()
__ESBMC_assume(x > 0)
assert result >= 0, "Result non-negative"
```

## Examples

### Example 1: Memory Safety

```c
// safe_array.c
#include <stdlib.h>

int main() {
    int *arr = malloc(10 * sizeof(int));
    __ESBMC_assume(arr != NULL);

    int idx = __ESBMC_nondet_int();
    __ESBMC_assume(idx >= 0 && idx < 10);

    arr[idx] = 42;
    __ESBMC_assert(arr[idx] == 42, "Value stored correctly");

    free(arr);
    return 0;
}
```

```bash
esbmc safe_array.c --memory-leak-check
```

### Example 2: Integer Overflow

```c
// safe_add.c
#include <limits.h>

int safe_add(int a, int b) {
    __ESBMC_assume(a >= 0 && a <= 1000);
    __ESBMC_assume(b >= 0 && b <= 1000);

    int result = a + b;
    __ESBMC_assert(result >= 0, "No overflow");

    return result;
}
```

```bash
esbmc safe_add.c --overflow-check
```

### Example 3: Concurrency

```c
// safe_counter.c
#include <pthread.h>

int counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *increment(void *arg) {
    pthread_mutex_lock(&mutex);
    counter++;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    __ESBMC_assert(counter == 2, "Counter correct");
    return 0;
}
```

```bash
esbmc safe_counter.c --deadlock-check --data-races-check --context-bound 2
```

### Example 4: C++ Verification

```cpp
// safe_stack.cpp
#include <vector>
#include <cassert>

class SafeStack {
    std::vector<int> data;
    int max_size;
public:
    SafeStack(int max) : max_size(max) {}
    bool push(int val) {
        if ((int)data.size() >= max_size) return false;
        data.push_back(val);
        return true;
    }
    int pop() {
        assert(!data.empty());
        int val = data.back();
        data.pop_back();
        return val;
    }
    bool empty() const { return data.empty(); }
};

int main() {
    SafeStack s(3);
    s.push(10);
    s.push(20);
    assert(!s.empty());
    int val = s.pop();
    assert(val == 20);
    return 0;
}
```

```bash
esbmc safe_stack.cpp --unwind 10
```

### Example 5: Python Verification

```python
# verify_factorial.py
def factorial(n: int) -> int:
    assert n >= 0, "Input non-negative"

    if n <= 1:
        return 1
    return n * factorial(n - 1)

def main():
    n: int = nondet_int()
    __ESBMC_assume(n >= 0 and n <= 10)

    result = factorial(n)
    assert result > 0, "Factorial positive"

if __name__ == "__main__":
    main()
```

```bash
esbmc verify_factorial.py --unwind 12
```

## Use Cases in Development

### 1. Pre-Commit Verification
Add ESBMC checks to your pre-commit hooks:

```bash
#!/bin/bash
for file in $(git diff --cached --name-only | grep '\.c$'); do
    esbmc "$file" --unwind 10 --timeout 30s || exit 1
done
```

### 2. CI/CD Integration
Add to your GitHub Actions workflow:

```yaml
- name: ESBMC Verification
  run: |
    for file in src/*.c; do
      esbmc "$file" --unwind 10 --timeout 60s
    done
```

### 3. Security Audits
Regular security audits on critical code:

```bash
./full-audit.sh src/crypto/*.c -r security-report.txt
```

### 4. Bug Hunting
Find bugs in existing codebases:

```bash
esbmc legacy_code.c --incremental-bmc --multi-property
```

### 5. Proving Correctness
Prove algorithms correct for all inputs:

```bash
esbmc algorithm.c --k-induction --max-k-step 100
```

### 6. Smart Contract Verification
Verify Solidity contracts:

```bash
esbmc --sol Token.sol --contract Token --overflow-check --reentry-check
```

### 7. Concurrent Code Review
Verify multi-threaded code:

```bash
esbmc threaded.c --deadlock-check --data-races-check --context-bound 3
```

## Interpreting Results

### VERIFICATION SUCCESSFUL
All checked properties hold within the given bounds. For BMC, this means no bugs were found up to the unwind limit. For k-induction, this is a proof that holds for all executions.

### VERIFICATION FAILED
A property violation was found. Examine the counterexample trace:
- Variable values at each step
- Execution path leading to the violation
- The specific property that failed

### VERIFICATION UNKNOWN
The verification could not complete due to:
- Timeout
- Memory limit
- Solver limitations

Try increasing resources or simplifying the verification.

## Troubleshooting

### "Command not found: esbmc"
Ensure ESBMC is installed and in your PATH:
```bash
export PATH=$PATH:/path/to/esbmc/bin
```

### Verification Timeout
- Reduce `--unwind` value
- Add more `__ESBMC_assume` constraints
- Use `--incremental-bmc` for bug hunting
- Increase `--timeout` value

### Out of Memory
- Reduce `--unwind` value
- Enable slicing (default)
- Use `--memlimit` to set bounds

### Python Verification Errors
- Ensure all functions have type annotations
- Use Python 3.10+
- Import ESBMC intrinsics correctly

## Reference Documentation

The plugin includes comprehensive reference files:

- `references/cli-options.md` - Complete CLI reference
- `references/verification-strategies.md` - Strategy guide
- `references/language-specific.md` - Language-specific features
- `references/intrinsics.md` - ESBMC intrinsics reference
- `references/adding-intrinsics.md` - Step-by-step guide for annotating code
- `references/fixing-failures.md` - Diagnosing and fixing verification failures

### Example Files

- `examples/memory-check.c` - Memory safety verification (C)
- `examples/overflow-check.c` - Integer overflow detection (C)
- `examples/concurrent.c` - Concurrency verification (C)
- `examples/cpp-verify.cpp` - C++ verification (classes, STL, RAII, templates)
- `examples/python-verify.py` - Python verification

## Contributing

Contributions are welcome! Please submit issues and pull requests to help improve this plugin.

## License

MIT License - See LICENSE file for details.

## Resources

- [ESBMC GitHub](https://github.com/esbmc/esbmc)
- [ESBMC Documentation](https://esbmc.org/)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
