---
name: esbmc-verification
description: This skill should be used when the user asks to "verify code", "run ESBMC", "model check", "check for bugs", "find memory leaks", "detect buffer overflow", "find undefined behavior", "check for race conditions", "detect deadlocks", "prove correctness", "add verification intrinsics", "add nondet values", "add preconditions", "make code verifiable", or mentions bounded model checking, SMT solving, formal methods, or safety properties. Provides guidance for verifying C, C++, Python, Solidity, CUDA, and Java/Kotlin programs with ESBMC and adding verification annotations.
version: 1.0.0
---

# ESBMC Verification Skill

ESBMC (Efficient SMT-based Context-Bounded Model Checker) detects bugs or proves their absence in C, C++, CUDA, CHERI-C, Python, Solidity, Java, and Kotlin programs.

## Prerequisites Check

Before running any ESBMC command, verify that ESBMC is installed by running `which esbmc` or `esbmc --version`. If ESBMC is not found, inform the user and provide installation instructions:

- **Pre-built binaries**: Download from [ESBMC releases](https://github.com/esbmc/esbmc/releases)
- **Build from source**: Clone https://github.com/esbmc/esbmc and follow the build instructions in its README
- **After installing**, ensure `esbmc` is in the PATH: `export PATH=$PATH:/path/to/esbmc/bin`

Do not proceed with verification commands until ESBMC is confirmed available.

## Verification Pipeline

Source Code → Frontend Parser → GOTO Program → Symbolic Execution (SSA) → SMT Formula → Solver → Result

## Quick Start

```bash
# C file with default checks
esbmc file.c

# C++ file
esbmc file.cpp

# Python (requires type annotations)
esbmc file.py

# Solidity contract
esbmc --sol contract.sol --contract ContractName

# Common safety checks
esbmc file.c --memory-leak-check --overflow-check --unwind 10 --timeout 60s

# Concurrency verification
esbmc file.c --deadlock-check --data-races-check --context-bound 2
```

## Supported Languages

| Language | Command | Notes |
|----------|---------|-------|
| C | `esbmc file.c` | Default, Clang frontend |
| C++ | `esbmc file.cpp` | C++11-20 via `--std` |
| Python | `esbmc file.py` | Requires type annotations, Python 3.10+ |
| Solidity | `esbmc --sol file.sol --contract Name` | Smart contracts |
| CUDA | `esbmc file.cu` | GPU kernel verification |
| Java/Kotlin | `esbmc file.jimple` | Requires Soot conversion from .class |

For language-specific options and examples, see `references/language-specific.md`.

## Safety Properties

### Default Checks (Always On)
- **Array bounds**: Out-of-bounds array access
- **Division by zero**: Integer and floating-point
- **Pointer safety**: Null dereference, invalid pointer arithmetic
- **Assertions**: User-defined `assert()` statements

### Optional Checks (Enable Explicitly)
- `--overflow-check` / `--unsigned-overflow-check`: Integer overflow
- `--memory-leak-check`: Memory leaks
- `--deadlock-check` / `--data-races-check`: Concurrency safety
- `--nan-check`: Floating-point NaN
- `--ub-shift-check`: Undefined shift behavior

### Disable Specific Checks
`--no-bounds-check`, `--no-pointer-check`, `--no-div-by-zero-check`, `--no-assertions`

For the full CLI reference, see `references/cli-options.md`.

## Loop Handling

```bash
# Unwind all loops N times
esbmc file.c --unwind 10

# Per-loop bounds (use --show-loops to find loop IDs)
esbmc file.c --show-loops
esbmc file.c --unwindset L1:5,L2:10

# Incremental unwinding (find bugs faster)
esbmc file.c --incremental-bmc

# K-induction for unbounded verification
esbmc file.c --k-induction
```

## Verification Strategies

| Goal | Strategy | Command |
|------|----------|---------|
| Quick bug finding | BMC | `--unwind 10` |
| Unknown loop bounds | Incremental BMC | `--incremental-bmc` |
| Prove correctness | K-induction | `--k-induction` |
| All violations | Multi-property | `--multi-property` |
| Large programs | Incremental SMT | `--smt-during-symex` |
| Concurrent code | Context-bounded | `--context-bound 3` |

For detailed strategy descriptions and configuration, see `references/verification-strategies.md`.

## Solver Selection

```bash
esbmc --list-solvers   # List available solvers
esbmc file.c --z3      # Z3 (default)
esbmc file.c --bitwuzla # Fast bit-vectors
esbmc file.c --boolector # Efficient bit-vectors
```

## ESBMC Intrinsics

Use these in source code to guide verification.

### Quick Reference

| Purpose | C/C++ | Python |
|---------|-------|--------|
| Symbolic int | `__ESBMC_nondet_int()` | `nondet_int()` |
| Symbolic uint | `__ESBMC_nondet_uint()` | N/A |
| Symbolic bool | `__ESBMC_nondet_bool()` | `nondet_bool()` |
| Symbolic float | `__ESBMC_nondet_float()` | `nondet_float()` |
| Assumption | `__ESBMC_assume(cond)` | `assume(cond)` |
| Assertion | `__ESBMC_assert(cond, msg)` | `esbmc_assert(cond, msg)` |
| Atomic section | `__ESBMC_atomic_begin/end()` | N/A |

### Basic Usage

```c
int x = __ESBMC_nondet_int();       // Symbolic input
__ESBMC_assume(x > 0 && x < 100);   // Constrain input
__ESBMC_assert(result >= 0, "msg");  // Verify property
```

```python
from esbmc import nondet_int, assume, esbmc_assert
x: int = nondet_int()
assume(x > 0 and x < 100)
esbmc_assert(result >= 0, "msg")
```

For the step-by-step process of adding intrinsics to code, see `references/adding-intrinsics.md`.
For the full intrinsics API, see `references/intrinsics.md`.

## Output and Debugging

```bash
esbmc file.c                          # Counterexample shown on failure
esbmc file.c --witness-output w.graphml # Generate witness file
esbmc file.c --show-vcc               # Show verification conditions
esbmc file.c --generate-testcase      # Generate test from counterexample
esbmc file.py --generate-pytest-testcase            # Generate pytest
```

## Resource Limits

```bash
esbmc file.c --timeout 300s   # Time limit
esbmc file.c --memlimit 4g    # Memory limit
```

## Common Workflows

### Bug Hunting
```bash
esbmc file.c --unwind 5 --timeout 60s          # Quick scan
esbmc file.c --incremental-bmc --timeout 120s   # If timeout
```

### Proving Correctness
```bash
esbmc file.c --k-induction --overflow-check
esbmc file.c --unwind 100 --multi-property      # Thorough bounded check
```

### Memory Safety Audit
```bash
esbmc file.c --memory-leak-check --unwind 10
```

### Concurrency Verification
```bash
esbmc threaded.c --deadlock-check --data-races-check --context-bound 2
```

For guidance on fixing verification failures, see `references/fixing-failures.md`.

## Additional Resources

### Reference Files
- **`references/cli-options.md`** — Complete CLI reference
- **`references/verification-strategies.md`** — Detailed strategy guide
- **`references/language-specific.md`** — Language-specific features and options
- **`references/intrinsics.md`** — Full ESBMC intrinsics API
- **`references/adding-intrinsics.md`** — Step-by-step guide for annotating code
- **`references/fixing-failures.md`** — Diagnosing and fixing verification failures

### Example Files
- **`examples/memory-check.c`** — Memory safety verification (C)
- **`examples/overflow-check.c`** — Integer overflow detection (C)
- **`examples/concurrent.c`** — Concurrency verification (C)
- **`examples/cpp-verify.cpp`** — C++ verification (classes, STL, RAII)
- **`examples/python-verify.py`** — Python verification

### Scripts
- **`scripts/quick-verify.sh`** — Quick verification wrapper
- **`scripts/full-audit.sh`** — Comprehensive security audit
