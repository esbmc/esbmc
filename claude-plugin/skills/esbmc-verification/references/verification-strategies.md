# ESBMC Verification Strategies Guide

## Overview

ESBMC provides multiple verification strategies optimized for different goals:
- **Bug finding**: Quickly find violations
- **Proof generation**: Prove properties hold for all executions
- **Coverage analysis**: Measure verification effectiveness

## Bounded Model Checking (BMC)

### How It Works

BMC unrolls loops a fixed number of times and converts the program to an SMT formula. If the formula is satisfiable, a bug exists; if unsatisfiable, no bug exists within those bounds.

### When to Use

- Initial bug hunting
- Programs with bounded loops
- When quick results are needed

### Configuration

```bash
# Basic BMC with loop bound
esbmc file.c --unwind 10

# Per-loop bounds (use --show-loops first)
esbmc file.c --show-loops
esbmc file.c --unwindset main.0:5,foo.1:10

# With unwinding assertions (default)
esbmc file.c --unwind 10
# Fails if loop executes more than bound

# Without unwinding assertions
esbmc file.c --unwind 10 --no-unwinding-assertions
# May miss bugs beyond bound
```

### Limitations

- Cannot prove unbounded properties
- May miss bugs beyond the unwind bound
- Requires choosing appropriate bounds

## K-Induction

### How It Works

K-induction proves properties hold for ALL executions by:
1. **Base case**: Property holds for k steps
2. **Forward condition**: No state repetition in k steps (optional)
3. **Inductive step**: If property holds for k steps, it holds for k+1

### When to Use

- Proving unbounded safety properties
- Programs with loops of unknown bounds
- When complete verification is required

### Configuration

```bash
# Basic k-induction
esbmc file.c --k-induction

# With custom parameters
esbmc file.c --k-induction --max-k-step 100 --k-step 2

# Run steps in parallel (faster on multi-core)
esbmc file.c --k-induction-parallel

# Start from specific k value
esbmc file.c --k-induction --base-k-step 5

# Individual steps (for debugging)
esbmc file.c --base-case --unwind 10
esbmc file.c --forward-condition --unwind 10
esbmc file.c --inductive-step --unwind 10
```

### Contractor Refinement

Enable interval-based refinement for better precision:

```bash
esbmc file.c --k-induction --goto-contractor
esbmc file.c --k-induction --goto-contractor-condition
```

### Interpreting Results

| Result | Meaning |
|--------|---------|
| VERIFICATION SUCCESSFUL | Property proved for all executions |
| VERIFICATION FAILED (base) | Bug found in first k executions |
| VERIFICATION FAILED (inductive) | Potential bug (may be spurious) |
| Max k-step reached | Inconclusive |

## Incremental BMC

### How It Works

Incrementally increases loop bounds from 1 upward, checking at each level. Stops when bug found or limit reached.

### When to Use

- Bug hunting with unknown loop bounds
- Finding the minimal counterexample
- When you don't know appropriate bounds

### Configuration

```bash
# Basic incremental BMC
esbmc file.c --incremental-bmc

# Bug-finding only (base case check, no forward condition / no safety proof)
esbmc file.c --falsification

# Termination proof focus
esbmc file.c --termination
```

### Advantages

- Finds bugs at minimal depth
- No need to guess bounds
- Good counterexample quality

## Multi-Property Verification

### How It Works

Verifies all assertions/properties in a single verification run, rather than stopping at the first failure.

### When to Use

- Comprehensive verification
- Understanding full violation landscape
- Regression testing

### Configuration

```bash
# Check all properties
esbmc file.c --multi-property

# Stop after N failures
esbmc file.c --multi-property --multi-fail-fast 5

# Parallel solving (faster)
esbmc file.c --parallel-solving

# Keep checking even after some verified
esbmc file.c --multi-property --keep-verified-claims
```

## Incremental SMT (Early Termination)

### How It Works

Performs SMT solving incrementally during symbolic execution, potentially finding bugs before full formula generation.

### When to Use

- Large programs with early bugs
- Reducing memory usage
- Interactive exploration

### Configuration

```bash
# Enable incremental SMT
esbmc file.c --smt-during-symex

# Check guards during exploration
esbmc file.c --smt-during-symex --smt-symex-guard

# Check assertions immediately
esbmc file.c --smt-during-symex --smt-symex-assert

# Full incremental checking
esbmc file.c --smt-during-symex --smt-symex-guard --smt-symex-assert --smt-symex-assume
```

## Interval Analysis

### How It Works

Static analysis that computes numeric bounds for variables, used to:
- Simplify verification conditions
- Prove some properties without SMT
- Reduce search space

### When to Use

- Numeric-heavy code
- Performance optimization
- Pre-filtering trivial cases

### Configuration

```bash
# Basic interval analysis
esbmc file.c --interval-analysis

# With arithmetic operations
esbmc file.c --interval-analysis --interval-analysis-arithmetic

# Dump computed intervals
esbmc file.c --interval-analysis-dump
esbmc file.c --interval-analysis-csv-dump intervals.csv

# Advanced options
esbmc file.c --interval-analysis \
  --interval-analysis-arithmetic \
  --interval-analysis-bitwise \
  --interval-analysis-simplify \
  --interval-analysis-narrowing
```

## Concurrency Verification

### How It Works

Explores thread interleavings systematically with optimizations like partial order reduction and state hashing.

### When to Use

- Multi-threaded programs
- Pthread-based code
- Finding race conditions and deadlocks

### Configuration

```bash
# Basic concurrency checks
esbmc threaded.c --deadlock-check --data-races-check

# Limit context switches
esbmc threaded.c --context-bound 3

# Enable state hashing (prune duplicates)
esbmc threaded.c --state-hashing

# Disable partial order reduction (more thorough)
esbmc threaded.c --no-por

# Check all interleavings
esbmc threaded.c --all-runs
```

### Specific Checks

```bash
# Data races only
esbmc threaded.c --data-races-check-only

# Lock ordering violations
esbmc threaded.c --lock-order-check

# Atomicity violations
esbmc threaded.c --atomicity-check
```

## Code Coverage Analysis

### How It Works

Instruments code to track which parts are covered during verification.

### When to Use

- Measuring verification quality
- Identifying unreachable code
- Test suite evaluation

### Configuration

```bash
# Assertion coverage
esbmc file.c --assertion-coverage

# Condition coverage
esbmc file.c --condition-coverage

# Branch coverage
esbmc file.c --branch-coverage

# Branch + function coverage
esbmc file.c --branch-function-coverage

# With claims listing
esbmc file.c --branch-coverage-claims
```

## Function Contracts

### How It Works

Uses function contracts (pre/postconditions) to modularize verification:
- **Enforce**: Verify function meets its contract
- **Replace**: Use contract instead of function body

### When to Use

- Large codebases
- Modular verification
- Library functions with specifications

### Configuration

```bash
# Check all function contracts
esbmc file.c --enforce-contract "*"

# Check specific function
esbmc file.c --enforce-contract "myfunction"

# Replace calls with contracts
esbmc file.c --replace-call-with-contract "library_func"

# Combine both
esbmc file.c --enforce-contract "impl" --replace-call-with-contract "helper"
```

## Strategy Selection Guide

| Goal | Strategy | Command |
|------|----------|---------|
| Quick bug finding | BMC | `--unwind 10` |
| Unknown loop bounds | Incremental BMC | `--incremental-bmc` |
| Prove correctness | K-induction | `--k-induction` |
| All violations | Multi-property | `--multi-property` |
| Large programs | Incremental SMT | `--smt-during-symex` |
| Concurrent code | Context-bounded | `--context-bound 3` |
| Modular verification | Contracts | `--enforce-contract` |

## Combining Strategies

```bash
# Thorough verification
esbmc file.c --k-induction --multi-property --overflow-check

# Fast bug hunting
esbmc file.c --incremental-bmc --smt-during-symex --timeout 60s

# Concurrent program proof
esbmc file.c --k-induction --deadlock-check --context-bound 2

# Coverage-guided verification
esbmc file.c --branch-coverage --multi-property --unwind 20
```

## Performance Tips

1. **Start with small bounds** and increase as needed
2. **Use incremental BMC** when unsure about bounds
3. **Enable parallel solving** on multi-core machines
4. **Use interval analysis** for numeric-heavy code
5. **Apply slicing** to remove irrelevant code: default on
6. **Set timeouts** to avoid runaway verification
7. **Use `--quiet`** to reduce I/O overhead
