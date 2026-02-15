# Fixing Verification Failures

## Interpreting Results

### VERIFICATION SUCCESSFUL
All checked properties hold within the given bounds. For BMC, no bugs found up to the unwind limit. For k-induction, a proof that holds for all executions.

### VERIFICATION FAILED
A property violation was found. Examine the counterexample trace showing:
- Variable values at each step
- Execution path leading to the violation
- The specific property that failed

### VERIFICATION UNKNOWN
Verification could not complete (timeout, resource limit, or solver limitation).

## Failure Diagnosis Process

### 1. Read the Counterexample
The trace shows variable values at each step leading to the violation.

### 2. Identify the Root Cause
- **Missing assumption**: Input was not properly constrained
- **Actual bug**: Code has a real defect
- **Incomplete specification**: Assertion is too strict

### 3. Apply the Fix
- **If missing assumption**: Add `__ESBMC_assume()` to constrain inputs
- **If actual bug**: Fix the code logic
- **If incomplete spec**: Adjust or remove the assertion

### 4. Re-verify
Run ESBMC again to confirm the fix works.

## Common Failure Patterns

| Failure | Likely Cause | Fix |
|---------|--------------|-----|
| Array bounds | Unconstrained index | Add `__ESBMC_assume(idx >= 0 && idx < size)` |
| Null pointer | Missing null check | Add `if (ptr != NULL)` or assume |
| Division by zero | Unconstrained divisor | Add `__ESBMC_assume(divisor != 0)` |
| Overflow | Large input values | Constrain inputs to safe ranges |
| Assertion failed | Bug or wrong spec | Fix code or adjust assertion |
| Memory leak | Missing free | Add `free(ptr)` on all paths |
| Deadlock | Lock ordering | Acquire locks in consistent order |
| Data race | Missing synchronization | Add mutex or atomic sections |

## Troubleshooting Common Issues

### Verification Timeout
- Reduce `--unwind` value
- Add more `__ESBMC_assume` constraints to prune the search space
- Use `--incremental-bmc` for bug hunting
- Increase `--timeout` value
- Enable `--quiet` to reduce I/O overhead

### Out of Memory
- Reduce `--unwind` value
- Enable slicing (on by default)
- Use `--memlimit` to set bounds
- Try a different solver (some are more memory-efficient)

### Python-Specific Errors
- Ensure all functions have type annotations
- Use Python 3.10+
- Import ESBMC intrinsics correctly: `from esbmc import nondet_int, assume, esbmc_assert`
- Use `--strict-types` for stricter type checking

### C++-Specific Errors
- Use `--full-inlining` for template-heavy code
- Set appropriate C++ standard with `--std c++17`
- Use `--no-abstracted-cpp-includes` if operational model issues arise

### "Command not found: esbmc"
Ensure ESBMC is installed and in the PATH:
```bash
export PATH=$PATH:/path/to/esbmc/bin
```

### Spurious Failures
If k-induction reports a failure in the inductive step, it may be spurious. Confirm with BMC at higher bounds:
```bash
esbmc file.c --unwind 50 --timeout 5m
```
