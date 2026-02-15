---
name: audit
description: Perform a comprehensive security audit on a source file using ESBMC
arguments:
  - name: file
    description: The source file to audit
    required: true
---

# ESBMC Security Audit Command

Perform a comprehensive security audit using multiple ESBMC verification passes.

## Instructions

1. Check that ESBMC is installed by running `which esbmc`. If not found, stop and tell the user:
   - Download pre-built binaries from https://github.com/esbmc/esbmc/releases
   - Or build from source: https://github.com/esbmc/esbmc
   - Ensure `esbmc` is in the PATH: `export PATH=$PATH:/path/to/esbmc/bin`

2. Read the source file to understand its structure and detect:
   - Language (C, C++, Python, Solidity)
   - Presence of concurrency (pthreads, std::thread, threading module)
   - Complexity indicators (loops, recursion, dynamic allocation)

3. Run multiple verification passes:

   **Pass 1: Quick Scan (30s timeout, unwind 5)**
   ```bash
   esbmc <file> --unwind 5 --timeout 30s
   ```

   **Pass 2: Memory Safety (60s timeout, unwind 10)**
   ```bash
   esbmc <file> --memory-leak-check --unwind 10 --timeout 60s
   ```

   **Pass 3: Integer Safety (60s timeout, unwind 10)**
   ```bash
   esbmc <file> --overflow-check --unsigned-overflow-check --unwind 10 --timeout 60s
   ```

   **Pass 4: Concurrency (if applicable) (60s timeout)**
   ```bash
   esbmc <file> --deadlock-check --data-races-check --context-bound 2 --unwind 10 --timeout 60s
   ```

   **Pass 5: Deep Verification (120s timeout, unwind 30)**
   ```bash
   esbmc <file> --memory-leak-check --overflow-check --unwind 30 --timeout 120s
   ```

   **Pass 6: K-Induction Proof Attempt (120s timeout)**
   ```bash
   esbmc <file> --k-induction --max-k-step 20 --timeout 120s
   ```

4. Compile results into a summary report:
   - Number of passes completed
   - Issues found per category
   - Recommendations for fixes

5. Present the audit results in a clear, structured format

## Output Format

```
ESBMC Security Audit Report
===========================
File: <filename>
Language: <language>
Concurrency: <yes/no>

Results:
--------
[Pass 1: Quick Scan]
  ✓ PASSED / ✗ FAILED: <details>

[Pass 2: Memory Safety]
  ✓ PASSED / ✗ FAILED: <details>

...

Summary:
--------
Total Issues: N
- Memory: X issues
- Integer: Y issues
- Concurrency: Z issues

Recommendations:
----------------
1. <recommendation>
2. <recommendation>
```
