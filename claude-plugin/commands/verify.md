---
name: verify
description: Verify a source file with ESBMC for bugs and safety violations
arguments:
  - name: file
    description: The source file to verify
    required: true
  - name: checks
    description: Additional checks to enable (memory, overflow, concurrent, all)
    required: false
---

# ESBMC Verification Command

Verify the specified source file using ESBMC bounded model checker.

## Instructions

1. Check that ESBMC is installed by running `which esbmc`. If not found, stop and tell the user:
   - Download pre-built binaries from https://github.com/esbmc/esbmc/releases
   - Or build from source: https://github.com/esbmc/esbmc
   - Ensure `esbmc` is in the PATH: `export PATH=$PATH:/path/to/esbmc/bin`

2. Determine the file type from the extension:
   - `.c` → C file
   - `.cpp`, `.cc`, `.cxx` → C++ file
   - `.py` → Python file (auto-detected by extension)
   - `.sol` → Solidity file (use `--sol`)
   - `.cu` → CUDA file

3. Build the ESBMC command based on the file type and requested checks:

   **Base command:**
   ```bash
   esbmc <file> --unwind 10 --timeout 60s
   ```

   **For Python** (no special flag needed, auto-detected by `.py` extension)**:**
   ```bash
   esbmc <file> --unwind 10 --timeout 60s
   ```

   **Additional checks based on `checks` argument:**
   - `memory` → add `--memory-leak-check`
   - `overflow` → add `--overflow-check --unsigned-overflow-check`
   - `concurrent` → add `--deadlock-check --data-races-check --context-bound 2`
   - `all` → add all of the above

4. Run the command using Bash tool

5. Interpret the results:
   - **VERIFICATION SUCCESSFUL** → All checked properties hold within bounds
   - **VERIFICATION FAILED** → Bug found, examine counterexample trace
   - **UNKNOWN/TIMEOUT** → Verification inconclusive

6. If verification fails, provide:
   - Summary of the violation type
   - Key parts of the counterexample trace
   - Suggestions for fixing the issue

## Examples

User: `/verify src/parser.c`
→ Run basic verification on C file

User: `/verify src/memory.c memory`
→ Run with memory leak checking

User: `/verify threaded.c concurrent`
→ Run with concurrency checks

User: `/verify contract.sol --contract MyContract`
→ Run Solidity verification
