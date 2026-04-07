# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESBMC (Efficient SMT-based Context-Bounded Model Checker) is a software model checker that detects bugs or proves their absence in C, C++, CUDA, CHERI-C, Python, Solidity, Java, and Kotlin programs. It works by parsing source → building AST → converting to GOTO program → symbolic execution (SSA) → encoding as SMT formula → solving with SMT solvers.

## Build Commands

**NEVER run cmake in the repo root (e.g., `cmake .` or `cmake -B. -H.`).** Always use `build/` or a subdirectory of it as the build directory (e.g., `-Bbuild`). The `.gitignore` only covers `build/` — in-tree builds pollute the source tree with hundreds of untracked artifacts.

```sh
# Minimal build with Z3 solver (at least one solver must be enabled for regression tests)
cmake -GNinja -Bbuild -H. \
  -DDOWNLOAD_DEPENDENCIES=On \
  -DENABLE_PYTHON_FRONTEND=On \
  -DENABLE_Z3=On \
  -DBUILD_TESTING=On \
  -DENABLE_REGRESSION=On \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build (778 targets, uses Ninja)
ninja -C build

# Install
ninja -C build install
```

Additional optional CMake flags:
- `-DENABLE_SOLIDITY_FRONTEND=On` — Solidity smart contract frontend
- `-DENABLE_JIMPLE_FRONTEND=On` — Java/Kotlin frontend (requires JDK 11+)
- `-DENABLE_BITWUZLA=On` — Bitwuzla solver backend
- `-DENABLE_BOOLECTOR=On` — Boolector solver backend
- Quality: `-DENABLE_WERROR=On`, `-DENABLE_CLANG_TIDY=On`, `-DENABLE_COVERAGE=On`

See `scripts/build.sh` for full platform-specific dependency setup and solver configuration.

Requires: CMake 3.18+, Ninja, Boost (date_time, program_options, iostreams, system, filesystem), LLVM 11-21+, Bison, Flex, Z3 (or another SMT solver).

## Testing

There are 236 unit tests and ~7200+ regression tests. Regression tests require at least one solver backend (e.g., Z3). All commands run from the `build/` directory.

```sh
# Run unit tests only (fast, ~4 seconds, tests 1-236)
ctest -j$(nproc) -I 1,236 --timeout 60

# Run all regression tests (slow, creates temp dirs in /tmp — see note below)
ctest -j$(nproc) -L regression --timeout 120

# Run a specific regression suite by label
ctest -j$(nproc) -L esbmc --timeout 120          # core C tests (~1046 tests)
ctest -j$(nproc) -L python --timeout 120          # Python tests (~2186 tests)
ctest -j$(nproc) -L "esbmc-cpp/cpp" --timeout 120 # C++ tests (~420 tests)
ctest -j$(nproc) -L floats --timeout 120          # floating-point tests (~141 tests)

# List all available test labels
ctest --print-labels

# Run a single named test
ctest -R "regression/esbmc/00_big_endian_01" --output-on-failure
```

**Important: Python frontend tests require `ast2json`.** ESBMC's Python frontend invokes `python3` from `PATH` to run `parser.py`, which imports `ast2json`. For Python regression tests to pass, either activate the uv venv first (`source .venv/bin/activate`) or ensure `ast2json` is installed in the system Python.

**Important: /tmp disk space.** Each regression test creates an `esbmc-headers-*` temp directory (~7.4MB) in `/tmp`. Running the full suite generates thousands of these (~70GB total). Clean them after test runs: `rm -rf /tmp/esbmc-headers-*`

Regression test format (`test.desc`): line 1 is `CORE`/`KNOWNBUG`/`FUTURE`/`THOROUGH` (THOROUGH is Linux-only), line 2 is the source file, line 3 is ESBMC flags, line 4+ are expected output regexes. Every PR should include at least two regression tests (one passing, one failing).

## Code Style

- **C++**: Clang-format (Clang 11), Allman braces, 80-col limit, 2-space indent, no tabs. Config in `.clang-format`.
- **Python**: YAPF, PEP 8 based, 100-col limit. Config in `.style.yapf`.
- Prefer modern C++ idioms (C++11+). Use const-correctness throughout. Prefer stack allocation over heap when possible. Follow existing patterns in the file being modified.
- CI enforces formatting on PRs via GitHub Actions.

## Coding Guidelines

- Write simple, clean, and readable code with minimal indirection.
- Each function should do one thing well. No redundant abstractions or duplicate code.
- Check the entire codebase to reuse existing methods before writing new ones.
- Tests MUST NOT use mocks, patches, or any form of test doubles. Integration tests are preferred.
- After implementation, simplify and clean up the code aggressively — remove unnecessary conditional checks while ensuring correctness.
- Run ESBMC over your solution to formally check that it works and does not introduce new errors.

## Code Review Priorities

1. **Critical**: Verification soundness, memory safety, undefined behavior
2. **High**: Logic errors in SMT encoding/symbolic execution, performance regressions, missing tests
3. **Medium**: Code quality, API consistency, documentation gaps
4. **Low**: Minor style if matching surrounding code

## Source Architecture

Key directories under `src/`:

- `esbmc/` — Main entry point and CLI driver
- `irep2/` — Internal representation (IRep2), the core data structure for expressions/types
- `goto-programs/` — GOTO intermediate representation and transformations
- `goto-symex/` — Symbolic execution engine (core verification logic)
- `solvers/` — SMT solver backends (z3, bitwuzla, boolector, cvc4, cvc5, yices, mathsat, smtlib)
- `langapi/` — Language API abstractions shared across frontends
- `pointer-analysis/` — Memory model and pointer safety analysis
- `util/` — Shared utilities and data structures

Frontends (each parses a language into the shared GOTO representation):
- `clang-c-frontend/` — C, CHERI-C, CUDA (via Clang)
- `clang-cpp-frontend/` — C++ (via Clang)
- `python-frontend/` — Python 3.10+ (AST→JSON→IRep2)
- `jimple-frontend/` — Java/Kotlin (via Soot/Jimple)
- `solidity-frontend/` — Solidity smart contracts

Tools:
- `c2goto/` — Converts C operational models to GOTO binaries
- `goto2c/` — Converts GOTO programs back to C

Other top-level directories:
- `unit/` — GoogleTest unit tests
- `regression/` — regression test suites (60+ categories)
- `scripts/` — build scripts and CMake modules (`scripts/cmake/`)
- `docs/` — generated documentation
- `website/` — Hugo-based project website

## Debugging Verification Issues

When ESBMC produces an unexpected VERIFICATION FAILED or SUCCESSFUL result, use these techniques:

**1. Inspect the GOTO program** — Use `--goto-functions-only` to dump the intermediate GOTO representation. This reveals exactly what code ESBMC is verifying, including how frontend constructs are lowered:
```sh
esbmc test.py --unwind 9 --goto-functions-only 2>&1 | grep -A50 "python_user_main"
```
Look for the `python_user_main` function to see how Python source maps to GOTO instructions (ASSIGN, FUNCTION_CALL, ASSERT). This is especially useful for catching compile-time optimizations that incorrectly pre-resolve values.

**2. Bisect with simpler test cases** — When a test fails, create variants that isolate the problem.

**3. Read the counterexample trace** — ESBMC's `[Counterexample]` section shows the state at each step. Track field assignments in structs (e.g., `PyObject`'s `.value`, `.type_id`, `.size`) through the trace.

**4. Key files for Python frontend debugging:**
- `src/python-frontend/python_converter.cpp` — Main expression/statement conversion
- `src/python-frontend/python_list.cpp` — List operations
- `src/python-frontend/function_call_expr.cpp` — Method call handling
- `src/c2goto/library/python/list.c` — C operational model for list operations

**5. Hypothesis tests** — Property-based tests in `tests/python-frontend/` test ESBMC's models against CPython. Run with: `uv run python -m pytest tests/python-frontend/ -v`

## Commit Conventions

Prefix commits with a category tag in brackets, e.g., `[python]`, `[build]`, `[solver]`, `[om]` (operational model). Title: one line, imperative mood, <72 chars. Description: 2–4 lines explaining what changed and why.

## PR Conventions

- Branch from `master` (the default branch)
- Target PRs to `master`
- Check formatting with clang-format before submitting

For module-specific instructions, subdirectory CLAUDE.md files can be added (they load automatically when working in those directories).
