# AGENTS.md

This file provides guidance to coding agents working with this repository. The same workflow rules also live in `CLAUDE.md` (which Claude Code loads automatically); update both files together when changing build, test, style, or post-implementation rules.

## Project Overview

ESBMC (Efficient SMT-based Context-Bounded Model Checker) is a software model checker that detects bugs or proves their absence in C, C++, CUDA, CHERI-C, Python, Solidity, Java, and Kotlin programs. It works by parsing source → building AST → converting to GOTO program → symbolic execution (SSA) → encoding as SMT formula → solving with SMT solvers.

## Build Commands

**NEVER run cmake in the repo root (e.g., `cmake .` or `cmake -B. -S.`).** Always use `build/` or a subdirectory of it as the build directory (e.g., `-Bbuild`). The `.gitignore` only covers `build/` — in-tree builds pollute the source tree with hundreds of untracked artifacts.

```sh
# Minimal build with Z3 solver (at least one solver must be enabled for regression tests)
cmake -GNinja -Bbuild -S . \
  -DDOWNLOAD_DEPENDENCIES=On \
  -DENABLE_PYTHON_FRONTEND=On \
  -DENABLE_Z3=On \
  -DBUILD_TESTING=On \
  -DENABLE_REGRESSION=On \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build (uses Ninja)
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

Requires: CMake 3.18+, Ninja, Boost (date_time, program_options, iostreams, system, filesystem), LLVM 11+ (tested up to 21), Bison, Flex, Z3 (or another SMT solver).

## Testing

Regression tests require at least one solver backend (e.g., Z3). All commands run from the `build/` directory.

```sh
# Run unit tests only (fast, excludes regression-labeled tests)
ctest -j$(nproc) -LE regression --timeout 60

# Run all regression tests (slow, creates temp dirs in /tmp — see note below)
ctest -j$(nproc) -L regression --timeout 120

# Run a specific regression suite by label
ctest -j$(nproc) -L esbmc --timeout 120          # core C tests
ctest -j$(nproc) -L python --timeout 120          # Python tests
ctest -j$(nproc) -L "esbmc-cpp/cpp" --timeout 120 # C++ tests
ctest -j$(nproc) -L floats --timeout 120          # floating-point tests

# List all available test labels
ctest --print-labels

# Run a single named test
ctest -R "regression/esbmc/00_big_endian_01" --output-on-failure
```

**Important: Python frontend tests require `ast2json`.** ESBMC's Python frontend invokes `python3` from `PATH` to run `parser.py`, which imports `ast2json`. For Python regression tests to pass, either activate the uv venv first (`source .venv/bin/activate`) or ensure `ast2json` is installed in the system Python.

**Important: /tmp disk space.** Each regression test creates an `esbmc-headers-*` temp directory (~7.4MB) in `/tmp`. Running the full suite generates thousands of these (~70GB total). Clean them after test runs: `rm -rf /tmp/esbmc-headers-*`

Regression test format (`test.desc`): line 1 is `CORE`/`KNOWNBUG`/`FUTURE`/`THOROUGH` (THOROUGH is Linux-only), line 2 is the source file, line 3 is ESBMC flags, line 4+ are expected output regexes. Every PR should include at least two regression tests (one passing, one failing).

**Before committing:**

- Always run the project's test suite. If tests fail, fix the failures before committing — never commit broken or untested code.
- **Regression suite cap.** When running the full regression suite, cap the run at **5 minutes** (300000 ms) — pass the timeout to the `Bash` tool's `timeout` parameter, or wrap the invocation with `timeout 5m …`. If the suite cannot complete in 5 minutes, narrow the scope (e.g. run only the affected subset) or ask the user before extending the limit.
- **Lint and typecheck.** Run lint and typecheckers and fix any errors. For Python code, use `pylint`. For C++ code, ensure clang-format compliance (CI enforces this).

## Branching

Before implementing any feature or bug fix, always work on a dedicated branch:

1. Check the current branch — never work directly on `master`.
2. Create a branch with a descriptive name (e.g. `feat/short-description` or `fix/short-description`).
3. Confirm the branch is active before making any changes.

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

## Post-implementation Pass

After implementing any non-trivial coding task, before committing:

1. **Simplify aggressively.** Remove unnecessary conditional checks, dead code, redundant abstractions, duplicate logic. Re-verify the code still works correctly. Apply the same pass to test code.
2. **Verify with ESBMC** when the task touches C/C++ code or ESBMC's own headers/frontends. Use the `esbmc-verifier` agent to confirm the patch works and introduces no new errors. For non-ESBMC tasks (e.g. Python frontend, build scripts), run the project's normal lint/typecheck/test commands.
3. **Code review.** Use the `code-reviewer` agent on the diff. Apply high-confidence findings; explain anything you skip.

## Available Subagents

These specialised agents are configured in `~/.claude/agents/` and should be preferred over ad-hoc Bash invocations when their description fits the task.

- **`esbmc-verifier`** — Recommended formal-verification tool for this repo. Verifies C/C++/Python with ESBMC: inspects GOTO IR (`--goto-functions-only`), VCCs (`--show-vcc`), and the symbol table; applies minimal patches; re-runs ESBMC to confirm `VERIFICATION SUCCESSFUL`; and produces nondet test cases under `regression/`. Invoke for the post-implementation ESBMC step (§Post-implementation Pass #2), for deterministic witnesses when sanitizers cannot reproduce a memory/UB bug (§Regression Tests for Memory/UB Bugs), and when diagnosing unexpected ESBMC results (§Debugging Verification Issues). Defaults to bitwuzla; honours `test.desc` flags when present. For one-shot sanity checks (`esbmc file.c --incremental-bmc`), call `esbmc` directly via Bash instead.
- **`esbmc-firmware-verifier`** — Three-phase firmware verification (language-level safety → contracts via k-induction → bug-specific negative proofs) with stub-shadowing for hardware dependencies. Use when the verification target is external embedded C/C++, not ESBMC's own code.
- **`code-reviewer`** — Diff review against the priorities in §Code Review Priorities. Invoke for the post-implementation review step (§Post-implementation Pass #3).
- **`creduce-reducer`** — Reduces C/C++ programs that trigger an ESBMC bug to a minimal reproducer using C-Reduce with property-preserving interestingness scripts. Use when filing or investigating ESBMC bug reports against large inputs.

## Regression Tests for Memory/UB Bugs

When fixing a memory-safety or undefined-behaviour bug in C/C++ code:

1. Before applying the fix, write a regression test that reproduces the bug under sanitizers (ASan, UBSan, or MSan as appropriate; TSan for data races).
2. Compile and run the regression test, and confirm it fails on the unfixed code — either via a clear sanitizer diagnostic or by tripping an embedded `assert` — so the failure mode is reproducible end-to-end, not just inferred.
3. Apply the fix and re-run the compiled test; confirm it now passes cleanly (assertion holds and no sanitizer diagnostic).
4. Skip this step for pure logic bugs, build/config issues, or non-C/C++ work — sanitizers do not apply.

If sanitizers do not reproduce the bug (e.g. timing-dependent races, allocator-dependent use-after-free, MSan without instrumented dependencies, optimisation-dependent UB, or input coverage gaps):

1. Try a different sanitizer (ASan ↔ TSan ↔ MSan ↔ UBSan) and vary build flags (`-O0` vs `-O2`, `_GLIBCXX_DEBUG`, `MALLOC_PERTURB_`, `ASAN_OPTIONS=detect_stack_use_after_return=1`).
2. If still not reproducible under sanitizers, fall back to ESBMC (`esbmc-verifier` agent) to obtain a deterministic witness.
3. As a last resort, write a regression test that reproduces the observable symptom (wrong output, assertion, crash) without relying on a sanitizer diagnostic, and note in the commit message why sanitizer-based reproduction was not feasible.

## Consulting the C/C++ Standard

When a C/C++ change concerns standard-defined semantics — undefined behaviour, implicit conversions, object lifetime, name lookup, overload resolution, constant evaluation, or similar — consult the relevant standard draft (e.g. the latest C or C++ working draft on open-std.org, or cppreference for a digestible summary) before implementing. Cite the section in the commit message or code comment when it clarifies a non-obvious choice. Skip for routine edits that do not depend on standard semantics.

## Incremental Patch Testing

When a fix involves multiple patches (e.g. N1, N2), apply and test them one at a time:

1. Apply patch N1, then run the relevant tests to check whether the bug is fixed.
2. If fixed, stop — do not apply further patches.
3. If not fixed, apply patch N2 and test again. Repeat until the bug is resolved or all patches are exhausted.
4. Do not apply all patches at once before testing.

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
- `unit/` — Catch2 unit tests
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

**5. Hypothesis tests** — Property-based tests in `unit/python-frontend/` test ESBMC's models against CPython. Run with: `uv run python -m pytest unit/python-frontend/ -v`

## Commit Conventions

Prefix commits with a category tag in brackets, e.g., `[python]`, `[build]`, `[solver]`, `[om]` (operational model). Title: one line, imperative mood, <72 chars. Description: 2–4 lines explaining what changed and why. Reference the relevant issue/PR with `Fixes #N` when applicable.

**Never squash commits.** Always preserve the full commit history — every individual commit must remain intact. Do not use `git merge --squash`, `git rebase` to squash, or any PR merge strategy that collapses commits.

## PR Conventions

- Branch from `master` (the default branch)
- Target PRs to `master`
- Check formatting with clang-format before submitting

## Issue and PR Labels

Always apply at least one label when creating an issue or PR. Pick the label that matches the affected area — e.g. `python`, `clang-c-frontend`, `solver`, `build`, `docs`. Use `gh label list --repo esbmc/esbmc` to see the available labels, then `gh issue edit <N> --add-label <label>` or `gh pr edit <N> --add-label <label>`. If no existing label fits, ask the user rather than creating a new one.

For module-specific instructions, subdirectory CLAUDE.md files can be added (they load automatically when working in those directories).
