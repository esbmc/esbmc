# AGENTS.md

This file provides guidance to coding agents working with this repository.
AGENTS.md is the canonical file; a harness that cannot read it gets an import
or a symlink to it, never a second copy. Every rule an agent needs lives here
or in `docs/agents/`, written for any harness; harness config (e.g.
`.claude/`) only adapts it and holds no rules of its own.

## Task-Specific Rules

These rules live in `docs/agents/`. Read the named file when its trigger
applies — before acting, not after.

- **Fixing a bug or changing verification behaviour:**
  `docs/agents/regression-tests.md` (test pairs, mutation check) and
  `docs/agents/bug-fixing.md` (sanitizer-first reproduction, incremental
  patches).
- **Adding or changing a regression test, or running the regression suite and
  judging its result:** `docs/agents/regression-tests.md` (per-test budgets,
  slowdown cap, runs that read green but measure nothing).
- **Changing an operational model, or anything that depends on C/C++ standard
  semantics:** `docs/agents/cpp-standard.md`.
- **ESBMC gives an unexpected verdict, crashes, or errors out:**
  `docs/agents/debugging.md`.
- **The change could move SV-COMP verdicts, or changes what ESBMC prints:**
  `docs/agents/svcomp.md`.
- **Before investigating or fixing a GitHub issue:**
  `docs/agents/issues-and-prs.md` (known-bug check first).
- **Before opening or labelling an issue or PR:**
  `docs/agents/issues-and-prs.md`.
- **After implementing a non-trivial change, before committing:**
  `docs/agents/post-implementation.md` (simplify, verify, review, coverage
  gate).

## Project Overview

ESBMC (Efficient SMT-based Context-Bounded Model Checker) is a software model
checker that detects bugs or proves their absence in C, C++, CUDA, CHERI-C,
Python, Solidity, Java, and Kotlin programs. It works by parsing source →
building AST → converting to GOTO program → symbolic execution (SSA) → encoding
as SMT formula → solving with SMT solvers.

## Build Commands

**NEVER run cmake in the repo root (e.g., `cmake .` or `cmake -B. -S.`).**
Always use `build/` or a subdirectory of it as the build directory (e.g.,
`-Bbuild`). The `.gitignore` only covers `build/` — in-tree builds pollute the
source tree with hundreds of untracked artifacts.

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
- Quality: `-DENABLE_WERROR=On`, `-DENABLE_CLANG_TIDY=On`,
  `-DENABLE_COVERAGE=On`

See `scripts/build.sh` for full platform-specific dependency setup and solver
configuration.

Requires: CMake 3.18+, Ninja, Boost (date_time, program_options, iostreams,
system, filesystem), LLVM 11+ (tested up to 21), Bison, Flex, Z3 (or another SMT
solver).

## Testing

Regression tests require at least one solver backend (e.g., Z3). All commands
run from the `build/` directory. `$(nproc)` is Linux; on macOS use
`$(sysctl -n hw.ncpu)`.

```sh
# Run unit tests only (fast, excludes regression-labeled tests)
ctest -j$(nproc) -LE regression --timeout 60

# Run all regression tests (slow; Python/Solidity suites leave /tmp/esbmc*)
ctest -j$(nproc) -L regression

# Run a specific regression suite by label
ctest -j$(nproc) -L esbmc           # core C tests
ctest -j$(nproc) -L python          # Python tests
ctest -j$(nproc) -L "esbmc-cpp/cpp" # C++ tests
ctest -j$(nproc) -L floats          # floating-point tests

# List all available test labels
ctest --print-labels

# Run a single named test
ctest -R "regression/esbmc/00_big_endian_01" --output-on-failure
```

**Important: the Python frontend needs `python3` on `PATH`.** ESBMC's Python
frontend invokes `python3` to run `parser.py`. `ast2json` is vendored in the
source tree (`src/python-frontend/libs/ast2json`), so it no longer needs to be
installed separately for Python regression tests. (`mypy` is an optional extra
for type checking.)

**Before committing:**

- Always run the project's test suite. If tests fail, fix the failures before
  committing — never commit broken or untested code.
- **Regression suite cap.** When running the full regression suite, cap the run
  at **10 minutes** (600000 ms) — set the harness's command timeout, or wrap
  the invocation with `timeout 10m …`. If the suite
  cannot complete in 10 minutes, narrow the scope (e.g. run only the affected
  subset) or ask the user before extending the limit. `ctest --timeout` does
  **not** cap this suite: CMake gives every test an explicit `TIMEOUT`
  property, and ctest's flag only supplies a default for tests that have none.
- **Lint and typecheck.** Run lint and typecheckers and fix any errors. For
  Python code, use `pylint`. For C++ code, ensure clang-format compliance (CI
  enforces this).
- **Cyclomatic complexity.** `python3 scripts/complexity/ccn_report.py --gate`
  reports what the branch adds against its merge base, the same check the
  Complexity workflow runs on the PR (needs `pip install lizard==1.23.0`). It is
  advisory while the thresholds are being calibrated.

## General Style Guidelines

The following applies to the tone and style when writing comments, code, documentation
pull requests and more.

- Please remove all mannered prose.

## Branching

Before implementing any feature or bug fix, always work on a dedicated branch:

1. Check the current branch — never work directly on `master`, and do not
   reuse a branch made for other work.
2. Create a dedicated branch from `master` with a descriptive name (e.g.
   `feat/short-description` or `fix/short-description`).
3. Confirm the branch is active before making any changes.

## Code Style

- **C++**: clang-format 23, Allman braces, 80-col limit, 2-space indent,
  no tabs. Config in `.clang-format`.
- **Python**: YAPF, PEP 8 based, 100-col limit. Config in `.style.yapf`.
- Prefer modern C++ idioms (C++11+). Use const-correctness throughout. Prefer
  stack allocation over heap when possible. Follow existing patterns in the file
  being modified.
- CI enforces formatting on PRs via GitHub Actions.

## Coding Guidelines

- Write simple, clean, and readable code with minimal indirection.
- Each function should do one thing well. No redundant abstractions or duplicate
  code.
- Check the entire codebase to reuse existing methods before writing new ones.
- Tests MUST NOT use mocks, patches, or any form of test doubles. Integration
  tests are preferred.

## Code Comments

Write few comments — favour self-explanatory code (clear names, small functions)
over narration. Keep added comments to a minimum in PRs; excess comments are
noise reviewers must wade through.

- **Do not** restate what the code plainly does (`i++; // increment i`), label
  structure (`// constructor`, `// helpers`), narrate the change or its history
  (`// added null check`, `// was: foo()`), or echo a function/variable name in
  prose.
- **Do** comment only when it adds what the code cannot convey: a non-obvious
  _why_ (rationale, trade-off, workaround with an issue/PR link), a caveat or
  invariant a caller must respect, a citation to the C/C++ standard or a
  solver/algorithm detail, or genuinely subtle logic. One line beats a
  paragraph.
- Preserve existing meaningful comments and the file's established doc
  conventions (e.g. Doxygen-style headers where already used). Match the
  surrounding comment density rather than exceeding it.

## Source Architecture

Key directories under `src/`:

- `esbmc/` — Main entry point and CLI driver
- `irep2/` — Internal representation (IRep2), the core data structure for
  expressions/types
- `goto-programs/` — GOTO intermediate representation and transformations
- `goto-symex/` — Symbolic execution engine (core verification logic), split
  into `engine/` (`goto_symext` and its statement handlers, including
  `engine/builtin_functions/`), `state/` (per-thread state and SSA naming),
  `scheduler/` (thread interleaving and the exploration tree), `equation/` (the
  SSA formula and its passes), and `trace/`, `witness/` and `testgen/` for what
  is built once a verdict exists. Only `testgen/` is unreachable from
  `symex_step`: `symex_printf` and the witness hooks call into `trace/` and
  `witness/` from inside symex
- `solvers/` — SMT solver backends (z3, bitwuzla, boolector, cvc4, cvc5, yices,
  mathsat, smtlib)
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

## Commit Conventions

Prefix commits with a category tag in brackets, e.g., `[python]`, `[build]`,
`[solver]`, `[om]` (operational model). Title: one line, imperative mood, <72
chars. Description: 2–4 lines explaining what changed and why. Reference the
relevant issue/PR with `Fixes #N` when applicable.

**Never squash commits.** Always preserve the full commit history — every
individual commit must remain intact. Do not use `git merge --squash`,
`git rebase` to squash, or any PR merge strategy that collapses commits.

**No agent attribution.** Do not add a `Co-Authored-By: Claude …` trailer, a
`Claude-Session:` trailer, or a "🤖 Generated with [Claude Code]" footer to a
commit message or a PR description. Agent harnesses inject these by default,
and some state that their attribution instruction replaces earlier guidance; it
does not. Omit them when writing the message, not afterwards — removing them
from a pushed branch costs a history rewrite and a force-push on a branch with
an open PR.
