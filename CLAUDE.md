# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESBMC (Efficient SMT-based Context-Bounded Model Checker) is a formal verification tool that detects bugs in C, C++, CUDA, CHERI-C, Python, Java/Kotlin, Solidity, and Rust programs. It works by parsing source code into an AST, converting to a GOTO intermediate representation, symbolically executing it to produce SSA constraints, encoding those as SMT formulas, and checking satisfiability to find property violations (or prove their absence).

## Build Commands

```bash
# Full build (install deps, configure, build, install)
./scripts/build.sh

# Individual steps
./scripts/build.sh deps      # Install dependencies + configure
./scripts/build.sh build     # Build only
./scripts/build.sh install   # Install to ./release/

# Common options
./scripts/build.sh -b Debug build          # Debug build
./scripts/build.sh -s address build        # With AddressSanitizer
./scripts/build.sh -C deps build install   # SV-COMP build (extra solvers)
```

The binary is installed to `./release/bin/esbmc`.

## Testing

Tests are run via CTest from the `build/` directory:

```bash
cd build/

# Run all regression tests
ctest -j$(nproc) --progress --output-on-failure

# Run a specific test suite (label matches "folder/" pattern)
ctest -j4 -L "esbmc-cpp/cpp"
ctest -j4 -L "python"

# Run a single test by name
ctest -R "regression/esbmc/00_bitshift_01"

# Exclude slow Python tests
ctest -j4 -LE python-intensive

# Run unit tests only
ctest -j4 -L unit
```

### Regression Test Format

Each test is a directory under `regression/` containing:
- A source file (e.g., `main.c`, `main.py`)
- A `test.desc` file with this format:
  ```
  CORE                          # Mode: CORE, THOROUGH, KNOWNBUG, or FUTURE
  main.c                        # Input file
  --no-slice --some-flag        # ESBMC command-line arguments
  ^VERIFICATION FAILED$         # Expected output regex (one per line)
  ```

Every PR should include at least two test cases: one that passes and one that fails verification.

## Code Formatting

- C/C++: clang-format with Clang 11
- Python: YAPF
- CMake: cmakelint

## Architecture

### Verification Pipeline

```
Source code → Frontend (AST) → GOTO program → Symbolic execution (SSA) → SMT formula → Solver → Result
```

### Key Source Directories (`src/`)

| Directory | Purpose |
|-----------|---------|
| `esbmc/` | Entry point (`main.cpp`), BMC orchestration (`bmc.cpp`), CLI parsing (`esbmc_parseoptions.cpp`) |
| `irep2/` | Core intermediate representation: typed expressions (`irep2_expr.h`) and types (`irep2_type.h`). Uses `expr2tc`/`type2tc` smart pointers. |
| `goto-programs/` | GOTO IR: control flow graph with instruction types (GOTO, ASSERT, ASSUME, ASSIGN, etc.). `goto_convert.cpp` transforms AST to GOTO. |
| `goto-symex/` | Symbolic execution engine. `symex_main.cpp` drives path exploration, generates SSA form via `symex_target_equation`. |
| `solvers/` | SMT/SAT solver backends. Abstract interface in `smt/smt_conv.h`; implementations in `z3/`, `bitwuzla/`, `boolector/`, `cvc5/`, `yices/`, `mathsat/`, `smtlib/`. |
| `clang-c-frontend/` | C frontend using Clang. `clang_c_convert.cpp` is the main AST-to-irep2 converter. |
| `clang-cpp-frontend/` | C++ frontend extending the C frontend. Handles classes, templates, virtual functions. |
| `python-frontend/` | Python frontend. Converts Python AST (via ast2json) to irep2. |
| `solidity-frontend/` | Solidity smart contract frontend. Core converter split by concern: `solidity_convert.cpp` (entry/init), `_expr` (expressions), `_call` (function calls/transfers), `_type` (types), `_decl` (declarations), `_util` (helpers), `_constructor`, `_contract` (instances/multi-contract verification), `_ref` (symbol resolution), `_mapping`, `_stmt`, `_modifier`, `_builtin` (msg/tx/block), `_tuple`, `_inheritance`, `_literals`. Single class `solidity_convertert` declared in `solidity_convert.h`. |
| `jimple-frontend/` | Java/Kotlin frontend via Soot's Jimple IR. |
| `c2goto/` | C library models and standard definitions for GOTO conversion. |
| `pointer-analysis/` | Static pointer analysis framework. |
| `util/` | Shared utilities: symbol table (`context.h`), config, expression simplifier, type casting. |

### Solver Architecture

The solver layer uses an abstract interface (`smt_convt`) with per-solver implementations. The conversion pipeline flattens irep2 expressions → lowers memory model/pointers/casts → encodes to solver-native AST → queries satisfiability. See `src/solvers/README.txt` for details.

### Expression System

The `irep2` layer defines 170+ expression types and 20+ type constructors. Expressions use `expr2tc` (shared pointer wrapper) and are enumerated in `ESBMC_LIST_OF_EXPRS`. Types use `type2tc`. This is the universal IR that all frontends target and all backends consume.

## Code Style

- Modern C++ (C++11+), prefer const-correctness and RAII
- Follow existing patterns in the file being modified
- Prioritize verification soundness and memory safety in changes
- SMT encoding correctness is critical -- changes to solver code require careful review
