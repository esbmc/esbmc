# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ESBMC (Efficient SMT-based Context-Bounded Model Checker) is a mature model checker that automatically detects or proves the absence of runtime errors in C, C++, CUDA, CHERI, Kotlin, Python, and Solidity programs. It uses SMT solvers to verify safety properties through symbolic execution.

## Building ESBMC

### Standard Build (Ubuntu 24.04)
```bash
# Install dependencies
sudo apt-get install -y clang-14 llvm-14 clang-tidy-14 python-is-python3 python3 \
  git ccache unzip wget curl bison flex g++-multilib linux-libc-dev \
  libboost-all-dev libz3-dev libclang-14-dev libclang-cpp-dev cmake

# Build with Z3 solver
# IMPORTANT: Always explicitly set CXX=g++ and CC=gcc to avoid compiler detection issues
mkdir build && cd build
CXX=g++ CC=gcc cmake .. -DENABLE_Z3=1
make -j4
```

### Build with All Features Enabled
```bash
CXX=g++ CC=gcc cmake .. -DENABLE_Z3=1 \
  -DENABLE_PYTHON_FRONTEND=1 \
  -DENABLE_SOLIDITY_FRONTEND=1 \
  -DENABLE_JIMPLE_FRONTEND=1 \
  -DBUILD_TESTING=On \
  -DENABLE_REGRESSION=1
make -j4
```

### Automated Build Script
For a full-featured build with dependencies, use:
```bash
./scripts/build.sh
```

### Python Frontend Support
To enable Python verification, install `ast2json`:
```bash
pip install ast2json
CXX=g++ CC=gcc cmake .. -DENABLE_Z3=1 -DENABLE_PYTHON_FRONTEND=1
```

## Testing

### Running Unit Tests
Unit tests use Catch2 framework and are located in `unit/`:
```bash
# From build directory with -DBUILD_TESTING=On
ctest -j4                    # Run all tests
ctest --progress             # Show progress
ctest --timeout 30           # Set timeout
```

### Running Regression Tests
Regression tests are in `regression/` and use CTest:
```bash
# From build directory with -DENABLE_REGRESSION=1
ctest -j4 -L esbmc-cpp/cpp              # Run specific test suite
ctest -L esbmc-cpp/*                    # Pattern matching
ctest -LE esbmc-cpp*                    # Exclude pattern
ctest -j4 -L python --progress          # Python tests with progress
```

### Python Regression Tests
Special script for Python tests:
```bash
./scripts/check_python_tests.sh
```

### Regression Test Structure
Each regression test directory contains:
- Source file(s) to verify (`.c`, `.cpp`, `.py`, `.sol`, etc.)
- `test.desc` file with test configuration
- Expected output/counterexample

## Code Architecture

### Verification Pipeline
1. **Frontend Parsing** → 2. **GOTO Conversion** → 3. **Symbolic Execution** → 4. **SMT Solving**

The architecture follows this flow:
- Source code (C/C++/Python/Solidity/etc.) is parsed into an Abstract Syntax Tree (AST)
- AST is transformed into GOTO intermediate representation (control-flow graph)
- Symbolic execution engine (`goto-symex`) explores execution paths
- Execution traces are converted to Static Single Assignment (SSA) form
- SSA is encoded as SMT formulas and passed to solvers

### Key Components

**Frontend Parsers** (in `src/`):
- `clang-c-frontend/` - C frontend using Clang
- `clang-cpp-frontend/` - C++ frontend using Clang
- `python-frontend/` - Python AST parser with type inference
- `solidity-frontend/` - Solidity smart contract parser
- `jimple-frontend/` - Java/Kotlin via Jimple intermediate representation

**Core Verification** (in `src/`):
- `goto-programs/` - GOTO intermediate representation and CFG manipulation
- `goto-symex/` - Symbolic execution engine
  - `goto_symext` - Main symbolic execution class
  - `execution_state` - Execution state management
  - `goto_symex_state` - State tracking during symbolic execution
- `solvers/` - SMT solver interfaces (Z3, Boolector, Bitwuzla, CVC4/5, MathSAT, Yices)
- `pointer-analysis/` - Pointer and alias analysis
- `irep2/` - Internal representation (version 2)

**Main Entry** (in `src/esbmc/`):
- `main.cpp` - Entry point
- `esbmc_parseoptions.cpp` - Command-line option parsing
- `bmc.cpp` - Bounded model checking orchestration

### Language Support Notes

**Python Frontend**: Requires type annotations (`var:type` syntax per PEP 484). Converts Python AST to JSON using `ast` and `ast2json` modules, performs type inference, then generates GOTO representation.

**Solidity Frontend**: Requires both `.sol` source and `.solast` (compact JSON AST from `solc`). Tested with Solidity 0.8.0. Use `--sol` flag and optionally `--reentry-check` for reentrancy detection.

## Code Style

### Formatting
- Use clang-format (Clang 11 or later)
- Configuration in `.clang-format`
- Key rules:
  - 80 column limit
  - 2-space indentation (no tabs)
  - Allman brace style
  - Break before binary operators on new lines
  - One parameter per line for functions

### Running clang-format
```bash
clang-format -i path/to/file.cpp
```

## Common Development Commands

### Single Test Run
```bash
# Run ESBMC on a single file
./build/src/esbmc/esbmc test.c --incremental-bmc

# With specific solver
./build/src/esbmc/esbmc test.c --boolector --incremental-bmc

# Python verification
./build/src/esbmc/esbmc test.py

# Solidity verification (requires .solast)
solc --ast-compact-json contract.sol > contract.solast
./build/src/esbmc/esbmc --sol contract.sol contract.solast --k-induction
```

### Debugging Options
```bash
# Show GOTO program
./build/src/esbmc/esbmc test.c --show-goto-functions

# Show verification conditions
./build/src/esbmc/esbmc test.c --show-vcc

# Generate HTML report
./build/src/esbmc/esbmc test.c --generate-html-report
```

### CMake Configuration Options
Key CMake flags for development:
- `-DENABLE_Z3=1` - Enable Z3 solver
- `-DENABLE_BOOLECTOR=1` - Enable Boolector solver
- `-DENABLE_BITWUZLA=1` - Enable Bitwuzla solver
- `-DENABLE_PYTHON_FRONTEND=1` - Enable Python support
- `-DENABLE_SOLIDITY_FRONTEND=1` - Enable Solidity support
- `-DENABLE_JIMPLE_FRONTEND=1` - Enable Java/Kotlin support
- `-DBUILD_TESTING=On` - Enable unit tests
- `-DENABLE_REGRESSION=1` - Enable regression tests
- `-DDOWNLOAD_DEPENDENCIES=On` - Auto-download dependencies

## Documentation

Generate Doxygen documentation:
```bash
doxygen .doxygen
# Output in docs/HTML/index.html
```

## Contributing Guidelines

1. Fork and clone the repository
2. Create a branch from `master`
3. Make changes following code style (check with clang-format)
4. Add at least two regression tests: one passing and one failing case
5. Push changes and create PR against `master` branch

### Keeping Fork Updated
```bash
git remote add upstream https://github.com/esbmc/esbmc
git checkout master
git fetch upstream
git pull --rebase upstream master
git push origin HEAD:master
```

## Verification Strategies

ESBMC supports multiple verification approaches:
- `--incremental-bmc` - Incremental bounded model checking (recommended)
- `--k-induction` - K-induction proof rule for unbounded verification
- `--context-bound N` - Limit context switches for concurrent programs (default: symbolic)

## Supported Error Classes

ESBMC detects:
- User-specified assertion failures
- Array bounds violations
- Pointer errors (null deref, out-of-bounds, double-free, misalignment)
- Integer overflows/underflows
- Division by zero
- Memory leaks
- Shift operation undefined behavior
- Floating-point NaN
- Deadlock (pthread mutexes/conditionals)
- Data races
- Atomicity violations

## Known Issues

### CMake Compiler Detection with ccache

**Problem**: When ccache is in the PATH, CMake's auto-detection may incorrectly select `gcc` instead of `g++` as the C++ compiler. This causes linking failures in yaml-cpp test programs with errors like:
```
undefined reference to `operator new(unsigned long)'
undefined reference to `operator delete(void*, unsigned long)'
```

**Root Cause**: When `gcc` is used to link C++ code (instead of `g++`), it doesn't automatically link the C++ standard library (`libstdc++`), resulting in undefined references to C++ runtime functions.

**Solution**: Always explicitly set compilers when running cmake:
```bash
CXX=g++ CC=gcc cmake .. [options]
```

This is already documented in all cmake commands above. The `scripts/build.sh` is used for CI and handles this correctly for automated builds.
