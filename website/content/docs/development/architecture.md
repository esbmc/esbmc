---
title: Architecture
weight: 2
prev: /docs/usage
---

# Project structure

The repository is laid out as follows:

- **.github** — files used by the repository itself, including the GitHub
  Actions workflows.
- **docs** — generated documentation and the development roadmaps.
- **regression** — the regression suites, over 60 categories of proof harnesses
  that validate ESBMC.
- **scripts** — helper tools, CMake modules and the competition wrappers; nothing
  here is used directly by ESBMC at run time.
- **src** — ESBMC's source.
- **unit** — Catch2 unit tests.
- **website** — this site (Hugo).

## Inside `src`

The verification pipeline is *parse → GOTO → symbolic execution → SMT*, and the
directories follow it:

| Directory | Role |
|---|---|
| `esbmc/` | The driver. Option handling, GOTO preparation, the BMC strategies and reporting are built as the `esbmc-driver` static library, so a unit test can link them; `main.cpp`, `globals.cpp` and the generated build-id object stay in the executable |
| `irep2/` | The internal representation (IRep2) for expressions and types |
| `goto-programs/` | The GOTO intermediate representation and its transformations |
| `goto-symex/` | The symbolic execution engine (see below) |
| `solvers/` | SMT backends: Z3, Bitwuzla, Boolector, CVC4, CVC5, Yices, MathSAT, SMT-LIB |
| `langapi/` | Language API abstractions shared across frontends |
| `pointer-analysis/` | The memory model and pointer-safety analysis |
| `util/` | Shared utilities and data structures |
| `big-int/` | Arbitrary-precision integers |
| `c2goto/`, `goto2c/` | C operational models compiled to GOTO, and GOTO back to C |
| `cpp/library/` | The C++ operational models |

Each frontend parses one language into the shared GOTO representation:
`clang-c-frontend/` (C, CHERI-C, CUDA), `clang-cpp-frontend/` (C++),
`python-frontend/` (Python 3.10+), `jimple-frontend/` (Java/Kotlin),
`solidity-frontend/` (Solidity) and `ld-frontend/` (IEC 61131-3 ladder logic).

### `goto-symex`

`src/goto-symex` is partitioned into seven subsystem directories rather than one
flat directory:

| Directory | Contents |
|---|---|
| `engine/` | `goto_symext` and its statement handlers, including `engine/builtin_functions/` |
| `state/` | Per-thread state and SSA naming |
| `scheduler/` | Thread interleaving and the exploration tree |
| `equation/` | The SSA formula and its passes |
| `trace/` | Counterexample traces |
| `witness/` | Witness generation |
| `testgen/` | Test-case generation |

Only `testgen/` is unreachable from `symex_step`; `symex_printf` and the witness
hooks call into `trace/` and `witness/` from inside symbolic execution.

## Continuous integration

ESBMC relies on [GitHub Actions](https://github.com/features/actions). The
workflow files in `.github/workflows` are grouped by trigger, since
subdirectories are not supported there and a filename prefix is the only
grouping available:

| Prefix | Trigger |
|---|---|
| `ci-` | The scheduled and event-driven gates: `ci-pull-request`, `ci-master`, `ci-nightly`, `ci-weekly`, plus release and stats refresh |
| `aux-` | Reusable workflows invoked with `workflow_call` — the per-platform builds, build info, and the symex oracles |
| `dispatch-` | Manual dispatch only — BenchExec runs and their diffs, benchmark bring-up, witness validation |
| `bot-` | Automation |

The pull-request gate keeps one Linux build; Windows, armv8, macOS, the
sanitizers and the symex oracles run on the weekly schedule. No `CORE`
regression test may exceed 30 s — the rule is recorded in
`regression/CMakeLists.txt` — so a slower test is marked `THOROUGH` instead.
