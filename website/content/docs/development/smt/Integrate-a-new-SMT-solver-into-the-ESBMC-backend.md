---
title: Integrating a New SMT Solver
---

This guide explains how to add a new SMT solver backend (call it `NewSolver`) to
ESBMC. The Z3 backend (`src/solvers/z3`) is the reference template throughout.

> [!NOTE]
> Line numbers drift. The anchors below name **files, classes, and functions**
> rather than line numbers so the guide stays valid as the code moves. Grep for
> the identifier if you cannot find it.

## 1. How ESBMC talks to a solver

ESBMC lowers every frontend to a GOTO program, symbolically executes it into SSA,
and encodes that SSA into SMT. The encoding goes through one abstraction:

- **`smt_solver_baset`** (`src/solvers/smt/smt_solver.h`) — the base class every
  backend derives from. It declares the ~100 virtual term builders
  (`mk_smt_bv`, `mk_extract`, `mk_bvadd`, `mk_ite`, `mk_smt_symbol`, the sort
  builders, …) plus the solving and model-extraction entry points (`dec_solve`,
  `push_ctx`/`pop_ctx`, `get_bool`, `get_bv`, `get_fpbv`, `get_array_elem`). This
  virtual surface **is** the contract you implement.
- **`smt_convt`** (`src/solvers/smt/smt_conv.h`) — the solver-independent driver.
  It *wraps* a `smt_solver_baset` (see `create_solver` in `src/solvers/solve.cpp`)
  and feeds it the encoded SSA. You do not subclass it.
- **Flatteners** — `array_conv`, `fp_conv`, and the tuple flatteners
  (`smt_tuple_node`/`smt_tuple_sym`) lower arrays, IEEE floats, and structs onto
  plain bit-vectors when the backend has no native support.

So the split is:

| Solver-independent (do not touch) | Solver-specific (you write) |
|---|---|
| `smt_convt`, flatteners, `solve.cpp` wiring | the `NewSolver` backend class under `src/solvers/newsolver/` |

The backend wraps the solver's native term type in `solver_smt_ast<T>` (cf.
`z3_smt_ast : solver_smt_ast<z3::expr>`) and implements the
`smt_solver_baset` virtuals against the solver's API. It may additionally inherit
`array_iface`, `fp_convt`, and/or `tuple_iface` — each one you inherit opts into a
*native* capability instead of the flattener fallback.

## 2. Copy the Z3 template

```sh
cp -r src/solvers/z3 src/solvers/newsolver
cd src/solvers/newsolver
git mv z3_conv.h   newsolver_conv.h
git mv z3_conv.cpp newsolver_conv.cpp
git mv try_z3.c    try_newsolver.c
```

Rename mechanically across the copied files:

- `z3_convt` → `newsolver_convt`, `z3_smt_ast` → `newsolver_smt_ast`
- `create_new_z3_solver` → `create_new_newsolver_solver`
- the include guard `_ESBMC_SOLVERS_Z3_Z3_CONV_H`
- the helper macro `#define new_ast new_solver_ast<z3_smt_ast>`

Keep the class inheriting `smt_solver_baset, tuple_iface, array_iface, fp_convt`
until you know which capabilities are native (see §9).

## 3. Replace the Z3 API calls

Re-implement each method body against `NewSolver`, in order of leverage:

1. **Context & sorts** — constructor, `mk_bv_sort`, `mk_bool_sort`,
   `mk_array_sort`.
2. **Terms** — the `mk_bvadd / mk_bvule / mk_concat / mk_extract / mk_ite / …`
   builders and the literal/symbol builders `mk_smt_bv`, `mk_smt_bool`,
   `mk_smt_symbol`.
3. **Solving** — `dec_solve()` asserts the accumulated constraints, calls
   check-sat, and maps the result to `smt_convt::P_SATISFIABLE` /
   `P_UNSATISFIABLE` / `P_SMTLIB`/error. `push_ctx`/`pop_ctx` drive the native
   assertion stack and **must** chain to `smt_solver_baset::push_ctx`/`pop_ctx`.
4. **Model extraction** — `get_bool`, `get_bv`, `get_fpbv`, `get_array_elem`
   read values back from the model.
5. **Factory** — `create_new_newsolver_solver` constructs the backend and sets
   `*tuple_api` / `*array_api` / `*fp_api` to the backend **only for the
   interfaces you implemented natively**; pass `nullptr` for the rest so
   `solve.cpp` installs the corresponding flattener.

**Record semantic differences explicitly.** Watch signed-vs-unsigned
division/remainder/shift, rotates, FP rounding modes, and array-of-array support.
If `NewSolver` cannot express something, leave its `*_api` null and rely on the
flattener — never emit subtly wrong semantics.

## 4. Build system

Create **`src/solvers/newsolver/CMakeLists.txt`** from `z3/CMakeLists.txt` (or
`bitwuzla/CMakeLists.txt` if the dependency is fetched via pkg-config /
`DOWNLOAD_DEPENDENCIES`):

- locate the library (`find_library`/`find_path` or `pkg_check_modules`), guarded
  by `if(ENABLE_NEWSOLVER)`; keep the `try_run` version probe
  (`try_newsolver.c`);
- `add_library(solvernewsolver newsolver_conv.cpp)`, set include dirs, then
  `target_link_libraries(solvernewsolver fmt::fmt <lib>)` and
  `target_link_libraries(solvers INTERFACE solvernewsolver)`;
- on success, propagate to the parent scope:
  ```cmake
  set(ESBMC_ENABLE_newsolver 1 PARENT_SCOPE)
  set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} newsolver" PARENT_SCOPE)
  ```

In **`src/solvers/CMakeLists.txt`** add `set(ESBMC_ENABLE_newsolver 0)` to the
top block and `add_subdirectory(newsolver)` to the solver list. Declare the
`ENABLE_NEWSOLVER` CMake option alongside the other `ENABLE_*` solver options.

## 5. Register the solver (`src/solvers/solve.cpp`)

1. Declare the factory next to the others:
   `solver_creator create_new_newsolver_solver;`
   (`solver_creator` is the `typedef smt_solver_baset *(...)` in `solve.h`).
2. Add a guarded entry to the `esbmc_solvers` map:
   ```cpp
   #ifdef NEWSOLVER
     {"newsolver", create_new_newsolver_solver},
   #endif
   ```
   (mind the trailing comma on the last live entry).
3. Add `"newsolver"` to the `all_solvers[]` array. **Order matters** — that array
   encodes default-selection priority (first compiled-in entry, excluding
   `smtlib`, wins when no solver is requested). `resolve_user_solver_choice` and
   `check_solver_availability` then recognise it automatically via
   `options.get_bool_option("newsolver")` — no further factory edits needed.

## 6. Configuration header (`src/solvers/solver_config.h.in`)

Append the block that defines the `NEWSOLVER` macro gating the `#ifdef`s above:

```c
#if @ESBMC_ENABLE_newsolver@
#define NEWSOLVER
#endif
```

## 7. CLI and options

- **`src/esbmc/options.cpp`** — in the solver option group (where `{"z3", NULL,
  "Use Z3"}` lives) add `{"newsolver", NULL, "Use NewSolver"},`, plus any
  `--newsolver-debug`-style sub-options.
- **`src/esbmc/esbmc_parseoptions.cpp`** — no mandatory factory edit; selection is
  data-driven from the map and `all_solvers[]`. `--list-solvers` and `--help` pick
  up the new backend automatically from `ESBMC_AVAILABLE_SOLVERS`. Optionally add
  `"newsolver"` to the `preferred[]` diagnostic array and the multi-solver
  `cmdline.isset(...)` checks so warnings mention it.

Users then select it with `esbmc file.c --newsolver`.

## 8. Documentation

Update [`BUILDING.md`](https://github.com/esbmc/esbmc/blob/master/BUILDING.md)
with a `NewSolver` subsection: how to obtain/build the library, the
`-DENABLE_NEWSOLVER=On` flag (and `-DNewSolver_DIR=` / `DOWNLOAD_DEPENDENCIES`
behaviour), the minimum supported version, and the `--newsolver` usage. Match the
depth of the existing Z3/Bitwuzla entries.

## 9. CI integration

- [`build.yml`](https://github.com/esbmc/esbmc/blob/master/.github/workflows/build.yml)
  — add a matrix entry that installs `NewSolver`, configures with
  `-DENABLE_NEWSOLVER=On`, and runs the regression label subset against it.
- [`release.yml`](https://github.com/esbmc/esbmc/blob/master/.github/workflows/release.yml)
  — ensure the released artifact links/bundles the library (and installs the DLL
  on Windows, cf. the Z3 `install(FILES …)` block). Decide and document
  static-vs-dynamic linking. **Watch for static-link symbol clashes**: the
  Bitwuzla CMake renames its bundled CaDiCaL symbols to avoid colliding with
  cvc5 — replicate that pattern if `NewSolver` statically links a shared SAT
  core.

## 10. Validate

```sh
cmake -GNinja -Bbuild -S . -DDOWNLOAD_DEPENDENCIES=On -DENABLE_NEWSOLVER=On \
  -DBUILD_TESTING=On -DENABLE_REGRESSION=On -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja -C build
build/src/esbmc/esbmc --list-solvers   # expect "newsolver" listed
```

- **Differential test against an established backend.** Run a representative
  regression subset with `--newsolver` and with `--z3`/`--bitwuzla`; the verdicts
  must agree (modulo timeouts). A divergence is almost always a wrong
  signed/unsigned or FP encoding (see §3).
- Add at least **two regression tests** (one passing `CORE`, one failing) whose
  `test.desc` flag line uses `--newsolver`.
- Clean the per-test temp dirs afterwards: `rm -rf /tmp/esbmc-headers-*`.

## 11. Maintainability notes

- **Implement natively only what pays off.** Start by inheriting *only*
  `smt_solver_baset` and returning `nullptr` for the tuple/array/fp APIs — the
  flatteners give a correct baseline immediately. Add `array_iface` / `fp_convt` /
  `tuple_iface` incrementally, each gated by the differential tests, so every
  native path is proven before it ships.
- **Reject unsupported sorts loudly.** A bit-vector-only `NewSolver` cannot serve
  `--ir` / `--ir-ieee` (integer/real arithmetic). `solve.cpp` already rejects
  that combination for Bitwuzla/Boolector with a clean `exit(1)`; add `newsolver`
  to that guard rather than letting the backend `abort()` at construction.
- **Don't deepen the CMake copy-paste.** `src/solvers/CMakeLists.txt` notes each
  backend's CMake is duplicated because CMake lacks indirect function calls.
  Factoring the shared find/version/link logic into an `esbmc_add_solver()` helper
  under `scripts/cmake/` is a worthwhile refactor to land alongside a new backend.

### Known limitations

- Integer/real arithmetic auto-selects Z3; a BV-only backend won't serve `--ir`.
- Quantifiers, FP, and tuples are optional surfaces — the flatteners keep results
  correct if `NewSolver` lacks them, at some performance cost.
- Static-link symbol collisions with cvc5/Bitwuzla bundled SAT cores are the most
  likely release-build failure and must be resolved in CMake, not at runtime.
