---
title: Implementing a New SMT Theory
---

This guide explains how to teach ESBMC's SMT layer a new **theory** — a family of
sorts and operations that the encoder can emit to the solver. The worked example
is ESBMC's integer/real arithmetic mode (`QF_AUFLIRA`), which is the theory the
`--ir` / `--ir-ieee` flags select.

> [!NOTE]
> Anchors below name **files, classes, and functions** rather than line numbers,
> which drift. Grep for the identifier if you cannot find it.

## Which guide do you actually need?

Three different tasks are easy to confuse:

| You want to… | Read |
|---|---|
| Add a new IR expression (irep2 → SMT) | [Adding New Expressions](../adding-new-expressions/) |
| Add a whole new solver backend | [Integrating a New SMT Solver](../integrate-a-new-smt-solver-into-the-esbmc-backend/) |
| Add a new **sort/operation family** the SMT layer can emit | **this page** |

A "theory" here means new SMT sorts (e.g. `Int`, `Real`, `String`, `Set`) and the
operations over them, plus the *mode* that decides when ESBMC emits them instead
of the default bit-vector encoding.

## Architecture

The relevant pieces live under `src/solvers/smt/`:

- **Sorts** — `smt_sort.h` enumerates sort kinds (`SMT_SORT_BV`, `SMT_SORT_INT`,
  `SMT_SORT_REAL`, `SMT_SORT_FIXEDBV`, …). A new theory usually needs a new
  `SMT_SORT_*` and a sort-builder.
- **Operation surface** — `smt_solver_baset` (`smt_solver.h`) declares the
  virtual term builders (`mk_smt_int`, `mk_smt_real`, `mk_add`, `mk_lt`, … and the
  bit-vector duals `mk_bvadd`, `mk_bvult`, …). This virtual set **is** the theory
  contract each backend implements.
- **Mode flag** — a theory that is an *alternative encoding* of existing types is
  gated by a flag. Integer/real mode uses `bool int_encoding`, a member of
  `smt_solver_baset` initialised from the `int-encoding` option in
  `smt_solver.cpp` (`int_encoding = options.get_bool_option("int-encoding")`).
  The encoder branches on it throughout `src/solvers/smt/*.cpp` (`smt_casts.cpp`,
  `smt_byteops.cpp`, `smt_bitcast.cpp`, `smt_fp_conv.cpp`, …) to pick Int/Real
  builders over bit-vector ones.
- **Backends** — each solver implements the new virtuals natively, or rejects the
  theory. Z3 implements `mk_smt_int`/`mk_smt_real`; bit-vector-only backends
  (Bitwuzla, Boolector) `abort()` with "does not support integer encoding mode".
- **Capability guard** — `create_solver`/`pick_solver` in `solve.cpp` reject
  `--ir`/`--ir-ieee` for bit-vector-only solvers with a clean `exit(1)` instead of
  letting the backend abort at construction.
- **SMT-LIB output** — `smtlib_conv.cpp` selects the logic string from the mode
  (`options.get_bool_option("int-encoding") ? "QF_AUFLIRA" : "QF_AUFBV"`) and
  emits `(set-logic …)`.

## Worked example: integer/real arithmetic

Use these touch points as a checklist when adding a comparable theory. Each is
named by file/identifier; grep to locate the current line.

1. **Define the user-facing option.** `src/esbmc/options.cpp` declares `--ir` and
   `--ir-ieee`; both set the internal `int-encoding` option.

2. **Plumb the option into the converter.** `smt_solver_baset` reads it once in
   its constructor: `int_encoding = options.get_bool_option("int-encoding")`
   (`smt_solver.cpp`). Everything downstream branches on this flag.

3. **Add sorts.** Ensure the sort kinds exist in `smt_sort.h`
   (`SMT_SORT_INT`, `SMT_SORT_REAL`) and that backends can build them.

4. **Add operations to the virtual surface.** Declare the new builders on
   `smt_solver_baset` in `smt_solver.h` (e.g. `mk_smt_int`, `mk_smt_real`, and the
   arithmetic/relational ops). Give them an `abort()`/diagnostic default so a
   backend that has not implemented them fails loudly rather than silently
   miscompiling.

5. **Dispatch the encoding.** In the SMT conversion (`smt_conv.cpp` and the
   per-operation files under `src/solvers/smt/`), branch on `int_encoding` to emit
   Int/Real builders instead of the bit-vector ones.

6. **Implement per backend.** Native support: `z3_convt::mk_smt_int` /
   `mk_smt_real` in `z3_conv.cpp`. No support: the BV-only pattern in
   `bitwuzla_conv.cpp` / `boolector_conv.cpp`, which logs and `abort()`s — and is
   pre-empted by the `solve.cpp` capability guard (step 7).

7. **Guard incompatible solvers.** Add the new mode to the rejection check in
   `solve.cpp` (the same place that already refuses `--ir` for Bitwuzla/Boolector),
   so an unsupported `--mytheory --bitwuzla` exits cleanly with a clear message.

8. **Emit the right SMT-LIB logic.** Extend the logic selection in
   `smtlib_conv.cpp` so the textual backend declares the correct `(set-logic …)`.

9. **Adjust the pointer/dereference model if addressing is affected.**
   `src/pointer-analysis/dereference.cpp` assumes a representation of addresses; a
   theory that changes how integers/pointers are modelled must be reconciled here.

10. **Add simplifications (optional).** Constant-folding or rewrites for the new
    operations go in `src/util/simplify_expr.cpp`. Optional for correctness, useful
    for performance.

## Validation

- Build with a backend that supports the theory (e.g. `-DENABLE_Z3=On`) and run a
  small program with and without the new mode; confirm the verdicts agree on
  cases where the theory is sound, and that an unsupported-solver combination
  exits cleanly (not via `abort()`).
- Add at least two regression tests (one passing `CORE`, one failing) whose
  `test.desc` flag line exercises the new mode.
- Differentially compare against the bit-vector encoding on a representative
  subset; divergence usually signals an over-approximation made unsound or a
  missed `int_encoding` branch.

## Maintainability & limitations

- **Capability negotiation, not silent fallback.** A backend that cannot express
  the theory must reject it (`solve.cpp` guard + a loud backend default), never
  emit wrong semantics. Where a *flattener* can lower the theory onto bit-vectors
  (as the array/FP/tuple flatteners do), prefer that to per-backend duplication.
- **Mode flags are viral.** `int_encoding` is read in many `src/solvers/smt/*.cpp`
  files; a new flag-gated theory will likely touch the same set. Keep the branch
  logic in the shared `smt/` layer, not copied into each backend.
- **Over-approximation is a soundness commitment.** Integer/real mode trades
  bit-precise wraparound for unbounded Int/Real (faster, over-approximating).
  Document precisely what your theory approximates, and which checks become
  unsound under it.
- **Addressing-sensitive theories need the pointer model.** Skipping
  `dereference.cpp` for a theory that changes integer/pointer representation
  produces subtle, hard-to-diagnose failures.
