# `src/solvers/` — ESBMC's SMT backend

*This document is intended for developers extending ESBMC with new SMT
backends or new SMT encodings.  If you are just trying to **use** an
existing solver, the project [`README.md`](../../README.md) and the
[setup docs](../../website/content/docs/setup.md) on the website are
the right starting points.*

This directory reduces SSA programs produced by ESBMC's symbolic execution
engine into SMT formulae, hands them to a back-end solver, and reads the
resulting model back out.

There is one backend, [`camada_conv.cpp`](camada_conv.cpp), which reaches
every solver through [camada](https://github.com/mikhailramalho/camada) — Z3,
cvc5, MathSAT, Yices and Bitwuzla natively, plus any other solver over
SMT-LIB2, interactively or as a one-shot subprocess. It implements
`smt_solver_baset`, the interface the rest of this directory is written
against.

The most thorough in-source documentation is the file-level Doxygen comment at
the top of [`smt_conv.h`](smt_conv.h) — start there if you want a deep dive.

## Where this fits in the ESBMC pipeline

If you are new to ESBMC, read the top-level
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) first — it has the diagram
and prose overview of the full verification pipeline.  The short version:

    source  →  AST  →  GOTO program  →  symex / SSA  →  SMT formula  →  solver  →  model
                                                       └──────────  src/solvers/  ──────────┘

This directory owns the last two stages.  Its inputs are SSA-form
expressions in ESBMC's typed internal representation; its outputs are
SMT terms in a back-end solver's API and, on `SAT`, a model that is
decoded back into ESBMC values.

Neighbouring layers that this module collaborates with — read these
when something in `smt_convt` is not behaving the way you expect:

- [`src/irep2/`](../irep2/README.md) — the typed expression / type IR
  (`expr2tc`, `type2tc`).  Every input to `smt_convt::convert` is an
  `expr2tc`; understanding the IR is a prerequisite for understanding
  what the converter is destructuring.
- [`src/pointer-analysis/`](../pointer-analysis/README.md) — ESBMC's
  memory model.  Pointer dereferencing happens *upstream* of this
  directory; `smt_convt` only sees the lowered byte / array operations.
- [`src/goto-symex/`](../goto-symex/) — symbolic execution.  Anything
  involving control-flow guards, dynamic allocation, or pointer
  liveness belongs there, not here.

User-facing theory and developer docs on the website mirror parts of
this pipeline and are worth a skim:

- [SMT Formula Generation](../../website/content/docs/theory/smt-formula-generation.md)
  — how to dump and inspect the formula ESBMC produces (`--smtlib
  --smt-formula-only`).  Essential for debugging a new backend or
  encoding.
- [Adding New Expressions](../../website/content/docs/development/Adding-new-expressions.md)
  — narrative walkthrough (quantifier example) of taking a new
  expression all the way from a frontend down to SMT.  Touches every
  layer above this directory and the encoding hooks inside it.
- [Architecture](../../website/content/docs/development/architecture.md)
  — project structure and conventions for contributors.

## Layout

One flat directory, one `solvers` library. There is a single backend now
(camada, covering every solver it supports plus its SMT-LIB text mode), so the
per-backend subdirectories the old structure was built around are gone.

| Path | Contents |
|------|----------|
| `smt_solver.{cpp,h}` | `smt_solver_baset`: the solver-agnostic core, and the interfaces a backend implements |
| `smt_conv.{cpp,h}` | `smt_convt`, the slim expr2tc-level facade the rest of ESBMC talks to |
| `smt_byteops.cpp`, `smt_casts.cpp`, `smt_bitcast.cpp`, `smt_overflow.cpp` | Lowering shared by every backend |
| `smt_memspace.cpp`, `pointer_logic.{cpp,h}` | The C memory address space and its bookkeeping |
| `smt_fp_conv.cpp` | IEEE754 constants, semantics and FP predicates |
| `ir_ieee_conv.{cpp,h}` | Real-arithmetic FP encoding (`--ir --ir-ieee`) |
| `smt_sort.h`, `smt_result.h` | Solver-handle and result vocabulary |
| `camada_conv.cpp` | The backend |
| `oneshot_options.{cpp,h}` | Option and temp-file policy for `--smtlib-oneshot-prog` |
| `solve.{cpp,h}` | Factory that picks and instantiates a backend |
| `solver_config.h.in` | Compile-time configuration (`#cmakedefine` per solver) |

## Architecture

Three abstract classes carry the design:

- **`smt_convt`** — represents a live solver context.  Owns the cache,
  the memory model, byte-op lowering, casts, union flattening, FP
  fall-backs, and the dispatcher `mk_func_app`.  Each backend subclasses
  it and implements solver-specific function/sort/literal builders.
- **`smt_astt`** / **`smt_sortt`** — aliases for camada's own `SMTExprRef`
  and `SMTSortRef` handles (`smt_sort.h`), not ESBMC wrappers around them.
  A camada expression carries its own sort, so nothing here stores one.
  Operations that depend on the operand's sort rather than its C++ type
  (`ast_eq`, `ast_assign`, `ast_update`, `ast_select`, `ast_project`) are
  methods on `smt_solver_baset`.

`smt_convt` flattens, *for every backend*: the C memory address space,
pointer representation, casts, byte extract/update, fixed-bv float
encoding, unions, and overflow detection.  Three optional interfaces let
a backend opt in to native handling where the solver supports it:

| Interface | If implemented | Fallback |
|-----------|----------------|----------|

If you find yourself flattening anything *more* than the items above —
for instance, anything that touches pointer dereferencing, control-flow
guards, or dynamic allocation — that work belongs in symbolic execution,
not here.

### Lifecycle of one query

Symex hands `smt_convt` an SSA program one expression at a time and
later asks for a verdict.  The sequence of calls is:

    smt_convt::assert_expr(expr2tc)
        └─> convert_ast(expr2tc)              // defined in smt_conv.cpp
                └─> mk_* family               // your overrides build native terms
        └─> assert_ast(smt_astt)              // your override hands term to solver

    ... repeat for every SSA assertion ...

    dec_solve()                               // your override invokes the solver
        └─> returns SAT / UNSAT / UNKNOWN

    if SAT:
        get_bool / get_bv / get_array_elem    // your overrides read the model

`push_ctx` / `pop_ctx` bracket incremental queries.  This is the only
temporal contract a backend must honour: terms produced by `mk_*` are
asserted via `assert_ast`, the solver is invoked exactly once per
verdict via `dec_solve`, and model values are extracted only after a
SAT result.

## Adding a new solver

There is no ESBMC-side backend to write any more: a new solver is added to
[camada](https://github.com/mikhailramalho/camada), and ESBMC picks it up
through `camada_conv.cpp`. Two routes:

**Linked in.** Implement camada's `SMTSolverImpl` for the solver's C/C++ API
(see camada's own `z3solver`, `bitwuzlasolver`, … for the pattern). On the
ESBMC side: add a `camada_backendt` enumerator, construct it in
`camada_convt`'s ctor switch, add a `create_new_<name>_solver` factory, and
register the name in `solve.cpp`'s `esbmc_solvers` map plus the `ENABLE_<NAME>`
plumbing in `CMakeLists.txt` and `solver_config.h.in`.

**Over SMT-LIB2, no code at all.** If the solver speaks SMT-LIB2 on stdin,
`--smtlib --smtlib-solver-prog "<cmd>"` already drives it. If it only reads a
file and prints a verdict, `--smtlib --smtlib-oneshot-prog "<cmd> %f"` does
that, with `--smtlib-oneshot-model-prog` supplying the model for
counterexamples and `--smtlib-logic` pinning the fragment it accepts. Mallob
and NeuroSym are driven this way.

Prefer the second route unless the solver's native API buys something the
text interface cannot.

## Adding a new SMT theory or encoding

Extending ESBMC with a *new encoding* — for example, a real-arithmetic
fragment, an integer-encoded bit-vector lowering, or a new logic — is a
different axis from adding a solver and touches different files.  The
wiki has a narrative walkthrough:
[Implement a new SMT theory into ESBMC][wiki-theory].

The in-tree exemplar is the `--ir-ieee` real-arithmetic FP mode
(summarised below); its entry point is
`smt_convt::apply_ieee754_semantics` in
[`smt_conv.cpp`](smt_conv.cpp).  Touch-points to expect:

- `src/esbmc/options.cpp` — new CLI option.
- `src/esbmc/esbmc_parseoptions.cpp` — propagate the option into the
  engine.
- `src/esbmc/bmc.cpp` — wire the option into the BMC pipeline.
- `src/pointer-analysis/dereference.cpp` — only if the encoding changes
  the pointer/memory model.
- `src/solvers/smt_conv.{h,cpp}` — encoding flag, any new
  `smt_func_kind` entries, and the encoding hook itself.
- `src/util/expr/expr_simplifier.cpp` — simplification rules for new operators,
  if any.
- `camada_conv.cpp` — the one backend, covering every solver camada
  supports plus its SMT-LIB text mode.

## Real-arithmetic FP mode (`--ir-ieee`)

**Intuition.** Floating-point operations are approximated using
real-arithmetic constraints with sound, symmetric error bounds: every
FP result is bracketed by `[r − ε, r + ε]` where `r` is the exact real
value and `ε` envelops the round-to-nearest rounding error.  The
encoding is cheaper for solvers without native FP, and never reports a
false `UNSAT`.

When `--ir-ieee` is set, floating-point operations are encoded in real
arithmetic rather than bit-precise FP.  `smt_convt::apply_ieee754_semantics`
(in `smt_conv.cpp`) wraps each real-valued FP result in a sound
symmetric error enclosure derived from the round-to-nearest model:

    |fl(r) - r| <= eps_rel * |r| + eps_abs

where `eps_rel` is half the machine epsilon (2⁻⁵³ double, 2⁻²⁴ single)
and `eps_abs` is the minimum positive subnormal (2⁻¹⁰⁷⁴ double,
2⁻¹⁴⁹ single), covering the underflow region.  The enclosure asserts
`r - (eps_rel * |r| + eps_abs) <= result <= r + (eps_rel * |r| + eps_abs)`
together with a sanity bound `lo <= hi`.  Bidirectional inequalities
are used rather than equalities so the bounds survive Z3's `solve-eqs`
tactic.

Epsilon constants come from four helpers in `smt_conv.cpp`
(`get_double_eps_rel`, `get_single_eps_rel`, `get_double_min_subnormal`,
`get_single_min_subnormal`), each rounded *upward* at the last decimal
digit so the parsed value is `>=` the true power of two — preserving
soundness of the enclosure.  Non-standard FP formats currently fall back
to an unconstrained (weak) enclosure.

## Debugging and validation

A few habits will save hours when bringing up a new backend or encoding:

- **Dump the formula and read it.**  Even when your backend is selected,
  point ESBMC at the text backend to inspect what was produced:

  ```sh
  esbmc t.c --smtlib --smt-formula-only --output t.smt2
  ```

  Cross-check the dumped term shape against what your backend's
  `mk_*` overrides produce.  See the website page
  [SMT Formula Generation](../../website/content/docs/theory/smt-formula-generation.md)
  for the supported dump options.
- **Use `dump_smt` / `print_model`.**  Both are virtual hooks on
  `smt_convt`; implementing them early turns "the solver said no" into
  "here is the assertion that failed and the model the solver returned".
- **Watch for sort mismatches.**  Most native solver APIs reject
  applications whose argument sorts disagree; ESBMC will surface that as
  an abort deep inside `mk_func_app`.  When you see one, log the sorts
  of the offending arguments before calling the solver — the smallest
  reproducer is usually an `assert_ast` on a single equality.
- **Validate model readback on bit-vectors of every width you support.**
  `get_bv` is invoked on widths from 1 up to 64+ (and beyond for
  multi-word integers); a backend that silently truncates large values
  will pass small regression tests and fail subtly on the full suite.
  The simplest regression is an `__ESBMC_assume(x == 0xDEADBEEFCAFEBABE)`
  followed by an `assert(x == 0)` — a wrong-width readback will pass.
- **Compare against `bitwuzla` and `z3`.**  Both are mature; if all
  three backends agree on a test the encoding is almost certainly
  right.  ESBMC's CI matrix does exactly this.  Disagreement is your
  signal to dump the formula.
- **Re-build operational-model files when relevant.**  Files under
  `src/c2goto/library/` and `src/cpp/library/` are mangled by
  `flail.py` and linked into the `esbmc` binary; edits there are
  invisible until the binary is rebuilt.
- **Sanitizers are your friend on the C++ side.**  A backend that
  forgets to ref-count or releases a term twice will only crash
  intermittently in CI; build with `-fsanitize=address,undefined` for
  the development loop.

## Further reading

In-tree, in this directory:

- File-level Doxygen comment in [`smt_conv.h`](smt_conv.h) —
  authoritative description of what `smt_convt` flattens and why.
- [`camada_conv.cpp`](camada_conv.cpp) — the backend, and the reference for
  how `smt_solver_baset` is implemented.
- [`solve.cpp`](solve.cpp) — factory plumbing and default-solver
  priority list.

In-tree, neighbouring layers:

- [`ARCHITECTURE.md`](../../ARCHITECTURE.md) — top-level pipeline.
- [`src/irep2/README.md`](../irep2/README.md) — the expression IR
  consumed here.
- [`src/pointer-analysis/README.md`](../pointer-analysis/README.md) —
  ESBMC's memory model.
- [SMT Formula Generation](../../website/content/docs/theory/smt-formula-generation.md)
  and [Adding New Expressions](../../website/content/docs/development/Adding-new-expressions.md)
  on the project website.
- [`CONTRIBUTIONS.md`](../../CONTRIBUTIONS.md) — general contribution
  workflow.

On the wiki:

- [Integrate a new SMT solver][wiki-solver] (long-form, written against
  `z3/`).
- [Implement a new SMT theory][wiki-theory] (long-form, narrative
  walkthrough).

[wiki-solver]: https://github.com/esbmc/esbmc/wiki/Integrate-a-new-SMT-solver-into-the-ESBMC-backend
[wiki-theory]: https://github.com/esbmc/esbmc/wiki/Implement-a-new-SMT-theory-into-ESBMC
