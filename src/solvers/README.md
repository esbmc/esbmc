# `src/solvers/` — ESBMC's SMT backend

*This document is intended for developers extending ESBMC with new SMT
backends or new SMT encodings.  If you are just trying to **use** an
existing solver, the project [`README.md`](../../README.md) and the
[setup docs](../../website/content/docs/setup.md) on the website are
the right starting points.*

This directory reduces SSA programs produced by ESBMC's symbolic execution
engine into SMT formulae, hands them to a back-end solver, and reads the
resulting model back out.  Backends are pluggable: each lives in its own
subdirectory and implements a small set of abstract interfaces defined in
`smt/`.

The cleanest reference backend is `bitwuzla/` (~1 kLoC, modern Bitwuzla
C API).  The most thorough in-source documentation is the file-level
Doxygen comment at the top of [`smt/smt_conv.h`](smt/smt_conv.h) — start
there if you want a deep dive.

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

## Directory layout

| Path | Contents |
|------|----------|
| `smt/` | Solver-agnostic core: `smt_convt`, `smt_ast`, `smt_sort`, byte ops, casts, memory model, overflow encoding |
| `smt/tuple/` | Tuple flattening for solvers that lack native tuples (node-based and symbol-based variants) |
| `smt/fp/` | Floating-point flattening (`fp_convt`) for solvers without native FP |
| `prop/` | Shared with SAT backends: `literal.h`, `pointer_logic.{cpp,h}` |
| `bitwuzla/`, `z3/`, `boolector/`, `cvc4/`, `cvc5/`, `mathsat/`, `yices/`, `smtlib/`, `minisat/`, `sat/` | Per-backend subclasses |
| `solve.{cpp,h}` | Factory that picks and instantiates a backend |
| `solver_config.h.in` | Compile-time configuration (`#cmakedefine` per solver) |

## Architecture

Three abstract classes carry the design:

- **`smt_convt`** — represents a live solver context.  Owns the cache,
  the memory model, byte-op lowering, casts, union flattening, FP
  fall-backs, and the dispatcher `mk_func_app`.  Each backend subclasses
  it and implements solver-specific function/sort/literal builders.
- **`smt_ast`** — wraps a single term in the backend's native AST type.
  Backends typically subclass the templated helper
  `solver_smt_ast<NativeTerm>` (see `bitw_smt_ast` in
  `bitwuzla/bitwuzla_conv.h` for the canonical pattern).
- **`smt_sort`** — wraps a sort.  Some backends (e.g. Boolector) get
  away without subclassing it.

`smt_convt` flattens, *for every backend*: the C memory address space,
pointer representation, casts, byte extract/update, fixed-bv float
encoding, unions, and overflow detection.  Three optional interfaces let
a backend opt in to native handling where the solver supports it:

| Interface | If implemented | Fallback |
|-----------|----------------|----------|
| `array_iface` | Solver's native arrays | `array_conv` (Kroening's decision procedure) |
| `tuple_iface` | Solver's native tuples / datatypes | `smt_tuple_node` or `smt_tuple_sym` |
| `fp_convt` (subclassed) | Solver's native FP theory | IEEE 754 bit-vector encoding |

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

## Adding a new SMT backend

The canonical small reference is `bitwuzla/`.  There is also a longer
narrative on the wiki:
[Integrate a new SMT solver into the ESBMC backend][wiki-solver].
That page is written against `z3/` as the template and some of its
line-number citations have drifted; treat `bitwuzla/` as the up-to-date
reference and the wiki as background.

### Quick start

1. Copy `bitwuzla/` to `<name>/` and rename `bitwuzla` → `<name>`
   throughout.  Replace Bitwuzla API calls with your solver's API; keep
   the class hierarchy intact.
2. Wire it into the build and CLI as described in *In-tree* and
   *Out-of-tree* below.
3. From a clean rebuild, smoke-test with a one-line C program:

   ```sh
   echo 'int main(){int x; assert(x == x);}' > t.c
   esbmc t.c --<name>          # expect: VERIFICATION SUCCESSFUL
   esbmc t.c --<name> --smtlib --smt-formula-only --output /dev/stdout
   ```

   The first command exercises the full lifecycle (`convert_ast` →
   `assert_ast` → `dec_solve` → no model needed); the second dumps the
   formula your backend would receive — invaluable when a real query
   misbehaves.  Once that works, run the regression suite filtered by
   `-L esbmc` with your `--<name>` flag wired in.

   If the first command does *not* return `VERIFICATION SUCCESSFUL`,
   the cause is almost always in Stage 1: a missing literal builder
   (`mk_smt_bool` / `mk_smt_bv` / `mk_smt_symbol`) or equality
   (`mk_eq`), or a broken solver hand-off (`assert_ast` / `dec_solve`).
   Start by implementing `dump_smt` and re-running with
   `--smt-formula-only` to see exactly what your backend produced.

### In-tree (under `src/solvers/`)

1. Create `src/solvers/<name>/` with `<name>_conv.{h,cpp}` and a
   `CMakeLists.txt`.
2. Declare your AST wrapper:
   `class <name>_smt_ast : public solver_smt_ast<NativeTerm>`.
3. Declare your converter:
   `class <name>_convt : public smt_convt, public array_iface, public fp_convt`
   (mirror `bitwuzla_convt`).  Drop `array_iface` / `fp_convt` if the
   solver lacks native arrays / FP and you intend to use the fall-backs.
4. Implement the overrides.  The base class declares ~50 virtual methods;
   build them up in this order — each stage produces a backend that can
   run progressively richer programs.

   **Stage 1 — Core (minimum to compile and solve a trivial query).**
   These methods are pure-virtual in `smt_convt`, or their defaults
   `abort()`.  Without them the class will not instantiate, or will die
   on the first call.

   - Solver lifecycle: `assert_ast`, `dec_solve`, `push_ctx`, `pop_ctx`,
     `solver_text`.
   - Sorts: `mk_bool_sort`, `mk_bv_sort`.
   - Literals and symbols: `mk_smt_bool`, `mk_smt_bv`, `mk_smt_symbol`.
   - Boolean glue: `mk_and`, `mk_or`, `mk_not`, `mk_eq`, `mk_neq`,
     `mk_ite`.
   - Bit slicing: `mk_extract`, `mk_sign_ext`, `mk_zero_ext`,
     `mk_concat`.
   - Model readback: `get_bool`, `get_bv`.

   At this point a trivial program with `--<name>` should reach
   `VERIFICATION SUCCESSFUL` / `FAILED` cleanly.  **This is your first
   milestone — the backend is correctly wired end-to-end and every
   subsequent stage is additive.**

   **Stage 2 — Common (any non-trivial C program needs these).**

   - Bit-vector arithmetic and logic: `mk_bv{add,sub,mul,sdiv,udiv,smod,umod,shl,ashr,lshr,neg,not,and,or,xor}`.
   - Bit-vector comparison: `mk_bv{ult,slt,ugt,sgt,ule,sle,uge,sge}`.
   - Boolean extras: `mk_xor`, `mk_implies`.
   - Arrays: `mk_array_sort`, `mk_array_symbol`, `mk_store`,
     `mk_select`, `convert_array_of`, `get_array_elem`.
   - Floats (BV-encoded): `mk_fbv_sort`, `mk_bvfp_sort`,
     `mk_bvfp_rm_sort`.

   After Stage 2, most of the `regression/esbmc` suite should run.

   **Stage 3 — Advanced and optional.**

   - Integer / real theories: `mk_smt_int`, `mk_smt_real` — only needed
     for solvers used with `--ir-ieee` or other non-BV encodings.
   - Quantifiers: `mk_quantifier` — required only if you intend to
     support `__ESBMC_forall` / `__ESBMC_exists`.
   - Overflow: `overflow_arith` — if your solver has native overflow
     predicates; otherwise the BV fall-back is fine.
   - Debug helpers: `dump_smt`, `print_model` — non-functional but
     greatly speed up triage; implement before Stage 2 if you can.

5. Expose a factory function
   `smt_convt *create_new_<name>_solver(const optionst &, const namespacet &, ...)`
   and register it in [`solve.cpp`](solve.cpp): add a `solver_creator`
   forward declaration, an entry in `esbmc_solvers` under
   `#ifdef <NAME>`, and add `"<name>"` to the `all_solvers` priority
   list (its position determines default-selection order when no
   solver is explicitly requested).
6. Wire CMake.  Mirror `bitwuzla/CMakeLists.txt`: a guarded
   `add_library(solver<name> ...)`, link it with
   `target_link_libraries(solvers INTERFACE solver<name>)`, then set
   `ESBMC_ENABLE_<name> 1 PARENT_SCOPE` and append `<name>` to
   `ESBMC_AVAILABLE_SOLVERS`.
7. Add `add_subdirectory(<name>)` to [`CMakeLists.txt`](CMakeLists.txt).

**Out-of-tree (elsewhere in the repo):**

- `src/solvers/solver_config.h.in` — add the `#cmakedefine <NAME>`.
- `src/esbmc/options.cpp` — register the `--<solver>` command-line flag.
- `src/esbmc/esbmc_parseoptions.cpp` — extend solver-selection.
- Top-level [`README.md`](../../README.md) and `scripts/build.sh` —
  install/dependency notes for the new solver.
- `.github/workflows/build.yml` and `.github/workflows/release.yml` —
  add the new solver to the CI matrix.
- `regression/esbmc/` — add at least one passing and one failing
  regression test exercising `--<solver>` (per the project's two-test
  minimum for new features).

## Adding a new SMT theory or encoding

Extending ESBMC with a *new encoding* — for example, a real-arithmetic
fragment, an integer-encoded bit-vector lowering, or a new logic — is a
different axis from adding a solver and touches different files.  The
wiki has a narrative walkthrough:
[Implement a new SMT theory into ESBMC][wiki-theory].

The in-tree exemplar is the `--ir-ieee` real-arithmetic FP mode
(summarised below); its entry point is
`smt_convt::apply_ieee754_semantics` in
[`smt/smt_conv.cpp`](smt/smt_conv.cpp).  Touch-points to expect:

- `src/esbmc/options.cpp` — new CLI option.
- `src/esbmc/esbmc_parseoptions.cpp` — propagate the option into the
  engine.
- `src/esbmc/bmc.cpp` — wire the option into the BMC pipeline.
- `src/pointer-analysis/dereference.cpp` — only if the encoding changes
  the pointer/memory model.
- `src/solvers/smt/smt_conv.{h,cpp}` — encoding flag, any new
  `smt_func_kind` entries, and the encoding hook itself.
- `src/util/simplify_expr.cpp` — simplification rules for new operators,
  if any.
- Each affected backend — typically `bitwuzla/`, `boolector/`, and the
  text backend `smtlib/smtlib_conv.cpp`.

## Real-arithmetic FP mode (`--ir-ieee`)

**Intuition.** Floating-point operations are approximated using
real-arithmetic constraints with sound, symmetric error bounds: every
FP result is bracketed by `[r − ε, r + ε]` where `r` is the exact real
value and `ε` envelops the round-to-nearest rounding error.  The
encoding is cheaper for solvers without native FP, and never reports a
false `UNSAT`.

When `--ir-ieee` is set, floating-point operations are encoded in real
arithmetic rather than bit-precise FP.  `smt_convt::apply_ieee754_semantics`
(in `smt/smt_conv.cpp`) wraps each real-valued FP result in a sound
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

- File-level Doxygen comment in [`smt/smt_conv.h`](smt/smt_conv.h) —
  authoritative description of what `smt_convt` flattens and why.
- [`bitwuzla/bitwuzla_conv.{h,cpp}`](bitwuzla/) — canonical small
  backend, recommended starting point for new integrations.
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
