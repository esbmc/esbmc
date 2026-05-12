# `src/solvers/` — ESBMC's SMT backend

This directory reduces SSA programs produced by ESBMC's symbolic execution
engine into SMT formulae, hands them to a back-end solver, and reads the
resulting model back out.  Backends are pluggable: each lives in its own
subdirectory and implements a small set of abstract interfaces defined in
`smt/`.

The cleanest reference backend is `bitwuzla/` (~1 kLoC, modern Bitwuzla
C API).  The most thorough in-source documentation is the file-level
Doxygen comment at the top of [`smt/smt_conv.h`](smt/smt_conv.h) — start
there if you want a deep dive.

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

## Adding a new SMT backend

The canonical small reference is `bitwuzla/`.  There is also a longer
narrative on the wiki:
[Integrate a new SMT solver into the ESBMC backend][wiki-solver].
That page is written against `z3/` as the template and some of its
line-number citations have drifted; treat `bitwuzla/` as the up-to-date
reference and the wiki as background.

**In-tree (under `src/solvers/`):**

1. Create `src/solvers/<name>/` with `<name>_conv.{h,cpp}` and a
   `CMakeLists.txt`.
2. Declare your AST wrapper:
   `class <name>_smt_ast : public solver_smt_ast<NativeTerm>`.
3. Declare your converter:
   `class <name>_convt : public smt_convt, public array_iface, public fp_convt`
   (mirror `bitwuzla_convt`).  Drop `array_iface` / `fp_convt` if the
   solver lacks native arrays / FP and you intend to use the fall-backs.
4. Implement the mandatory overrides — by family:
   - Bit-vector arithmetic and logic: `mk_bv{add,sub,mul,sdiv,udiv,smod,umod,shl,ashr,lshr,neg,not,and,or,xor}`.
   - Boolean: `mk_{and,or,xor,not,implies}`, `mk_ite`.
   - Comparison and equality: `mk_bv{ult,slt,ugt,sgt,ule,sle,uge,sge}`, `mk_eq`, `mk_neq`.
   - Arrays: `mk_store`, `mk_select`, `convert_array_of`.
   - Sort constructors: `mk_bool_sort`, `mk_bv_sort`, `mk_array_sort`, `mk_fbv_sort`, `mk_bvfp_sort`, `mk_bvfp_rm_sort`.
   - Literals and symbols: `mk_smt_{bool,int,real,bv,symbol}`, `mk_array_symbol`.
   - Bit slicing: `mk_extract`, `mk_sign_ext`, `mk_zero_ext`, `mk_concat`.
   - Solver control: `assert_ast`, `dec_solve`, `push_ctx`, `pop_ctx`, `solver_text`.
   - Model readback: `get_bool`, `get_bv`, `get_array_elem`.
   - Overflow + quantifiers: `overflow_arith`, `mk_quantifier`.
   - Debug helpers: `dump_smt`, `print_model`.
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
- `BUILDING.md` — install/dependency notes for the new solver.
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
- `src/solvers/smt/simplify_expr.cpp` — simplification rules for new
  operators, if any.
- Each affected backend — typically `bitwuzla/`, `boolector/`, and the
  text backend `smtlib/smtlib_conv.cpp`.

## Real-arithmetic FP mode (`--ir-ieee`)

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

## Further reading

- File-level Doxygen comment in [`smt/smt_conv.h`](smt/smt_conv.h) —
  authoritative description of what `smt_convt` flattens and why.
- [`bitwuzla/bitwuzla_conv.{h,cpp}`](bitwuzla/) — canonical small
  backend, recommended starting point for new integrations.
- [`solve.cpp`](solve.cpp) — factory plumbing and default-solver
  priority list.
- Wiki: [Integrate a new SMT solver][wiki-solver] (long-form, written
  against `z3/`).
- Wiki: [Implement a new SMT theory][wiki-theory] (long-form, narrative
  walkthrough).

[wiki-solver]: https://github.com/esbmc/esbmc/wiki/Integrate-a-new-SMT-solver-into-the-ESBMC-backend
[wiki-theory]: https://github.com/esbmc/esbmc/wiki/Implement-a-new-SMT-theory-into-ESBMC
