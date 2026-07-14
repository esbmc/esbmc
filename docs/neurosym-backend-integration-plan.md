# Integrating NeuroSym as an SMT Backend in ESBMC — Technical Design Document

**Status:** Draft design / plan
**Author:** Lucas C. Cordeiro (ESBMC team)
**Scope:** Add the NeuroSym neural-guided SMT solver (GAN + Z3 fallback, QF_BV / QF_LIA over SMT-LIB2) as a selectable ESBMC backend.
**Grounding (ESBMC side):** All ESBMC references below were read from the current `master` tree. Key files: `src/solvers/solve.cpp`, `src/solvers/solve.h`, `src/solvers/solver_config.h.in`, `src/solvers/smtlib/smtlib_conv.{h,cpp}`, `src/solvers/bitwuzllob/bitwuzllob_conv.{h,cpp}`, `src/esbmc/options.cpp`, `src/esbmc/parseoptions/{command_line_options,driver}.cpp`, `src/solvers/CMakeLists.txt`.
**Grounding (NeuroSym side):** The NeuroSym claims in this document — the `python main.py --stdin` invocation, the QF_BV/QF_LIA-only fragment (no arrays, no reals, no quantifiers), the GAN-candidate + Z3-fallback control flow, the fixed-dimension feature encodings (8576 QF_LIA / 10752 QF_BV), and the existing C++ code being a KLEE bridge that shells out to Python — describe the NeuroSym research prototype and are **not yet pinned** to a public reference (NeuroSym is not publicly indexed at the time of writing). Before implementation starts, replace this placeholder with the exact source: **NeuroSym repository:** `<repo URL @ commit hash — TBD>`; **paper:** `<DOI / venue — TBD, if published>`. Until pinned, every NeuroSym-side claim must be re-validated against the actual NeuroSym checkout (the Phase 1 PoC and the §8 open questions do exactly this).

---

## 0. Executive summary and the one decisive fact

ESBMC **already contains** a production subprocess-over-SMT-LIB2 backend and a worked example of subclassing it:

- `smtlib_convt` (`src/solvers/smtlib/smtlib_conv.{h,cpp}`) is the solver-independent SMT-LIB2 serializer. It implements every `smt_solver_baset` term builder by *printing* SMT-LIB2 and can drive an external solver two ways: an interactive **stdin/stdout pipe** (`process_emitter`, selected with `--smtlib-solver-prog <cmd>`) and/or a **file** (`file_emitter`, `--output`).
- `bitwuzllob_convt` (`src/solvers/bitwuzllob/bitwuzllob_conv.{h,cpp}`) is a ~450-line backend (414-line `.cpp` + 42-line header) that **derives from `smtlib_convt`**, reuses its serializer, spawns an external solver as a subprocess, parses `sat`/`unsat`/`unknown` from its stdout, and uses a second interactive SMT-LIB2 process only when a model is needed. It is registered in `solve.cpp` exactly like a native backend.

NeuroSym is invoked as `python main.py --stdin`, speaks SMT-LIB2, and returns `sat`/`unsat` (+ model) on stdout. **This is the bitwuzllob shape almost exactly.** The recommended design is therefore a new `neurosym_convt : public smtlib_convt`, modelled line-for-line on `bitwuzllob_convt`.

Three tiers of effort, stated up front and kept distinct throughout:

| Tier | What | Effort |
|---|---|---|
| **A — usable today, zero ESBMC code** | `esbmc file.c --smtlib --smtlib-solver-prog "python main.py --stdin"` — *works only if* NeuroSym implements ESBMC's interactive protocol (incremental `push`/`pop`, `(get-value …)`, `set-logic QF_AUFBV`). It almost certainly does **not** (NeuroSym is batch, QF_BV/QF_LIA only, no arrays). Useful as a first smoke test, not as the product. | hours |
| **B — the recommended deliverable** | `neurosym_convt` backend + CLI + build + tests. All work is in ESBMC; **no changes to NeuroSym required** if we accept batch/one-shot solving and force array/FP/tuple flattening to pure QF_BV. | ~1–2 weeks |
| **C — requires NeuroSym changes** | Incremental (`push`/`pop`) support, native `(get-value)` for every sort, warm-started process reuse, QF_ABV arrays. Needed for k-induction and low-overhead model extraction. | depends on NeuroSym authors |

---

## 1. Architecture assessment

### 1.1 How ESBMC talks to a solver (verified)

Pipeline: frontend → GOTO → symbolic execution → SSA → **SMT encoding** → solver. The encoding funnels through one abstraction pair:

- **`smt_solver_baset`** (`src/solvers/smt/smt_solver.h`) — abstract base every backend derives from. Declares the ~100 virtual term/sort builders (`mk_smt_bv`, `mk_bvadd`, `mk_extract`, `mk_concat`, `mk_ite`, `mk_smt_symbol`, `mk_bv_sort`, …) plus solving/model entry points (`dec_solve`, `push_ctx`/`pop_ctx`, `get_bool`, `get_bv`, `get_fpbv`, `get_array_elem`). **This virtual surface is the contract.**
- **`smt_convt`** (`src/solvers/smt/smt_conv.h`) — solver-independent driver. It *wraps* a `smt_solver_baset` and feeds it the encoded SSA. **You do not subclass it.** `create_solver` in `solve.cpp` constructs the backend, attaches flatteners, and wraps it: `return new smt_convt(std::unique_ptr<smt_solver_baset>(ctx));`
- **Flatteners** lower features a backend lacks onto plain bit-vectors:
  - `array_convt` (`src/solvers/smt/array_conv.h`) — arrays → BV.
  - `fp_convt` (`src/solvers/smt/fp/fp_conv.h`) — IEEE floats → BV (`--fp2bv`).
  - `smt_tuple_node_flattener` / `smt_tuple_sym_flattener` — structs/tuples → BV.

`create_solver` (read from `solve.cpp`) decides per-capability whether to use the backend's native interface or a flattener:

```cpp
if (tuple_api != nullptr && !node_flat && !sym_flat) ctx->set_tuple_iface(tuple_api);
else ctx->set_tuple_iface(new smt_tuple_node_flattener(ctx, ns));   // default
if (array_api != nullptr && !array_flat) ctx->set_array_iface(array_api);
else ctx->set_array_iface(new array_convt(ctx));                    // default
if (fp_api == nullptr || fp_to_bv) ctx->set_fp_conv(new fp_convt(ctx));
else ctx->set_fp_conv(fp_api);
```

A backend factory signals "I have no native arrays/FP/tuples" by leaving `*array_api`/`*fp_api`/`*tuple_api` = `nullptr`, and ESBMC installs the flattener automatically. **This is exactly how we constrain everything down to NeuroSym's QF_BV fragment.**

### 1.2 Registration mechanism (verified, `solve.cpp`)

- A backend exposes a factory `smt_solver_baset *create_new_<name>_solver(const optionst&, const namespacet&, tuple_iface**, array_iface**, fp_convt**)` (`solver_creator` typedef in `solve.h`).
- It is entered in the compile-time map `esbmc_solvers` under `#ifdef <MACRO>`, and its name is added to `all_solvers[]` (whose order encodes default-selection priority). `pick_default_solver` additionally skips backends that depend on external programs (`smtlib`, `bitwuzllob`) so they are never chosen implicitly — NeuroSym must join this exclusion list (§2.2).
- `resolve_user_solver_choice` reads `options.get_bool_option("<name>")`, so a `--<name>` boolean flag selects it — **no factory edit needed** beyond the map/array entries.

### 1.3 The SMT-LIB backend, in detail (verified, `smtlib_conv.{h,cpp}`)

`class smtlib_convt : public smt_solver_baset, public array_iface, public fp_convt`. Relevant machinery we inherit for free:

- **Two output sinks**: `struct process_emitter emit_proc` (interactive pipe to `--smtlib-solver-prog`) and `struct file_emitter emit_opt_output` (`--output` file). `emit(...)` / `flush()` fan out to both.
- **Solve path**: `dec_solve()` (line 665) calls `pre_solve()`, emits `(check-sat)`, then `read_check_sat_response()` (line 683) which maps stdout to `P_SATISFIABLE` / `P_UNSATISFIABLE`; on unrecognized output it logs "Unrecognized check-sat output" and **aborts** (`smtlib_conv.cpp:707`). When no interactive solver is attached it returns `P_SMTLIB` (formula-only mode).
- **Header emitted before the formula** (line ~402–405): logic string is
  `options.get_bool_option("int-encoding") ? "QF_AUFLIRA" : "QF_AUFBV"`, then
  `(set-option :produce-models true)` and `(set-logic <logic>)`.
  **`QF_AUFBV` = arrays + UF + BV. This is the single biggest incompatibility with NeuroSym (QF_BV only) and must be overridden.**
- **Model extraction**: `get_value(smt_astt)` emits `(get-value (…))`, parses the returned s-expr (Bison/Flex parser in `smtlib.ypp`/`smtlib_tok.lpp`), and `get_bv`/`get_bool`/`get_array_elem` decode it. Requires the solver to answer `(get-value)` interactively.
- **Failure signalling**: `struct external_process_died : std::runtime_error` thrown on EPIPE / unparseable EOF — the mechanism by which a dead subprocess is turned into a clean ESBMC error rather than a crash.

### 1.4 The bitwuzllob precedent, in detail (verified, `bitwuzllob_conv.{h,cpp}`)

This is the template. Salient reusable patterns:

- **Ctor delegation**: `bitwuzllob_convt(ns, options)` → `smtlib_convt(ns, options, model_prog, formula_path)`, i.e. it configures the pipe/file sinks independently of `--smtlib-solver-prog`/`--output`.
- **One-shot guard**: `bool solved`; `dec_solve()` aborts on a second `(check-sat)` — "supports a single (check-sat) query per run; incremental strategies are not supported."
- **Strategy rejection at factory time**: it refuses `--incremental-bmc`, `--falsification`, `--k-induction`, `--k-induction-parallel`, `--termination`, `--smt-during-symex`, `--multi-property`, and `--parallel-solving` with a clear error (`bitwuzllob_conv.cpp:32-40`). NeuroSym (batch) needs the same guard, copied verbatim.
- **Verdict parsing**: `parse_verdict_line` accepts `sat`/`unsat`/`unknown` amid log noise; `unknown → P_ERROR`.
- **Signal hygiene**: forks the solver into its own process group and registers it for `killpg` on timeout/signal (`register_pgroup_for_cleanup`). Discards a verdict if the child died on a signal (`WIFSIGNALED`). **We inherit these lessons directly.**
- **Model via a side process**: because Mallob's one-shot process can't answer `(get-value)`, bitwuzllob spins up a *second* interactive SMT-LIB2 solver (`--bitwuzllob-model-prog`, e.g. `z3 -in`) fed the same formula, and only reads its model when a counterexample is required. **NeuroSym has the identical limitation and can use the identical trick** — with the elegant extra that NeuroSym's own internal Z3 fallback could serve the model directly if it exposed `(get-value)`.

### 1.5 Options comparison and recommendation

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Native in-memory wrapper** (copy the Z3 template; link a C/C++ API) | Lowest per-query overhead; native incremental; native `get-value` | **NeuroSym has no C/C++ API** (it is Python; the existing C++ is a KLEE bridge that itself shells out to Python). Building one is a NeuroSym project, not an ESBMC one. | ❌ not feasible now |
| **SMT-LIB2 file → subprocess** (`python main.py file.smt2`) | Simple; matches NeuroSym's file interface; trivially reproducible (keep the `.smt2`) | Process spin-up per query; model needs a second query/process; no incrementality | ✅ viable, but stdin is strictly better for latency |
| **SMT-LIB2 stdin/stdout subprocess** (`python main.py --stdin`) | Reuses `smtlib_convt` pipe machinery *as-is*; one write, one read; matches NeuroSym's documented `--stdin`; model-solver side-channel already implemented in bitwuzllob | Batch (one check-sat per process) unless NeuroSym adds incremental; Python startup cost per solve | ✅ **recommended** |

**Recommendation: Tier B — a `neurosym_convt : public smtlib_convt` subprocess backend**, modelled on `bitwuzllob_convt`, feeding NeuroSym via `--stdin`, with **all** non-BV features flattened to QF_BV, one-shot solving, and an optional interactive model side-solver. This is the only option that is (a) implementable entirely within ESBMC today, (b) sound (flatteners preserve semantics; NeuroSym's Z3 fallback preserves completeness on QF_BV/QF_LIA), and (c) aligned with an existing, reviewed ESBMC pattern.

---

## 2. Implementation plan

### 2.1 Files to create

```
src/solvers/neurosym/
  neurosym_conv.h          # class neurosym_convt : public smtlib_convt
  neurosym_conv.cpp        # dec_solve(), verdict parse, model side-solver wiring, factory
  CMakeLists.txt           # mirrors bitwuzllob/CMakeLists.txt
```

### 2.2 Files to modify

| File | Change |
|---|---|
| `src/solvers/CMakeLists.txt` | add `set(ESBMC_ENABLE_neurosym 0)` to the top block; `add_subdirectory(neurosym)` **after** `smtlib` (it reuses the smtlib library, like bitwuzllob). |
| `src/solvers/solver_config.h.in` | add `#if @ESBMC_ENABLE_neurosym@` / `#define NEUROSYM` / `#endif`. |
| `src/solvers/solve.cpp` | declare `solver_creator create_new_neurosym_solver;`; add `#ifdef NEUROSYM {"neurosym", create_new_neurosym_solver},#endif` to `esbmc_solvers`; add `"neurosym"` to `all_solvers[]` **immediately after `"bitwuzllob"`**, grouping the external-program backends at the head of the array. **Critically, also extend the never-implicit exclusion in `pick_default_solver` (`solve.cpp:71-77`)** — it skips `smtlib` and `bitwuzllob` because they "depend on external programs the user must configure, so they are never picked implicitly", and NeuroSym (a Python subprocess needing a NeuroSym checkout + PyTorch at runtime) is exactly this class: without the exclusion, an `-DENABLE_NEUROSYM=On` build would silently make an unconfigured Python subprocess the default solver. Add `"neurosym"` to the int-encoding-rejection guard **only if** we do not support QF_LIA (see §3.4). |
| `src/esbmc/options.cpp` | in the solver option group (near `{"bitwuzllob", …}`, lines ~526–538) add: `{"neurosym", NULL, "Use NeuroSym (neural-guided GAN + Z3 fallback, QF_BV/QF_LIA)"}`, `{"neurosym-prog", value<std::string>(), "Command to run NeuroSym (default: python main.py --stdin)"}`, `{"neurosym-model-prog", value<std::string>(), "Interactive SMT-LIB2 solver for counterexample models (e.g. \"z3 -in\")"}`, and optionally `{"neurosym-timeout", value<unsigned>(), …}`. |
| `src/esbmc/parseoptions/driver.cpp` | optional, cosmetic: add `cmdline.isset("neurosym")` to the `user_picked_solver` flag list in the Solidity solver auto-selection block (`driver.cpp:252-256`). Without it, a Solidity run with `--neurosym` logs a misleading "auto-selecting bitwuzla" message — harmless, since the explicit flag still wins in `resolve_user_solver_choice`, but confusing. No edit is needed for `--list-solvers`: it prints `ESBMC_AVAILABLE_SOLVERS` directly (`parseoptions/command_line_options.cpp:341`), which §2.7 extends via CMake (consistent with §2.6). |
| `BUILDING.md` | document `-DENABLE_NEUROSYM=On`, the Python/PyTorch/Z3 runtime prerequisites, and `--neurosym` usage. |
| `.github/workflows/build.yml` | (Phase 4/5) matrix entry that installs NeuroSym + PyTorch and runs a regression subset with `--neurosym`. |
| CMake option declaration | add `option(ENABLE_NEUROSYM "…" ON/OFF)` alongside the other `ENABLE_*` solver options (where `ENABLE_BITWUZLLOB` is declared). |

### 2.3 The `neurosym_convt` backend (skeleton)

```cpp
// neurosym_conv.h
#include <solvers/smtlib/smtlib_conv.h>

class neurosym_convt : public smtlib_convt
{
public:
  neurosym_convt(const namespacet &ns, const optionst &options);
  ~neurosym_convt() override;

  smt_resultt dec_solve() override;
  const std::string solver_text() override;
  std::string dump_smt() override;

private:
  neurosym_convt(const namespacet &ns, const optionst &options,
                 const std::string &formula_path);
  smt_resultt run_neurosym();            // spawn + parse verdict (copy run_bitwuzllob)
  std::string formula_path;
  bool solved = false;
};
```

`neurosym_conv.cpp` is a near-copy of `bitwuzllob_conv.cpp` with three deltas:

1. **Command**: default `python main.py --stdin` (configurable via `--neurosym-prog`). Because NeuroSym's `--stdin` reads a *whole formula then answers once*, we can either (a) render to a temp `.smt2` and run `python main.py <file>` (file mode — simplest, mirrors bitwuzllob's file+mono flow), or (b) write the formula to the process's stdin pipe and read stdout (stdin mode). Start with (a); it is the exact bitwuzllob code path and avoids pipe-deadlock subtleties.
2. **Logic string override**: NeuroSym rejects `QF_AUFBV`. Override header emission to emit `QF_BV` (BV mode) / `QF_LIA` (int mode). The cleanest hook: add a small `virtual std::string logic_string(const optionst&)` to `smtlib_convt` returning `QF_AUFBV`/`QF_AUFLIRA`, and override it in `neurosym_convt` to return `QF_BV`/`QF_LIA`. (Minimal, isolated change to the base class; alternative is post-processing the formula file to rewrite the `set-logic` line, which is hackier.)
3. **Verdict + model**: reuse `parse_verdict_line`; reuse the `--neurosym-model-prog` side-solver exactly as bitwuzllob uses `--bitwuzllob-model-prog`. If `--result-only`, no model needed.

**Factory** (bottom of `neurosym_conv.cpp`), critical for forcing the BV fragment:

```cpp
smt_solver_baset *create_new_neurosym_solver(
  const optionst &options, const namespacet &ns,
  tuple_iface **tuple_api, array_iface **array_api, fp_convt **fp_api)
{
  // reject incremental strategies NeuroSym's one-shot model can't serve
  static const char *incompatible[] = {
    "incremental-bmc","falsification","k-induction","k-induction-parallel",
    "termination","smt-during-symex","multi-property","parallel-solving"};
  for (const char *opt : incompatible)
    if (options.get_bool_option(opt)) { log_error(...); abort(); }

  neurosym_convt *conv = new neurosym_convt(ns, options);
  *tuple_api = nullptr;   // -> smt_tuple_node_flattener  (tuples -> BV)
  *array_api = nullptr;   // -> array_convt               (arrays -> BV)
  *fp_api    = nullptr;   // -> fp_convt                  (floats -> BV, automatic)
  return conv;
}
```

Leaving all three `*_api` = `nullptr` is what pins the whole query to pure QF_BV — every array/struct/float is flattened before it reaches NeuroSym. (Contrast bitwuzllob, which sets `*array_api`/`*fp_api` to itself because Bitwuzla is natively QF_AUFBV+FP.)

### 2.4 Translation ESBMC → SMT-LIB2

**No new serializer is written.** `smtlib_convt`'s `mk_*` builders already emit SMT-LIB2 for BV, and the flatteners feed only BV/bool terms into them. The only serialization delta is the `set-logic` line (§2.3.2). Everything else — `(declare-fun)` for symbols, `(assert)` per VCC, `(bvadd)`/`(concat)`/`(extract)`/… — is inherited verbatim.

### 2.5 Parsing responses back

- **SAT/UNSAT/UNKNOWN**: `run_neurosym()` scans NeuroSym's stdout with `parse_verdict_line` → `P_SATISFIABLE` / `P_UNSATISFIABLE` / `P_ERROR`.
- **Model → counterexample**: two supported paths, in preference order:
  1. **Side model-solver** (`--neurosym-model-prog "z3 -in"`): inherited `emit_proc` + `read_check_sat_response()` + `get_value()` reconstruct the model via ESBMC's existing s-expr parser (`smtlib.ypp`). Zero new parsing code. This is the Phase-3 default.
  2. **Native NeuroSym `(get-value)`** (Tier C): if NeuroSym is extended to answer `(get-value)` on its `--stdin` channel, point `emit_proc` at NeuroSym itself and drop the side-solver. Deferred pending NeuroSym changes.

### 2.6 Command-line exposure

`esbmc file.c --neurosym` selects the backend (data-driven via `all_solvers[]`). `--neurosym-prog`, `--neurosym-model-prog`, `--neurosym-timeout`, `--result-only`, `--output` behave as the bitwuzllob analogues. `--list-solvers` shows `neurosym` automatically from `ESBMC_AVAILABLE_SOLVERS`.

### 2.7 Build system

`src/solvers/neurosym/CMakeLists.txt` mirrors `bitwuzllob/CMakeLists.txt`: **no library to find/link** (the "solver" is a Python subprocess), requires `ENABLE_SMTLIB`, degrades gracefully if smtlib is off, links `solverneurosym` against `smtlib fmt::fmt`, and propagates `ESBMC_ENABLE_neurosym 1` + appends `neurosym` to `ESBMC_AVAILABLE_SOLVERS`. Because there is no native library, **build portability is trivial** — the runtime dependency (Python + PyTorch + a NeuroSym checkout) is a *deployment* concern, discovered at solve time, not link time.

---

## 3. Feature compatibility analysis

### 3.1 What real ESBMC workloads need vs what NeuroSym offers

ESBMC's default encoding is **QF_AUFBV** (`smtlib_conv.cpp:402`): quantifier-free arrays + uninterpreted functions + bit-vectors. Typical C/C++ verification uses:

- **Bit-vectors** — pervasive (all integer/pointer arithmetic). ✅ NeuroSym supports QF_BV.
- **Arrays** — pervasive (the memory model, `__ESBMC_alloc`, every heap object, `memcpy`, string buffers) via the theory of arrays. ❌ NeuroSym has no arrays → **must flatten with `array_convt`**.
- **Tuples/structs** — common. ❌ → flatten with `smt_tuple_node_flattener`.
- **Floating point** — common in numeric code. ❌ → flatten with `fp_convt` (installed automatically when `*fp_api` is left null — `solve.cpp:228`; no `--fp2bv` flag needed).
- **Uninterpreted functions** — used for some abstractions/nondeterminism. ❌ NeuroSym support unknown → see §3.3.

**Verdict:** NeuroSym's *native* fragment is far narrower than a typical ESBMC query, but ESBMC's flatteners can reduce arrays + tuples + FP to **pure QF_BV**, which NeuroSym does support. So the practical question is not "does NeuroSym support arrays?" (no, but it doesn't need to) but "how large/hard is the flattened QF_BV formula, and does neural guidance help on it?" (§6).

### 3.2 Compatibility matrix

| Feature / theory | ESBMC emits | NeuroSym native | Bridge strategy | Status |
|---|---|---|---|---|
| Bit-vectors (QF_BV) | yes | **yes** | direct | ✅ works |
| Booleans / propositional | yes | yes (subset of BV logics) | direct | ✅ works |
| Linear integer arith (QF_LIA) | only under `--ir`/`--ir-ieee` | **yes** | direct (needs `set-logic QF_LIA`) | ⚠️ see §3.4 |
| Arrays (memory model) | yes (default) | no | `array_convt` → BV | ✅ flattened |
| Structs / tuples | yes | no | tuple flattener → BV | ✅ flattened |
| IEEE floating-point | yes (default `--floatbv`) | no | `fp_convt` → BV (automatic, §1.1) | ✅ flattened |
| Uninterpreted functions | sometimes | unknown | avoid, or fail loudly / fall back | ⚠️ §3.3 |
| Quantifiers | rarely (some intrinsics) | no (QF_*) | not supported | ❌ fall back |
| Strings (SMT string theory) | no (ESBMC models strings as arrays) | no | already arrays → BV | ✅ N/A |
| Incremental push/pop | yes (k-induction, incremental-BMC) | no (batch) | reject strategy at factory | ⚠️ §3.5 |
| `(get-value)` models | required for CEX | unknown/no | side model-solver | ⚠️ §2.5 |

### 3.3 Uninterpreted functions

If ESBMC emits `(declare-fun f (…) …)` + applications and NeuroSym cannot parse UF, the query is outside its fragment. Options: (a) detect UF emission in `neurosym_convt` and route the whole query to the fallback model-solver / return `P_ERROR` cleanly; (b) enable ESBMC options that avoid UF where possible. **Open question for NeuroSym authors: does the SMT-LIB2 front-end accept `declare-fun`/`apply` at all, or strictly QF_BV/QF_LIA terms over declared constants?** Until answered, treat UF as unsupported and fall back.

### 3.4 QF_LIA / integer-real mode

NeuroSym supports QF_LIA, and ESBMC has an integer mode (`--ir`), but the base smtlib logic string for int mode is `QF_AUFLIRA` (arrays + LIA + **reals**). NeuroSym does not do reals or arrays. To use NeuroSym for LIA we would need: array flattening in int mode (ESBMC's `array_convt` is BV-oriented — LIA arrays are a gap) and a pure `QF_LIA` (no reals) subset. This is materially harder than the BV path. **Recommendation: ship BV mode first; add QF_LIA only if there is demand, and until then add `"neurosym"` to the `--ir`/`--ir-ieee` rejection guard in `solve.cpp`** (mirroring the existing Bitwuzla/Boolector guard) so `--neurosym --ir` fails with a clean message instead of emitting a logic NeuroSym rejects.

### 3.5 Incrementality

k-induction, incremental-BMC, `--falsification`, `--multi-property`, `--parallel-solving`, and `--smt-during-symex` all issue repeated `(check-sat)` queries. NeuroSym's `--stdin` is batch (one formula, one answer). **Reject these at factory time** (copy bitwuzllob's `incompatible[]` list verbatim — including `"falsification"`, which is an incremental strategy like the rest). Plain single-shot BMC (`esbmc file.c --unwind N --neurosym`) is the supported mode.

### 3.6 Fallback mechanisms (layered)

1. **Internal (free):** NeuroSym *itself* falls back to Z3 when the GAN's candidate fails verification — so for any formula in QF_BV/QF_LIA, NeuroSym is already sound and complete without ESBMC doing anything. This is the primary fallback and the reason the integration is sound. **Explicit trust asymmetry:** the two verdict directions are *not* equally protected. A **SAT** verdict is independently confirmed by the side model-solver (§2.5) — divergence aborts, as in bitwuzllob. An **UNSAT** verdict, however, becomes `VERIFICATION SUCCESSFUL` on nothing but NeuroSym's word that failed GAN candidates always route through Z3 — a claim about a research prototype's internal control flow. A false `unsat` is ESBMC's worst failure mode. Until §8 Q10 is answered affirmatively, treat NeuroSym's `unsat` as a **trust assumption**, mitigated two ways: (a) the §5.3 differential harness compares verdicts against Z3/Bitwuzla; (b) an optional `--neurosym-verify-unsat` mode (Phase 5) re-checks every `unsat` with the side solver during the evaluation phase.
2. **ESBMC-level (unsupported fragment):** if a query uses UF/quantifiers/reals NeuroSym can't parse, or NeuroSym crashes/times out, `neurosym_convt` returns `P_ERROR`/`external_process_died`, and we **optionally re-dispatch the same VCC to a linked solver** (Z3/Bitwuzla). A clean design: a `--neurosym-fallback z3` option that, on `P_ERROR`, constructs a Z3 `smt_convt` for that query. (Simplest first cut: no auto-redispatch — surface the error and let the user pick a different `--solver`. Auto-fallback is a Phase-5 nicety.)
3. **Model fallback:** if NeuroSym can't produce a model, the `--neurosym-model-prog` side-solver does (§2.5).

---

## 4. Incremental development roadmap

### Phase 1 — Proof of concept (SMT-LIB2 + subprocess)
- **Tasks:** Get ESBMC to emit a `QF_BV` `.smt2` for a tiny program and have `python main.py file.smt2` answer it. Fastest route needs **no new code**: `esbmc t.c --smtlib --output q.smt2 --smt-formula-only`, then hand-rewrite `set-logic QF_AUFBV`→`QF_BV`, run NeuroSym manually, eyeball the verdict. Confirms NeuroSym ingests ESBMC-shaped SMT-LIB2.
- **Code changes:** none (or a throwaway script).
- **Testing:** 2–3 hand-picked BV programs; compare NeuroSym verdict to `--z3`.
- **Risks/opens:** Does NeuroSym accept ESBMC's `(declare-fun)` symbol style and `_ BitVec n` sorts? Does it need the `set-logic` line at all? Does it choke on arrays if we *don't* flatten (it will — hence Phase 4 flatteners)?
- **Complexity:** **Low** (½–1 day).

### Phase 2 — Basic SAT/UNSAT + robust errors
- **Tasks:** Create `neurosym_convt` (copy bitwuzllob), the factory with the strategy-rejection guard, the `logic_string` override, and `run_neurosym()` verdict parsing. Wire `--neurosym`, `--neurosym-prog`, build files, `solver_config.h.in`.
- **Code changes:** all of §2.1–2.3, §2.7; `solve.cpp` + `options.cpp` entries.
- **Testing:** `esbmc t.c --neurosym` returns SUCCESSFUL/FAILED matching Z3 on a BV set; malformed output / missing `main.py` / non-zero exit / signal-kill all yield clean `P_ERROR`, not crashes (reuse bitwuzllob's `WIFSIGNALED` discard + `external_process_died`).
- **Risks/opens:** batch-vs-interactive stdin deadlock (mitigate by using the temp-file path first); Python startup cost; verdict noise in NeuroSym's stdout.
- **Complexity:** **Medium** (2–3 days).

### Phase 3 — Model extraction for counterexamples
- **Tasks:** Wire `--neurosym-model-prog` side-solver exactly as bitwuzllob; on `P_SATISFIABLE`, feed the same formula to `z3 -in` and reconstruct the model via inherited `get_value`/`get_bv`/`get_array_elem`. Handle `--result-only` (no model).
- **Code changes:** the model-prog branch in `dec_solve()` (copy bitwuzllob's) + the `model_prog()`/`result-only` guards.
- **Testing:** unsafe programs produce a *correct* counterexample trace under `--neurosym --neurosym-model-prog "z3 -in"`; diff the trace against `--z3`.
- **Risks/opens:** model divergence (NeuroSym says sat, side-solver says unsat) → abort/refuse (bitwuzllob already does this). Reconstructing arrays from a flattened BV model — verify `array_convt` round-trips.
- **Complexity:** **Medium** (2–3 days).

### Phase 4 — Full backend integration + regression tests
- **Tasks:** Force array/tuple/fp flattening (the `nullptr` factory), add `--neurosym` regression tests, CI matrix entry, BUILDING.md docs, `--ir` rejection guard.
- **Code changes:** finalize factory; `regression/esbmc/neurosym_basic{,_fail}/` (§5.1); `build.yml`.
- **Testing:** ≥2 regression tests (one `CORE` pass, one failing) with `--neurosym` in the `test.desc` flag line; differential run of a BV regression subset `--neurosym` vs `--z3`/`--bitwuzla` — verdicts must agree modulo timeout.
- **Risks/opens:** flattened QF_BV formulas may be large/slow; CI runners need PyTorch (heavy). Gate CI behind a manual/nightly job initially.
- **Complexity:** **Medium–High** (3–5 days).

### Phase 5 — Optimization, caching, timeouts, benchmarking
- **Tasks:** `--neurosym-timeout` (SIGTERM the process group — infra already present); optional formula caching (hash → verdict) to skip re-solving identical VCCs; optional auto-fallback to Z3 on `P_ERROR`; an optional `--neurosym-verify-unsat` cross-check mode that re-solves every `unsat` with the side solver during evaluation (§3.6 trust asymmetry — drop it only once §8 Q10 is answered and differentially validated); the benchmarking harness (§6). Investigate persistent NeuroSym process to amortize Python/PyTorch startup (**needs NeuroSym incremental mode — Tier C**).
- **Code changes:** timeout plumbing; a small verdict cache; `--neurosym-fallback`.
- **Testing:** benchmark suite; verify timeout kills the whole subtree (no orphan Python/PyTorch).
- **Risks/opens:** GAN model-load time may dominate small queries (measure startup overhead explicitly, §6); reproducibility of GAN results across runs (§7).
- **Complexity:** **High** (1 week+, much of it evaluation not coding).

---

## 5. Verification and testing strategy

### 5.1 Regression tests to reuse
- **QF_BV:** `regression/esbmc/` integer-arithmetic/bit-op tests, `regression/esbmc-cpp/` BV-heavy cases, `regression/bitwuzla*` analogues. Pick small, purely-BV programs (no FP, no big arrays) for the first agreement runs.
- **QF_LIA:** only if §3.4 is pursued — `--ir` tests.
- Author 2 new tests per ESBMC convention, with descriptive names (the `github_<N>` prefix is reserved for issue reproducers): `regression/esbmc/neurosym_basic/` (safe, expect `VERIFICATION SUCCESSFUL`) and `regression/esbmc/neurosym_basic_fail/` (unsafe, expect `VERIFICATION FAILED`), each with `--neurosym` on the flag line. **Note:** CI on macOS runs only `regression/python` — these C tests validate on Linux only.

### 5.2 Unit tests (Catch2, `unit/`)
- **SMT-LIB2 generation:** feed a fixed small formula through `neurosym_convt`'s (inherited) serializer with the logic override; assert the emitted text contains `(set-logic QF_BV)` and the expected `(bv…)` terms, and **not** `QF_AUFBV` or `declare-fun … Array`.
- **Response parsing:** unit-test `parse_verdict_line` on `sat`, `unsat`, `unknown`, `s SATISFIABLE`, leading/trailing whitespace, interleaved log lines, and garbage → correct `smt_resultt`/`nullopt`.
- **No mocks** (per repo policy): drive the real serializer; for parsing, call the real function on literal strings.

### 5.3 Integration / differential tests
- Wrapping NeuroSym in a deterministic stub for PyTorch-less CI is disallowed (no test doubles); instead run real NeuroSym in a nightly job, and for PR CI run a **differential harness**: same programs under `--neurosym` and `--z3`; assert identical `VERIFICATION SUCCESSFUL/FAILED`. Divergence ⇒ almost always a wrong signed/unsigned or flattening bug (per the integrate-solver guide's §10 warning).

### 5.4 Expected behaviour table

| Situation | NeuroSym stdout | `neurosym_convt` result | ESBMC verdict |
|---|---|---|---|
| SAT (unsafe program) | `sat` (+ model via side-solver) | `P_SATISFIABLE` | VERIFICATION FAILED + trace |
| UNSAT (safe program) | `unsat` | `P_UNSATISFIABLE` | VERIFICATION SUCCESSFUL |
| UNKNOWN | `unknown` | `P_ERROR` | error surfaced; optional Z3 fallback |
| Solver crash | signal / non-zero + no verdict | `external_process_died` / `P_ERROR` | clean error, no ESBMC crash |
| Timeout | killed by `--neurosym-timeout` | `P_ERROR` | timeout reported; process group reaped |
| Unsupported logic (UF/reals/quant) | parse error / `unknown` | `P_ERROR` | clean error + guidance to use `--z3` |
| Malformed output | no recognizable verdict line | `P_ERROR` (last-tail logged) | clean error |

**Footnote — "clean error, no ESBMC crash" is a property of the bitwuzllob-style path, not the base class.** The graceful rows above hold because `run_neurosym()` parses verdicts with `parse_verdict_line` (tolerant; unmatched → `P_ERROR`). The base `smtlib_convt::read_check_sat_response` instead **aborts** on unrecognized output (`smtlib_conv.cpp:707`). This matters for Tier A (`--smtlib --smtlib-solver-prog`, which routes through the base implementation) and for the §2.5 native-`(get-value)` future path (which would point `emit_proc` at NeuroSym and again use the base reader) — either NeuroSym's stdout must be verdict-clean on that channel, or the reader needs hardening first.

---

## 6. Performance evaluation

### 6.1 Methodology
Differential benchmark: identical programs, `--neurosym` vs `--z3` vs `--bitwuzla` (vs `--boolector` for pure-BV). Use `benchexec` (as in the SV-COMP flow) or a scripted `ctest` subset. Pin ESBMC version and NeuroSym commit. Same `--unwind`, same machine, single-threaded, no incremental mode (NeuroSym can't).

### 6.2 Benchmark categories
- BV-heavy C (bit-twiddling, hashing, crypto kernels).
- Integer-arithmetic (loops, counters) — BV-encoded.
- Safe (expect UNSAT) **and** unsafe (expect SAT) tasks, separately — GAN guidance is a SAT-finder; its edge (if any) should show on **unsafe** tasks.
- Small formulas (fast-feedback) — here Python/PyTorch startup likely **dominates and loses** to linked solvers; report honestly.
- Large formulas — where neural guidance may or may not help; the interesting regime.

### 6.3 Metrics (instrument in `neurosym_convt`)
- Total verification time (ESBMC wall clock).
- Solver time (subprocess wall clock) vs **SMT-LIB2 serialization overhead** (time in `smtlib_convt::flush`) vs **subprocess start-up overhead** (fork/exec + Python + PyTorch model load — measure separately; expected to be the headline cost for small queries).
- Model-generation overhead (side-solver query time).
- Counts: SAT, UNSAT, UNKNOWN, timeout, fallback-to-Z3.
- GAN-hit vs Z3-fallback ratio inside NeuroSym (**requires NeuroSym to report which path answered** — open question §8).

### 6.4 Expected outcome (conservative)
For ESBMC's flattened QF_BV, expect NeuroSym to be **slower on small queries** (startup-bound) and, at best, competitive on large SAT instances where GAN guidance pays. The honest framing: this integration is a **research vehicle** to test whether neural guidance helps *ESBMC-shaped* formulas, not a drop-in speedup. Report per-category, not aggregate.

---

## 7. Risk analysis

| Risk | Detail | Mitigation |
|---|---|---|
| **Python subprocess comms** | Pipe deadlock (write-all-then-read on a full pipe), zombie/orphan Python+PyTorch on timeout | Use the **temp-file** path first (bitwuzllob's flow) to avoid deadlock; reuse the **process-group + `killpg` + `register_pgroup_for_cleanup`** machinery; discard verdicts on `WIFSIGNALED`. |
| **SMT-LIB2 serialization overhead** | Flattened QF_BV can be much larger than native QF_AUFBV | Measure (§6); keep native array flattening efficient; consider caching identical VCCs (Phase 5). |
| **Model parsing / CEX reconstruction** | Reconstructing arrays/structs from a flattened BV model; NeuroSym has no `(get-value)` | Route models through the `z3 -in` side-solver (inherited, proven in bitwuzllob); refuse to build a CEX if side-solver disagrees (bitwuzllob already aborts on divergence). |
| **Unsupported theories** | UF, reals, quantifiers, arrays-in-LIA | Flatten what's flattenable (arrays/tuples/FP→BV); reject `--ir` (guard); on genuinely unsupported queries return clean `P_ERROR` + optional Z3 fallback. |
| **Semantic divergence** | NeuroSym's QF_BV vs ESBMC's expected signed/unsigned div/rem/shift/rotate semantics | **Differential testing** (§5.3) against Z3/Bitwuzla is the gate; a divergence is a bug to fix before shipping (integrate-guide §10). |
| **GAN reproducibility** | Non-determinism (seeds, GPU float nondeterminism) could make a query sat on one run, error on another | NeuroSym's Z3 fallback makes the *final* verdict deterministic in principle; pin seeds; require the model side-solver to *confirm* every SAT before it becomes a CEX (already the design). Document any residual nondeterminism. |
| **Neural component scalability** | GAN encoding is fixed-dimension (8576 QF_LIA / 10752 QF_BV features); large formulas may exceed it and always hit Z3 fallback | Acceptable — fallback keeps it correct; benchmark to see where GAN stops contributing; do not claim speedups it doesn't deliver. |
| **Dependency / build portability** | PyTorch + Z3 + a NeuroSym checkout at *runtime*; heavy, platform-specific | Backend links **nothing** (subprocess), so ESBMC builds anywhere; runtime deps discovered at solve time with a clear error if `--neurosym-prog` isn't runnable. Document in BUILDING.md. |
| **CI integration** | PyTorch install is large/slow; GPU absent on runners | Nightly/manual CI job, CPU-only PyTorch, small formula set; keep `--neurosym` out of the default PR matrix. macOS PR CI won't run C tests anyway. |

---

## 8. Open questions for the NeuroSym authors

1. **Interactive protocol:** Does `--stdin` support *incremental* SMT-LIB2 (`push`/`pop`, multiple `check-sat`), or strictly one formula → one answer? (Determines k-induction/incremental-BMC support — Tier C.)
2. **`(get-value)`:** Can NeuroSym answer `(get-value (x …))` after `sat`, for every declared BV/bool constant? (Would remove the need for a side model-solver.)
3. **`(declare-fun)` / symbols:** Does the SMT-LIB2 front-end accept arbitrary `(declare-fun name () (_ BitVec n))` constants and ESBMC's symbol naming? Any UF (`declare-fun` with arguments) support at all?
4. **`set-logic` handling:** Is the `(set-logic …)` line required, ignored, or validated? Will it reject `QF_AUFBV`/`QF_AUFLIRA`? Is there a lenient/"any BV" mode?
5. **Verdict format:** Exact stdout for sat/unsat/unknown, and on internal error/timeout. Any stderr diagnostics we should surface?
6. **Which path answered:** Can NeuroSym emit (e.g. on stderr or via a flag) whether the GAN or the Z3 fallback produced the answer? (Essential for §6.3 evaluation.)
7. **Startup cost / warm process:** Is there a server/daemon mode that keeps PyTorch and the GAN model loaded across queries? (Amortizing startup is the key to competitiveness on small formulas.)
8. **Determinism:** Are seeds fixed? Is the verdict guaranteed reproducible run-to-run given the Z3 fallback?
9. **QF_LIA arrays/reals:** Any plan to support arrays, or pure `QF_LIA` (no reals) so ESBMC's `--ir` mode could target it?
10. **UNSAT soundness — the single most safety-relevant question:** Is the Z3 fallback *always* consulted before emitting `unsat`, or can the pipeline emit `unsat` heuristically (e.g. after the GAN exhausts candidates without a Z3 confirmation)? For a verifier, a false `unsat` silently turns a buggy program into `VERIFICATION SUCCESSFUL` (§3.6); the SAT direction is independently confirmed by the model side-solver, but UNSAT currently rests entirely on this control-flow guarantee.

---

## 9. Effort and complexity summary

| Phase | Deliverable | Complexity | Est. effort | Depends on NeuroSym? |
|---|---|---|---|---|
| 1 | Manual SMT-LIB2 PoC | Low | ½–1 day | Answers Q3/Q4 |
| 2 | `neurosym_convt` + CLI + build, SAT/UNSAT | Medium | 2–3 days | No |
| 3 | Model extraction (side-solver) | Medium | 2–3 days | No (Q2 removes side-solver) |
| 4 | Flattening, regression + CI + docs | Medium–High | 3–5 days | No |
| 5 | Timeout/cache/fallback/benchmark | High | 1 week+ | Q6/Q7 for real gains |
| **Total (Tier B, BV-only, batch)** | **shippable backend** | **Medium** | **~2–3 weeks** | **No NeuroSym changes** |
| Tier C add-ons | incremental (k-induction), native get-value, warm process, QF_LIA | High | open | **Yes** |

### What's immediately implementable (no NeuroSym changes)
Tier B in full: a `--neurosym` subprocess backend for single-shot BV BMC, arrays/tuples/FP flattened to QF_BV, models via a `z3 -in` side-solver, regression tests, docs. Sound and complete on QF_BV because NeuroSym's own Z3 fallback backstops the GAN.

### What requires NeuroSym changes (Tier C)
Incremental `push`/`pop` (⇒ k-induction, incremental-BMC, multi-property), native `(get-value)` (⇒ drop the side-solver), a warm/daemon process (⇒ competitive small-query latency), and richer QF_LIA (arrays/pure-int) for `--ir` mode.

### What requires deeper ESBMC changes
Only two, both small and isolated: (1) a `virtual logic_string()` hook in `smtlib_convt` so a subclass can override `QF_AUFBV`→`QF_BV`; (2) adding `"neurosym"` to the `--ir` rejection guard in `solve.cpp`. An optional larger change is a generic `--<solver>-fallback` re-dispatch mechanism (Phase 5), and the guide-suggested `esbmc_add_solver()` CMake helper to stop copy-pasting solver CMakeLists.

---

## Appendix A — Verified ESBMC integration points (quick reference)

- Factory typedef `solver_creator` — `src/solvers/solve.h`
- Registration map `esbmc_solvers`, priority array `all_solvers[]`, `pick_default_solver`, int-encoding rejection guard — `src/solvers/solve.cpp`
- Capability wiring (`set_tuple_iface`/`set_array_iface`/`set_fp_conv`) — `create_solver` in `src/solvers/solve.cpp`
- Config macro — `src/solvers/solver_config.h.in`
- SMT-LIB2 serializer, pipe/file sinks, `dec_solve`, `read_check_sat_response`, `get_value`, logic string `QF_AUFBV`/`QF_AUFLIRA`, `external_process_died` — `src/solvers/smtlib/smtlib_conv.{h,cpp}`
- Subprocess-backend precedent (`solved` guard, `incompatible[]` strategy rejection, `run_bitwuzllob`, process-group cleanup, `--*-model-prog` side-solver) — `src/solvers/bitwuzllob/bitwuzllob_conv.{h,cpp}`
- Solver option group (`{"z3",…}`, `{"bitwuzllob",…}`, `--smtlib-solver-prog`) — `src/esbmc/options.cpp` (~lines 512–566)
- Backend CMake pattern — `src/solvers/bitwuzllob/CMakeLists.txt`, `src/solvers/smtlib/CMakeLists.txt`, `src/solvers/CMakeLists.txt`
- Test convention — `regression/esbmc/*/test.desc` (line 1 CORE/KNOWNBUG, line 3 flags), `unit/` (Catch2)
