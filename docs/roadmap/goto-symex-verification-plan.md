# GOTO-SYMEX Formal Verification Plan

**Subsystem:** `src/goto-symex` — ESBMC's symbolic execution engine: the stage
that turns a GOTO program into an SSA formula (`symex_target_equationt`) for the
SMT backend.
**Verifier:** ESBMC itself (BMC + k-induction) on extracted kernels; Catch2
property/differential tests on the real classes (`unit/goto-symex/`);
whole-tool metamorphic oracles over `regression/`; sanitizers for the rest.
**Status:** Plan / **not yet executed**. Every harness below is a *proposal*;
nothing here asserts a proof has been discharged. Findings R1–R12 are
*hypotheses with cited evidence*, not confirmed end-to-end bugs.
**Audience:** An engineer who will implement the harnesses and run the
verification tasks directly from this document.
**Companion:** `docs/irep2-verification-plan.md` (branch
`docs/irep2-verification-plan`). Scope split is stated in §2.4 — this document
does **not** re-verify irep2 internals.

> **Framing.** goto-symex is the last stage at which a defect is still
> *invisible*. Everything downstream — SMT encoding, the solver — faithfully
> answers the question symex asked. If symex asks the wrong question (a dropped
> guard conjunct, a stale SSA index, a sliced-away assignment, a pruned
> interleaving), the solver's answer is *correct* and ESBMC's verdict is
> *wrong*, with no diagnostic anywhere. Two failure modes, asymmetric in cost:
>
> | Mode | Symptom | Cost |
> |---|---|---|
> | **Unsound** — lost behaviour | `VERIFICATION SUCCESSFUL` on a buggy program | Silent. No user-visible signal. **P0.** |
> | **Incomplete** — added behaviour | `VERIFICATION FAILED` with a spurious counterexample | Noisy, triageable. P1. |
>
> This plan treats goto-symex as safety-critical and adopts the conservative
> stance: **assume subtle unsoundness exists until formally disproven.**

---

## 1. Verification objectives

1. **SSA well-formedness.** Every symbol written to the equation carries a
   fresh, monotonically increasing L2 index for its (base name, L1 activation,
   thread) key; no two assignment steps define the same SSA name; every symbol
   *read* is either previously defined, a declared free/nondet symbol, or a
   global with an initialiser. A violation silently aliases two distinct program
   values.
2. **Path-condition soundness.** The guard attached to every emitted step, and
   the guard produced by every merge (`merge_state_guards`, `phi_function`),
   must be *logically equivalent* to the disjunction of the concrete paths it
   represents — never stronger (lost behaviour) and never weaker (spurious
   behaviour).
3. **Merge completeness.** Every deferred path snapshot pushed into
   `framet::merge_state_map` is consumed by exactly one `merge_gotos` at its
   join point. No path is dropped, none is merged twice.
4. **Transformation equisatisfiability.** Every equation-level transformation
   that *removes* information — the slicer (`symex_slicet`, `simple_slice`,
   `claim_slicer`), constant propagation, `do_simplify` — preserves the
   satisfiability of every retained claim.
5. **Memory safety and absence of ESBMC-checkable UB** in the engine itself:
   no null dereference (the unchecked `*ns.lookup(...)` family), no dangling
   reference into a rehashed container, no container underflow
   (`previous_frame`), no unsigned wrap in counters/bounds.
6. **Abstraction soundness.** Every documented over-/under-approximation
   (value-set filtering, uninterpreted-function fallback, POR/MPOR pruning,
   state hashing, interval-domain guard pruning) must be *proven to be in the
   claimed direction*. An "over-approximation" that actually removes behaviour
   is an unsoundness wearing a comment.
7. **Determinism.** Two symex runs over the same GOTO program in the same
   configuration must produce byte-identical equations. Non-determinism
   (iteration order over pointer-keyed containers) makes every other result
   unreproducible and invalidates regression pinning.
8. **Release-mode enforcement.** Invariants that matter must hold in the
   *shipped* binary, not only under `assert()`. See **R1**.
9. **Regression durability.** Each discharged property is pinned by a
   check-in-able artefact (Catch2 case, `regression/` harness, or CI oracle)
   that fails if the property is re-broken.

---

## 2. Verification scope definition

### 2.1 The reach of ESBMC on this codebase — measured, not assumed

Two probes were run against `build/src/esbmc/esbmc` (ESBMC 8.4.0, this tree):

| Probe | Input | Result |
|---|---|---|
| **P-1** | `std::unordered_map<std::string, POD>` + `std::make_shared` + `assert`, `--unwind 4` | `VERIFICATION SUCCESSFUL`, 2132 VCCs, 4.6 s. ESBMC's C++ operational models cover a useful STL subset. |
| **P-2** | `#include <goto-symex/renaming.h>` with the real include paths | **`ERROR: PARSING ERROR`** — `no template named 'is_standard_layout' in namespace 'std'`, raised inside `immer/detail/combine_standard_layout.hpp` (pulled in by `level1_map.h`). |

`src/goto-symex` is compiled `-std=gnu++23` (`build/compile_commands.json`) and
links `immer`, `fmt`, Boost, `BigInt`, `yaml-cpp` and the whole of `irep2`.
ESBMC's own C++ frontend regression tops out at C++20 (`regression/esbmc-cpp20`).
**Verifying a real goto-symex translation unit in place is not possible today**,
and claiming otherwise would produce vacuous "proofs".

P-2 is not merely a caveat to work around: it is a **bounded, enumerable
backlog of ESBMC operational-model gaps**, measured in §13. Closing it is
scheduled work in this plan (§13.6), because it converts part of Tier A from
*transcription* (which carries drift risk, §9.1) into direct verification of the
real code. §13 also records what closing it does **not** buy — the tractability
measurements there show whole-TU verification stays out of reach regardless.

The plan therefore splits targets into four honest tiers.

### 2.2 Tiers

| Tier | What it covers | Technique | Why this technique |
|---|---|---|---|
| **A — ESBMC-verifiable kernels** | Self-contained algorithmic cores lifted out of the template/library layer: SSA counter algebra, guard-merge selection, merge-queue conservation, slicer dependency closure, unwind-bound selection, MPOR dependency relation, frame lifecycle, equation context stack | Standalone reduced C/C++ harnesses driven by `nondet_*()` and `__ESBMC_assume`, verified with the full property flag-set | Once lifted, these are ordinary imperative code. ESBMC excels here and returns a *deterministic witness* on failure. |
| **B — Property / differential tests on the real classes** | Contracts that need the genuine engine: SSA well-formedness of produced equations, slicer equisatisfiability, renaming round-trip, phi-count laws, value-set merge monotonicity, run-to-run determinism | Catch2 in `unit/goto-symex/`, driving real symex via `unit/testing-utils/goto_factory.h` | The engine cannot be reduced without losing fidelity, but its *observable output* (the equation) is a first-class inspectable object. |
| **C — Whole-tool metamorphic / differential oracles** | Properties of the composed pipeline: transformation parity, solver parity, unwind monotonicity, POR parity, multi-property parity | Scripted sweeps over the existing `regression/` corpora comparing verdicts under semantically-equivalent flag pairs | These catch *composition* bugs that no unit test sees, at zero modelling cost, over the 1400 `CORE` inputs already in `regression/esbmc`. |
| **D — Not effectively verifiable here** | `immer` internals, `BigInt`/`fmt` internals, `std::atomic` ordering in irep2, the value-set/pointer-analysis fixpoint (`src/pointer-analysis`), the SMT backends | Sanitizers (existing `.github/workflows/sanitizers.yml`), hand proof, the sibling irep2 plan | Documented, **not** claimed as proven. See §14. |

### 2.3 Prioritised component scope

The areas named in the verification brief, mapped to this tree and ranked. `P0`
= a defect is silently unsound; `P1` = wrong-but-visible or unsound only under a
non-default flag; `P2` = localised; `P3` = diagnostic only.

| Area (brief) | Primary code | Rank | Rationale |
|---|---|---|---|
| **Core symbolic execution logic** | `symex_main.cpp` (`symex_step`, `claim`, `assume`, `do_simplify`, `replace_nondet`), `symex_other.cpp` | **P0** | The dispatch that decides what becomes a VCC at all. |
| **State management** | `goto_symex_state.{h,cpp}`, `renaming.{h,cpp}`, `level1_map.h` | **P0** | SSA naming *is* the value semantics of the formula. |
| **Assignment / expression handling** | `symex_assign.cpp` (`symex_assign` + 13 `symex_assign_*` recursors) | **P0** | Structural lvalue decomposition; a wrong byte-extract/concat/bitfield update writes the wrong bits. |
| **Path exploration & merging** | `symex_goto.cpp` (`symex_goto`, `merge_gotos`, `merge_state_guards`, `phi_function`, `loop_bound_exceeded`, `get_unwind`) | **P0** | Lost path = missed bug, with no signal. |
| **Constraint generation** | `symex_target_equation.{h,cpp}`, `slice.{h,cpp}` | **P0** | The slicer is the only stage that *deletes* constraints. |
| **Function-call handling** | `symex_function.cpp`, `symex_stack.cpp` | **P1** | Frame/argument binding, recursion bounds, function-pointer target enumeration. |
| **Memory modelling** | `dynamic_allocation.cpp`, `symex_valid_object.cpp`, `symex_mem`/`symex_free`/`symex_realloc` in `symex_main.cpp` | **P1** | Allocation validity arrays; documented gaps (R8, R9). |
| **Pointer reasoning** | `symex_dereference.cpp` + `symex_dereference_statet` | **P1** | Consumes `src/pointer-analysis`; the *use* is in scope, the analysis is Tier D. |
| **Concurrency handling** | `execution_state.{h,cpp}`, `reachability_tree.{h,cpp}`, `symex_symmetry.cpp` | **P0 (when enabled)** | POR/MPOR, state hashing and symmetry reduction all *prune* interleavings. |
| **Solver-interface interaction** | `runtime_encoded_equationt` (`--smt-during-symex`), `symex_goto`'s `dual_unsat_exception` path | **P1** | Symex asking the solver mid-flight; a wrong `tvt` prunes a live branch. |
| **Trace / witness output** | `goto_trace.cpp`, `build_goto_trace.cpp`, `witnesses.cpp`, `sarif.cpp`, `html.cpp`, `json.cpp`, `ctest.cpp`, `pytest.cpp` | **P3** | ~6.8 kLOC. Wrong output is visible and non-load-bearing for the verdict. Explicitly **out of scope** except where a trace crash masks a verdict. |

Size context: ~22.4 kLOC total in `src/goto-symex`; ~10.2 kLOC is the execution
core listed above, the remainder is counterexample/trace/witness rendering.

### 2.4 Boundary with the irep2 plan

`guard2tc`'s set-algebra (`operator-=`, `operator|=`, the O(Δ) prefix walk) lives
in `src/irep2/irep2_guard.cpp` and is owned by `docs/irep2-verification-plan.md`
(H-A9/H-B4 there). **This plan verifies goto-symex's *use* of it** — that
`merge_state_guards` composed with `phi_function`'s `tmp_guard -= cur_guard`
yields a sound path condition *given* the algebra's contract (H-A2 below).
Likewise `expr2tc` lifetime/CRC/ordering are irep2's obligations. Where a
harness needs both, this document states which contract it *assumes* and cites
the irep2 harness that discharges it (§7.3).

---

## 3. Architecture overview

```
 goto_functionst ──► reachability_treet            (interleaving search: DFS over
                          │                         execution states; POR, state
                          │                         hashing, context bound)
                          ▼
                    execution_statet : goto_symext (one per explored interleaving;
                       │      │                     owns N thread states + the
                       │      │                     shared L2 renaming + value set)
                       │      └── threads_state[]  ──► goto_symex_statet  (per thread:
                       │                                 pc, guard, call_stack of
                       │                                 framet{level1, merge_state_map,
                       │                                 local_variables, va_*}, value_set)
                       │
                       ▼  goto_symext operations
        symex_step ─► symex_assign* / symex_goto / symex_function_call /
                      symex_decl / symex_dead / symex_assume / symex_assert /
                      symex_malloc|free|realloc / run_intrinsic
                       │
                       ▼  symex_targett interface (assignment/assumption/
                       │   assertion/branching/renumber/output)
              symex_target_equationt          (list<SSA_stept>)
                   or runtime_encoded_equationt (--smt-during-symex: push/pop
                       │                          straight into a live smt_convt)
                       ▼
             ssa_step_algorithm passes: symex_slicet / simple_slice / claim_slicer
                       ▼
                  smt_convt  (solvers/)
```

**Anchor convention.** Every reference names the enclosing symbol
(`Class::member` or free function) alongside a line hint. Treat the **symbol
name as authoritative** and the line number as a hint; re-`grep` before relying
on a line.

| Mechanism | Source of truth | One-line contract |
|---|---|---|
| L1 renaming (activation records) | `renaming::level1t::rename` / `::get_ident_name`, `renaming.cpp:96,43` | L0 symbol → per-frame instance; globals → `level1_global`, `__thread` globals → per-thread L1. |
| L1 storage | `persistent_map` (immer HAMT), `level1_map.h` | O(1) copy of `framet` on every branch snapshot; `find` returns a pointer into shared storage **invalidated by the next mutation**. |
| L2 renaming (SSA) | `renaming::level2t::make_assignment` / `::rename` / `::coveredinbees`, `renaming.cpp:344,155,230` | `make_assignment` bumps `entry.count`, publishes the new index and caches a propagated constant. |
| L2 key | `level2t::name_record`, `renaming.h:143` | `(base_name, rlevel, l1_num, t_num)` with a derived `hash` used as the primary compare key. |
| Constant propagation | `goto_symex_statet::constant_propagation{,_reference}`, `goto_symex_state.cpp` | An L2 entry may hold a `constant` that `rename` substitutes *in place of* the SSA symbol. |
| Branch split | `goto_symext::symex_goto`, `symex_goto.cpp:~200-291` | Push a `merge_statet` snapshot at the join pc; add `guard_expr` to one side and `!guard_expr` to the other. |
| Join | `goto_symext::merge_gotos` + `merge_state_guards` + `phi_function` + `merge_locality` + `merge_value_sets`, `symex_goto.cpp:326,293,393,370,382` | Disjoin guards, emit `ite` phi assignments for every variable whose L2 index differs, union locals and value sets. |
| Loop bound | `goto_symext::get_unwind` / `loop_bound_exceeded`, `symex_goto.cpp:525,497` | Precedence: loop-specific > function-specific > global; on exhaustion emit an unwinding *assertion*, or (with `--no-unwinding-assertions`) an *assumption* that truncates the path. |
| Equation | `symex_target_equationt::SSA_stept`, `symex_target_equation.h:94` | Flat list; `ignore` = sliced, `hidden` = suppressed from trace, `cond` = the encoded formula. |
| Slicing | `symex_slicet::run` + `run_on_assignment`, `slice.h:101`, `slice.cpp:~230-300` | Reverse reverse-taint closure over `depends`; additionally elides array stores to indices proven never read (`index_reads`, `array_disqualified`). |
| Interleaving search | `reachability_treet`, `reachability_tree.{h,cpp}` | DFS over `execution_statet` clones; prunes by POR, `hit_hashes` state hashing, `--context-bound`. |
| POR | `execution_statet::check_mpor_dependency` / `calculate_mpor_constraints`, `execution_state.cpp` | Drops an interleaving when the last transitions of two threads are independent (no WW/RW/WR intersection). |
| State hashing | `state_hashing_level2t::make_assignment` / `generate_l2_state_hash`, `execution_state.cpp:~1342,1359` | Fingerprints (variable → value-hash); equal fingerprint ⇒ prune. Keyed by **L0** name (see R6). |
| Live solver queries | `runtime_encoded_equationt::ask_solver_question`, `symex_target_equation.h:307` | `--smt-during-symex`; `symex_goto` prunes a branch when the solver proves the guard constant. |

---

## 4. Component analysis

### 4.1 Classes, data structures, algorithms

**C1. `renaming::level1t` / `level2t`** (`renaming.{h,cpp}`, `level1_map.h`).
L1 = `persistent_map<name_record, unsigned>`; L2 = `std::unordered_map<name_record, valuet>`
where `valuet = {count, constant, node_id}`. Operations: `rename`,
`get_ident_name`, `get_original_name`, `make_assignment`, `coveredinbees`,
`current_number`, `rename_to_record`, `clone`.

**C2. `goto_symex_statet`** (`goto_symex_state.{h,cpp}`). Per-thread state:
`guard`, `global_guard`, `source`, `call_stack`, `num_instructions`,
`loop_iterations`, `function_unwind`, `realloc_map`, references to the shared
`level2` and `value_set`. Methods: `rename`, `rename_address`, `rename_type`,
`assignment`, `constant_propagation*`, `fixup_renamed_type`, `gen_stack_trace`.

**C3. `goto_symex_statet::framet`** — one activation record: `level1`,
`merge_state_map`, `local_variables`, `declaration_history`, `return_value`,
`entry_guard`, `va_index`/`va_cursor`, `stack_frame_total`, function-pointer
target list. **C3′. `merge_statet`** — the deliberately-narrower deferred-path
snapshot (`level2` clone, `value_set`, `guard`, `num_instructions`,
`thread_id`, `local_variables`, `interval_snapshot`).

**C4. `goto_symext`** (`goto_symex.h`, ~1455 lines of declarations). The
operation set: `symex_step`, `symex_assign` + 13 `symex_assign_*` members (the
`_rec` dispatcher plus 12 structural cases: `symex_assign_{symbol,structure,
array_structure,union,extract,typecast,array,member,if,byte_extract,concat,
bitfield}`), `symex_goto`, `symex_function_call*`,
`symex_decl`/`symex_dead`, `symex_assume`/`symex_assert`/`claim`,
`symex_malloc`/`alloca`/`realloc`/`free`/`cpp_new`/`cpp_delete`, `symex_va_arg`,
`symex_printf`, `run_intrinsic` (22 `intrinsic_*` members incl. `memset`/`memcpy`/
`memmove`/`memcmp`/`memchr`, thread primitives, atomics).

**C5. Merge machinery** — `merge_gotos`, `merge_state_guards`, `phi_function`,
`merge_locality`, `merge_value_sets`, `record_branch_sibling`.

**C6. Loop/recursion bounding** — `get_unwind`, `get_unwind_recursion`,
`loop_bound_exceeded`, `unwind_set` / `unwind_func_set` /
`loop_id_to_func_index`.

**C7. `symex_target_equationt`** — the SSA step list, `convert` (incl. the
`vacuity_mode` encoding), `clear_assertions`, `push_ctx`/`pop_ctx`,
`reconstruct_symbolic_expression`/`replace_rec`, `check_for_duplicate_assigns`,
`clone`. **C7′. `runtime_encoded_equationt`** — incremental variant with
`scoped_end_points`, `cvt_progress`, `ask_solver_question`.

**C8. Slicers** (`slice.{h,cpp}`) — `symex_slicet` (reverse taint + per-index
array-store elision), `simple_slice`, `claim_slicer`.

**C9. `execution_statet`** — N-thread container, `ex_state_level2t` /
`state_hashing_level2t`, thread creation/termination, atomic sections, monitors,
MPOR bookkeeping (`thread_last_reads`/`writes`, `dependency_chain`),
`get_expr_globals`, `analyze_assign`/`analyze_read`.

**C10. `reachability_treet`** — DFS over execution states, `hit_hashes`,
`ever_written_globals`, `address_taken_globals`, `--schedule` mode,
`setup_next_formula`.

**C11. Memory model** — `dynamic_allocation.cpp`
(`default_replace_dynamic_allocation`: `valid_object`, `deallocated_object`,
`dynamic_size`, `invalid_pointer` lowering against `__ESBMC_alloc` /
`__ESBMC_is_dynamic`), `symex_valid_object.cpp`, `track_new_pointer`,
`dynamic_memory` list, `add_memory_leak_checks`.

**C12. Pointer handling** — `symex_dereference_statet` (`dereference_failure`,
`get_value_set`, `has_failed_symbol`, `is_live_variable`, `rename`),
`goto_symext::dereference`.

**C13. Symmetry reduction** — `symex_symmetry.cpp` (thread symmetry over the
produced equation).

### 4.2 Representation invariants — the properties to assert

| ID | Invariant | Established / relied on by | Enforcement **today** |
|---|---|---|---|
| **I1** | For a fixed `name_record`, `valuet::count` is strictly increasing across `make_assignment` calls; every published L2 index is used by exactly one defining step. | `level2t::make_assignment` | `assert(entry.count <= count)` in `coveredinbees` — **debug only** |
| **I2** | An L2 `name_record` key is *stable* across the `rename(lhs,count)` call inside `make_assignment` — i.e. the callee's `current_names[...]` hits the same bucket, so the caller's `valuet &entry` reference stays valid. | `make_assignment` → `coveredinbees` | **unenforced**; comment "This'll update entry beneath our feet" (R3) |
| **I3** | `rename` is idempotent: a symbol already at `level2`/`level2_global` is returned unchanged. | `level2t::rename` early-return | by construction |
| **I4** | `get_original_name` ∘ `rename` = identity on the L0 form; renaming level never decreases along a path. | `renaming_levelt::get_original_name` | **unenforced** |
| **I5** | Constant propagation is semantics-preserving: substituting `valuet::constant` for the SSA symbol yields an equisatisfiable formula. | `level2t::rename` (`expr = it->second.constant`) | **unenforced** |
| **I6** | Every `merge_statet` pushed at pc *P* is consumed by exactly one `merge_gotos` at *P*; `pop_frame` never discards a non-empty `merge_state_map`. | `symex_goto`, `merge_gotos`, `framet::merge_state_map` | `assert(call_stack.back().merge_state_map.size()==0)` in `pop_frame` — **debug only** (R2) |
| **I7** | Post-merge guard ≡ (guard_cur ∨ guard_merge); the phi `ite` selects `merge` value exactly under the merge path condition. | `merge_state_guards`, `phi_function` | **unenforced** |
| **I8** | A phi assignment is emitted iff the variable's L2 index differs between the two states **and** the variable exists in both; the excluded names (`goto_symex::guard!`, `symex::invalid_object`) are provably safe to skip. | `phi_function` filter chain | **unenforced** |
| **I9** | `merge_value_sets` produces a **superset** of both inputs (over-approximation direction). | `merge_value_sets` → `value_sett::make_union` | **unenforced** |
| **I10** | No two assignment steps in an equation define the same SSA name. | whole engine | `check_for_duplicate_assigns` **logs** a message; never fails (R5) |
| **I11** | Slicing preserves equisatisfiability of every retained claim: a step marked `ignore` is not read by any retained step. | `symex_slicet::run_on_*` | **unenforced** |
| **I12** | Array-store elision precondition: index *i* of array version *v* is never read downstream **and** *v* is not disqualified. | `run_on_assignment`, `index_reads`, `array_disqualified` | **unenforced** |
| **I13** | `push_ctx`/`pop_ctx` on the equation are LIFO and restore the exact step-list prefix. | `symex_target_equationt::push_ctx`, `scoped_end_points` | **unenforced** |
| **I14** | POR drops an interleaving only when the two transitions are genuinely independent (`check_mpor_dependency` false ⇒ commutative). | `check_mpor_dependency`, `calculate_mpor_constraints` | **unenforced** |
| **I15** | State-hash equality implies concrete-state equality (no false collision ⇒ no over-pruning). | `generate_l2_state_hash`, `hit_hashes` | **unenforced**; hash keyed by L0 name (R6) |
| **I16** | `previous_frame()` is called only with `call_stack.size() ≥ 2`. | `goto_symex_statet::previous_frame` (`*(--(--end()))`) | **unenforced** (R7) |

### 4.3 Criticality ranking

Classified along the four axes named in the brief.

| Rank | Component | Soundness | Completeness | Performance | Solver interaction |
|---|---|---|---|---|---|
| **P0** | C1 renaming / SSA (I1–I5) | ✱ silent aliasing of distinct values | — | — | wrong terms |
| **P0** | C5 merge / phi (I6–I9) | ✱ lost path, wrong `ite` | spurious CE | — | wrong path condition |
| **P0** | C8 slicer (I11, I12) | ✱ dropped constraint ⇒ SAT where UNSAT | — | ✱ the whole point | fewer/wrong asserts |
| **P0** | C4 `symex_assign_*` recursors | ✱ wrong bits written | — | — | wrong encoding |
| **P0** | C9/C10 POR, state hashing, symmetry (I14, I15) | ✱ pruned interleaving | — | ✱ the whole point | — |
| **P1** | C6 unwind bounding | truncation under `--no-unwinding-assertions` | ✱ bounded by design | ✱ | assumption steps |
| **P1** | C7′ `runtime_encoded_equationt` | branch pruned on a wrong `tvt` | — | ✱ | ✱ direct |
| **P1** | C11/C12 memory & pointers | over-/under-approx (R8, R9) | ✱ | — | validity arrays |
| **P2** | C2/C3 frame lifecycle (I16) | crash, not wrong answer | — | — | — |
| **P3** | trace/witness renderers | — | — | — | — |

---

## 5. Property catalogue

Every Tier-A harness runs the full property flag-set. Mapping of the brief's
required property list onto ESBMC flags and goto-symex targets.

| # | Property | Why it matters here | Applies to | ESBMC flag / encoding |
|---|---|---|---|---|
| **P1** | Memory safety (bounds, invalid deref, UAF, leak) | The engine crashes mid-verification ⇒ no verdict; worse, a masked crash under `NDEBUG` ⇒ corrupt verdict | C1 (`persistent_map::find` pointer lifetime), C2/C3 (`previous_frame`), C11 | bounds/deref checks **on by default**; `--memory-leak-check` |
| **P2** | Absence of null dereference | 8 unchecked `*ns.lookup(...)` sites (R4) | `phi_function`, `symex_function_call_code`, `dynamic_allocation`, `symex_valid_object` | default deref check; model `lookup` as returning nondet-null |
| **P3** | Bounds safety | index math in intrinsics (`memcpy`/`memcmp` byte loops), `dependency_chain[t][t]`, `threads_state[i]` | C4 intrinsics, C9 | default; `--unsigned-overflow-check` for index arithmetic |
| **P4** | Invariant preservation (I1–I16) | These *are* the correctness argument of symex | all P0 components | `assert` in harness; §4.2 table is the checklist |
| **P5** | State consistency | A `merge_statet` must stay consistent with the state it snapshotted (L2 clone vs shared `value_set`) | C3′, C5 | modelled state pair + `assert` after each op |
| **P6** | Correct symbolic-state updates | `assignment()` must update L2 **and** value set **and** emit exactly one step | C1, C2 | conservation assertions (counts before/after) |
| **P7** | Correct path-condition propagation | Objective 2 | C5, C6 | equivalence to a naive reference over all valuations |
| **P8** | Correct SSA generation | Objective 1 | C1, C7 | freshness/monotonicity assertions; Tier-B validator |
| **P9** | Preservation of symbolic-expression semantics | Constant propagation, `do_simplify`, `fixup_renamed_type` must not change meaning | C2, `do_simplify` | differential: with/without `--no-simplify` (H-C2) |
| **P10** | Deterministic behaviour under equivalent inputs | Objective 7 | whole engine | H-B2 (in-process double run), H-C timestamped reruns |
| **P11** | Internal consistency of generated constraints | No duplicate SSA definition; no use-before-def; guard symbols defined before use | C7 | Tier-B equation validator (H-B1) |
| **P12** | Arithmetic overflow / underflow | `count + 1`, `num_instructions`, `stack_frame_total`, unwind counters, `idx - it` style index math | C1, C6, C4 | `--overflow-check --unsigned-overflow-check` |
| **P13** | Division / modulo by zero | element-size divisions in `memcpy`/`memset`/byte-extract lowering | C4 | on by default |
| **P14** | Uninitialised data | `level2t::name_record`'s defaulted ctor leaves `lev`/`l1_num`/`t_num`/`hash` indeterminate (R10) | C1 | model read-before-write explicitly; MSan for the real class |
| **P15** | UB (shift, conversion, lifetime) | hash combining, `renaming_level` enum casts, reference-into-container lifetime | C1, C9 | `--ub-shift-check`, `--overflow-check` |
| **P16** | Termination of engine loops | the slicer worklists, `run_next_function_ptr_target`, MPOR fixpoint | C8, C4, C9 | `--k-induction` **with convergence** (never `--no-unwinding-assertions`) |

---

## 6. Harness design strategy

### 6.1 Rules (each is a review gate)

1. **Lift, don't inline the world.** For a Tier-A kernel, transcribe the exact
   arithmetic/indexing/decision logic from the cited symbol into a
   self-contained `.c`/`.cpp` file, driven by `nondet_*()`. Model `expr2tc` as
   an integer handle, `irep_idt` as an integer, a guard as a conjunct bitmask.
   Fidelity is only required for what the property observes.
2. **Every assumption cites a source.** Each `__ESBMC_assume(...)` in a Tier-A
   harness carries a comment naming the `file:symbol` that establishes it. An
   assumption with no citation is a defect in the harness.
3. **Every Tier-A assumption is discharged by a Tier-B check.** This is the
   central anti-over-constraining device: if H-A1 assumes "the L2 key is stable
   across `rename`", then a Tier-B Catch2 case must assert that the *real*
   engine never violates it over the `regression/` corpus. Undischarged
   assumptions are listed in §7.3 as an explicit debt register.
4. **Bound the shape, not the values.** Cap sequence lengths / thread counts /
   variable counts with a small nondet bound (`__ESBMC_assume(n <= K)`), prove
   at `K`, then re-run at `K+1`; for the unbounded claim, switch to
   `--k-induction` and **require convergence**.
5. **One passing + one failing harness per property.** The `_fail` twin
   perturbs the kernel (removes the guard, drops the `+1`) and must produce
   `VERIFICATION FAILED`. A property whose `_fail` twin still passes has no
   discriminating power and is rejected.
6. **Reachability probe in every harness.** End each Tier-A harness body with a
   `__ESBMC_assert(0, "reachability probe")` behind a nondet flag, or run a
   companion `--unwind`-identical variant asserting `0` at the same point; the
   probe **must** report `FAILED`. This is what distinguishes "proved" from
   "the code was never reached".
7. **Never pair `--no-unwinding-assertions` with a reachability or
   unbounded-safety claim** (repo rule; empirically yields false SUCCESSFUL on
   truncated loops).
8. **Dual-solver agreement** (Bitwuzla + Z3) for every P0 result.

### 6.2 Verification boundaries and stub inventory

| Stub | Replaces | Fidelity kept | Fidelity dropped | Justification |
|---|---|---|---|---|
| `sym_t = {name:int, l1:int, tid:int, l2:int}` | `symbol2t` / `expr2tc` | the four identity fields the L2 key uses | type, operands, refcounting | The SSA property is a property of *names*; operand structure is irep2's obligation. |
| `map_t` = fixed-capacity array + linear probe | `std::unordered_map<name_record, valuet>` | key equality semantics, insert/find/erase, **rehash-invalidates-references** | bucket policy, allocator | Reference invalidation is exactly what I2/R3 is about; a faithful model must keep it. |
| `guard_t` = `unsigned` bitmask over ≤ 6 conjunct literals | `guard2tc` | conjunction, disjunction, difference, `is_false` | expression sharing, simplification | Path-condition *logic* is what P7 asserts; the algebra's implementation is irep2's H-A9/H-B4. |
| `step_t = {kind, lhs, rhs_reads[], guard, ignore}` | `SSA_stept` | dependency structure and `ignore` flag | source locations, traces, output payloads | The slicer's contract is purely dependency-structural. |
| `solver_answer()` → nondet `{TRUE,FALSE,UNKNOWN}` | `ask_solver_question` | the three-valued contract | actual satisfiability | Models a *correct* solver; a wrong solver is out of scope. Under-constraining here is deliberate: it forces symex to be correct for every legal answer. |
| `lookup(name)` → nondet `sym*` **including null** | `namespacet::lookup` | nullability | symbol table contents | Nullability is the property under test (R4). Assuming non-null would be the classic unrealistic assumption. |

### 6.3 Avoiding over-constraint: what harnesses must *not* assume

A forbidden-assumption list, enforced at review:

- ✗ `assume(lookup(n) != NULL)` — that is R4's bug.
- ✗ `assume(call_stack_size >= 2)` before `previous_frame` — that is R7's bug,
  unless the caller-side precondition is *itself* cited and separately proven.
- ✗ `assume(guard_cur && guard_merge` are disjoint`)` — true for a 2-way `if`,
  **false** for N-way merges at a shared join and for function-pointer merges.
- ✗ `assume(array indices are constant)` in the slicer harness — the
  `array_disqualified` path exists precisely for symbolic indices.
- ✗ `assume(single thread)` in any harness whose property is claimed for
  concurrent runs.

---

## 7. Proposed harnesses

Each entry: **target**, **preconditions**, **assertions**, **invocation**,
**anti-vacuity twin**. Code is a faithful sketch; the implementer transcribes the
cited symbol exactly.

### 7.1 Tier A — ESBMC-verifiable kernels

#### H-A1 — SSA freshness & monotonicity (P0, I1)
**Target:** `renaming::level2t::make_assignment` + `::coveredinbees`
(`renaming.cpp:344,230`).
**Model:** `map_t` keyed by `(name,l1,tid)`, `valuet{count,node}`; a nondet
sequence of ≤ K assignments to nondet keys; a `seen[]` set of published
`(name,l1,tid,l2)` quadruples.
**Assertions:**
- `A1.1` after each `make_assignment`, `count_after == count_before + 1` and no
  unsigned wrap;
- `A1.2` the published quadruple was never published before (freshness);
- `A1.3` two distinct keys never publish the same quadruple (injectivity);
- `A1.4` `coveredinbees`' `assert(entry.count <= count)` holds as a *proved*
  postcondition, not a debug assert.
**Invocation:** `--overflow-check --unsigned-overflow-check --memory-leak-check`
(+ a `--k-induction` variant for unbounded K).
**Twin (`_fail`):** replace `entry.count + 1` with `entry.count` → A1.2 fails.

#### H-A2 — Merge guard & phi selection soundness (P0, I7/I8)
**Target:** `merge_state_guards` + `phi_function`'s selection chain
(`symex_goto.cpp:293,393`).
**Model:** guards as bitmasks over ≤ 6 nondet boolean literals; `-=` modelled by
its *contract* (difference such that `(g_cur ∨ g_mrg) → (diff ↔ g_mrg)`, the
obligation discharged by irep2 H-A9), `|=` as union; two nondet values
`v_cur`,`v_mrg`; the three-way selection exactly as coded.
**Assertions:** for every valuation of the literals,
- `A2.1` `g_cur ∧ ¬g_mrg → merged == v_cur`;
- `A2.2` `g_mrg ∧ ¬g_cur → merged == v_mrg`;
- `A2.3` `g_cur ∧ g_mrg → merged ∈ {v_cur, v_mrg}` and matches the *later*
  writer under the code's own ordering;
- `A2.4` post-merge guard ≡ `g_cur ∨ g_mrg`, **including** the
  `merge_state_guards` shortcut branches (one side false ⇒ return the other);
- `A2.5` no valuation satisfies the post-merge guard but neither input guard
  (no invented path).
**Twin:** drop the `state.guard.is_false()` special case in
`merge_state_guards` → A2.4 fails on the truncated-path input.
**Note:** A2.3 is the highest-value assertion in this plan; N-way merges are
exercised by parameterising the harness to 3 incoming states.

#### H-A3 — Merge-queue conservation (P0, I6)
**Target:** `symex_goto`'s push into `framet::merge_state_map`, `merge_gotos`'
reverse drain + `erase`, `goto_symex_statet::pop_frame`.
**Model:** map pc → list; nondet interleaving of push/drain/pop-frame ops.
**Assertions:** `pushed == drained` at end; no drain of an empty list; **a
`pop_frame` with a non-empty map is a violation** (promoting the debug-only
`assert` to a proved obligation — R2).
**Twin:** allow `pop_frame` to discard → conservation fails.

#### H-A4 — Slicer dependency closure & dead-store elision (P0, I11/I12)
**Target:** `symex_slicet::run_on_assignment` / `run_on_assume` /
`run_on_renumber` / `collect_dependencies` / `scan_array_uses`
(`slice.cpp`, `slice.h:101-252`).
**Model:** ≤ 6 `step_t`s over ≤ 3 scalars and 1 array of ≤ 4 constant indices,
plus one symbolic-index read to exercise `array_disqualified`; two evaluators —
*full* (all steps) and *sliced* (steps with `ignore` skipped, elided stores
rewritten to identity).
**Assertions:** for all nondet inputs, the retained assert's truth value is
identical under both evaluators (equisatisfiability, `A4.1`); no retained step
reads a symbol defined only by an ignored step (`A4.2`); `index_reads`
propagation is monotone (`A4.3`).
**Twin:** delete the `array_disqualified` consultation → `A4.1` fails on the
symbolic-index input.

#### H-A5 — Unwind-bound selection & truncation semantics (P1, C6)
**Target:** `goto_symext::get_unwind`, `loop_bound_exceeded`
(`symex_goto.cpp:525,497`).
**Assertions:** precedence loop-specific > function-specific > global (`A5.1`);
`max_unwind == 0` ⇒ never stop (`A5.2`); in `loop_bound_exceeded`, exactly one
of {unwinding assertion, unwinding assumption, partial-loops no-op} is taken,
and the assumption branch is taken **only** when `no_unwinding_assertions`
(`A5.3`); after either branch the state guard is strengthened by
`¬cond` (`A5.4`).
**Twin:** swap the loop-specific / function-specific precedence → A5.1 fails.

#### H-A6 — MPOR independence relation (P0, I14)
**Target:** `execution_statet::check_mpor_dependency` +
`calculate_mpor_constraints`.
**Model:** T ≤ 3 threads, V ≤ 3 variables, nondet read/write sets as bitmasks.
**Assertions:** `A6.1` symmetry `dep(j,l) == dep(l,j)`; `A6.2` **completeness**:
if a genuine conflict exists (WW, WR, or RW on a shared variable) then `dep` is
true — a missed dependency is the unsound direction; `A6.3` read-read never
forces a dependency (the optimisation is retained and justified); `A6.4` the
`dependency_chain` update preserves transitive closure.
**Twin:** delete the "we wrote what that reads" clause → A6.2 fails.

#### H-A7 — Frame lifecycle & va_list cursor (P2, I16)
**Target:** `framet` ctor, `new_frame`, `pop_frame`, `previous_frame`,
`va_index`/`va_cursor` (`goto_symex_state.h:151-244,301-322`), `argument_assignments`.
**Assertions:** `previous_frame` is never evaluated with size < 2 (`A7.1` — the
`*(--(--end()))` UB); `va_cursor ≥ va_index` monotone and never wraps (`A7.2`);
`local_variables` of a popped frame are removed from L2 exactly once (`A7.3`);
`stack_frame_total` accumulation does not overflow (`A7.4`).
**Twin:** call `previous_frame` on a 1-frame stack → deref failure.

#### H-A8 — Equation context stack LIFO (P1, I13)
**Target:** `symex_target_equationt::push_ctx`/`pop_ctx`,
`runtime_encoded_equationt::scoped_end_points`/`cvt_progress`.
**Assertions:** balanced push/pop restores the exact step prefix (`A8.1`);
`pop_ctx` without a matching `push_ctx` is a violation (`A8.2`);
`cvt_progress` never points past the list end (`A8.3`).
**Twin:** pop one extra context → A8.2 fails.

#### H-A9 — Reference-into-container lifetime (P0-mem, I2/R3)
**Target:** the `valuet &entry` reference in `make_assignment` held across the
virtual `rename(lhs,count)` call.
**Model:** the `map_t` stub with an explicit `rehash-on-insert` that invalidates
outstanding references (modelled as a generation counter); the callee either
re-keys (insert ⇒ possible rehash) or hits the same key.
**Assertions:** `A9.1` the callee's key equals the caller's key (I2); `A9.2` if
`A9.1` is violated, the subsequent `entry.count`/`entry.constant` write is a
use-after-invalidation — the harness *demonstrates* the failure mode, pinning
why I2 must be asserted.
**Twin:** the harness's `_fail` variant is the re-keying callee.

#### H-A10 — Null-safe symbol lookup (P2/R4)
**Target:** the eight `*ns.lookup(...)` sites (`phi_function` `symex_goto.cpp:433`;
`symex_function.cpp:159`; `symex_valid_object.cpp:47`; `dynamic_allocation.cpp:66,92,105,118,143`).
**Model:** `lookup` returns nondet-null; the surrounding decision logic as coded.
**Assertions:** no null dereference on any path (`A10.1`); for each site, the
*claimed* precondition ("the name is always in the symbol table") is stated as a
citation or the site is reported as a defect.
**Twin:** none needed — the unfixed transcription *is* the failing variant; land
as `KNOWNBUG` until the fix, per §11.

### 7.2 Tier B — property / differential tests on the real engine (`unit/goto-symex/`)

All use `goto_factory::get_goto_functions` (`unit/testing-utils/goto_factory.h`)
to build real GOTO programs from C source strings, then run real symex and
inspect the produced `symex_target_equationt`.

| ID | Property | Method | Detects |
|---|---|---|---|
| **H-B1** | **SSA well-formedness validator** (P8/P11, I10) | Walk `SSA_steps`: (a) every assignment `lhs` is a `symbol2t` at `level2`/`level2_global`; (b) no SSA name is defined twice — promote `check_for_duplicate_assigns` from `log_status` to a hard check (R5); (c) every symbol read is previously defined, a `nondet$`/`NULL`/`INVALID` free symbol, or a global; (d) guard symbols are defined before use | The entire class of stale-index / aliasing bugs. **Highest-value single artefact in this plan** — reusable as an assertion inside every other Tier-B test. |
| **H-B2** | **Determinism** (P10) | Run symex twice in one process over the same program; compare the two equations step-for-step (kind, `crc` of `cond`, `guard`, `ignore`) | Iteration-order nondeterminism over pointer-keyed containers (`std::set<expr2tc>` in `thread_last_reads/writes`; the `generate_l2_state_hash` comment already concedes cross-run instability) |
| **H-B3** | **Slicer equisatisfiability** (P0, I11) | Build equation; clone; slice the clone; solve both per-claim with the real backend; assert identical per-claim verdicts on ≥ 30 small programs incl. arrays with symbolic indices | Slicer unsoundness on real formulas — the honest complement to H-A4 |
| **H-B4** | **Renaming round-trip** (I3/I4) | For each `SSA_stept`, `get_original_name` of `lhs` equals the L0 symbol; `rename` is idempotent; the level never decreases along the step list | `fixup_renamed_type` / `rename_address` regressions |
| **H-B5** | **Phi laws** (I8) | For 2-branch programs: #phi assignments == #variables written in exactly one branch + #variables written differently in both; **zero** phi for untouched variables | Over- and under-generation of phi nodes |
| **H-B6** | **Value-set merge monotonicity** (I9) | After `merge_value_sets`, assert the result ⊇ both inputs (using `value_sett` API) | An accidental intersection — a silent unsoundness |
| **H-B7** | **Assumption-discharge suite** (§6.1 rule 3) | For each Tier-A assumption in §7.3, an assertion on the real engine that it holds over the corpus | Over-constrained Tier-A proofs |
| **H-B8** | **Incremental-equation parity** (I13) | Same program with and without `--smt-during-symex`; assert identical claim count and per-claim verdicts | `runtime_encoded_equationt` ctx-stack bugs |

**Infrastructure note (blocking).** `unit/goto-symex/CMakeLists.txt` currently
has its single `new_unit_test(...)` **commented out** ("Our current CMake is
having some weird dependencies… It makes no sense for this test to depend on
solvers"). Tier B cannot start until this is fixed: the link set
`symex;solvers;gotoalgorithms;pointeranalysis;util_esbmc;langapi` must build.
This is milestone **M0** and is a genuine prerequisite, not boilerplate.

### 7.3 Undischarged-assumption register

Maintained as the harnesses land; an entry may not be closed without a Tier-B
discharge or an explicit, reviewed waiver.

| Assumption used by | Statement | Discharged by |
|---|---|---|
| H-A1, H-A9 | The L2 `name_record` key is stable across `make_assignment`'s inner `rename` (I2) | H-B7 assertion on the real `level2t` |
| H-A2 | `guard2tc::operator-=` satisfies `(g_cur ∨ g_mrg) → (diff ↔ g_mrg)` | **irep2 plan** H-A9/H-B4 — cross-document dependency |
| H-A2 | Incoming merge guards may overlap (no disjointness assumed) | by construction (not assumed) |
| H-A4 | Every `with2t` store the slicer elides has a `symbol2t` source and constant index | H-B7 counts the shapes reaching that branch |
| H-A6 | `thread_last_reads/writes` contain *all* accesses of the last transition, including through pointers | H-B7 + `get_expr_globals` audit (**open risk**, R11) |
| H-A8 | `push_ctx`/`pop_ctx` calls are balanced by the caller (`reachability_treet`) | H-B8 |
| all Tier A | `nondet` solver answers are *sound* (no wrong TRUE/FALSE) | out of scope — solver backends are Tier D |

### 7.4 Tier C — whole-tool metamorphic oracles

Scripted sweeps over existing corpora. Each is a *pure verdict comparison*: no
modelling, no assumptions, and a divergence is always a real bug in one of the
two configurations.

| ID | Relation | Corpus | Detects |
|---|---|---|---|
| **H-C1** | verdict(default) == verdict(`--no-slice`) | `regression/esbmc` CORE (1400 of 1543 dirs) | Slicer unsoundness/incompleteness end-to-end |
| **H-C2** | verdict(default) == verdict(`--no-simplify`) | same | Simplifier / constant-propagation semantic drift (P9) |
| **H-C3** | verdict(bitwuzla) == verdict(z3) | same | Encoding assumptions that only one solver tolerates |
| **H-C4** | verdict(default) == verdict(`--no-por`) and == verdict(`--state-hashing`) | `regression/esbmc-unix`, `regression/esbmc` concurrency tests | POR / state-hashing over-pruning (I14, I15) |
| **H-C5** | verdict(default) == verdict(`--no-interval-symex-guard`) | `regression/esbmc`, `regression/k-induction` | Interval-domain guard pruning (the documented hazard at `symex_goto.cpp:57-79`) |
| **H-C6** | **Unwind monotonicity**: FAILED at `--unwind k` ⇒ FAILED at every `k' > k` | loop-bearing subset | Lost counterexamples when the bound grows — a pure soundness relation, no oracle needed |
| **H-C7** | per-claim verdicts under `--multi-property` == the individual `--claim N` runs | `regression/esbmc` multi-assert tests | Claim/slice interaction bugs (cf. recent `multi_property_check` fixes) |

**Cost control.** H-C1/C2/C3 are the cheap wins (one extra run per test).
Run them as a scheduled (nightly/weekly) CI job, not per-PR; H-C6 needs 2–3 runs
per test and should be scoped to a curated ~200-test subset.

---

## 8. Property matrix

Rows = components (§4.1); columns = the brief's property list. Cell = harness
ID(s). `—` = not applicable; `§14` = deferred with rationale.

| Component ＼ Property | Mem-safety | Null-deref | Bounds | Invariant | State consistency | Symbolic update | Path condition | SSA | Expr semantics | Determinism | Constraint consistency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **C1** renaming / L2 | H-A9 | — | — | H-A1 | H-A1 | H-A1,H-B4 | — | **H-A1,H-B1** | H-B4 | H-B2 | H-B1 |
| **C2** `goto_symex_statet` | H-A7 | — | — | H-B4 | H-B4 | H-A1 | — | H-B1 | H-C2 | H-B2 | H-B1 |
| **C3/C3′** frames / merge_statet | **H-A7** | — | H-A7 | H-A3 | H-A3 | — | — | — | — | — | — |
| **C4** `symex_assign_*` + intrinsics | H-A10 | H-A10 | H-A10 | — | — | H-B1 | — | H-B1 | H-C2 | H-B2 | H-B1 |
| **C5** merge / phi | — | **H-A10** | — | **H-A2** | H-A3 | H-B5 | **H-A2** | H-B5 | — | H-B2 | H-B1 |
| **C6** unwind bounding | — | — | — | H-A5 | — | — | H-A5 | — | — | — | H-C6 |
| **C7** equation | H-A8 | — | H-A8 | H-A8 | H-A8 | — | — | **H-B1** | — | H-B2 | **H-B1** |
| **C7′** runtime-encoded | H-A8 | — | H-A8 | H-A8 | H-B8 | — | — | — | — | — | H-B8 |
| **C8** slicer | — | — | H-A4 | **H-A4** | — | — | H-A4 | — | — | — | **H-B3,H-C1** |
| **C9** execution_state / POR | — | — | H-A6 | **H-A6** | H-A6 | — | — | — | — | H-B2 | H-C4 |
| **C10** reachability_tree | — | — | — | H-A6 | — | — | — | — | — | H-B2 | H-C4 |
| **C11** memory model | H-A10 | **H-A10** | — | R8/R9 | — | — | — | — | — | — | — |
| **C12** pointer / deref | — | — | — | H-B6 | H-B6 | H-B6 | — | — | — | — | — |
| **C13** symmetry reduction | — | — | — | §14 | — | — | — | — | — | — | H-C4 |

**Coverage summary.** Every P0 component has at least one Tier-A *and* one
Tier-B/C artefact. The pointer analysis behind C12, the SMT backends, and
symmetry reduction's algebraic argument are explicitly deferred (§14), not
silently claimed.

---

## 9. Soundness risk assessment

### 9.1 Risks intrinsic to the harnesses

| Risk | Concrete failure mode in *this* subsystem | Mitigation | Detection |
|---|---|---|---|
| **Over-constraining** | H-A2 assumes incoming merge guards are disjoint (true for a 2-way `if`, false at an N-way join or a function-pointer combine) ⇒ the `ite` selection bug is unreachable in the model and "proved" | Rule 2 (cite every assumption) + Rule 3 (Tier-B discharge) + §6.3 forbidden list; parameterise the merge arity to 3 | H-B7 asserts the shapes actually occurring; if a shape appears that the model excluded, the harness is invalid |
| **Under-constraining** | H-A1 lets `count` start anywhere ⇒ a counterexample at `count = UINT_MAX` that real symex cannot reach; triage pressure then weakens the property | Triage every CE by trying to realise it as a C input; only tighten with a *cited* precondition | CE-realisation log; an unrealisable CE that cannot be excluded by a cited precondition is escalated, not silently assumed away |
| **Missed reachable behaviour** | Modelling only 2 threads (real POR bugs need 3); only constant array indices (the `array_disqualified` path needs a symbolic one); only forward gotos (backward = loops) | Bound-and-increment (Rule 4): prove at K, re-prove at K+1, k-induct for the general case; explicitly include one symbolic index and one backward goto in every relevant harness | A property that holds at K but fails at K+1 is reported as *bound-dependent*, never as proved |
| **Unrealistic assumptions** | `assume(ns.lookup != NULL)`; assuming `assert()` is active (it is **not** — R1); assuming single-threaded symex | §6.3 forbidden list is a review gate; every harness states its build mode and must be reasoned about under `NDEBUG` | Review checklist (§11.4) |
| **Masking defects (transcription drift)** | A Tier-A kernel is a *copy*; the real `phi_function` changes and the harness keeps proving the old code | Each harness header records `file:symbol` **plus a checksum of the cited region**; a CI job recomputes it and fails when the region changes without the harness being touched | `scripts/verification/symex/drift_check.py` (deliverable D7) |
| **Vacuous proofs** | `assume(false)` from contradictory preconditions; the assert sits after an unreachable branch; `--no-unwinding-assertions` truncates the loop and everything after is unreachable | Mandatory `_fail` twin (Rule 5); mandatory reachability probe (Rule 6); banned flag pairing (Rule 7); **reject any run whose log shows `0 remaining` VCCs after simplification** | Acceptance criteria §11.3 parse the `Generated N VCC(s), M remaining` line |
| **Cross-document gap** | H-A2 assumes the irep2 guard-algebra contract; if the irep2 plan is never executed the assumption is load-bearing and unproven | §7.3 register lists it as a *cross-document dependency*; it may not be closed locally | Register review at each milestone |
| **False assurance from Tier C** | Tier-C parity passing means "the two configurations agree", **not** "both are correct" (a bug present in both is invisible) | State this in every Tier-C report; pair with Tier A/B, never cite Tier C alone as a soundness proof | Report template |

### 9.2 Code-level findings (hypotheses with cited evidence)

Ranked by (impact × likelihood). **None is confirmed as a live end-to-end bug by
this document** — each is a prioritised target for the cited harness.

| ID | Severity | Finding / hypothesis | Evidence | Harness | Recommended action |
|---|---|---|---|---|---|
| **R1** | **High (systemic)** | **The shipped binary enforces none of goto-symex's invariants.** There are **113 `assert(...)` in `src/goto-symex/*.cpp` and 5 in the headers**, and **all 674 TUs in this build carry `-DNDEBUG`** (`build/compile_commands.json`, RelWithDebInfo). Every invariant in §4.2 marked "debug only" — including `pop_frame`'s merge-map emptiness (I6) and `coveredinbees`' monotonicity (I1) — is a **no-op in release**. A violation is silent and unbounded. | `grep -c 'assert(' src/goto-symex/*.cpp`; `-DNDEBUG` in all 674 compile commands | H-A1, H-A3 | Introduce a release-checked `SYMEX_INVARIANT(cond, msg)` (CBMC's `INVARIANT` pattern) and promote the ~10 load-bearing asserts (I1, I2, I6, I16) to it. Measure the cost; gate the rest behind an `--expensive-asserts` build option. |
| **R2** | **High (soundness)** | `pop_frame` discards `merge_state_map` under a debug-only `assert`. In release, a frame popped with pending merges **silently drops those paths** ⇒ missed bug, no diagnostic. | `goto_symex_statet::pop_frame`, `goto_symex_state.h:310` | H-A3 | Promote to `SYMEX_INVARIANT`; add H-B1-adjacent runtime check counting pushed vs merged snapshots per frame. |
| **R3** | **Medium–High (memory safety)** | `make_assignment` holds `valuet &entry` — a reference **into** `current_names` (`std::unordered_map`) — across the virtual call `rename(lhs_symbol, entry.count + 1)`, which reaches `coveredinbees` and performs `current_names[key]`. Today this is safe *only because* the recomputed key is identical (the symbol is still L1 at that point), so `operator[]` finds rather than inserts. **The invariant is unasserted**; any change that re-keys before the call ⇒ insert ⇒ possible rehash ⇒ **dangling `entry`**, then `entry.count = …; entry.constant = …` is a use-after-free. The code comment concedes the hazard. | `renaming::level2t::make_assignment` `renaming.cpp:344-369`; `::coveredinbees` `:230-246`; comment "This'll update entry beneath our feet" | H-A9 | Restructure to avoid the held reference (re-`find` after the call), or assert key stability. ASan/H-A9 pin the failure mode. |
| **R4** | **Medium (crash → no verdict)** | **Eight unchecked `*ns.lookup(...)` dereferences.** `namespacet::lookup` returns `nullptr` on miss (as `renaming.cpp:15-21` itself demonstrates by checking). A miss ⇒ null deref ⇒ SIGSEGV mid-verification. `phi_function`'s site is the most exposed: it filters only `goto_symex::guard!` and `symex::invalid_object` before looking up an arbitrary merged variable's base name. | `symex_goto.cpp:433`; `symex_function.cpp:159`; `symex_valid_object.cpp:47`; `dynamic_allocation.cpp:66,92,105,118,143` | H-A10 | Add checked lookups with a diagnostic (`log_error` + controlled abort) or prove the precondition per site and record it as a cited comment. |
| **R5** | **Medium (soundness detector disabled)** | `check_for_duplicate_assigns` — the *only* in-tree checker for the core SSA invariant I10 — merely `log_status`es duplicates and then reports "Checked N insns". It never fails, and nothing calls it in a normal run. | `symex_target_equationt::check_for_duplicate_assigns`, `symex_target_equation.cpp` | H-B1 | Turn it into a validator returning a bool; run it under a debug/CI flag over the whole regression corpus. |
| **R6** | **Medium (unsound pruning, opt-in flag)** | `state_hashing_level2t::make_assignment` keys `current_hashes` by the **L0** original name, acknowledged in-code ("XXX — consider whether to use l1 names instead. Recursion, reentrancy."). Two states that differ only in the L1 activation of a recursive local therefore fingerprint identically ⇒ `hit_hashes` prunes a genuinely different state ⇒ missed interleaving. Severity is bounded by `--state-hashing` being opt-in. | `execution_state.cpp:~1342-1378`; `reachability_treet::hit_hashes`, `reachability_tree.h:352` | H-A8-style model + **H-C4** | Key by the L1 name record; H-C4 parity sweep quantifies the current gap. |
| **R7** | **Low–Medium (UB)** | `previous_frame()` computes `*(--(--call_stack.end()))` with no size check — UB when `call_stack.size() < 2`, and it returns a reference to a `framet` that a subsequent `pop_frame` invalidates. | `goto_symex_statet::previous_frame`, `goto_symex_state.h:319` | H-A7 | Add a size precondition (release-checked) or return `std::optional`. |
| **R8** | **Medium (documented model gap)** | `is_valid_object` returns `false` for **every** non-static, non-dynamic symbol: the stack-scope branch is `#if 0`'d out with "XXX re-enable to be able to check for stack-var-out-of-scope problems". Stack-object validity is therefore not modelled, and `dynamic_allocation.cpp` compensates by *assuming* `invalid_pointer` applies only to dynamic objects ("we never update `__ESBMC_alloc` for stack ptrs"). Net effect on stack-lifetime bugs (use-after-scope) is a **missed-bug** direction. | `goto_symext::is_valid_object`, `symex_valid_object.cpp:85-118`; `dynamic_allocation.cpp:110-116` | H-A10 + a targeted `regression/esbmc` use-after-scope corpus | Quantify with a dedicated corpus before attempting a fix; the fix is a model change, not a patch. |
| **R9** | **Low–Medium (approximation direction unproven)** | Three documented "sound over-approximation" claims are unproven: value-set filtering after a pointer havoc (`symex_assign.cpp:~550-570`), the non-scalar uninterpreted-function fallback (`symex_function.cpp:~410-430`), and the function-pointer target enumeration over an over-approximated value set (`symex_function.cpp:~766`). Each *argues* the direction in a comment; none is checked. | cited lines | H-B6 + H-C1/H-C3 | For each, state the claim as a checkable predicate and add a Tier-B assertion (e.g. filtered set ⊆ original **and** the dropped entries are `unknown`/`invalid` only). |
| **R10** | **Low (latent UB)** | `renaming::level2t::name_record`'s `name_record() = default` leaves `lev`, `l1_num`, `t_num` **and the derived `hash`** indeterminate (contrast `level1t::name_record`, which initialises `base_name("")`). No current default-construction site was found, but a future one (`std::optional`, map default-insert, array of records) would read indeterminate memory in `compare`/`hash`. | `renaming.h:143-214` | MSan (Tier D) + a `static_assert`-style unit check | Add default member initialisers; near-zero cost. |
| **R11** | **Open question (concurrency soundness)** | MPOR's independence decision consumes `thread_last_reads`/`thread_last_writes`, populated via `get_expr_globals`, which resolves pointer operands through the *current* value set. If a write through a pointer whose value set is incomplete (or whose entry is `unknown`) is missed, the dependency is missed and an interleaving is dropped — **unsound**. `get_expr_globals` also early-returns entirely under `--data-races-check-only`. | `execution_statet::get_expr_globals`, `check_mpor_dependency`; `reachability_treet::ever_written_globals`/`address_taken_globals` | H-A6 (relation) + **H-C4** (end-to-end) | Determine whether an `unknown` value-set entry forces a conservative dependency; if not, that is a concrete unsoundness to fix. Highest-uncertainty item in this plan. |
| **R12** | **Info (bounded by design)** | With `--no-unwinding-assertions`, `loop_bound_exceeded` emits an *assumption* that truncates the path; a `VERIFICATION SUCCESSFUL` then covers only the truncated prefix. This is intended BMC behaviour, but the repo has already been bitten by it in *verification harnesses* (`CLAUDE.md` bans pairing it with reachability checks). | `goto_symext::loop_bound_exceeded`, `symex_goto.cpp:497-523` | H-A5 | No code change; encode as an acceptance criterion (§11.3) so no harness in this plan ever uses that flag. |

---

## 10. Incremental verification roadmap

Ordered by (risk × tractability). Each milestone yields check-in-able artefacts;
no milestone depends on a later one. The ESBMC extension work items (WI-1…WI-6,
§13.6) run **in parallel** with this track: no property claimed in §8 is blocked
on them.

**M0 — Infrastructure (0.5 wk).**
Re-enable `unit/goto-symex/CMakeLists.txt` (fix the `symex;solvers;
gotoalgorithms;pointeranalysis;util_esbmc;langapi` link set). Stand up the
`regression/esbmc/symex_*` harness skeleton + `test.desc` template. Land the
drift-check script. In parallel, start **WI-1** (`<shared_mutex>` operational
model) and **WI-2** (`<type_traits>` completion), and file G1–G8 as issues (D12).
*Artefact:* one green + one deliberately-red smoke harness proving the pipeline
detects an injected bug; a building `unit/goto-symex` target; WI-1 merged.

**M1 — Low-level kernels: SSA algebra (1 wk).** H-A1, H-A9, H-A7.
Retires I1/I2/I16; produces the R3 and R7 verdicts. In parallel: **WI-3**
(`initializer_list` / `iterator_traits` / `this_thread` / `aligned_storage`),
which completes the parse path to `renaming.h`. *Artefact:* three Tier-A harness
pairs + the R3 restructure PR if H-A9 confirms the hazard; WI-2/WI-3 merged.

**M2 — Isolated core algorithms: merging and bounding (1.5 wk).**
H-A2 (the highest-value harness), H-A3, H-A5. Dual-solver mandatory.
*Artefact:* merge-soundness proof at arity 2 **and** 3; R2 `SYMEX_INVARIANT` PR.

**M3 — Release-mode enforcement (0.5 wk).** R1: introduce `SYMEX_INVARIANT`,
promote the ~10 load-bearing asserts, measure the runtime cost on
`regression/esbmc` (accept if < 2 %). *Artefact:* invariant macro + benchmark
note. This is a prerequisite for M1/M2's proofs to mean anything in the shipped
binary.

**M4 — Individual symex operations on the real engine (1.5 wk).**
H-B1 (SSA validator — build it first, reuse everywhere), H-B4, H-B5, H-B2.
Run the **WI-4** Tier-B′ pilot here (include the real `renaming.h`, drive the
real `level1t`) — gated on parse **and** < 60 s verification; a negative result
is recorded in §13.3 and Tier A is kept.
*Artefact:* `unit/goto-symex/{ssa_wellformed,renaming,phi,determinism}.test.cpp`;
R5 promoted to a real validator; WI-4 verdict.

**M5 — Constraint generation: the slicer (1 wk).** H-A4, H-B3, then the
**H-C1** sweep over all 1400 `regression/esbmc` CORE tests. *Artefact:* slicer
equisatisfiability suite + the first whole-corpus parity report.

**M6 — Subsystem interaction: concurrency (1.5 wk).** H-A6, H-C4, and the R11
investigation (pointer-mediated writes in `get_expr_globals`). Highest
uncertainty — budget for the answer being "there is a real gap". *Artefact:* MPOR
relation proof + POR/state-hashing parity report + an R11 verdict.

**M7 — End-to-end scenarios and regression pinning (1 wk).** H-C2, H-C3, H-C5,
H-C6, H-C7 wired as a scheduled CI job; H-B8. *Artefact:* the oracle job + a
per-oracle baseline of known divergences (each triaged to a filed issue or a
justified waiver — **an untriaged divergence is a blocker, not a baseline**).

**M8 — Previously-reported bugs and regression cases (0.5 wk, continuous).**
Convert every historical goto-symex issue with a reproducer into a Tier-A or
Tier-B case; start from the tree's own `KNOWNBUG` inventory. *Artefact:* a
`regression/esbmc/symex_regressions/` index mapping issue → harness.

Total ≈ 9 engineer-weeks for the verification track, plus ≈ 2 weeks for the
ESBMC extension critical path (WI-1…WI-3, §13.6) running alongside it.
**M1–M3 alone** retire the systemic finding (R1) and the two memory-safety
hypotheses (R3, R7); **M5's H-C1** is the single cheapest high-yield artefact in
the plan and can be run opportunistically from M0.

---

## 11. Automation, regression, and acceptance

### 11.1 Organisation and naming

```
docs/roadmap/goto-symex-verification-plan.md  this document (+ verdict log, §9.2/§10)
regression/esbmc/symex_<area>_<nn>/           Tier A, passing
    ├── symex_<area>_<nn>.c
    └── test.desc
regression/esbmc/symex_<area>_<nn>_fail/      Tier A, anti-vacuity twin
unit/goto-symex/<area>.test.cpp               Tier B  (wire in CMakeLists.txt)
scripts/verification/symex/
    ├── oracle_slice_parity.sh                H-C1
    ├── oracle_simplify_parity.sh             H-C2
    ├── oracle_solver_parity.sh               H-C3
    ├── oracle_por_parity.sh                  H-C4
    ├── oracle_unwind_monotonic.sh            H-C6
    └── drift_check.py                        transcription-drift guard
.github/workflows/symex-oracles.yml           scheduled Tier-C job
```

`<area>` ∈ `{ssa, merge, mergequeue, slice, unwind, mpor, frame, eqctx,
lookup, refalias}` — one area per harness family, matching §7.

### 11.2 CI integration

| Artefact | Trigger | Budget |
|---|---|---|
| Tier A (`regression/esbmc/symex_*`) | every PR, via the existing `ctest -L esbmc` path | each harness < 30 s; the suite's 120 s per-test harness cap is hard |
| Tier B (`unit/goto-symex`) | every PR, `ctest -LE regression` | < 60 s total |
| Tier C oracles | **scheduled** (nightly for C1/C2/C3, weekly for C4/C6/C7) + `workflow_dispatch` | ≤ 90 min per leg, mirroring `sanitizers.yml` |
| Drift check | every PR touching `src/goto-symex/**` | seconds |
| Sanitizers (Tier D) | existing `sanitizers.yml` (asan/ubsan/tsan) — add an **msan** leg for R10 | existing budget |

Per repo convention the local regression cap is **5 minutes**; a full-corpus
Tier-C sweep is a CI-only activity and must never be run inside a PR loop.
Remember `rm -rf /tmp/esbmc-headers-*` after large sweeps (~7.4 MB per test).

### 11.3 Acceptance / rejection criteria

A Tier-A result is **ACCEPTED** only if *all* hold:

1. `VERIFICATION SUCCESSFUL` (or the expected `FAILED` for a `_fail` twin);
2. the log line `Generated N VCC(s), M remaining after simplification` shows
   **M ≥ 1** — `M == 0` is a vacuity smell and is rejected;
3. the anti-vacuity twin reports the *opposite* verdict;
4. the reachability probe fires (`FAILED` where expected);
5. **no** unwinding-assertion failure in the log, and `--no-unwinding-assertions`
   is **absent** from the invocation;
6. **dual-solver agreement** (Bitwuzla and Z3) for any P0 property;
7. the ESBMC commit hash and solver versions are recorded in the verdict log.

Otherwise the result is **INCONCLUSIVE — do not claim**, with a per-criterion
breakdown. "Timed out" is never "SUCCESSFUL".

A Tier-C divergence is **never** auto-waived: it is triaged to (a) a filed
issue, (b) a documented and reviewed semantic difference between the two flag
settings, or (c) a fix. Adding it to a baseline without one of those is
prohibited.

### 11.4 Review checklist for new/changed harnesses

- [ ] Every `__ESBMC_assume` cites the `file:symbol` establishing it.
- [ ] No assumption from the §6.3 forbidden list.
- [ ] Assumptions not discharged by Tier B are listed in §7.3.
- [ ] `_fail` twin exists and produces the opposite verdict.
- [ ] Reachability probe present.
- [ ] The cited source region's checksum is recorded (drift guard).
- [ ] `test.desc` has **no comments** — line 1 mode, line 2 source, line 3 flags,
      line 4+ are expected-output regexes consumed verbatim (Appendix B).
- [ ] Runtime < 30 s; the harness does not depend on `-DNDEBUG` being absent.
- [ ] For a harness that motivates a *code* fix adding or removing a branch:
      Mode C (C-Live / C-Dead) discharged per `CLAUDE.md`.

### 11.5 Maintenance

Each harness file header carries: target `file:symbol`, the property IDs it
discharges, its assumption list, and the cited-region checksum. When
`src/goto-symex` changes, the drift guard fails the PR and forces one of:
re-transcribe, widen the harness, or explicitly retire it (with the property
matrix updated in the same PR). The property matrix (§8) is the source of truth
for what is claimed; a claim may not outlive its harness.

---

## 12. Deliverables

| # | Deliverable | Location | Milestone |
|---|---|---|---|
| **D1** | This plan (architecture, scope, properties, risks, roadmap) | `docs/roadmap/goto-symex-verification-plan.md` | — |
| **D2** | Prioritised verification backlog | §9.2 (R1–R12) × §10 milestones | M0 |
| **D3** | Component-to-harness mapping | §8 property matrix + §7 harness list | M0 |
| **D4** | Property matrix | §8 | M0 |
| **D5** | Risk assessment (harness-design + code-level) | §9.1 / §9.2 | M0 |
| **D6** | Tier-A harnesses: 10 kernels × {ok, fail} | `regression/esbmc/symex_*/` | M1–M6 |
| **D7** | Tier-B suites (8 files) + working `unit/goto-symex` CMake wiring + drift guard | `unit/goto-symex/`, `scripts/verification/symex/drift_check.py` | M0, M4 |
| **D8** | Tier-C oracle scripts + scheduled workflow | `scripts/verification/symex/`, `.github/workflows/symex-oracles.yml` | M5, M7 |
| **D9** | `SYMEX_INVARIANT` release-checked macro + promoted invariants + cost benchmark | `src/goto-symex/` | M3 |
| **D10** | Fix PRs for confirmed findings (R2–R5, R7, R10 are tractable; R6/R8/R11 are investigations first) | code PRs, each with Mode-C proof where a branch changes | M1–M6 |
| **D11** | Verdict log — per-harness result, ESBMC commit, solver versions, date — appended to this document | §9.2 / §10 | continuous |
| **D12** | Issues against ESBMC's C++ operational model, one per gap G1–G8 (§13.2), each with a reproducer and two regression tests | GitHub, label `clang-cpp-frontend` | M0 |
| **D13** | ESBMC extension work items WI-1…WI-3 (`<shared_mutex>`, `<type_traits>` completion, `<compare>`/`std::unreachable`, `initializer_list`/`iterator_traits`/`this_thread`) | `src/cpp/library/`, `regression/esbmc-cpp*` | M0–M1 |
| **D14** | Tier B′ pilot result (WI-4) — a harness including the real `renaming.h`, **or** a recorded negative result keeping Tier A | `unit/goto-symex/` or §13.3 | M4 |

---

## 13. Extending ESBMC to verify goto-symex

Tier A is *transcription* only because ESBMC cannot ingest the real headers, and
transcription carries a permanent drift risk (§9.1). This section turns that
caveat into a scheduled work programme: what exactly blocks ESBMC today
(measured, §13.2), what closing it does and does **not** buy (§13.3), which
verification capabilities are genuinely missing versus already present
(§13.4), the bridge to use meanwhile and its hazard (§13.5), and the sequenced
work items (§13.6).

Every work item below is an **independent, upstreamable ESBMC improvement with
its own regression tests** — each benefits any user verifying modern C++, not
only this plan.

### 13.1 Method

```sh
esbmc probe.cpp -Wc,-include,shim.h -I<esbmc includes> --std c++20 --parse-tree-only
```

`-Wc,OPT,...` forwards options to the clang frontend (`--help`: "options
directly to the C/C++ frontends"), so a shim header can stage facilities the
operational model lacks. Starting from `#include <goto-symex/renaming.h>`, each
missing declaration was added to the shim and the probe re-run, until the error
set stopped naming new facilities. A separate 26-case sweep isolated
language-level from library-level support.

### 13.2 Measured gap

**Language support is not the problem.** The 26-case sweep passed on: concepts
and constrained templates, fold expressions, `if constexpr`, `consteval`,
structured bindings, virtual dispatch with `override`/virtual destructor,
exceptions (`throw`/`catch` through `std::exception`), `std::variant`,
`std::optional`, `std::function` with lambdas, `std::tuple` + `std::apply`,
`std::unordered_map`, `std::list`, `std::atomic` with explicit memory orders,
`std::void_t`, `std::alignment_of`. Every failure was a *library* facility.

| ID | Missing facility | Where it blocks | Evidence |
|---|---|---|---|
| **G1** | `<type_traits>`: `is_standard_layout`, `is_trivial`, `is_aggregate`, `is_assignable`, `is_copy_assignable`, `is_copy_constructible`, `is_destructible`, `is_nothrow_move_constructible`, `is_nothrow_move_assignable`, `conjunction`/`disjunction`/`negation`, `aligned_storage`/`aligned_storage_t` | `immer/detail/combine_standard_layout.hpp`, `immer/detail/type_traits.hpp:144-145`, `immer/detail/hamts/champ.hpp:145` | `src/cpp/library/type_traits` (811 lines) defines 34 `is_*` traits; these are absent |
| **G2** | **`<shared_mutex>` — header absent entirely** | `src/util/base/string_pool.h:9` ← `irep_idt.h` ← `irep.h` ← … — reached by **every** ESBMC header | `fatal error: 'shared_mutex' file not found`; no `shared_mutex` in `src/cpp/library/` |
| **G3** | `std::initializer_list` not usable as a template in namespace `std` | `immer/detail/hamts/champ.hpp:290`, `immer/map.hpp:178` | `no template named 'initializer_list' in namespace 'std'` |
| **G4** | `std::iterator_traits<T>::difference_type` | `immer/detail/type_traits.hpp:148` | `expected unqualified-id` at the `difference_type` member |
| **G5** | `std::this_thread::yield` | `immer/lock/spinlock_policy.hpp:44` | `no member named 'this_thread' in namespace 'std'` |
| **G6** | `<compare>`: `std::strong_ordering` for a defaulted `operator<=>` | any C++20 defaulted three-way comparison | probe: `cannot default 'operator<=>' because type 'std::strong_ordering' was not found` |
| **G7** | `std::unreachable` (C++23) | `irep2`'s exhaustive-dispatch tails (`std::unreachable()`) | probe: `no member named 'unreachable' in namespace 'std'` |
| **G8** | C++23 library surface generally | the subsystem builds at `-std=gnu++23`; frontend regression stops at `regression/esbmc-cpp20` | `build/compile_commands.json` |

**G2 is the first-order blocker** and is independent of `immer`: it stops
*any* file that reaches `irep_idt`, which is essentially every ESBMC header.
It is also the cheapest to close.

### 13.3 Tractability — parsing is necessary, not sufficient

Closing G1–G8 makes the code *parse*. Whether ESBMC can then *verify* it is a
separate axis, and the STL operational model's cost scales steeply:

| Probe | Flags | Result |
|---|---|---|
| 1-key `unordered_map<string,POD>` + `make_shared` | `--unwind 4` | `SUCCESSFUL`, **4.6 s**, 2132 VCCs |
| 4-key `unordered_map` insert+read loop | `--unwind 5` | `SUCCESSFUL`, **85.9 s** |
| same | `--unwind 8` | **> 280 s — timeout** |

**Conclusion, stated plainly: even with G1–G8 closed, whole-translation-unit
verification of `src/goto-symex` will not be tractable.** The `immer` HAMT alone
is far beyond the measured envelope. What the operational-model work actually
buys is narrower and still worthwhile:

- **Tier B′** — reduced harnesses that `#include` the *real* header and drive
  the *real* class at small bounds, removing transcription drift for the
  smallest components. Realistic candidates: `renaming::level1t` / `level2t`
  operations. Not realistic: anything whose control flow runs through the immer
  HAMT or `irep2`'s dispatch layer.
- Independent value to ESBMC's users (G2, G6, G7 are plain defects).

Tier A transcription therefore remains the workhorse, and the drift guard
(§9.1, D7) remains mandatory. **No milestone in §10 is blocked on §13.6.**

### 13.4 Verification capabilities: genuinely missing vs already present

| ID | Capability | Status | Impact on this plan |
|---|---|---|---|
| **E1** | **Container reference / iterator invalidation semantics.** R3 and H-A9 need "a reference into an `unordered_map` is invalidated by a rehash". `src/cpp/library/unordered_map` (562 lines) contains **zero** mentions of rehash or invalidation. | **Missing** | The property is unstatable on the real class; Tier A must model invalidation by hand (a generation counter in the `map_t` stub, §6.2). Proposed model: per-container generation counter bumped by every mutator; dereference of a reference captured at an older generation is a deref failure. |
| **E2** | **Relational (2-safety) verification.** H-A4 (sliced vs unsliced equisatisfiability), H-B2 (determinism) and every Tier-C parity oracle are two-run properties. No equivalence / product-program mode exists (`src/esbmc/options.cpp` has no such option). | **Missing** | Worked around by self-composition inside a single Tier-A harness and by scripted verdict comparison in Tier C. A native mode would promote H-C1/H-C2 from *sweep* to *proof*. Stretch goal, valuable well beyond this plan. |
| **E3** | **Quantifiers** — `__ESBMC_forall`, `__ESBMC_exists` | **Present** (`src/clang-c-frontend/clang_c_language.cpp:612-613`) | I10 ("no duplicate SSA name in the equation") is expressible today. **Do not re-request.** |
| **E4** | **Function contracts** — `__ESBMC_requires`, `__ESBMC_ensures`, `__ESBMC_old` | **Present** (`clang_c_language.cpp:659-663`; `regression/function_contract`) | The §4.2 invariants can be written as contracts directly; the Phase-2 contract regressions required by `CLAUDE.md` have first-class support. |
| **E5** | **k-induction with convergence** | **Present** (`--k-induction`) | Required by §6.1 rule 4 and the acceptance criteria (§11.3). |

E3–E5 are recorded explicitly so that no work item re-requests a feature ESBMC
already has.

### 13.5 The shim bridge — and its hazard

Until G1–G8 land, `-Wc,-include,<shim>.h` can stage the missing declarations for
an individual harness. This is how §13.2 was measured, and it is a legitimate
bring-up tool.

**It is also a soundness hazard, and is governed accordingly.** A shim that
defines `is_standard_layout<T>::value = true` unconditionally forces every
branch on that trait to take one arm — precisely the over-constraining /
defect-masking failure mode of §9.1. Rules:

1. A shim may only supply facilities the harness under test **does not branch
   on**. If the code selects behaviour by the trait, the shim is invalid.
2. Every shimmed name is recorded in the §7.3 undischarged-assumption register.
3. **A shimmed harness may never be cited as a P0 proof** — it is bring-up
   evidence only, until the corresponding G-item lands.

### 13.6 Work items

| WI | Work | Effort | Milestone | Unblocks |
|---|---|---|---|---|
| **WI-1** | `<shared_mutex>` operational model (G2) | ~2 d | M0 | Including *any* ESBMC header in a harness. Highest ratio in this section. |
| **WI-2** | `<type_traits>` completion (G1) + `<compare>` `strong_ordering` (G6) + `std::unreachable` (G7) | ~1 wk | M0–M1 | `immer`-dependent and `irep2`-dependent headers; two of the three are plain user-facing defects |
| **WI-3** | `std::initializer_list` template form (G3), `iterator_traits::difference_type` (G4), `this_thread::yield` (G5), `aligned_storage[_t]` | ~2 d | M1 | Parsing `level1_map.h` → `renaming.h` end-to-end |
| **WI-4** | **Tier B′ pilot**: a reduced harness that `#include`s `renaming.h` and drives the real `level1t`. **Gate:** must parse *and* verify in < 60 s. If it does not, record the negative result in §13.3 and keep Tier A — do not force it. | ~1 wk | M4 | Removes transcription drift for C1 |
| **WI-5** | E1 container reference/iterator invalidation modelling | ~2–3 wk | M6 | Stating R3/H-A9 on the real class; benefits all STL verification |
| **WI-6** | E2 native 2-safety / equivalence mode | unscoped | post-M7 | Promotes H-C1/H-C2 from sweep to proof |

**Critical path:** WI-1 → WI-2 → WI-3, ≈ 2 weeks, and it is *parallel to* M1–M3
rather than blocking them. WI-4 is a gated experiment with an explicit
accept-the-negative-result branch. WI-5/WI-6 are stretch goals; neither is a
precondition for any property claimed in §8.

Each WI ships with the repo-mandated two regression tests (one passing, one
failing) under the appropriate `regression/esbmc-cpp*` suite, and is filed as a
separate issue labelled `clang-cpp-frontend` so the work is reviewable
independently of this plan.

---
## 14. Undischargeable today — and why

Stated plainly, to avoid over-claiming:

1. **Real goto-symex translation units.** Two independent obstacles, and only
   the first is being removed. *(a) Parsing* — probe P-2: the C++ operational
   model is missing the facilities enumerated as G1–G8 in §13.2, starting with
   `<shared_mutex>` (which every ESBMC header reaches via `irep_idt`). This is a
   bounded backlog and §13.6 schedules it. *(b) Tractability* — the measurements
   in §13.3 (a 4-key `unordered_map` loop takes 86 s at `--unwind 5` and times
   out at `--unwind 8`) put whole-TU verification out of reach **even after (a)
   is fixed**. Tier A is therefore *transcription*, and its fidelity rests on the
   drift guard (§11.1), not on the compiler. **This remains the single largest
   soundness caveat of the whole plan and must be stated in every report** —
   closing §13.6 narrows it (Tier B′, §13.3) but does not eliminate it.
2. **The pointer analysis / value-set fixpoint** (`src/pointer-analysis`).
   goto-symex *consumes* `value_sett`; its correctness (that the points-to set is
   a genuine over-approximation) is a separate obligation. H-B6 checks only that
   goto-symex's *merge* of value sets is monotone.
3. **The SMT backends** (`src/solvers`) and the solvers themselves. Modelled as
   a sound oracle; a wrong `tvt` from `ask_solver_question` is out of scope.
4. **irep2 internals** — refcounting, CRC, ordering, guard set-algebra.
   Delegated to `docs/irep2-verification-plan.md`; §7.3 records the one
   load-bearing cross-document dependency (H-A2 on irep2 H-A9/H-B4).
5. **Symmetry reduction** (`symex_symmetry.cpp`). Its soundness is an algebraic
   argument about thread-symmetric formulas; no faithful reduced model was
   identified. Covered only end-to-end by H-C4-style parity.
6. **Data-race freedom of ESBMC's own parallelism.** `--parallel-solving` solves
   an already-built equation; the plan *assumes* symex itself is single-threaded
   (the assumption `level1_map.h` and `guard_seq.h` explicitly rely on). TSan
   (existing CI) is the tripwire; **if symex is ever parallelised, this plan and
   the irep2 plan must both be re-audited.**
7. **Absolute (unbounded) correctness of the engine.** Every Tier-A result is a
   proof at a bound, or a k-induction proof with convergence. Where convergence
   is not achieved, the result is reported as *bounded*, never as *proved*.

---

## Appendix A — Methodological basis

- **Design by contract.** Every harness is precondition (`__ESBMC_assume`) →
  operation → postcondition (`assert`). §4.2's invariants are the class
  invariants a DbC discipline would attach; several (I1, I6, I16) are *documented
  and asserted only in debug* — the plan promotes them to release-checked
  obligations (R1/D9).
- **Anti-vacuity via mutation.** Each property ships a perturbed twin that must
  fail, giving the check demonstrated discriminating power (mutation testing
  adapted to model checking).
- **Differential / metamorphic verification.** Where an optimisation has a naive
  reference (slicer vs no-slice, simplifier vs `--no-simplify`, POR vs `--no-por`),
  verify *equivalence to the reference* rather than re-deriving correctness.
  H-C6's unwind monotonicity is a metamorphic relation requiring no oracle at all.
- **Bounded proof + inductive step.** Prove at a small bound with full unwinding;
  generalise with `--k-induction` requiring convergence.
- **Dual-solver agreement** for all P0 results, matching the repo's Mode-C gate.
- **Assumption discharge.** Tier-A assumptions are obligations on Tier B, not
  free parameters (§6.1 rule 3, §7.3). This is what keeps a reduced model from
  quietly becoming a different program.

## Appendix B — `test.desc` format and invocation recipes

`test.desc` has **no comment syntax**: line 1 is the mode
(`CORE`/`KNOWNBUG`/`FUTURE`/`THOROUGH`), line 2 the source file, line 3 the ESBMC
flags, and **every line from 4 onward is a verbatim expected-output regex**
consumed by `regression/testing_tool.py`. A stray `#…` becomes a regex that never
matches. There is **no `EXIT=` line** — `testing_tool.py` never emits one.

Passing Tier-A harness (straight-line kernel, e.g. H-A1):

```
CORE
symex_ssa_01.c
--overflow-check --unsigned-overflow-check --memory-leak-check
^VERIFICATION SUCCESSFUL$
```

Loop-bearing kernel needing the inductive step (e.g. H-A4's worklist):

```
CORE
symex_slice_01.c
--overflow-check --unsigned-overflow-check --k-induction
^VERIFICATION SUCCESSFUL$
```

Anti-vacuity twin (identical flags; only the expected verdict differs):

```
CORE
symex_ssa_01_fail.c
--overflow-check --unsigned-overflow-check --memory-leak-check
^VERIFICATION FAILED$
```

A finding recorded before its fix is ready (never land an R-harness as
`CORE`-SUCCESSFUL ahead of its fix; prefer fix-and-prove in one PR):

```
KNOWNBUG
symex_lookup_01.c
--memory-leak-check
^VERIFICATION SUCCESSFUL$
```

Local iteration:

```sh
build/src/esbmc/esbmc regression/esbmc/symex_ssa_01/symex_ssa_01.c \
  --overflow-check --unsigned-overflow-check --memory-leak-check
# then repeat with --z3 for the dual-solver gate
```

Tier-C parity sweep (H-C1 sketch):

```sh
for d in regression/esbmc/*/; do
  head -1 "$d/test.desc" | grep -q '^CORE' || continue
  src=$(sed -n 2p "$d/test.desc"); flags=$(sed -n 3p "$d/test.desc")
  a=$(timeout 120 build/src/esbmc/esbmc "$d/$src" $flags 2>&1 | grep -Eo 'VERIFICATION (SUCCESSFUL|FAILED)')
  b=$(timeout 120 build/src/esbmc/esbmc "$d/$src" $flags --no-slice 2>&1 | grep -Eo 'VERIFICATION (SUCCESSFUL|FAILED)')
  [ "$a" = "$b" ] || echo "DIVERGENCE $d: default=$a no-slice=$b"
done
rm -rf /tmp/esbmc-headers-*
```

Cross-check a Tier-A kernel against its source before transcription:

```sh
grep -n "make_assignment\|coveredinbees" src/goto-symex/renaming.cpp
grep -n "phi_function\|merge_state_guards\|merge_gotos" src/goto-symex/symex_goto.cpp
```

## Appendix C — Reproducing the §13 measurements

All figures in §13.2 and §13.3 come from `build/src/esbmc/esbmc` (ESBMC 8.4.0)
on this tree. Re-run before citing them; they will move as the operational
model improves.

**Language / library support sweep (§13.2).** One probe per feature, verdict is
"does it parse":

```sh
probe() {  # $1 = name, $2 = code
  printf '%s\n' "$2" > "p_$1.cpp"
  out=$(timeout 120 build/src/esbmc/esbmc "p_$1.cpp" --parse-tree-only --std c++20 2>&1)
  echo "$out" | grep -q "PARSING ERROR" && echo "$1 FAIL" || echo "$1 OK"
}
probe is_standard_layout '#include <type_traits>
struct S{int a;}; int main(){ static_assert(std::is_standard_layout<S>::value); }'
probe concepts_cpp20 'template<class T> concept Num = requires(T a){ a+a; };
template<Num T> T f(T x){return x;} int main(){ return f(1); }'
```

**Distance-to-parse for a real header (§13.2 gap table).** `-Wc,...` forwards
options to the clang frontend, so a shim header stages what the operational
model lacks; add each missing name to the shim and re-run to reveal the next
blocker:

```sh
cat > shim.h <<'EOF'
namespace std {
template <class T> struct is_standard_layout { static constexpr bool value = true; };
template <class T> struct is_trivial { static constexpr bool value = true; };
}
EOF
cat > probe.cpp <<'EOF'
#include <goto-symex/renaming.h>
int main() { renaming::level1t l1; (void)l1; return 0; }
EOF
build/src/esbmc/esbmc probe.cpp -Wc,-include,shim.h \
  -Ibuild/src -Isrc -Isrc/util -Isrc/util/lib \
  -Ibuild/_deps/fmt-src/include -Ibuild/_deps/immer-src \
  -Ibuild/_deps/json-src/include --std c++20 --parse-tree-only 2>&1 |
  grep "error:" | sed 's/.*error: //' | sort -u
```

Read §13.5 before using a shim for anything other than measurement.

**Tractability scaling (§13.3).**

```sh
cat > scale.cpp <<'EOF'
#include <unordered_map>
#include <string>
#include <cassert>
struct rec { unsigned count; unsigned node; };
int main() {
  std::unordered_map<std::string, rec> m;
  const char *k[4] = {"a","b","c","d"};
  for (int i = 0; i < 4; i++) { rec r; r.count = i; r.node = 0; m[k[i]] = r; }
  for (int i = 0; i < 4; i++) assert(m[k[i]].count == (unsigned)i);
  return 0;
}
EOF
/usr/bin/time -f "WALL=%es" build/src/esbmc/esbmc scale.cpp --unwind 5   # 85.9 s here
/usr/bin/time -f "WALL=%es" build/src/esbmc/esbmc scale.cpp --unwind 8   # > 280 s, timeout
rm -rf /tmp/esbmc-cpp-headers-* /tmp/esbmc-headers-*
```

**Release-build assert census (R1).**

```sh
cat src/goto-symex/*.cpp | grep -c '\bassert('   # 113
cat src/goto-symex/*.h   | grep -c '\bassert('   # 5
python3 -c "import json;d=json.load(open('build/compile_commands.json'));\
print(sum(1 for e in d if '-DNDEBUG' in e['command']),'of',len(d),'TUs with -DNDEBUG')"
```
