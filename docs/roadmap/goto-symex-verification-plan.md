# GOTO-SYMEX Formal Verification Plan

**Subsystem:** `src/goto-symex` — ESBMC's symbolic execution engine: the stage
that turns a GOTO program into an SSA formula (`symex_target_equationt`) for the
SMT backend.
**Verifier:** ESBMC itself (BMC + k-induction) on extracted kernels; Catch2
property/differential tests on the real classes (`unit/goto-symex/`);
whole-tool metamorphic oracles over `regression/`; sanitizers for the rest.
**Status:** **M0–M5 closed** (§15 verdict log); M6–M8 not yet executed. §6.4
records the tier-ordering rule M1 produced. Except where §15 records a
discharged result, every harness below is a *proposal* and nothing here asserts
a proof. Findings not marked discharged in §9.2 remain *hypotheses with cited
evidence*, not confirmed end-to-end bugs; R14 and R15 are pinned by a failing
test or an explicit assertion but neither is fixed.
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
| **I2** | An L2 `name_record` key is *stable* across the `rename(lhs,count)` call inside `make_assignment` — i.e. the callee's `current_names[...]` resolves to the caller's entry, so the index `coveredinbees` stores is the one `entry.count` then publishes. | `make_assignment` → `coveredinbees` | **unenforced** in code; comment "This'll update entry beneath our feet" (R3). Pinned by `unit/goto-symex/renaming.test.cpp` (§15 M1) |
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

### 6.4 Tier B first — a rule learned from M1

The tier table in §2.2 lists A before B, and M0/M1 read that as an ordering.
It is not one. **Tier A proves things about a transcription; only Tier B and
Tier C observe the shipped C++.** A Tier-A harness is therefore justified only
when a property cannot be observed from outside the engine — and §15's M1 entry
shows the cost of ignoring that: the moment H-A1's stub grew from an array index
to the real `name_record` key behind a hash-probed map, it went from 163 VCCs
and 4 s to 4371 VCCs and no verdict inside the plan's own 30 s budget. Fidelity
to the real key and ESBMC-tractability pull in opposite directions, so a Tier-A
harness faithful enough to be worth trusting is often one ESBMC cannot discharge.

The rule for every remaining harness in §7:

1. **Ask what the equation already shows.** Any property expressible over the
   `symex_target_equationt` the engine actually produced — SSA well-formedness,
   phi counts, slicer equisatisfiability, determinism — belongs in
   `unit/goto-symex/`, with zero modelling and zero drift.
2. **Then ask what a whole-tool oracle shows** (§7.4), for composition
   properties over the 1400 `regression/esbmc` CORE inputs.
3. **Only then write Tier A**, for internal decisions the equation does not
   expose (the held-reference hazard of R3, a `previous_frame` precondition, an
   MPOR independence relation), and keep the stub at the smallest fidelity the
   property observes — a Tier-A harness that times out proves nothing at all.

§11.3's acceptance criteria already say "timed out is never SUCCESSFUL". This
section says the same thing one step earlier: prefer the tier that cannot time
out, and cannot drift.

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
| **H-B2** | **Determinism** (P10) | Run symex twice in one process over the same program; compare the two equations step-for-step (kind, `crc` of `cond`, `guard`, `ignore`). **Qualified, §15 M4 (H-B2)**: strict equality holds only for programs creating no dynamic/invalid object; elsewhere the comparison must canonicalise object numbering (R15) | Iteration-order nondeterminism over pointer-keyed containers (`std::set<expr2tc>` in `thread_last_reads/writes`; the `generate_l2_state_hash` comment already concedes cross-run instability) |
| **H-B3** | **Slicer equisatisfiability** (P0, I11) | Build equation; clone; slice the clone; solve both per-claim with the real backend; assert identical per-claim verdicts on ≥ 30 small programs incl. arrays with symbolic indices. **Discharged by H-C1 instead, §15 M5 (H-A4/H-B3)**: 1328 corpus inputs at 0 divergences beats 30 programs; the per-claim residue moves to H-C7 | Slicer unsoundness on real formulas — the honest complement to H-A4 |
| **H-B4** | **Renaming round-trip** (I3/I4) | For each `SSA_stept`, `get_original_name` of `lhs` equals the L0 symbol; `rename` is idempotent; the level never decreases along the step list | `fixup_renamed_type` / `rename_address` regressions |
| **H-B5** | **Phi laws** (I8) | For 2-branch programs: the set of *program variables* receiving a phi == the set written by at least one arm; **zero** phi for untouched variables. **Corrected, §15 M4 (H-B5)**: this row originally said "written *differently* in both", which the code does not do — `phi_function` filters on the L2 index differing, not the value | Over- and under-generation of phi nodes |
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
| **H-C1** | verdict(default) == verdict(`--no-slice`) | `regression/esbmc` CORE (1430 of 1574 dirs) | Slicer unsoundness/incompleteness end-to-end. **Run, §15 M5: 1328 agreed, 0 diverged**, 67 inconclusive, 35 skipped |
| **H-C2** | verdict(default) == verdict(`--no-simplify`) | same | Simplifier / constant-propagation semantic drift (P9). **Run, §15 M5: 1174 agreed, 11 diverged** — R16 (10, incompleteness) and R17 (1, false SUCCESSFUL composed with `--no-slice`) |
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
| **R1** | **High (systemic)** — discharged, §15 M3 | **The shipped binary enforces none of goto-symex's invariants.** There are **113 `assert(...)` in `src/goto-symex/*.cpp` and 5 in the headers**, and **all 674 TUs in this build carry `-DNDEBUG`** (`build/compile_commands.json`, RelWithDebInfo). Every invariant in §4.2 marked "debug only" — including `pop_frame`'s merge-map emptiness (I6) and `coveredinbees`' monotonicity (I1) — is a **no-op in release**. A violation is silent and unbounded. | `grep -c 'assert(' src/goto-symex/*.cpp`; `-DNDEBUG` in all 674 compile commands | H-A1, H-A3 | Introduce a release-checked `SYMEX_INVARIANT(cond, msg)` (CBMC's `INVARIANT` pattern) and promote the ~10 load-bearing asserts (I1, I2, I6, I16) to it. Measure the cost; gate the rest behind an `--expensive-asserts` build option. |
| **R2** | **High (soundness)** — enforced, §15 M3 | `pop_frame` discards `merge_state_map` under a debug-only `assert`. In release, a frame popped with pending merges **silently drops those paths** ⇒ missed bug, no diagnostic. | `goto_symex_statet::pop_frame`, `goto_symex_state.h:310` | H-A3 | Promote to `SYMEX_INVARIANT`; add H-B1-adjacent runtime check counting pushed vs merged snapshots per frame. |
| **R3** | **Medium (soundness) — re-characterised, §15 M1** | `make_assignment` holds `valuet &entry` — a reference **into** `current_names` (`std::unordered_map`) — across the virtual call `rename(lhs_symbol, entry.count + 1)`, which reaches `coveredinbees` and performs `current_names[key]`. This is safe *only because* the recomputed key is identical (the symbol is still L1 at that point, as `make_assignment` sets `symbol.rlevel` only *after* the call), so `operator[]` finds rather than inserts. **The invariant is unasserted.** The originally-hypothesised consequence — rehash ⇒ dangling `entry` ⇒ use-after-free — **does not hold**: [unord.req.general]/9 states rehashing "does not invalidate pointers or references to elements", and only erasing an element invalidates references to it, which `coveredinbees` never does. The real consequence of a re-keying callee is a **correctness** one: `coveredinbees` would bump a different entry, `make_assignment` would then publish the caller's *stale* `entry.count`, and two distinct program values would share an SSA name — an I1/I10 violation, silently unsound. | `renaming::level2t::make_assignment`; `::coveredinbees`; comment "This'll update entry beneath our feet"; [unord.req.general]/9 | `unit/goto-symex/renaming.test.cpp` (Tier B, discharged) | Severity downgraded from memory safety to soundness. The invariant still deserves an assertion — promote to `SYMEX_INVARIANT` with R1 (M3). No restructure needed: re-`find`ing after the call would buy nothing the standard does not already give. |
| **R4** | **Medium (crash → no verdict)** | **Eight unchecked `*ns.lookup(...)` dereferences.** `namespacet::lookup` returns `nullptr` on miss (as `renaming.cpp:15-21` itself demonstrates by checking). A miss ⇒ null deref ⇒ SIGSEGV mid-verification. `phi_function`'s site is the most exposed: it filters only `goto_symex::guard!` and `symex::invalid_object` before looking up an arbitrary merged variable's base name. | `symex_goto.cpp:433`; `symex_function.cpp:159`; `symex_valid_object.cpp:47`; `dynamic_allocation.cpp:66,92,105,118,143` | H-A10 | Add checked lookups with a diagnostic (`log_error` + controlled abort) or prove the precondition per site and record it as a cited comment. |
| **R5** | **Medium (soundness detector disabled)** — discharged, §15 M4 | `check_for_duplicate_assigns` — the *only* in-tree checker for the core SSA invariant I10 — merely `log_status`es duplicates and then reports "Checked N insns". It never fails, and nothing calls it in a normal run. | `symex_target_equationt::check_for_duplicate_assigns`, `symex_target_equation.cpp` | H-B1 | Turn it into a validator returning a bool; run it under a debug/CI flag over the whole regression corpus. |
| **R6** | **Medium (unsound pruning, opt-in flag)** | `state_hashing_level2t::make_assignment` keys `current_hashes` by the **L0** original name, acknowledged in-code ("XXX — consider whether to use l1 names instead. Recursion, reentrancy."). Two states that differ only in the L1 activation of a recursive local therefore fingerprint identically ⇒ `hit_hashes` prunes a genuinely different state ⇒ missed interleaving. Severity is bounded by `--state-hashing` being opt-in. | `execution_state.cpp:~1342-1378`; `reachability_treet::hit_hashes`, `reachability_tree.h:352` | H-A8-style model + **H-C4** | Key by the L1 name record; H-C4 parity sweep quantifies the current gap. |
| **R7** | **Low–Medium (UB) — refined, §15 M1** | `previous_frame()` computes `*(--(--call_stack.end()))` with no size check. `call_stackt` is a `std::vector<framet>`, so at size 1 this evaluates `--begin()`, forming a pointer before the start of the array — undefined by [expr.add]/4 **whether or not it is dereferenced**, not merely a bad read. The second clause of the original finding ("returns a reference a subsequent `pop_frame` invalidates") does **not** hold: `pop_back` invalidates only the reference to the erased last element, and `previous_frame` returns the second-to-last. The precondition holds today by construction — the sole call site does `new_frame(...)` on the preceding line — but nothing states it in the shipped binary (R1). | `goto_symex_statet::previous_frame`; sole caller `goto_symext::symex_function_call_code`; [expr.add]/4 | `unit/goto-symex/frame_lifecycle.test.cpp` (Tier B, discharged) | Add a release-checked precondition **as part of R1's `SYMEX_INVARIANT` work in M3**, so the macro lands once with its cost measured; index (`call_stack[size() - 2]`) rather than decrementing an iterator. |
| **R8** | **Medium (documented model gap)** | `is_valid_object` returns `false` for **every** non-static, non-dynamic symbol: the stack-scope branch is `#if 0`'d out with "XXX re-enable to be able to check for stack-var-out-of-scope problems". Stack-object validity is therefore not modelled, and `dynamic_allocation.cpp` compensates by *assuming* `invalid_pointer` applies only to dynamic objects ("we never update `__ESBMC_alloc` for stack ptrs"). Net effect on stack-lifetime bugs (use-after-scope) is a **missed-bug** direction. | `goto_symext::is_valid_object`, `symex_valid_object.cpp:85-118`; `dynamic_allocation.cpp:110-116` | H-A10 + a targeted `regression/esbmc` use-after-scope corpus | Quantify with a dedicated corpus before attempting a fix; the fix is a model change, not a patch. |
| **R9** | **Low–Medium (approximation direction unproven)** | Three documented "sound over-approximation" claims are unproven: value-set filtering after a pointer havoc (`symex_assign.cpp:~550-570`), the non-scalar uninterpreted-function fallback (`symex_function.cpp:~410-430`), and the function-pointer target enumeration over an over-approximated value set (`symex_function.cpp:~766`). Each *argues* the direction in a comment; none is checked. | cited lines | H-B6 + H-C1/H-C3 | For each, state the claim as a checkable predicate and add a Tier-B assertion (e.g. filtered set ⊆ original **and** the dropped entries are `unknown`/`invalid` only). |
| **R10** | **Low (latent UB)** | `renaming::level2t::name_record`'s `name_record() = default` leaves `lev`, `l1_num`, `t_num` **and the derived `hash`** indeterminate (contrast `level1t::name_record`, which initialises `base_name("")`). No current default-construction site was found, but a future one (`std::optional`, map default-insert, array of records) would read indeterminate memory in `compare`/`hash`. | `renaming.h:143-214` | MSan (Tier D) + a `static_assert`-style unit check | Add default member initialisers; near-zero cost. |
| **R11** | **Open question (concurrency soundness)** | MPOR's independence decision consumes `thread_last_reads`/`thread_last_writes`, populated via `get_expr_globals`, which resolves pointer operands through the *current* value set. If a write through a pointer whose value set is incomplete (or whose entry is `unknown`) is missed, the dependency is missed and an interleaving is dropped — **unsound**. `get_expr_globals` also early-returns entirely under `--data-races-check-only`. | `execution_statet::get_expr_globals`, `check_mpor_dependency`; `reachability_treet::ever_written_globals`/`address_taken_globals` | H-A6 (relation) + **H-C4** (end-to-end) | Determine whether an `unknown` value-set entry forces a conservative dependency; if not, that is a concrete unsoundness to fix. Highest-uncertainty item in this plan. |
| **R13** | **Medium (silent under-verification) — confirmed and fixed, §15 M2 (cont.)** | **`--unwindsetname` never matched a loop.** `unwind_func_set` was keyed by `user_name_to_usr(name)`, which appends a `#` terminator (clang's C++ USR spelling), while `loop_id_to_func_index` was keyed by the goto function-map id, which for a C function is `c:@F@f` with no terminator. The `count(unwind_key)` in `get_unwind` therefore always missed and the global `--unwind` silently won, so a user raising the bound for one function got the lower global bound and a verdict covering less than they asked for. A second defect in the same option: the `name:index:bound` field split scanned left-to-right, so the documented USR form (`c:@F@f#:0:11`) split inside the `c:` prefix. Neither was caught because all five `unwindsetname` regression tests ran without a global `--unwind` and so passed vacuously. | `goto_symext::goto_symext`, `symex_assign.cpp:66-120`; `get_unwind`, `symex_goto.cpp:525`; `user_name_to_usr`, `usr_utils.cpp:29` | `unit/goto-symex/unwind.test.cpp` (Tier B, discharged) | Fixed: both sides now key on the name `--show-loops` prints (`usr_to_user_name`), and the field split scans from the right. Three non-vacuous regression tests added; `unwindsetname_03_priority` corrected to the loop number the program actually has. |
| **R14** | **Open (I10 violated on a real input)** — found by R5's repaired detector, §15 M4 | With `--double-assign-check` made to fail, `regression/esbmc/github_286_3` produces an equation that **defines one SSA name twice**: `…@F@getNumbers2@numbers2?1!0&0#1`, the L2 index 1 of a local array in a function that returns a dangling pointer to it. Two definitions of one name are two constraints `x#1 == e1` and `x#1 == e2` on the same variable; where the right-hand sides disagree the conjunction is unsatisfiable, which silently removes that path from the formula — the missed-bug direction. One input in ~900 swept. Not yet characterised: which two steps emit it, and whether the two right-hand sides can differ. | `symex_target_equationt::check_for_duplicate_assigns` under `--double-assign-check`; `regression/esbmc/double_assign_check_local_array` (KNOWNBUG) | H-B1 | Find the two emitting steps (the local's scope exit is the first suspect), then decide whether the second definition is a stale re-emission or a legitimate step that must take a fresh index. |
| **R15** | **Low (reproducibility, latent collision)** — found by H-B2, §15 M4 (H-B2) | **Object numbering leaks across symex runs in one process.** `execution_statet::dynamic_counter` and `dereferencet::invalid_counter` are `static thread_local` and reset nowhere, so a second exploration in the same process names its objects from where the first stopped: the same program under the same options yields `symex_dynamic::dynamic_1_array` on the first run and `dynamic_2_array` on the second. The sibling `nondet_count` is a plain instance member the constructor zeroes, so the asymmetry is unintended rather than a design choice. The equation is therefore not a function of (program, options) alone. No wrong verdict follows — the names only need to be *fresh*, and monotonic counters are fresh — so this is a reproducibility defect, and objective 7's "byte-identical" wording is unachievable as stated. **Latent second-order risk:** `thread_local` means two threads each start at 0, so if symex is ever parallelised (§14.6) two threads would mint *colliding* object names into a shared context. | `execution_state.cpp:21`, `execution_state.h:583`; `dereference.cpp:23,538`, `dereference.h:281`; contrast `nondet_count` reset at `execution_state.cpp:104` | `unit/goto-symex/determinism.test.cpp` (Tier B, pinned) | Reset both counters per exploration — in `setup_for_new_explore`, **not** in the `execution_statet` constructor, which the reachability tree copies per interleaving and where a reset would mint colliding names. Expect churn in `test.desc` files whose expected output names a dynamic object; run the full corpus before landing. |
| **R16** | **Medium (incompleteness under a non-default flag)** — found by H-C2, §15 M5 (H-C2) | **`--no-simplify` is not verdict-preserving: 10 corpus inputs where the default proves SUCCESSFUL and `--no-simplify` does not.** Nine report a spurious counterexample (`github_1174_{hex,lmod,oct,pass}`, `github_2341_3`, `github_2357_5`, `github_2566_1`, `github_785-2`, `realloc13`) and one returns UNKNOWN (`github_252`, under `--k-induction`). In every case the *default* leg matches the verdict the test's own `test.desc` expects, so the fault is in the `--no-simplify` configuration, not the default. Spot-confirmed on `github_2341_3`: `--no-simplify` reports a violated `assert(temp != NULL)` the default discharges. The noisy direction — P1 — but it means `do_simplify` is load-bearing for *correctness of the encoding*, not merely for formula size, which is not how an "optimisation" flag reads. | `oracle_flag_parity.py --b=--no-simplify` over `regression/esbmc` CORE | **H-C2** | Triage one input down to the expression shape the encoder mishandles unsimplified. Until then `--no-simplify` is a debugging aid, not a semantics-preserving flag. |
| **R17** | **High (false SUCCESSFUL, non-default flag pair)** — found by H-C2, §15 M5 (H-C2) | **`--no-simplify --no-slice` misses a reachable `assert(0)`.** Three lines reproduce it: `void *b = malloc(-4); assert(0);` returns **`VERIFICATION SUCCESSFUL`**. Neither flag alone does this (both give FAILED), nor does a positive size, nor no allocation — so it is a **composition** defect, the class §7.4 says Tier C exists to catch. The negative size widens to a huge `size_t`; one VCC is generated and the solver returns UNSAT, so the path to the assertion is **vacuously infeasible** and every later assertion in such a program is silently unreachable. Reached in the corpus via `github_1631_compact`, whose `--compact-trace` sets `no-slice` implicitly (`command_line_options.cpp:410`) — that indirection is why the pair is easy to hit without naming it. | minimal reproducer above; `regression/esbmc/no_simplify_no_slice_huge_malloc` (KNOWNBUG, observed output `VERIFICATION SUCCESSFUL`) and `..._malloc` (CORE, positive size, passes today) | **H-C2** | Find which allocation-model constraint becomes contradictory unsimplified — the `__ESBMC_alloc_size` update and any size-overflow guard are the first suspects. Pinned, not fixed. |
| **R12** | **Info (bounded by design)** | With `--no-unwinding-assertions`, `loop_bound_exceeded` emits an *assumption* that truncates the path; a `VERIFICATION SUCCESSFUL` then covers only the truncated prefix. This is intended BMC behaviour, but the repo has already been bitten by it in *verification harnesses* (`CLAUDE.md` bans pairing it with reachability checks). | `goto_symext::loop_bound_exceeded`, `symex_goto.cpp:497-523` | H-A5 | No code change; encode as an acceptance criterion (§11.3) so no harness in this plan ever uses that flag. |

---

## 10. Incremental verification roadmap

Ordered by (risk × tractability). Each milestone yields check-in-able artefacts;
no milestone depends on a later one. The ESBMC extension work items (WI-1…WI-6,
§13.6) run **in parallel** with this track: no property claimed in §8 is blocked
on them.

**M0 — Infrastructure (0.5 wk). — DONE, see §15.**
Re-enable `unit/goto-symex/CMakeLists.txt` (fix the `symex;solvers;
gotoalgorithms;pointeranalysis;util_esbmc;langapi` link set). Stand up the
`regression/esbmc/symex_*` harness skeleton + `test.desc` template. Land the
drift-check script. In parallel, start **WI-1** (`<shared_mutex>` operational
model) and **WI-2** (`<type_traits>` completion), and file G1–G8 as issues (D12).
*Artefact:* one green + one deliberately-red smoke harness proving the pipeline
detects an injected bug; a building `unit/goto-symex` target; WI-1 merged.
*Delivered:* the harness triple, the drift guard + its PR gate, and the
`unit/goto-symex` target. WI-1/WI-2 and D12 remain open and move to M1.

**M1 — Low-level kernels: SSA algebra (1 wk).** H-A1, H-A9, H-A7.
Retires I1/I2/I16; produces the R3 and R7 verdicts. In parallel: **WI-3**
(`initializer_list` / `iterator_traits` / `this_thread` / `aligned_storage`),
which completes the parse path to `renaming.h`. *Artefact:* three Tier-A harness
pairs + the R3 restructure PR if H-A9 confirms the hazard; WI-2/WI-3 merged.
**Revised, §15 M1.** H-A1's Tier-A form was built and **rejected on
tractability** — see §15 and the rule it produced in §6.4. I1/I10/P11 are
instead discharged against the *real* `goto_symext` by H-B1
(`unit/goto-symex/ssa_wellformed.test.cpp`), and I1/I2 against the real
`renaming::level2t` by `unit/goto-symex/renaming.test.cpp`, which also produces
the **R3 verdict** (re-characterised: soundness, not memory safety). H-A7's
I16/R7 obligation is discharged by `unit/goto-symex/frame_lifecycle.test.cpp`
plus a call-site argument. **M1 closed.**

**M2 — Isolated core algorithms: merging and bounding (1.5 wk).**
H-A2 (the highest-value harness), H-A3, H-A5. Dual-solver mandatory.
*Artefact:* merge-soundness proof at arity 2 **and** 3; R2 `SYMEX_INVARIANT` PR.
**Revised, §15 M2.** H-A2's and H-A3's obligations (I8, I6) are observable in the
produced equation, so both are discharged at Tier B by
`unit/goto-symex/merge.test.cpp` per §6.4 — no transcription, no dual-solver gate
needed because no SMT query is involved. H-A5 is likewise Tier B
(`unit/goto-symex/unwind.test.cpp`) and found R13. **M2 closed**; R2's
`SYMEX_INVARIANT` remains M3.

**M3 — Release-mode enforcement (0.5 wk).** R1: introduce `SYMEX_INVARIANT`,
promote the ~10 load-bearing asserts, measure the runtime cost on
`regression/esbmc` (accept if < 2 %). *Artefact:* invariant macro + benchmark
note. This is a prerequisite for M1/M2's proofs to mean anything in the shipped
binary. **Done, §15 M3** — 7 promotions covering I1, I2, I6 and R7, with
`unit/goto-symex/invariant.test.cpp` as the evidence that each fires under
`NDEBUG`; cost below measurement resolution. **M3 closed.**

**M4 — Individual symex operations on the real engine (1.5 wk).**
H-B1 (SSA validator — build it first, reuse everywhere), H-B4, H-B5, H-B2.
Run the **WI-4** Tier-B′ pilot here (include the real `renaming.h`, drive the
real `level1t`) — gated on parse **and** < 60 s verification; a negative result
is recorded in §13.3 and Tier A is kept.
*Artefact:* `unit/goto-symex/{ssa_wellformed,renaming,phi,determinism}.test.cpp`;
R5 promoted to a real validator; WI-4 verdict.
**Revised, §15. M4 closed.** H-B1 (#6487), H-B4 (#6491), H-B5 and H-B2 all
discharged; H-B5's cases live in `merge.test.cpp`, which already owned I8, so
there is no `phi.test.cpp`. H-B2 found R15 and refuted objective 7's
"byte-identical" wording. **WI-4 was not run** and carries to M5 with D14.

**M5 — Constraint generation: the slicer (1 wk).** H-A4, H-B3, then the
**H-C1** sweep over all 1430 `regression/esbmc` CORE tests. *Artefact:* slicer
equisatisfiability suite + the first whole-corpus parity report.
**Closed, §15 M5.** H-C1 (1328 agreed, 0 diverged) and H-C2 (1174 agreed, **11
diverged** → R16, R17) via `oracle_flag_parity.py`, which also covers H-C3/H-C5
by argument; H-A4's A4.2 at Tier B in `unit/goto-symex/slice.test.cpp`; H-B3's
equisatisfiability via H-C1, with the per-claim residue passed to H-C7. H-A4 and H-B3 remain;
per §6.4, H-A4's obligations should be discharged at Tier B rather than
transcribed. The scheduled CI job (§11.2) is not yet wired.

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
docs/roadmap/goto-symex-verification-plan.md  this document (+ verdict log, §15)
regression/esbmc/symex_<area>_<nn>/           Tier A, passing — owns the kernel
    ├── symex_<area>_<nn>.c
    └── test.desc
regression/esbmc/symex_<area>_<nn>_fail/      Tier A, anti-vacuity twin (§6.1 r5)
regression/esbmc/symex_<area>_<nn>_probe/     Tier A, reachability probe (§6.1 r6)
unit/goto-symex/<area>.test.cpp               Tier B — prefer this, see §6.4
scripts/verification/symex/
    ├── oracle_flag_parity.py                 H-C1, H-C2, H-C3, H-C5
    ├── oracle_por_parity.sh                  H-C4
    ├── oracle_unwind_monotonic.sh            H-C6
    └── drift_check.py                        transcription-drift guard
.github/workflows/symex-oracles.yml           scheduled Tier-C job
```

`<area>` ∈ `{ssa, merge, mergequeue, slice, unwind, mpor, frame, eqctx,
lookup, refalias}` — one area per harness family, matching §7. `<nn> = 00` is
the M0 template.

H-C1, H-C2, H-C3 and H-C5 are the *same* relation — verdict(A) == verdict(B)
over a corpus — so §15 M5 replaced the four planned shell scripts with one
parameterised `oracle_flag_parity.py`, invoked as `--b=--no-slice`,
`--b=--no-simplify`, `--a=--bitwuzla --b=--z3` and
`--b=--no-interval-symex-guard`. It builds each argument list through
`regression/testing_tool.py`'s `TestCase` rather than re-parsing `test.desc`, so
a sweep invokes each input exactly as `ctest` does. H-C4 and H-C6 stay separate:
one compares three configurations, the other is monotonicity across bounds.

The `_fail` and `_probe` directories hold a one-line
`#include "../symex_<area>_<nn>/symex_<area>_<nn>.c"` and select their variant
with `-DSYMEX_HARNESS_PERTURB` / `-DSYMEX_HARNESS_PROBE` on `test.desc` line 3.
The kernel therefore has exactly one copy: a twin cannot silently stop
perturbing the code it is meant to perturb.

Every harness citing `src/goto-symex` carries, in its header comment, one
`SYMEX-HARNESS-TARGET: <path>::<symbol>` line per transcribed symbol and a
`SYMEX-HARNESS-SHA256:` line holding the checksum of that symbol's definition.
`drift_check.py` re-extracts and re-hashes each; `--update` refreshes them after
a reviewed re-transcription.

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
| **D11** | Verdict log — per-harness result, ESBMC commit, solver versions, date — appended to this document | §15 | continuous |
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

## 15. Verdict log (D11)

Append-only. One row per discharged (or attempted) harness run, with the exact
artefact it was run against. A row here is the *only* place this document claims
a result; §7's harness descriptions remain proposals until they appear below.

**Environment.** ESBMC 8.4.0, tree at `ecf26b5312`, `RelWithDebInfo` (`-DNDEBUG`
present — see R1), Bitwuzla 0.9.0, Z3 4.13.3, Linux x86_64.

### M0 — 2026-07-27

| Artefact | Invocation | Verdict | Acceptance (§11.3) |
|---|---|---|---|
| `regression/esbmc/symex_ssa_00` | `--overflow-check --unsigned-overflow-check --memory-leak-check` | `VERIFICATION SUCCESSFUL` | 1 ✓ · 2 ✓ (45 of 163 VCCs remain) · 3 ✓ · 4 ✓ · 5 ✓ · 6 ✓ (Bitwuzla and Z3 agree) · 7 ✓ |
| `regression/esbmc/symex_ssa_00_fail` | same + `-DSYMEX_HARNESS_PERTURB` | `VERIFICATION FAILED` at the `I1: L2 index advances by exactly one` claim | anti-vacuity twin for the row above |
| `regression/esbmc/symex_ssa_00_probe` | same + `-DSYMEX_HARNESS_PROBE` | `VERIFICATION FAILED` at `reachability probe` | the harness body is reached |

**Scope of the claim.** `symex_ssa_00` is the M0 *template*, not H-A1. It proves
I1 for the counter algebra alone, over a 4-key map and a 4-assignment sequence,
with the map modelled as a direct-indexed array. It does **not** yet model the
`name_record` key, the `rename`-beneath-our-feet reference hazard (I2/R3), or
node ids. H-A1 (M1) subsumes it.

**Infrastructure delivered.**

- `unit/goto-symex` builds and runs again (`intrinsic-utils-test`, 4 assertions).
  The link failure was cvc5's exported target naming `cadical`/`picpoly`/
  `picpolyxx` as bare `-l` flags with no search path: `src/esbmc/CMakeLists.txt`
  carried a private `target_link_directories` workaround, so `esbmc` linked and
  every other consumer of `solvers` did not. Moved onto `solvercvc5` as a
  `PUBLIC` link directory, which fixes all consumers at once.
- `scripts/verification/symex/drift_check.py` + a `pull_request.yml` gate.
  Verified end-to-end: inserting one comment into `level2t::make_assignment`
  turns the check red.

**Carried into M1.** WI-1 (`<shared_mutex>`), WI-2 (`<type_traits>` completion)
and D12 (issues G1–G8) were scheduled for M0 and are not started.

### M1 — 2026-07-27

**Result: I1, I10 and P11 discharged against the real engine.** Not by a
transcription — by running the real `goto_symext` and validating the
`symex_target_equationt` it produced.

| Artefact | What it drives | Result |
|---|---|---|
| `unit/goto-symex/ssa_wellformed.test.cpp` (H-B1) | real `goto_factory` → real `reachability_treet` → real `goto_symext`, over straight-line, branching, nested-branch, unwound-loop, and call/recursion programs | 6 cases, 37 assertions, **pass**, 1.6 s |

Properties checked on every produced equation:

- **I1 / P8** — per `(base_name, l1_num, thread_num)`, the L2 index of each
  definition strictly increases in equation order.
- **I10 / P11** — no two assignment steps define the same SSA name. This is the
  invariant `check_for_duplicate_assigns` exists for and never enforces (R5);
  the validator is the enforcing version R5 asks for.
- **P11** — no step reads an SSA name before the step defining it. A name never
  defined in the equation is a free symbol and is correctly not a violation.

Anti-vacuity: the last case takes an equation the engine produced, appends a
copy of one of its own assignment steps, and requires the validator to report
exactly one duplicate definition and one non-monotonic index. Without it the
five passing cases would be indistinguishable from a validator that checks
nothing.

**One harness-side defect found and fixed, no engine defect.** The first run
reported use-before-def on every first definition. Cause: for an assignment
step, `symex_target_equationt::assignment` sets
`SSA_step.cond = equality2tc(lhs, rhs)`, so reading `cond` on an assignment
reports the definition as a use of itself. The reads of an assignment are in
its `rhs`; `cond` is for assume/assert steps. Recorded because it is the exact
failure mode a Tier-A transcription would have hidden — a stub has no `cond`
field to get wrong.

**Rejected: H-A1 as a Tier-A harness.** Built as specified in §7.1 — the full
`name_record` key `(base_name, lev, l1_num, t_num)`, `hash` modelled as an
arbitrary function of those fields so collisions stay possible, and a
linear-probed map, at 3 assignments over a 16-key space. Measured:

| Harness | VCCs | Wall | Verdict |
|---|---|---|---|
| `symex_ssa_00` (M0 template, array-indexed key) | 163 | 4 s | `SUCCESSFUL` |
| H-A1 Tier-A (real `name_record` key, hash-probed map) | 4371 | **> 200 s** | none |
| H-A1 `_fail` twin | 4368 | 108 s | `FAILED` |

Rejected under §11.3 criterion 1 (no verdict) and the §11.2 budget (< 30 s per
harness; the regression harness cap is a hard 120 s). Not committed. The green
harness never returned, so it proves nothing — while the `_fail` twin returning
`FAILED` in half the time is the shape you would expect: finding one violating
trace is easy, proving none exists is not.

The general lesson is written up as §6.4: raising a Tier-A stub's fidelity
toward the real data structure raises its cost superlinearly, so the tier that
verifies the shipped C++ is also the tier that scales. §6.4 reorders the
remaining harness work accordingly.

### M1 (cont.) — R3 verdict

**Result: I1 and I2 discharged on the real `renaming::level2t`; R3
re-characterised from memory safety to soundness.**

| Artefact | What it drives | Result |
|---|---|---|
| `unit/goto-symex/renaming.test.cpp` | the `renaming::level2t` owned by a real `execution_statet`, its real `current_names`, and the real `make_assignment` → `rename` → `coveredinbees` chain | 4 cases, 28 assertions, **pass** |

Only the input symbols are constructed; the class under test is the shipped one.

- **I1** — five successive `make_assignment` calls on one key publish 1…5, and
  `current_names.at(key).count` equals each published index.
- **I2** — a first assignment to a fresh key grows `current_names` by exactly
  one entry, and later assignments to it by none. A callee that recomputed a
  *different* key would default-insert a second entry through the nested
  `current_names[...]`, so this is I2's direct observable.
- **non-aliasing** — keys differing in exactly one of `l1_num`, `t_num` or
  `lev` keep independent counters.
- **R3's memory-safety claim** — a `valuet *` taken into `current_names` is held
  across 256 further insertions that provably rehash the table
  (`bucket_count()` grows), then dereferenced.

**R3 as written was wrong about the failure mode.** It hypothesised
"insert ⇒ possible rehash ⇒ dangling `entry`, then `entry.count = …` is a
use-after-free". `current_names` is a `std::unordered_map`, and
[unord.req.general]/9 says rehashing "invalidates iterators, changes ordering
between elements, and changes which buckets elements appear in, **but does not
invalidate pointers or references to elements**"; only erasing an element
invalidates references to it, and `coveredinbees` never erases. The test above
confirms this empirically for the real container. There is no use-after-free
here and no restructure to do.

What survives is a **soundness** hazard, and it is the one worth asserting: if
the key recomputed inside `coveredinbees` ever differed from the caller's, the
callee would bump a *different* entry, `make_assignment` would then publish the
caller's stale `entry.count`, and two distinct program values would share an SSA
name — an I1/I10 violation with no diagnostic. Today the key is stable only
because `make_assignment` sets `symbol.rlevel` *after* the `rename` call.
Nothing enforces that ordering. §9.2's R3 row is updated; the action moves from
"restructure to avoid the held reference" to "promote I2 to `SYMEX_INVARIANT`"
alongside R1 in M3.

**A note on the Tier-A form of H-A9.** §7.1 specifies its stub as "the `map_t`
stub with an explicit rehash-on-insert that invalidates outstanding references
(modelled as a generation counter)". That stub models a container `std::
unordered_map` is not: it would have "proved" a hazard the standard rules out,
and its `_fail` twin would have passed the §11.3 gate while demonstrating
nothing real. A second, sharper instance of §6.4 — a stub encodes the author's
belief about the real type, and that belief is exactly what needed checking.

### M1 (cont.) — R7 verdict, M1 closed

**Result: I16 discharged by construction plus a Tier-B frame-balance pin; R7
survives as a latent hazard with a corrected severity basis.**

| Artefact | What it drives | Result |
|---|---|---|
| `unit/goto-symex/frame_lifecycle.test.cpp` | real symex over nested calls, recursion, calls through a function pointer, and calls inside an unwound loop | 5 cases, 17 assertions, **pass** |

**The precondition holds, by construction, at the only call site.**
`previous_frame()` has exactly one caller —
`goto_symext::symex_function_call_code` (`symex_function.cpp`) — and it reads:

```cpp
assert(!cur_state->call_stack.empty());
goto_symex_statet::framet &frame = cur_state->new_frame(...);
frame.level1 = cur_state->previous_frame().level1;
```

`new_frame` pushes unconditionally on the line before, so `size() >= 2` at the
call. The engine starts at depth 1 (`goto_symex_statet::initialize` →
`new_frame`), which the first test pins.

**The severity basis in R7 was understated.** `call_stackt` is
`std::vector<framet>`, so `*(--(--call_stack.end()))` at size 1 evaluates
`--begin()` — forming a pointer before the start of the array, which is
undefined by [expr.add]/4 regardless of whether it is dereferenced. The
"returns a reference a subsequent `pop_frame` invalidates" clause does **not**
apply: `pop_back` invalidates only the reference to the erased last element,
and `previous_frame` returns the second-to-last.

**What the tests pin.** The engine's residual call-stack depth after symex is a
constant of the entry sequence (`__ESBMC_main`'s frame plus `main`'s, whose
`END_FUNCTION` is the last instruction) and **not** a function of call nesting.
The discriminating case runs the same recursive program at depth 1 and depth 3
and requires equal residual depth: if `pop_frame` did not match `new_frame`, the
deeper program would end with more frames standing, and the size-1 precondition
would stop being a structural property. Recursion is additionally required to
produce more than one L1 activation of the callee's local — which is what
`previous_frame().level1` exists to seed.

**A held-`framet&` audit, since `call_stack` is a vector.** `emplace_back` can
reallocate and invalidate every outstanding reference into it, so every
`framet &` in `src/goto-symex` was checked against the one push site
(`symex_function_call_code`). Six sites hold one: `symex_function.cpp:599, 908,
955, 1008`, `symex_other.cpp:79`, `symex_goto.cpp:328`. None is used after a
call that can push a frame. The closest is `run_next_function_ptr_target`, where
`cur_frame` is live across `symex_function_call_code(state_call)` but has no use
after it. **No live defect; the pattern is fragile rather than broken**, and is
the kind of thing a `SYMEX_INVARIANT` on frame identity would pin in M3.

**Action for R7 unchanged in substance, sharpened in wording:** the fix belongs
with R1/M3 — a release-checked precondition. It is deliberately not applied here
so that `SYMEX_INVARIANT` lands once, in one place, with its cost measured.

**M1 is closed.** Its three harnesses are discharged at Tier B — H-A1's property
by `ssa_wellformed.test.cpp`, H-A9/R3 by `renaming.test.cpp`, H-A7/R7 by
`frame_lifecycle.test.cpp` — with the Tier-A forms of H-A1 and H-A9 rejected on
tractability and on modelling fidelity respectively (§6.4). Still open from M0:
WI-1, WI-2, D12. Next is **M2** (H-A2 merge-guard soundness, H-A3 merge-queue
conservation, H-A5), which §6.4 requires be scoped Tier-B-first: `phi_function`
and `merge_state_guards` are observable in the produced equation through phi
assignment counts and step guards.
### M2 (partial) — 2026-07-27

**Result: I8 and I6 pinned against the real engine; R2's failure mode confirmed
as unobserved on every shape tried.**

| Artefact | What it drives | Result |
|---|---|---|
| `unit/goto-symex/merge.test.cpp` (H-A2 + H-A3) | real symex over two-armed, one-armed, nested, in-callee, early-return and in-loop branches | 7 cases, 31 assertions, **pass** |

§4.3 ranks the merge machinery P0 — a lost path is a missed bug with no
diagnostic — and both of its unenforced invariants turned out to be visible in
the produced equation, so neither needed a transcription (§6.4).

- **I8, emission** — a two-armed branch over nondet values produces exactly one
  `ite` definition for the merged variable; a variable untouched in both arms
  produces none; nested branches produce one per join; a branch in a 3×-unwound
  loop produces at least one per iteration.
- **I8, freshness** — the phi's L2 index is strictly greater than every index
  previously defined for that key. A phi that reused one of its own inputs would
  alias two distinct values under one SSA name, which is the I1/I10 violation
  R3's stale-count scenario also produces.
- **I8, one-armed merge** — for `if (c) x = …;` with no `else`, the emitted ite's
  arms are distinct. This is the lost-behaviour direction: if the *pre-branch*
  value were not one of the arms, the not-taken path would simply vanish from
  the formula.
- **I6 / R2** — after symex, every live frame's `merge_state_map` is empty. The
  cases are chosen for where a snapshot could be orphaned: a branch inside a
  callee (so the join and the `pop_frame` belong to the same frame), an early
  `return` that jumps past a join, and a branch inside an unwound loop.

Non-vacuity: the pending-merge count `REQUIRE`s a non-empty call stack before
summing, so a zero cannot come from having examined nothing. Arm values are
`nondet_int()` throughout — with constant arms, `simplify` folds the ite and the
phi disappears, which would have made every emission assertion vacuous.

**R2 is not retired by this.** These tests show the invariant holding on the
shapes tried; R2 is that *nothing enforces it in the shipped binary*, since
`pop_frame`'s `assert(merge_state_map.size() == 0)` is a no-op under NDEBUG.
That remains true and remains M3's `SYMEX_INVARIANT` work. What the tests add is
a durable regression: if a future change starts orphaning snapshots on any of
these six shapes, this fails rather than silently dropping paths.

**Still open in M2.** H-A5 (unwind bounding, `get_unwind` /
`loop_bound_exceeded`). Also still open from M0: WI-1, WI-2, D12.

### M2 (cont.) — 2026-07-28, M2 closed

**Result: H-A5 discharged at Tier B — and it found a live bug (R13).**

| Artefact | What it drives | Result |
|---|---|---|
| `unit/goto-symex/unwind.test.cpp` (H-A5) | real symex under `--unwind`, `--unwindsetname`, `--unwindset`, `--no-unwinding-assertions`, `--partial-loops` | 8 cases, 39 assertions, **pass** |

Both targets are observable from outside the engine, so §6.4 puts them at Tier B:
`get_unwind`'s decision is the *number of loop-body assignments* in the equation,
and `loop_bound_exceeded`'s three arms are which step it appends. No case makes a
reachability claim, so R12 and §11.3 are respected by construction.

- **A5.1, precedence** — the three bounds are given distinct values (global 5,
  function-specific 3, loop-specific 2) so that a precedence swap lands on a
  different iteration count rather than an equal one. The loop id passed to
  `--unwindset` is read out of the built GOTO program, not assumed.
- **A5.2, `0` means unbounded** — a loop-specific `0` exempts one loop from a
  global `--unwind 2`, and the loop then runs to its natural exhaustion. Its
  non-vacuity twin drops only the exemption and gets truncation back.
- **A5.3, exactly one arm** — `--no-unwinding-assertions` must trade the claim
  for an assumption *together*: the unwinding-assertion count goes 1 → 0 while
  the assume count goes n → n+1, at an unchanged iteration count.
  `--partial-loops` moves neither.
- **A5.4, guard strengthening** — with a condition still true at the bound, the
  `¬cond` that `loop_bound_exceeded` adds is `false`, so the code after the loop
  becomes unreachable and is not emitted. Under `--partial-loops` — the one arm
  that skips the `guard.add` — the same post-loop assignment *is* emitted. That
  difference is the guard strengthening, observed rather than modelled.

**R13, found by writing A5.1.** The function-specific arm of `get_unwind`'s
precedence chain was dead: `unwind_func_set` was keyed by
`user_name_to_usr(name)`, which appends clang's C++ USR `#` terminator, and
`loop_id_to_func_index` by the goto function-map id, which for a C function has
none. Every `--unwindsetname` lookup missed and the global `--unwind` silently
won. The five existing `unwindsetname` regression tests all passed vacuously:
none set a global `--unwind`, so all of them verify identically whether the
option works or is ignored. Both sides now key on the name `--show-loops`
prints, and the `name:index:bound` split scans from the right so a USR name's
own `c:` prefix no longer splits the field.

This is the plan's premise landing for the first time on a shipped-binary
defect: the harness was written to state A5.1, not to hunt for a bug, and the
bug fell out of stating it.

**M2 is closed.** H-A2/H-A3 (§15 M2) and H-A5 are discharged at Tier B, all
three without a transcription. Still open from M0: WI-1, WI-2, D12. Next is
**M3** (R1: `SYMEX_INVARIANT`), which R2 and R7 both wait on.

### M3 — 2026-07-28, M3 closed

**Result: R1 discharged. goto-symex's load-bearing invariants are now checked in
the shipped binary, at no measurable cost.**

| Artefact | What it drives | Result |
|---|---|---|
| `src/goto-symex/symex_invariant.{h,cpp}` | `SYMEX_INVARIANT(cond, reason)`: located diagnostic + `abort()` | 7 call sites |
| `unit/goto-symex/invariant.test.cpp` | each promoted invariant violated on the real engine state | 5 cases, 33 assertions, **pass** |

M1 and M2 proved things about invariants the shipped binary did not check — R1
is what makes those proofs mean anything at runtime. Promoted:

| Invariant | Site | Was |
|---|---|---|
| I1 (L2) | `coveredinbees`, counter monotonicity + the L1-key precondition | `assert`, the precondition additionally behind `#ifndef NDEBUG` |
| I1 (L1) | `level1t::rename`, activation-index monotonicity | `assert` on a `[[maybe_unused]]` read |
| I2 | `make_assignment`, the key held across the virtual `rename` | **unchecked** |
| I6 / R2 | `pop_frame`, no unmerged snapshots | `assert` |
| R7 | `previous_frame`, a caller frame exists | **unchecked**, and the body formed `begin() - 1` |
| — | `top()` ×2, non-empty call stack | `assert` |

**I2 had no check at all.** R3 (§15 M1) established that the hazard is not a
dangling reference but a *stale count*: if the callee re-keyed, `make_assignment`
would publish this entry's old index and two program values would share an SSA
name. The cheapest witness that the key held is the count itself — the caller
now records `expected_count` before the call and requires the entry to carry it
after, an integer compare that catches exactly the failure R3 describes.

**R7's UB is gone, not just diagnosed.** `previous_frame` now indexes
`call_stack[size() - 2]` behind a size precondition, so the size-1 case is a
located abort rather than a pointer formed before the start of a vector
([expr.add]/4).

**Evidence the promotion is real.** A death test that merely observes `SIGABRT`
cannot distinguish a live `SYMEX_INVARIANT` from a libc `assert` in a build that
forgot `-DNDEBUG`. Each case therefore captures the child's stderr and requires
*our* wording, so the test proves release-mode enforcement rather than assuming
it. A fifth case runs a well-formed symex and requires that none of the seven
fires — without it, an invariant that aborted unconditionally would pass all
four death tests.

**Cost: below measurement resolution.** Benchmark: 8 000 calls and ~40 k L2
assignments (400 × 20 nested calls, `--unwind 512 --show-vcc`), which excludes
solver time so the symex fraction — and hence the invariant fraction — is at its
maximum. Six interleaved A/B pairs on the same machine: baseline mean 12.55 s
(min 12.34), promoted mean 12.45 s (min 12.34). The promoted binary measures
*faster* on the mean, i.e. the difference is inside run-to-run noise (±2 %).
Well under §10's 2 % bar, and that bar is measured here against an upper bound.

**Not promoted.** The other ~111 `assert`s in `src/goto-symex` stay as they are:
§10 asks for the load-bearing ones, and the macro's contract (see its header) is
that a check costs a comparison or less. Anything walking a container or
rebuilding a key belongs behind `assert` or a debug flag.

**M3 is closed.** R1 is discharged, and R2, R7 and I2 — all three deferred here
from M1/M2 — are now enforced. Still open from M0: WI-1, WI-2, D12. Next is
**M4** (H-B1 the SSA validator first, then H-B4, H-B5, H-B2).

### M4 (partial) — 2026-07-28

**Result: R5 discharged — and the repaired detector immediately found an I10
violation on a real input (R14).**

| Artefact | What it drives | Result |
|---|---|---|
| `symex_target_equationt::check_for_duplicate_assigns` | I10 over the produced equation, opt-in via `--double-assign-check` | now returns a verdict; fails the run |
| `regression/esbmc/double_assign_check_clean` | the check on a well-formed equation | **CORE, passes** |
| `regression/esbmc/double_assign_check_local_array` | the equation that violates I10 | **KNOWNBUG** |

R5 recorded that the only in-tree checker for I10 "merely `log_status`es
duplicates and then reports Checked N insns. It never fails." That is now a
`bool`, each offender is an error rather than a status line, and the
`--double-assign-check` call site turns a false into a `SYMEX_INVARIANT` stop —
the M3 macro doing what it was introduced for. The body also reads `step.lhs`
rather than destructuring `to_equality2t(step.cond).side_1`, which assumed a
shape the step already exposes directly.

**A detector that cannot fail teaches you nothing about the corpus.** Sweeping
~900 `regression/esbmc` inputs with the check enabled — each with its own
`test.desc` flags, 10 s cap — produced exactly one violation: `github_286_3`,
recorded as R14. The remaining ~899 are the first evidence in this plan that
I10 holds broadly rather than merely on the shapes M1's `ssa_wellformed` tests
construct.

**R14 is pinned, not fixed.** `double_assign_check_local_array` is `KNOWNBUG`:
it states the verdict the input should produce and does not today. Diagnosing
which two steps emit the duplicate is H-B1 work and is deliberately not
attempted here — the value of this entry is that the violation is now visible
and cannot regress unnoticed.

### M4 (H-B1) — 2026-07-28
### M4 (closed) — 2026-07-28

**Result: H-B1 closed. The validator is shared, and the corpus sweep was made
to answer a question rather than only to collect verdicts.**

| Artefact | What it drives | Result |
|---|---|---|
| `unit/goto-symex/ssa_validator.h` | I1 / I10 / P11 over any equation | shared header, `symex_ssa::require_well_formed` |
| `merge.test.cpp`, `frame_lifecycle.test.cpp` | the same checks on the equations they already build | asserted in `engine::run()` |
| `ssa_wellformed.test.cpp` "the shipped I10 detector reports a duplicate" | `check_for_duplicate_assigns` itself | pins the positive path |

The last row matters because the detector was otherwise pinned only by
`double_assign_check_local_array`, which is KNOWNBUG — it passes when the
output fails to match *for any reason* — over an input R14 is slated to fix.
Duplicating a definition in an equation the real engine produced and requiring
`check_for_duplicate_assigns()` to return false keeps that coverage after R14
is closed.

§7.2 wanted H-B1 reusable as an assertion inside every other Tier-B test rather
than private to the test that introduced it. It now is: the checks moved to
`ssa_validator.h`, and the two other tests that drive the real engine call
`require_well_formed` from their `run()`, so every equation they build is an
additional I1/I10/P11 sample for free. Assertion counts: `merge` 31 → 59,
`frame_lifecycle` → 41, `ssa_wellformed` 37 → 42. `merge.test.cpp`'s private
copy of `is_ssa_symbol` is gone.

**The `is_symbol2t` guard was dead, and the sweep is what proved it.** The
repaired `check_for_duplicate_assigns` initially *skipped* an assignment step
whose lhs was not a symbol. Instrumenting that arm and re-running all **1547**
`regression/esbmc` inputs produced **zero** hits: no ASSIGNMENT step in the
corpus carries a non-symbol lhs — which is what `assignment()` documents
("lhs must be a symbol") and only partly checks (`assert(!is_nil_expr(lhs))`).
Per the Mode C C-Live obligation an unreachable new branch is dead
instrumentation and must be removed, so the arm is now a `SYMEX_INVARIANT`
stating the precondition. A *silent skip inside a validator* is R5's defect one
level down: it quietly narrows the check the function exists to perform.

Both checks ran in the same pass, so the same 1547 inputs re-confirm R14 as the
sole I10 violation in the corpus (`github_286_3`, 1 of 1547). The "~900" figure
in the M4 (partial) entry above counted only inputs that reached a verdict
inside the 10 s cap; 1547 is the number actually swept, and is the figure to
quote.

**Still open in M4.** H-B4, H-B5, H-B2. R14 remains pinned, not diagnosed.
Also still open from M0: WI-1, WI-2, D12.

*(This entry was originally headed "M4 (closed)". It was not — only H-B1 was.
The heading is corrected above; M4 closes when H-B4, H-B5 and H-B2 land.)*

### M4 (H-B4) — 2026-07-28

**Result: I3 and I4 discharged on the real `renaming::level2t`. No defect
found — and the mutation testing is the reason that statement is worth
anything.**

I4 was listed **unenforced** in §4.2 and C1 (renaming/SSA) is P0 in §4.3
("silent aliasing of distinct values"), so this is the highest-ranked component
in the plan whose round-trip property had no test at all. Five cases added to
`unit/goto-symex/renaming.test.cpp` (which already owned I1/I2/R3, so the
subject keeps one file): 4 → 9 cases, 28 → 520 assertions.

| Case | Property |
|---|---|
| `rename is idempotent on an L2 symbol` | I3, early-return path |
| `rename of an L1 symbol reaches a fixed point` | I3, through the real `current_names` lookup |
| `get_original_name inverts rename` | I4, for `level1` **and** `level1_global` |
| `stripping to L0 keeps name and type` | I4, fields zeroed / name and type preserved |
| `every equation definition strips cleanly to L0` | I4 + level monotonicity over a real equation |

**Two of these were vacuous when first written, and only mutation testing said
so.** Deliberately breaking `get_original_name` (dropping the `level1_num`
reset) left the whole-equation sweep **green**, for two independent reasons:

1. The test program had no callee local with a non-zero L1 activation, so
   `level1_num == 0` held for every definition regardless of the bug. Fixed by
   giving the program repeated and recursive calls, plus an explicit
   `REQUIRE(with_activation > 0)` so the weaker vacuity cannot return.
2. The sweep asserted `rlevel`, `thename`, `type` and idempotence but never
   asserted the *zeroed fields* — the very thing the mutation changed.

Both are now fixed and the mutation is caught by both cases (39 failing
assertions). A second mutation — removing `level2t::rename`'s early return for
already-L2 symbols — is caught by both I3 cases.

**Generalising: a Tier-B sweep over "every step in a real equation" feels
strong and is easy to make vacuous.** Its coverage is the product of what the
loop asserts and what shapes the input program actually produces, and neither
is visible from reading the test. The M2 note already recorded the input half
(constant phi arms get folded away); this is the assertion half. Recommend
every whole-equation sweep in this plan carry (a) a counter proving the shape
it targets was present, and (b) a recorded mutation that it catches.

**Still open in M4.** H-B5, H-B2. R14 remains pinned, not diagnosed. Also
still open from M0: WI-1, WI-2, D12.

### M4 (H-B5) — 2026-07-30

**Result: I8's counting law discharged on the real `phi_function`. No defect
found; §7.2's statement of the law was wrong and is corrected there.**

Five cases added to `unit/goto-symex/merge.test.cpp`, which already owned I8's
freshness direction (H-A2), so per the M4 (H-B4) precedent the subject keeps one
file — no `phi.test.cpp`, contrary to §10's artefact list. 7 → 12 cases,
59 → 97 assertions.

| Case | Property |
|---|---|
| `the phi set is exactly the variables an arm wrote` | set equality over `{both, then_only, else_only}` with `untouched` absent |
| `the same value in both arms still gets a phi` | the index-keyed filter, *not* value-keyed |
| `a straight-line program gets no phi at all` | no join ⇒ no phi |
| `a variable declared inside an arm gets no phi` | the `merge_variables.find(...) == end()` arm |
| `an unwound loop merges only what it writes` | untouched stays 0 across 4 unwindings and a callee |

The set-equality form is what makes this stronger than the per-variable cases
above it: over-generation and under-generation are caught by one assertion, and
a phi for a variable the test never named cannot slip through.

**§7.2's law did not match the code.** It required a phi for variables "written
*differently* in both" arms. `phi_function` compares
`merge_state.level2.current_number(variable)` against the current state's — an
L2 *index* comparison — so `if (c) same = v; else same = v;` still emits an
ite-shaped phi over two distinct SSA names that happen to hold equal values,
which `simplify` cannot see through. Benign (a redundant ite the solver
discharges), but a test written to the original wording would have asserted the
absence of a phi that is really there, and failed for the wrong reason.

**Two discarded discriminators, both of which produced green-but-wrong tests
before being caught.** Identifying a phi step is the whole difficulty here:

1. *rhs is an ite* — what the pre-existing `phis_for` helper used. A phi is
   ite-shaped only when neither incoming guard is false, so the first of an
   if/else's **two** joins yields a symbol-shaped phi that this misses. The
   helper is now defined in terms of the phi predicate and keeps the ite filter
   only where the test wants the two-live-values case.
2. *hidden + unguarded* — correct for phis, but symex's own bookkeeping
   (`__ESBMC_alloc`, `$tmp::return_value$_*`) is written the same way, so a
   straight-line program appeared to have eleven "phis". `original_rhs` looked
   like the fix and is nil on every step in this configuration, discriminating
   nothing. The predicate now also requires the `@F@<func>@<var>` program-
   variable shape, and the law is stated over program variables only.

An if/else lowers to two joins, so a variable written in the then-arm receives
two phis; the loop case shows 12 for one variable at `--unwind 4`. Those counts
are properties of GOTO lowering rather than of the merge, so the cases assert
the phi *set* and a `> 1` lower bound, never an exact per-variable count.

**Mutation testing.** Deleting the `continue` on the "not changed" index test
fails 6 cases (5 of the 6 are these new ones); deleting the "deleted in this
branch" `continue` fails exactly 1 — `a variable declared inside an arm gets no
phi`, the only coverage in the suite for that filter.

Artefact: ESBMC 8.4.0 at master `31aee3387a`; 589/589 `ctest -LE regression`
pass. Test-only change, no `src/` line touched, so no Mode C obligation arises.
The 5 `regression/esbmc-unix` failures in the sampled subset reproduce on the
unmodified tree (pre-existing, macOS).

**Still open in M4.** H-B2. R14 remains pinned, not diagnosed. Also still open
from M0: WI-1, WI-2, D12.

### M4 (H-B2) — 2026-07-30, M4 closed

**Result: P10 discharged in the form the code actually supports, and objective
7's wording refuted. Two process-global counters leak across runs — R15.**

New file `unit/goto-symex/determinism.test.cpp`, 6 cases, 26 assertions. It does
not reuse the `engine` the other Tier-B tests carry: comparing two runs means
holding two equations alive at once, and each equation's
`symex_target_equationt` holds its `namespacet` **by reference**, so each run
must own its whole context bundle for the equation's lifetime.

| Case | Property |
|---|---|
| `the comparators distinguish two programs` | control — both comparators separate two different programs |
| `two runs of a branching program agree` | strict step-for-step equality |
| `two runs of a loop with calls agree` | strict, across 4 unwindings and a callee |
| `two runs over addressed locals agree once object ids are normalised` | equality modulo R15, value-set path |
| `two runs of a heap program agree once object ids are normalised` | equality modulo R15, allocation path |
| `object numbering leaks across runs (R15)` | pins the leak, so it cannot widen unnoticed |

**Objective 7 asks for byte-identical equations and that is not achievable.**
The first written form of this harness compared crcs of
`(type, ignore, hidden, guard, lhs, rhs, cond)` and **failed on 2 of 4
programs** — every failing one involving a pointer or the heap, at the first
`malloc`. The cause was not the container-iteration-order hazard §7.2
anticipated: the symbol *names* differ between runs, `symex_dynamic::dynamic_1_
array` versus `dynamic_2_array`, because `execution_statet::dynamic_counter` and
`dereferencet::invalid_counter` are `static thread_local` and reset nowhere.
Recorded as R15 with the fix location (`setup_for_new_explore`, not the
`execution_statet` constructor — the reachability tree copies that per
interleaving, so resetting there would mint *colliding* names). Not fixed here:
the fix renames objects in counterexample output and needs a full-corpus
`test.desc` sweep, so it is pinned exactly as R14 was.

The property therefore splits: strict equality where the program creates no such
object, and equality after canonicalising `dynamic_<n>`/`symex::invalid_object
<n>` where it does. Canonicalisation works on the pretty-printed step rather
than a crc, because the numbering sits *inside* symbol names, which a structural
hash cannot reach without rewriting the expressions.

**What this harness can and cannot catch, stated because a determinism test
invites over-reading.** It catches only divergence *between two runs*. A
consistent reordering — iterating `current_names` backwards, say — reorders both
runs identically and is invisible here; that is correct behaviour for a
determinism oracle, not a gap in the cases, but it means P10 passing says nothing
about step order being *right*. It is also same-process only: cross-process
variation (ASLR changing an address-ordered iteration) is out of reach and
belongs to a Tier-C oracle.

**Mutation testing.** A process-global skip injected into `phi_function`'s filter
chain (`static int mut; if (++mut % 9 == 0) continue;`) — a faithful model of the
leak R15 documents — is caught by both strict-equality cases. The two
canonicalising cases do not catch it, which is the expected sensitivity ordering:
canonicalisation trades precision for tolerating R15.

Artefact: ESBMC 8.4.0 at master `31aee3387a`; 595/595 `ctest -LE regression`
pass. Test-only change plus one CMake line; no `src/` line touched, so no Mode C
obligation arises.

**M4 closed** — H-B1, H-B4, H-B5, H-B2 all discharged. WI-4 (the Tier-B′ pilot)
was **not** run and is carried to M5 with D14. R14 and R15 are pinned, not fixed.
Also still open from M0: WI-1, WI-2, D12.

**Cleanup owed (not done here).** Five Tier-B files now carry a near-identical
`engine` fixture, and `determinism.test.cpp` adds a sixth variant. Extracting
one shared fixture is blocked on a real semantic conflict, recorded so the next
attempt does not rediscover it: `invariant`, `renaming` and `frame_lifecycle`
read the live state *before* `run()` and so need `setup_for_new_explore()` in the
constructor, while `unwind` defers it precisely so its `with()` options are set
first. Reconciling those is a five-file refactor with no property gain, so it
should land on its own.

### M5 (H-C1) — 2026-07-30

**Result: slicer verdict-parity holds over the corpus — 1328 inputs agree, 0
diverge. The first whole-corpus parity report in this plan, and the first
artefact of D8.**

| | |
|---|---|
| Corpus | `regression/esbmc`, 1430 `CORE` of 1574 dirs |
| Relation | verdict(default flags) == verdict(default flags + `--no-slice`) |
| **agreed** | **1328** |
| **diverged** | **0** |
| inconclusive | 67 — no verdict in one or both legs |
| skipped | 35 — `test.desc` already names `--no-slice` |
| Cap | 15 s per leg, 12 concurrent; ESBMC 8.4.0 at master `31aee3387a` |

**The denominators matter more than the headline.** 1328/1430 is the coverage
this run actually achieved, not 1430. Of the 67 inconclusive, 66 produce **no
verdict in either leg on this build** — an uncaught `irep2_cast_error` on the
`memcpy`/bitfield inputs, and `_BitInt` widths this target rejects — so they are
CORE tests already failing locally, unrelated to slicing, and correctly excluded
from a parity comparison rather than counted as agreement. The 67th,
`github_1600`, is `A=FAILED, B=timeout`: it exceeds 15 s only with `--no-slice`,
which is the slicer working as the performance optimisation it also is. A Linux
CI build should convert most of the 66 into real comparisons, so the sweep should
be re-run there before this row is quoted as corpus-wide.

**Four planned scripts collapsed into one.** H-C1, H-C2, H-C3 and H-C5 are the
same relation over different flags, so `oracle_flag_parity.py` takes both
flag-sets as arguments instead of §11.1's four near-identical shell scripts
(§11.1 updated). It builds each argument list through
`regression/testing_tool.py`'s `TestCase` rather than re-parsing `test.desc` —
which also inherits the `--timeout`/`--memlimit` stripping and the
path-resolution rule. Two bugs found while getting that right, both of which
would have produced a *falsely clean* or *falsely alarming* sweep:

1. Running each pair in a scratch cwd (needed so the two legs cannot collide on
   output files) broke the repo-relative input paths `TestCase` generates, so
   **every** test returned "no verdict" — a run that reports 0 divergences
   because it never verified anything. Fixed by resolving the suite to an
   absolute path. The lesson is that "0 diverged" is only meaningful next to a
   non-trivial `agreed` count, which is why the summary prints both.
2. A test whose own flags already name the compared flag would compare a
   configuration against itself; 35 such tests exist and are reported as
   skipped, not silently folded into agreement.

Per the no-silent-caps rule, every skipped, inconclusive and diverging test is
printed by name, and the script exits non-zero only on a real divergence.

**Still open in M5.** H-A4 and H-B3. Per §6.4 and the M2 precedent, H-A4 should
not be built as a Tier-A transcription: its A4.1 obligation *is* H-B3, and A4.2
(no retained step reads a symbol defined only by an ignored step) and A4.3 are
observable on the real sliced equation, so both belong at Tier B. The scheduled
CI job of §11.2 is not yet wired, so D8 remains partial.

### M5 (H-C2) — 2026-07-30

**Result: 11 divergences, the fault in the `--no-simplify` leg in every one. One
is a false SUCCESSFUL (R17); the other ten are spurious or absent verdicts
(R16). The first real defects this plan's Tier C has produced.**

| | |
|---|---|
| Relation | verdict(default) == verdict(default + `--no-simplify`) |
| **agreed** | **1174** |
| **diverged** | **11** |
| inconclusive | 197 — 156 of them a timeout in the `--no-simplify` leg |
| skipped | 48 |

**Triage came almost free, and it is what makes the result actionable.** Each
divergence was checked against the verdict its own `test.desc` expects, and in
all 11 the *default* leg matches it. `--no-simplify` is therefore the faulty
configuration in every case, and no default-configuration soundness bug is
implied. Ten are R16 (nine spurious counterexamples, one UNKNOWN under
`--k-induction`); one is R17.

**R17 is the one that matters**, and no single flag exhibits it:

| Configuration | Verdict on `void *b = malloc(-4); assert(0);` |
|---|---|
| default | FAILED ✓ |
| `--no-slice` | FAILED ✓ |
| `--no-simplify` | FAILED ✓ |
| `--no-simplify --no-slice` | **SUCCESSFUL ✗** |

A reachable `assert(0)` is missed: one VCC is generated and the solver returns
UNSAT, so the path became vacuously infeasible. Minimised from
`github_1631_compact` by dropping `--force-malloc-success` and the second
allocation; a *positive* size does not reproduce, so the trigger is the negative
size widening to a huge `size_t`. Pinned by
`regression/esbmc/no_simplify_no_slice_huge_malloc` (KNOWNBUG) with a
positive-size CORE companion, and not fixed — root-causing which allocation
constraint turns contradictory is its own task.

**One inference I made and had to retract**, recorded because it would have
become a wrong bug report. The corpus reproducer needs `--compact-trace`, and I
first read that as a *trace-formatting flag changing a verdict*, which would
contradict §2.3's ranking of trace output as P3 and non-load-bearing for the
verdict. Reading the option handling showed `--compact-trace` sets `no-slice`
implicitly (`command_line_options.cpp:410`). §2.3's ranking stands; what the
episode actually shows is that an output flag quietly enabling a *semantic* one
makes flag-composition defects reachable without the user naming either.

**H-C2's cost is the timeout tail, not the runs.** 156 of the 197 inconclusive
are a timeout in the `--no-simplify` leg alone — unsimplified formulas are much
harder, so the 15 s cap that suffices for H-C1 truncates ~11 % of this sweep.
Those inputs are reported by name rather than folded into agreement, so the
honest coverage figure is 1185 of 1382 compared, and a scheduled run should give
this oracle a longer cap than its siblings.

### M5 (H-A4 / H-B3) — 2026-07-30, M5 closed

**Result: the slicer's closure obligation is discharged against the real
`symex_slicet`. No defect found — and the first version of the harness could not
have found one, which mutation testing is the only reason I know.**

New file `unit/goto-symex/slice.test.cpp`, 5 cases, 141 assertions. Per §6.4,
H-A4 was **not** built as the Tier-A transcription §7.1 specified: A4.2 is
entirely observable on the equation the real slicer rewrites, so transcribing
`symex_slicet` would add drift risk (§9.1) to verify a copy. This is the third
harness to take that route after H-A1 (M1) and H-A2/H-A3 (M2); §7.1's Tier-A
entries should now be read as obligations, not as prescribed implementations.

| Case | Property |
|---|---|
| `the slicer keeps every definition it still reads` | A4.2 on straight-line code |
| `closure survives a branch and a join` | A4.2 across phi nodes |
| `closure survives a call and an unwound loop` | A4.2 across frames and 4 unwindings |
| `closure survives constant array indices` | A4.2, and the dead-store elision fires |
| `a symbolic array index disqualifies the array` | no store elided when the read index is unknown |

**The harness was blind in its first version, and passed anyway.** The slicer
discards information *two* ways: setting `step.ignore`, and rewriting an
assignment's `cond` to `lhs == src` so a dead array store encodes the identity.
Version one audited only `ignore` flags, so deleting the `array_disqualified`
consultation in `slice.cpp` — §7.1's prescribed twin for this very harness —
left every `ignore` flag unchanged and **all five cases still passed**. A
`store_elided` predicate comparing each assignment's `cond` against
`equality2tc(lhs, rhs)` closed the hole, and the twin now fails the
symbolic-index case. Generalising: where a component has more than one mechanism
for discarding information, a harness keyed on one of them looks like coverage
and is not.

**Input vacuity, caught the same way.** The assertions were first written
`x == x`, which folds to `true` before the slicer runs, so the assert read
nothing, the read set was empty, and closure held for free. Rewritten as
`x != 424242`. This is the trap the M4 (H-B4) entry recorded for phi arms, now
hit in a second harness, so it is worth a standing rule: an assertion written to
be trivially true is usually also trivially *unread*.

**Mutation testing.** Removing the `array_disqualified` consultation fails the
symbolic-index case and only it. Over-slicing every fifth assignment
(`|| ++mut % 5 == 0` on the tracked-lhs test) fails 3 cases via
`dangling_reads`. Every case also requires `ignored > 0`, `retained > 0` and
`live_reads > 0`, so neither an inert slicer nor an unread equation can pass.

**H-B3's equisatisfiability obligation is discharged by H-C1, not by this file.**
§7.2 asked for per-claim solver verdicts over "≥ 30 small programs"; H-C1
compared whole-run verdicts with and without slicing over **1328** real corpus
inputs at 0 divergences, which is the stronger corpus. What H-C1 does not give is
*per-claim* granularity: a compensating pair of claim-level divergences within
one run would cancel. Closing that is exactly H-C7's `--multi-property`
comparison, so the residue is passed there rather than claimed here.

**M5 closed** — H-C1, H-C2, H-A4 (at Tier B) and H-B3 (via H-C1, per-claim
residue to H-C7). R16 and R17 are pinned, not fixed. D8 stays partial: the §11.2
scheduled job is still unwired, which is the one thing keeping these oracles from
running unasked.

The `symex_run::equation` fixture added for H-B2 is now shared with this file
(`unit/goto-symex/symex_run.h`), so the two harnesses that own an equation rather
than live state share one copy. The five-file `engine` consolidation recorded in
M4 (H-B2) is still owed, still blocked on the same conflict.

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
