# GOTO-SYMEX Formal Verification Plan

**Subsystem:** `src/goto-symex` — ESBMC's symbolic execution engine: the stage
that turns a GOTO program into an SSA formula (`symex_target_equationt`) for the
SMT backend.
**Verifier:** ESBMC itself (BMC + k-induction) on extracted kernels; Catch2
property/differential tests on the real classes (`unit/goto-symex/`);
whole-tool metamorphic oracles over `regression/`; sanitizers for the rest.
**Status:** **M0–M8 closed**, **M9 in progress** (§15 verdict log). §6.4 records
the tier-ordering rule M1 produced. Except where §15 records a discharged
result, every harness below is a *proposal* and nothing here asserts a proof.
Findings not marked discharged in §9.2 remain *hypotheses with cited evidence*,
not confirmed end-to-end bugs.

M9 closed the §7.3 assumption register (H-A8 was its last live row), pinned all
three of R9's approximation claims, and fixed R10 and R15. Three rows were
**re-characterised rather than fixed**, which is the milestone's more useful
output: R4's unchecked lookups have no witness in 352 corpus inputs, so guarding
them would add branches nothing can show reachable; R8's "missed-bug direction"
is false, because stack lifetime is checked by `is_live_variable` and the
disabled block is superseded; and R29 — found by M9's own access-shape census —
is a **new High-severity false SUCCESSFUL**, partly fixed, with its residual
traced out of this subsystem into `src/pointer-analysis`.

**Still open:** R29's two bare-struct-member shapes and the pre-existing
R16/R19–R27 rows §9.2 records individually. R6 got its witness and its fix
(#6785); A6.4, carried since M6, is discharged by the run-order invariant the
engine now checks in release.
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
   unreproducible and invalidates regression pinning. Held only modulo object
   numbering until R15 was fixed (§15 M9 (R15)); the wording is now literal.
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
| **H-B6** | **Value-set merge monotonicity** (I9) | After `merge_value_sets`, assert the result ⊇ both inputs (using `value_sett` API). **Run, §15 M9: I9 discharged, no defect** — `unit/goto-symex/value_set_merge.test.cpp`; an *intersecting* `make_union` is caught by 3 of 5 cases, while *deleting* the union is caught by none, and cannot be | An accidental intersection — a silent unsoundness |
| **H-B7** | **Assumption-discharge suite** (§6.1 rule 3) | For each Tier-A assumption in §7.3, an assertion on the real engine that it holds over the corpus | Over-constrained Tier-A proofs |
| **H-B8** | **Incremental-equation parity** (I13) | Same program with and without `--smt-during-symex`; assert identical claim count and per-claim verdicts. **Run at Tier C instead, §15 M7: 1358 agreed, 3 diverged → R19**, a per-property false PASSED | `runtime_encoded_equationt` ctx-stack bugs |

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
| H-A1, H-A9 | The L2 `name_record` key is stable across `make_assignment`'s inner `rename` (I2) | **Discharged**, §15 M9 (H-B7) — `renaming.test.cpp`'s "make_assignment publishes a fresh increasing L2 index" asserts the entry `coveredinbees` updated is the one keyed by the caller's key, over five successive publications |
| H-A2 | `guard2tc::operator-=` satisfies `(g_cur ∨ g_mrg) → (diff ↔ g_mrg)` | **irep2 plan** H-A9/H-B4 — cross-document dependency |
| H-A2 | Incoming merge guards may overlap (no disjointness assumed) | by construction (not assumed) |
| H-A4 | Every `with2t` store the slicer elides has a `symbol2t` source and constant index | **Discharged**, §15 M9 (H-B7) — `assumption_discharge.test.cpp` checks it on every elided store and censuses the excluded shapes; struct member stores are excluded by their `constant_string2t` field, which the census now pins |
| H-A6 | `thread_last_reads/writes` contain *all* accesses of the last transition, including through pointers | **Refuted twice.** R11 → **R18** (one-level resolution losing a nested dereference) was fixed by **#6550**. The completeness this row asks for is now *checked* rather than assumed — a 21-shape census, §15 M9 (H-A6) — and it failed: five shapes holding the pointer in an aggregate were missed, recorded as **R29**. R29 is now fixed (§15 M9 (R29 fix), (R29 residual)) and the census re-runs **21/21 agreeing with `--no-por`**, dual-solver, §15 M9 (H-A6 re-census). The row stays **refuted**: extending the same census by two shapes immediately found **R31** (`int **pp = &s.p; **pp = 1`, a false SUCCESSFUL on ordinary C) plus a struct-punning shape that is UB. R31 is now fixed and the census runs **27/28**, §15 M9 (R31 fix) — but that round also showed the 21/21 above was over-stated, since an array element reached through a pointer into the array was failing all along under a spelling the enumeration had recorded as passing. R31's own section then declared its one remaining gap witnessless, and the next probe witnessed it as **R32** (a symbolic array index, false SUCCESSFUL on both solvers, §15 M9 (R32)). Every extension of this census has found a defect, one extension falsified the census's own result, and one falsified the *closing claim* of the round that made it — which is the argument against ever discharging this row by enumeration. **R33** sharpens that argument from the other side: code review of R31's fix found a false SUCCESSFUL on `&s.v[1]`, a shape more common than several the census does cover, which all twenty-eight enumerated shapes walked past. Enumeration is bounded by the model of failure that generates it; reading the neighbouring branch is not. With R31, R32 and R33 fixed the repository's 22 shapes all agree with `--no-por`, §15 M9 (census re-run) — the first round whose count is reproducible from the tree rather than from a scratch directory. The only divergences left are two strict-aliasing shapes that C11 6.5p7 leaves undefined |
| H-A8 | `push_ctx`/`pop_ctx` calls are balanced by the caller (`reachability_treet`) | **Discharged**, §15 M9 (H-A8) — `context_stack.test.cpp` on a real `runtime_encoded_equationt` over a real solver: an exhausted 49-interleaving exploration lands back on depth 0, having reached 9. Deleting the setup `push_ctx` fails it *and* SIGSEGVs, which is the UB the row's failure mode predicted |
| all Tier A | `nondet` solver answers are *sound* (no wrong TRUE/FALSE) | out of scope — solver backends are Tier D |

### 7.4 Tier C — whole-tool metamorphic oracles

Scripted sweeps over existing corpora. Each is a *pure verdict comparison*: no
modelling, no assumptions, and a divergence is always a real bug in one of the
two configurations.

| ID | Relation | Corpus | Detects |
|---|---|---|---|
| **H-C1** | verdict(default) == verdict(`--no-slice`) | `regression/esbmc` CORE (1430 of 1574 dirs) | Slicer unsoundness/incompleteness end-to-end. **Run, §15 M5: 1328 agreed, 0 diverged**, 67 inconclusive, 35 skipped |
| **H-C2** | verdict(default) == verdict(`--no-simplify`) | same, minus tests selecting an approximate arithmetic encoding | Simplifier / constant-propagation semantic drift (P9). **Run, §15 M5: 1174 agreed, 11 diverged** — R16 (10, incompleteness) and R17 (1, false SUCCESSFUL, since **fixed**; the `--no-slice` composition turned out to be a symptom, and its residual is R25). **Re-measured §15 M9 (R16): 1198 agreed, 3 diverged**, 206 inconclusive, 55 skipped, 42 abstract — seven of the M5 entries agree again, and what is left is `github_252` (R16) plus the `github_1257` pair (**R28**). The relation's premise fails for `--ir`/`--ir-ieee`/`--fixedbv`, where the encoding rather than the simplifier decides the verdict, and the oracle now reports those as `abstract` rather than comparing them. The inconclusive count is **not** the loaded machine it was first attributed to: it is `--no-simplify` costing two orders of magnitude more on the inputs it strands, so no practical bound recovers them — §15 M9 (H-C2 residue). **Re-measured whole-corpus after #6781 and #6783, §15 M9 (H-C2 re-measured): 1299 agreed, 2 diverged, 120 inconclusive** (37 no-verdict, 83 timeout), 55 skipped, 42 abstract. 101 inputs the relation had never been able to compare now agree and **no new divergence appeared**, so the residue concealed no defect; the 2 are R28's `github_1257` pair, and R16's `github_252` now agrees |
| **H-C3** | verdict(bitwuzla) == verdict(z3) | same | Encoding assumptions that only one solver tolerates. **Run, §15 M7: 1269 agreed, 0 diverged** |
| **H-C4** | verdict(default) == verdict(`--no-por`) and == verdict(`--state-hashing`) | `regression/esbmc-unix`, `regression/esbmc` concurrency tests | POR / state-hashing over-pruning (I14, I15). **Run, §15 M6: 258/0 and 255/0 — clean, but the R18 witness is a program the corpus does not contain** |
| **H-C5** | verdict(default) == verdict(`--no-interval-symex-guard`) | `regression/esbmc`, `regression/k-induction` | Interval-domain guard pruning (the documented hazard at `symex_goto.cpp:57-79`). **Run, §15 M7: 1360 agreed, 0 diverged** |
| **H-C6** | **Unwind monotonicity**: FAILED at `--unwind k` ⇒ FAILED at every `k' > k` | loop-bearing subset | Lost counterexamples when the bound grows — a pure soundness relation, no oracle needed |
| **H-C7** | per-claim verdicts under `--multi-property` == the individual `--claim N` runs | `regression/esbmc` multi-assert tests | Claim/slice interaction bugs (cf. recent `multi_property_check` fixes). **Run, §15 M7 (cont.): 326 compared, 1 diverged** (`github_1655` = R19); the 4 first reported were oracle artefacts |

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
| **R4** | **Medium (crash → no verdict)** — swept, no witness, §15 M9 (R4) | **Eight unchecked `*ns.lookup(...)` dereferences.** `namespacet::lookup` returns `nullptr` on miss (as `renaming.cpp:15-21` itself demonstrates by checking). A miss ⇒ null deref ⇒ SIGSEGV mid-verification. `phi_function`'s site is the most exposed: it filters only `goto_symex::guard!` and `symex::invalid_object` before looking up an arbitrary merged variable's base name. | `symex_goto.cpp:433`; `symex_function.cpp:159`; `symex_valid_object.cpp:47`; `dynamic_allocation.cpp:66,92,105,118,143` | H-A10 | Add checked lookups with a diagnostic (`log_error` + controlled abort) or prove the precondition per site and record it as a cited comment. |
| **R5** | **Medium (soundness detector disabled)** — discharged, §15 M4 | `check_for_duplicate_assigns` — the *only* in-tree checker for the core SSA invariant I10 — merely `log_status`es duplicates and then reports "Checked N insns". It never fails, and nothing calls it in a normal run. | `symex_target_equationt::check_for_duplicate_assigns`, `symex_target_equation.cpp` | H-B1 | Turn it into a validator returning a bool; run it under a debug/CI flag over the whole regression corpus. |
| **R6** | **Medium (unsound pruning, opt-in flag)** — mechanism pinned, §15 M6 (cont.); witnessed and **FIXED** by **#6785**, §15 M9 (R6) | `state_hashing_level2t::make_assignment` keys `current_hashes` by the **L0** original name, acknowledged in-code ("XXX — consider whether to use l1 names instead. Recursion, reentrancy."). Two states that differ only in the L1 activation of a recursive local therefore fingerprint identically ⇒ `hit_hashes` prunes a genuinely different state ⇒ missed interleaving. Severity is bounded by `--state-hashing` being opt-in. | `execution_state.cpp:~1342-1378`; `reachability_treet::hit_hashes`, `reachability_tree.h:352` | H-A8-style model + **H-C4** | Key by the L1 name record (or mix call-stack depth into `generate_hash`). The unproven step is the "equal fingerprints are bisimilar" claim at `reachability_tree.cpp:420`: the fingerprint omits call-stack depth, so two states at one pc with equal L0→value maps but different recursion depths collide while resuming into different continuations. H-C4's state-hashing leg is clean (255/0) and four targeted programs produced no verdict-changing prune, so a witness must make the bug reachable *exclusively* behind the colliding state. **Witness found and fixed, §15 M9 (R6): recursion is not required.** One function called from two sites stands at one pc with one L0 map and returns to different places, which collides just as well and is far easier to build; `f(); f(); assert(0);` behind a cswitch point is missed under `--state-hashing` and reported without it, on both solvers. Depth would not have separated those two states either — they sit at equal depth — so the remedy this row proposed is insufficient, and **#6785** mixes each frame's `calling_location` instead. |
| **R7** | **Low–Medium (UB) — refined, §15 M1** | `previous_frame()` computes `*(--(--call_stack.end()))` with no size check. `call_stackt` is a `std::vector<framet>`, so at size 1 this evaluates `--begin()`, forming a pointer before the start of the array — undefined by [expr.add]/4 **whether or not it is dereferenced**, not merely a bad read. The second clause of the original finding ("returns a reference a subsequent `pop_frame` invalidates") does **not** hold: `pop_back` invalidates only the reference to the erased last element, and `previous_frame` returns the second-to-last. The precondition holds today by construction — the sole call site does `new_frame(...)` on the preceding line — but nothing states it in the shipped binary (R1). | `goto_symex_statet::previous_frame`; sole caller `goto_symext::symex_function_call_code`; [expr.add]/4 | `unit/goto-symex/frame_lifecycle.test.cpp` (Tier B, discharged) | Add a release-checked precondition **as part of R1's `SYMEX_INVARIANT` work in M3**, so the macro lands once with its cost measured; index (`call_stack[size() - 2]`) rather than decrementing an iterator. |
| **R8** | **Medium (documented model gap)** — **re-characterised, §15 M9 (R8): the missed-bug clause is false** | `is_valid_object` returns `false` for **every** non-static, non-dynamic symbol: the stack-scope branch is `#if 0`'d out with "XXX re-enable to be able to check for stack-var-out-of-scope problems". Stack-object validity is therefore not modelled, and `dynamic_allocation.cpp` compensates by *assuming* `invalid_pointer` applies only to dynamic objects ("we never update `__ESBMC_alloc` for stack ptrs"). Net effect on stack-lifetime bugs (use-after-scope) is a **missed-bug** direction. | `goto_symext::is_valid_object`, `symex_valid_object.cpp:85-118`; `dynamic_allocation.cpp:110-116` | H-A10 + a targeted `regression/esbmc` use-after-scope corpus | Quantify with a dedicated corpus before attempting a fix; the fix is a model change, not a patch. |
| **R9** | **Low–Medium (approximation direction unproven)** — **all three pinned**, §15 M9 (R9) and M9 (R9 cont.) | Three documented "sound over-approximation" claims are unproven: value-set filtering after a pointer havoc (`symex_assign.cpp:554-576`), the non-scalar uninterpreted-function fallback (`symex_function.cpp:418-449`), and the function-pointer target enumeration over an over-approximated value set (`symex_function.cpp:806-839`). Each *argues* the direction in a comment; none was checked. | cited lines; `unit/goto-symex/overapproximation.test.cpp` | H-B6 + H-C1/H-C3 | All three are now Tier-B predicates over the produced equation, each mutation-confirmed: disabling the compatibility filter, and dropping its empty-list guard, fail one case apiece. **Claim 1 is pinned too** — the `pc->inductive_step_instruction` gate is reachable after all, since `goto_k_induction` is a free function a fixture can run over a `goto_factory` program (§15 M9 (R9 cont.)). Its two cases separate three mutants: keeping the sinks, keeping only the sinks, and deleting the `!filtered.empty()` guard, the last of which silently costs a dereference its safety checks. |
| **R10** | **Low (latent UB)** — **FIXED**, §15 M9 (R10) | `renaming::level2t::name_record`'s `name_record() = default` leaves `lev`, `l1_num`, `t_num` **and the derived `hash`** indeterminate (contrast `level1t::name_record`, which initialises `base_name("")`). No current default-construction site was found, but a future one (`std::optional`, map default-insert, array of records) would read indeterminate memory in `compare`/`hash`. | `renaming.h:143-214` | `unit/goto-symex/renaming.test.cpp` (Tier B, discharged) | **Fixed.** Default member initialisers on all four fields, and the hash computation factored into a private `compute_hash()` both constructors call — the fix has to give `hash` a value *consistent with* the fields, not merely a defined one, because `compare()` short-circuits on it. Not latent in the weak sense the row implied: with the original `= default`, a test that merely default-constructs a record **traps** (SIGTRAP, exit 133). |
| **R11** | **Confirmed — mechanism corrected, see R18, §15 M6** | MPOR's independence decision consumes `thread_last_reads`/`thread_last_writes`, populated via `get_expr_globals`, which resolves pointer operands through the *current* value set. If a write through a pointer whose value set is incomplete (or whose entry is `unknown`) is missed, the dependency is missed and an interleaving is dropped — **unsound**. `get_expr_globals` also early-returns entirely under `--data-races-check-only`. | `execution_statet::get_expr_globals`, `check_mpor_dependency`; `reachability_treet::ever_written_globals`/`address_taken_globals` | H-A6 (relation) + **H-C4** (end-to-end) | **Answered.** An `unknown` entry does not force a conservative dependency — the `dest` loop skips anything that is not an `object_descriptor2t` over a `symbol2t`, with no fallback. But that is *not* the reachable defect: the witness in R18 shows the missed dependency comes from resolving only **one** pointer level, so a nested dereference is recorded against the intermediate pointer. R11's suspicion was right and its stated mechanism was wrong. Superseded by R18. |
| **R13** | **Medium (silent under-verification) — confirmed and fixed, §15 M2 (cont.)** | **`--unwindsetname` never matched a loop.** `unwind_func_set` was keyed by `user_name_to_usr(name)`, which appends a `#` terminator (clang's C++ USR spelling), while `loop_id_to_func_index` was keyed by the goto function-map id, which for a C function is `c:@F@f` with no terminator. The `count(unwind_key)` in `get_unwind` therefore always missed and the global `--unwind` silently won, so a user raising the bound for one function got the lower global bound and a verdict covering less than they asked for. A second defect in the same option: the `name:index:bound` field split scanned left-to-right, so the documented USR form (`c:@F@f#:0:11`) split inside the `c:` prefix. Neither was caught because all five `unwindsetname` regression tests ran without a global `--unwind` and so passed vacuously. | `goto_symext::goto_symext`, `symex_assign.cpp:66-120`; `get_unwind`, `symex_goto.cpp:525`; `user_name_to_usr`, `usr_utils.cpp:29` | `unit/goto-symex/unwind.test.cpp` (Tier B, discharged) | Fixed: both sides now key on the name `--show-loops` prints (`usr_to_user_name`), and the field split scans from the right. Three non-vacuous regression tests added; `unwindsetname_03_priority` corrected to the loop number the program actually has. |
| **R14** | **High (missed bug, default configuration)** — I10 violated on a real input, found by R5's repaired detector, §15 M4; **FIXED** by **#6650** | With `--double-assign-check` made to fail, `regression/esbmc/github_286_3` produces an equation that **defines one SSA name twice**: `…@F@getNumbers2@numbers2?1!0&0#1`, the L2 index 1 of a local array in a function that returns a dangling pointer to it. Two definitions of one name are two constraints `x#1 == e1` and `x#1 == e2` on the same variable; where the right-hand sides disagree the conjunction is unsatisfiable, which silently removes that path from the formula — the missed-bug direction. One input in ~900 swept. Not yet characterised: which two steps emit it, and whether the two right-hand sides can differ. | `symex_target_equationt::check_for_duplicate_assigns` under `--double-assign-check`; `regression/esbmc/double_assign_check_local_array` (CORE since #6650) | H-B1 | **Fixed by #6650.** The scope exit was the emitter, as this entry guessed: `pop_frame` *erased* the local's L2 record rather than retiring it, so the counter restarted and a later write through the returned dangling pointer re-issued the index the declaration had already used. The fix advances the counter on frame exit, leaving reads through the dangling pointer unconstrained. Re-measured 2026-08-03: the pin is CORE and `github_286_3 --double-assign-check` matches its `test.desc`. |
| **R15** | **Low (reproducibility, latent collision)** — found by H-B2, §15 M4 (H-B2); **FIXED**, §15 M9 (R15) and §15 M9 (R15 regression) | **Object numbering leaks across symex runs in one process.** `execution_statet::dynamic_counter` and `dereferencet::invalid_counter` are `static thread_local` and reset nowhere, so a second exploration in the same process names its objects from where the first stopped: the same program under the same options yields `symex_dynamic::dynamic_1_array` on the first run and `dynamic_2_array` on the second. The sibling `nondet_count` is a plain instance member the constructor zeroes, so the asymmetry is unintended rather than a design choice. The equation is therefore not a function of (program, options) alone. No wrong verdict follows — the names only need to be *fresh*, and monotonic counters are fresh — so this is a reproducibility defect, and objective 7's "byte-identical" wording is unachievable as stated. **Latent second-order risk:** `thread_local` means two threads each start at 0, so if symex is ever parallelised (§14.6) two threads would mint *colliding* object names into a shared context. | `execution_state.cpp:21`, `execution_state.h:583`; `dereference.cpp:23,538`, `dereference.h:281`; contrast `nondet_count` reset at `execution_state.cpp:104` | `unit/goto-symex/determinism.test.cpp` (Tier B, pinned) | **Fixed**, and the first placement of the fix was wrong in the way the prescription predicted it would not be. `reset_dynamic_counter()` and `reset_object_counter()` are called from `setup_for_new_explore` rather than the `execution_statet` constructor, but unguarded that regressed `--incremental-bmc`: `bmct::run` calls `setup_for_new_explore` once per k iteration while the symbol state persists, so iteration k+1 re-mints a name iteration k bound at another type and Bitwuzla aborts on a sort-width mismatch in `mk_eq`. The reset is therefore taken only when the context holds no `symex_dynamic::` object yet (`reachability_tree.cpp`, `!context_has_objects`), which is where an *independent* run begins rather than where any exploration does — landed with #6774. The predicted `test.desc` churn did not materialise: nothing in the corpus matches `dynamic_[0-9]+` or `invalid_object[0-9]+`, so the numbering was never pinned. Objective 7's "byte-identical" wording is achievable and the determinism harness asserts it strictly |
| **R27** | **High (missed bug, default configuration)** — found by M8 triage, §15 M8 (cont. 6); filed as **#6558**, fixed by **#6571**; **renumbered from R23**, which §15 uses for the compound-assignment defect | **A real race is lost when the guarded branch writes its own guard variable back to the falsifying value.** Nine lines: `t1` loops twice over `if (receive) { assert(i < 1); receive = 0; }` while `main` sets `receive = 1` after `pthread_create`. The racy schedule — `i=0` reads 0 and skips, `main` writes 1, `i=1` reads 1 and the assertion fires — is explored and reported when the branch body is empty, writes a *different* variable, or writes `receive = 1`. Only `receive = 0` loses it. **On that schedule the added write never executes before the violation**, since the body is skipped at `i=0` and the assertion fires before reaching the write at `i=1`, so provably-unexecuted code is removing a counterexample. **No flag recovers it:** `--no-por`, `--context-bound 8`, `--state-hashing`, `--no-slice` and `--no-interval-symex-guard` all still report SUCCESSFUL, which rules out POR, the context bound, state hashing, the slicer and the interval-domain guard pruning that was the natural suspect. **Narrowed further, §15 M8 (cont. 7):** the loss needs all three of — `t1` guards on X, `t1` writes X in the guarded body, and `main` writes X *exactly once*. Splitting the guard from the written variable in either direction restores detection, and so does giving `main` a second, redundant `receive = 1`. Since a duplicated identical write cannot change the formula's meaning but does add a scheduling point, the incompleteness is in the **interleaving set**, not the encoding. The GOTO programs of the detected and missed variants are instruction-for-instruction identical apart from the assignment target, so nothing upstream of symex differs. Mechanism still unknown. | discriminator table above; `regression/esbmc-unix/race_guard_self_clear` (CORE since #6571) and `race_guard_other_write` (CORE, different variable, caught today); `regression/esbmc-unix/03_circular_reduce` (pre-existing KNOWNBUG) | M8 triage | **Violability proven, gate hypothesis retracted, §15 M8 (cont. 8–9):** under `--data-races-check` the program reports FAILED **on the assertion itself**, so the schedule is reachable and the claim genuinely violable — the miss is a real incompleteness, not a modelling artefact. The `main_thread_ended` cutoff in `check_if_ileaves_blocked` was proposed as the cause and is **refuted**: keeping `main` alive past its write, with up to four further global writes, still reports SUCCESSFUL. `--data-races-check` both bypasses that gate *and* adds race instrumentation, so it does not isolate either. **Fixed by #6571**, which found three composing defects in the branch-merge path and flipped the pin to CORE. |
| **R29** | **High (false SUCCESSFUL, default configuration)** — found by the H-A6 shape census, §15 M9 (H-A6); **renumbered from R28** on merging master, which uses that number for the `--no-simplify` truncation defect; this branch's commit titles predate the renumbering | **A pointer held in an aggregate defeats MPOR's access resolution.** `get_expr_globals` gates its pointer-chain resolution on `is_symbol2t(expr)`, so a pointer reached through any aggregate step never enters it: the write is keyed on the *aggregate* while another thread keys on the target, MPOR calls the two transitions independent and prunes the racy interleaving. `*(s.p) = 1` against a concurrent `g = 2` reports **SUCCESSFUL** by default and **FAILED** under `--no-por`, both under Bitwuzla and Z3. Five shapes reproduce it — struct member (`s.p`), arrow (`sp->p`), array element (`pa[0]`), nested struct (`o.in.p`), union member (`u.p`) — and the boundary is exact: copying the pointer to a local first (`int *lp = s.p;`) restores detection, so the gate is syntactic, not a value-set limitation. This is **R18/#6539 generalised**: that fix followed chains of *symbols*, and an aggregate step between the pointer and its name was left outside. | census of 21 access shapes, 16 + 5 boundary probes; `regression/esbmc-unix/mpor_aggregate_ptr_race` (KNOWNBUG when filed, now CORE) and `..._local` (CORE control) | H-A6 | **Fixed.** §15 M9 (R29 fix) closed the array-element, arrow and union shapes with a `dereference2t` arm; §15 M9 (R29 residual) closed the two bare struct-member shapes in `src/pointer-analysis`. The cause was neither the arm asymmetry nor the `get_reference_set` count this row first recorded — constant propagation leaves the aggregate a `constant_struct2t`, whose members' values sit in the expression itself, so there is no suffixed symbol to key on. Review of that fix found two further false SUCCESSFULs, both fixed: splitting the value-set suffix on `.` breaks clang's anonymous member names, and the constant-union arm never consumed its suffix. `mpor_aggregate_ptr_race` is **CORE**, joined by `_nested`, `_anon`, `_prefix` and `_union_struct`; the census re-runs 21/21 dual-solver, §15 M9 (H-A6 re-census). |
| **R26** | **Medium–High (missed check, non-default flag)** — found by M8 triage, §15 M8 (cont. 3); **renumbered from R22**, which §15 uses for the return-value interleaving defect | **`--overflow-check` does not check arithmetic on a bitfield member.** `struct { int a : 3; } b = {3}; b.a += nondet_int();` reports **SUCCESSFUL**, though `3 + INT_MAX` overflows. The same statement on a *plain* member of the same struct is checked and reports FAILED, as is a plain local (`int x = 3; x += nondet_int()`). The gap is the bitfield, not the union, the sign, or the operand: struct and union bitfields, signed and unsigned, all miss it, while struct and union plain members are all checked. Attributes the pre-existing `github_162_fail` KNOWNBUG. | boundary probes above; `regression/esbmc/overflow_bitfield_member` (CORE, flipped from KNOWNBUG by R23's fix) and `overflow_plain_member` (CORE, plain member, caught today); `regression/esbmc/github_162_fail` (pre-existing KNOWNBUG) | M8 triage | Done, as a side effect of R23: the compound assignment was narrowing `b.a` to its 3-bit type before the addition, so the overflow claim was unfalsifiable. Performing the operation in the computation type emits `!overflow("+", (signed int)b.a, a)` and the check fires. |
| **R21** | **Medium (incompleteness, default configuration)** — found by M8 triage, §15 M8 (cont.); filed as **#6545** | **Multiplying an address-derived integer loses object identity, so the reconstructed pointer is rejected.** `uintptr_t u = (uintptr_t)&s; u *= 2; u -= (uintptr_t)&s; *(int *)u = 3;` recovers `&s` exactly, yet reports **FAILED**. The boundary: an *additive* round-trip (`u += 4; u -= 4`) is tracked, multiplying a *pure integer* offset and adding it to an address is tracked, and `u = u * 1` folds away — only a genuine multiplication of an address-derived term defeats recovery. `offsetof` counts as address-derived, since it expands to `(size_t)&((S *)0)->m`: the same program with a literal `4` in place of `offsetof(struct S, y)` verifies, with `offsetof` it does not. Attributes three pre-existing KNOWNBUGs to one cause — `github_426_2` (multiplies an `offsetof`), `github_426_3` and `github_426_4` (multiply an address). Noisy direction, so P1: a spurious counterexample, not a missed bug — and the exact complement of R20, which *accepts* a computed address it should reject. | boundary probes above; `regression/esbmc/ptr_int_multiply_roundtrip` (KNOWNBUG) and `ptr_int_additive_roundtrip` (CORE); `regression/esbmc/github_426_{2,3,4}` (pre-existing KNOWNBUGs) | M8 triage; H-A10's `symex_dereference` obligation | **Closed as a stated limitation** (#6545): `docs/design/pointer-integer-provenance.md` records why the obvious fix is unsound, and `ptr_int_multiply_roundtrip` stays KNOWNBUG as that limitation's marker rather than as an open defect. Still reproduces by design. |
| **R20** | **Medium–High (missed bug, default configuration)** — found by M8 triage, §15 M8 (cont.); filed as **#6544** | **A dereference through a constant non-null integer address is unchecked.** One line reproduces it: `int *p = (int *)65; return *p;` reports **`VERIFICATION SUCCESSFUL`**. The boundary is narrow and is what makes this a defect rather than a modelling choice: `(int *)0` is caught by the null check, `(int *)nondet_ulong()` is caught, and `(int *)(unsigned long)&x` is correctly accepted as a valid round-trip — only the *constant* non-null address escapes, for reads and for writes alike. Attributes two pre-existing KNOWNBUGs to one cause: `github_1175_9` casts `'A'` (65) and `github_1175_11` casts a constant-folded `strlen("Hello")` (5). **The obvious mechanism is refuted:** `--no-propagation` and `--no-simplify`, together and separately, leave the verdict SUCCESSFUL, so constant propagation is not what loses the check. | one-line reproducer above; `regression/esbmc/deref_constant_int_address` (KNOWNBUG) and `deref_nondet_int_address` (CORE, nondet address, caught today); `regression/esbmc/github_1175_{9,11}` (pre-existing KNOWNBUGs) | M8 triage; belongs to H-A10's `symex_dereference` obligation | **Fixed by #6554**, which compares object ids in the invalid-pointer check. Re-measured 2026-08-03: `(int *)65` now reports FAILED and `deref_constant_int_address` is CORE. |
| **R19** | **High (per-property false PASSED, non-default flag pair)** — **confirmed with a minimal reproducer** by H-B8, §15 M7; filed as **#6540** | **With `--multi-property --smt-during-symex`, a violable claim that is not the last property is individually reported as `✓ PASSED`.** Seven lines reproduce it: two non-trivial properties where the violable one comes first. ESBMC prints `✓ PASSED` for the violable claim, `Properties: 2 verified ✓ 2 passed`, and `VERIFICATION SUCCESSFUL`. Swapping the two assertions so the violable one is **last** restores `FAILED`, so the defect is positional. Neither flag alone loses the counterexample — `--multi-property` alone and `--smt-during-symex` alone both report FAILED — making this a flag *composition* defect like R17. This is I13 exactly as H-B8 hypothesised it: the per-claim solve reuses a `runtime_encoded_equationt` whose context stack still carries the preceding claim's state, so a non-final claim is discharged against the wrong formula. Worse than a verdict flip: the per-property report actively asserts the claim holds. | `oracle_flag_parity.py --b=--smt-during-symex` (3 corpus divergences: `github_1408`, `github_1890_1`, `github_2629`, all `--multi-property` tests); reproducer in #6540; **no portable regression pin** — see §15 M7 (CI) | **H-B8** | **Fixed by #6565**, which scoped the per-claim solves on the shared runtime solver — exactly the `push_ctx`/`pop_ctx` pairing this entry named. Re-measured 2026-08-03: both claim orderings now report the violable claim FAILED. |
| **R22** | **High (false SUCCESSFUL, default configuration)** — **confirmed with a minimal reproducer** by M8 triage, **confirmed and fixed**, §15 M8 (cont. 6) | **A shared write performed by a function's return-value assignment creates no interleaving point.** Six lines reproduce it: one thread runs `x = notify(); x = 2;` (`notify` returns `1`), another asserts `x != 1`. Default reports **SUCCESSFUL** — no schedule can observe the intermediate value, because no context switch is offered between the two writes. Three controls make the boundary exact: writing `x = 1; x = 2;` inline reports FAILED; splitting the call off the shared write (`int v = notify(); x = v; x = 2;`) reports FAILED; and inserting *any* other shared write between them (`x = notify(); g = 5; x = 2;`) reports FAILED. The value therefore reaches the equation — `x = notify();` alone reports FAILED, and an in-thread `assert(x == 1)` after it holds — so what is lost is the *scheduling point*, not the write. Not POR (`--no-por` unchanged), not the context bound (`--context-bound 10` unchanged), and not constant propagation (the split control propagates identically and still catches it). `x = notify()` lowers to a `FUNCTION_CALL` instruction carrying the lhs, so the write is performed by the `RETURN` case's `make_return_assignment` path; `execution_statet::symex_step` calls `analyze_assign(assign)` there **after** `symex_return(thecode)`, whose last statement is `cur_state->guard.make_false()`, and `analyze_assign` early-returns on a false guard. That is the same mistake #6558 fixed at `symex_goto`, and instrumentation confirms the reorder does exactly what the argument predicts — the `RETURN` step goes from `writes=0 cswitch=false` to `writes=1 cswitch=true`. **It is still not sufficient**, and the second half is now identified: `execute_guard` emits `assume(false)` and kills the interleaving whenever a switch is taken away from a thread whose guard is false, which `symex_return` guarantees at a return boundary. That is **#6558's defect at a second boundary** — the `last_transition.branch` arm chains the pre-branch guard for gotos and nothing does so for returns. Both halves must be fixed together; see §15 M8 (cont. 4) and (cont. 5). **Fixed in §15 M8 (cont. 6)**, where the two halves collapse into one change: a return parks its continuation exactly as a branch parks its sibling arm, so `symex_return` now records that parked path through the same hook `symex_goto` uses, and the existing branch arms in `execute_guard` and `preserve_last_paths` cover returns unchanged. Fixing only the first two halves exposed a third — the returning thread was marked `thread_ended` at the boundary — which the same change removes. | reproducer and controls above; `execution_statet::execute_guard`, `execution_state.cpp:712-755`; `execution_statet::symex_step` `RETURN` case, `execution_state.cpp:339-356`; `goto_symext::symex_return`, `symex_function.cpp:1041-1066`; `execution_statet::analyze_assign`, `execution_state.cpp:819-838`; `regression/esbmc-unix/symex_return_value_cswitch` (CORE, flipped from KNOWNBUG by the fix), `..._split` (CORE) and `..._resume` (CORE, added by the fix to pin the thread-survival half) | M8 triage; **H-A6**'s A6.2 completeness obligation | Done. All three pins are CORE and the whole `esbmc-unix` suite is clean. |
| **R18** | **High (false SUCCESSFUL, default configuration)** — **FIXED**, §15 M6 (fix); filed as **#6539**, fixed by **#6550** | **POR drops a racy interleaving when the write goes through a nested dereference.** `get_expr_globals` resolves *one* pointer level (`get_reference_set` on a single `dereference2tc`), so a write spelled `*(*gpp) = 1` is recorded against the intermediate pointer `gp` rather than its target `g`. A second thread writing `g` directly records `g`, the two keys do not alias, `check_mpor_dependency` returns *independent*, and the interleaving is pruned — **a real race missed in the default configuration, with no diagnostic**. Twelve lines reproduce it: writer does `*(*gpp) = 1`, `main` does `g = 2; seen = g;`, and `assert(seen == 2)` is reachable. Default reports **SUCCESSFUL**; `--no-por` reports FAILED. The mechanism is pinned by a decisive pair: with *both* threads using the nested form the race is found again (matching keys), while writer-nested/main-direct misses it. Splitting the nested access into `int *q = *gpp; *q = 1;` also restores detection, so the key depends on the syntactic nesting depth of the access rather than on the object touched. This is precisely the completeness direction H-A6's A6.2 names — a missed dependency — and it is **not** in the relation but upstream in the key construction feeding it. | `execution_statet::get_expr_globals`, `execution_state.cpp:868-918`; `check_mpor_dependency`, `:1050`; `mpor_set_conflicts`, `:231`; `regression/esbmc-unix/mpor_nested_deref_race` (KNOWNBUG) and `..._nopor` (CORE) | **H-A6**, **H-C4** | **Fixed in #6550** by following the chain and recording every shared object along it. The cost gate the entry called for was run: H-C4 agreement *rose* (258→259 on `--no-por`, 255→257 on `--state-hashing`) at 0 divergences, and the concurrency suite timing was unchanged (24.10 s vs 24.12 s). |
| **R16** | **Medium (incompleteness under a non-default flag)** — found by H-C2, §15 M5 (H-C2); **re-measured, all but one entry retired**, §15 M9 (R16) | **`--no-simplify` is not verdict-preserving: 10 corpus inputs where the default proves SUCCESSFUL and `--no-simplify` does not.** Nine report a spurious counterexample (`github_1174_{hex,lmod,oct,pass}`, `github_2341_3`, `github_2357_5`, `github_2566_1`, `github_785-2`, `realloc13`) and one returns UNKNOWN (`github_252`, under `--k-induction`). In every case the *default* leg matches the verdict the test's own `test.desc` expects, so the fault is in the `--no-simplify` configuration, not the default. Spot-confirmed on `github_2341_3`: `--no-simplify` reports a violated `assert(temp != NULL)` the default discharges. The noisy direction — P1 — but it means `do_simplify` is load-bearing for *correctness of the encoding*, not merely for formula size, which is not how an "optimisation" flag reads. | `oracle_flag_parity.py --b=--no-simplify` over `regression/esbmc` CORE | **H-C2** | **Re-measured, §15 M9 (R16).** Seven of the nine spurious-counterexample entries agree on a current binary. #6660, #6675 and #6676 all landed in the interval and each removed a modelling decision gated on `do_simplify` — the shape this whole list turned out to share — but the per-test attribution was not re-derived, so treat that as the likely cause rather than a measured one. Two more are not the simplifier's doing at all: `github_2357_5` and `github_2566_1` select `--ir`, and `github_562` (which the original list missed) selects `--fixedbv`; dropping the encoding flag makes both legs agree again in all three, so the oracle now excludes approximate encodings rather than comparing them. What is left of R16 proper is `github_252` — UNKNOWN under `--k-induction`, the sound direction. **Closed by #6781** (§15 M9 R16 closed): the forward condition could not close because the loop it reasoned about never exited under the flag, so the last R16 entry was a symptom of #6778 rather than a simplifier gap of its own, and its baseline entry is removed. The re-run also surfaced a divergence pair the original list did not contain, which is **R28**. |
| **R28** | **Medium (false SUCCESSFUL, non-default flag combination)** — **confirmed with a ten-line reproducer** by H-C2, §15 M9 (R16) | **`--no-simplify` can put a bounded loop back where the default folded it away, and with `--no-unwinding-assertions` the resulting truncation discharges every claim on the path in silence.** `calloc`'s model ends in `memset(res, 0, total_size)`. With a constant `total_size` the default folds that to the byte-wise `gen_value_by_byte` form and no loop survives; under `--no-simplify` it takes `__memset_impl`'s loop, which needs `total_size` iterations. `github_1257-memcleanup` pins `--unwind 1` and passes `--no-unwinding-assertions`, so the truncation becomes an `assume(false)` that cuts every path through `calloc`, and a genuine CWE-401 leak reports **SUCCESSFUL** where the default reports FAILED. Ten lines reproduce it — `p = calloc(100, 8); if (!p) abort(); *p = 5; g = p;` — and the discriminator is exactly `calloc`: the same leak spelled `malloc(800)` is caught under both legs, because no memset is involved. Three controls pin the mechanism rather than the leak logic: raising the bound to `--unwind 801` reports the leak again and names `dynamic_2_array`, the object `calloc`'s non-zero path allocates; leaving unwinding assertions **on** turns the same run into `unwinding assertion loop 3` in `__memset_impl`; and the sibling `github_1257-memsafety`, which differs only by keeping them on, is the same mechanism surfacing honestly as a bound complaint (`SUCCESSFUL` → FAILED, the sound direction). Not a new unsoundness in symex — it is the documented truncated-loop hazard — but the route to it is a flag pair a user would not expect to change loop *structure*, an `--unwind` calibrated against the default program silently under-covers the `--no-simplify` one, and nothing warns. **Wider than the flag pair, §15 M9 (H-C2 residue):** the lost constant is what bounds the loop at all, so with *no* `--unwind` the same mechanism does not truncate — it fails to terminate, and that is what H-C2's 206 "inconclusive" results are. Confirmed on `__memcpy_impl` (`string.c:284`) and on two loops outside `string.c` entirely — a test's own `myMemcpy` and `__ESBMC_atexit_handler` (`stdlib.c:38`) — so the rule is any loop whose trip count `do_simplify` folds, not the string models. | `oracle_flag_parity.py --b=--no-simplify`; `regression/esbmc/github_1257-memcleanup` and `github_1257-memsafety`; `calloc` in `src/c2goto/library/stdlib.c`; `__memset_impl` at `src/c2goto/library/string.c:304` | **H-C2** | Filed as **#6778**; the guard-fold gate is fixed by **#6781** (300 CORE tests, measured against an unpatched build: 49 non-terminating -> 18, 238 agreed -> 269, no new divergence). Cheapest honest fix is a diagnostic: an `--unwind` that truncates a loop while `--no-unwinding-assertions` is set should say so, since the two flags together turn every over-bound path into a vacuous proof. That does not address the unbounded form, which needs the trip count to survive `--no-simplify` rather than a warning. Baselined meanwhile — see `baselines/simplify-parity.txt`. |
| **R30** | **Medium–High (no verdict, default configuration)** — **confirmed with a five-line reproducer**, §15 M9 (R30) | **A loop whose trip count is statically determined but not *syntactically* a constant node never terminates, with no flags set.** `symex_goto` decides a branch by `is_false(new_guard)` (`symex_goto.cpp:23`), a syntactic test that only holds once `do_simplify` has folded the renamed guard to a literal, and nothing else in the default configuration can decide a loop exit: `--smt-symex-guard` asks the solver but is off, and the interval guard prunes only when the guard is provably *true* and never sets `new_guard_false`, by design. So the default configuration terminates exactly on the loops `simplify()` happens to fold. Five lines find one it does not — a pointer difference between two constant offsets into the same object: `int a[5]; int *p=&a[0], *q=&a[4]; unsigned n=q-p; for (unsigned i=0;i<n;i++) s++;` reaches **iteration 867405 in 20 s** and is still unwinding. The value is not in doubt: `assert(n == 4)` on its own proves SUCCESSFUL, the same program with the bound written `4` proves SUCCESSFUL, and adding `--smt-symex-guard` stops the loop at `iteration 4` in 0.004 s. This is R28's mechanism reached without `--no-simplify`, so the flag was never the cause — it only widened the set of guards that fail to fold. Not unsoundness: the tool returns no verdict rather than a wrong one, but a five-line program with a statically known bound hanging under default flags is a completeness defect a user meets as a hang. | `symex_goto.cpp:20-23`; `do_simplify` at `symex_assign.cpp:221`; reproducer above | **H-C2** | Filed as **#6779**, fixed by **#6783** (fold `&base[i] - &base[j]` to `i - j`, per C23 6.5.6p9). Same fix direction as R28's unbounded form: the exit decision should not rest on whether an *optimisation* folded the guard — either fold unconditionally for that decision, or fall back to the solver question `--smt-symex-guard` already implements. |
| **R17** | **High (false SUCCESSFUL, default configuration)** — found by H-C2, §15 M5 (H-C2); **FIXED**, §15 M5 (R17 root cause) | **An allocation the address space cannot lay out is encoded as a contradiction instead of a failed allocation, so the whole formula goes UNSAT and every assertion is discharged vacuously.** Found as `void *b = malloc(-4); assert(0);` returning **`VERIFICATION SUCCESSFUL`** under `--no-simplify --no-slice`, and first recorded as a flag-*composition* defect. It is not one, and the sign is not the trigger: `malloc(0xFFFFFFFFFFFFFFFCUL)` reproduces it under `--no-slice` alone. `--no-simplify` merely disabled the pre-existing negative-size guard (`do_simplify` is a no-op under it, so the guard never saw a constant) and `--no-slice` merely kept the otherwise-dead allocation in the equation. The real boundary is a layout limit and is exact: `1UL<<63` is fine, every size `>= 2^64 - 16` is vacuous, because `init_pointer_obj` asserts `end == start + size` **and** `end >= start` while `start` is past the NULL object at address 0 and aligned to `max_alignment()` (16). Reached in the corpus via `github_1631_compact`, whose `--compact-trace` sets `no-slice` implicitly (`command_line_options.cpp:410`). **No flag is needed at all**: an underflowing size such as `malloc(len - 4)` with `len < 4` widens to a huge `size_t`, and when the result is *used* the slicer keeps the allocation, so plain `esbmc file.c` goes vacuous. `default_underflow_malloc` pins that. | `smt_memspace.cpp` `init_pointer_obj`; fixed in `symex_mem`, `src/goto-symex/builtin_functions/memory_alloc.cpp`. `regression/esbmc/no_simplify_no_slice_huge_malloc` (KNOWNBUG → **CORE**), `default_underflow_malloc` (CORE, default flags), `no_slice_unrepresentable_malloc` (CORE, positive literal), `..._malloc` (CORE control) | **H-C2** | Fixed: classify the request on an unconditionally simplified copy so `--no-simplify` cannot blind it, and fail any allocation the address space cannot lay out by returning NULL, as real allocators do. Residual **R25** covers the symbolic-size form. |
| **R23** | **High (false SUCCESSFUL *and* false FAILED, default configuration)** — **confirmed with a two-line reproducer** by M8 triage, §15 M8 (cont. 7); filed as **#6589** | **Compound assignment narrows the right operand to the left operand's type before the operation.** C11 **6.5.16.2p3**: "A compound assignment of the form E1 op= E2 is equivalent to the simple assignment expression E1 = E1 op (E2), except that the lvalue E1 is evaluated only once". ESBMC violates that equivalence for every left operand narrower than `int`. `char b; b += a;` emits `!overflow("+", (signed int)b, (signed int)((signed char)a))` — the right operand cast to `char` — where `b = b + a` correctly emits `!overflow("+", (signed int)b, a)`. Both directions are reachable and both are wrong: with `b = 3, a = INT_MAX`, `b += a` reports **SUCCESSFUL** (the overflow claim is unfalsifiable, a **missed bug**) while `b = b + a` reports FAILED; and with `char b = 100; int a = 256`, `b /= a` reports **FAILED "division by zero"** because the divisor narrows to `(char)256 == 0`, where C gives `100 / 256 == 0` and gcc/UBSan agree. Not bitfield-specific — `char`, `short`, struct members and bitfields all reproduce; the discriminator is *narrower than the promoted type*, not the member/bitfield spelling. `github_162_fail` is where it was found, and its claim is vacuous for exactly this reason — but that entry is a *wrong test* independently of R23, see §15 M8 (cont. 8). **Frontend, not goto-symex**, so it is outside §2.3's scope, but it is a soundness defect in extremely common C. **Fixed, §15 M8 (cont. 8).** | `clang_c_convertert::get_compound_assign_expr`, `clang_c_convert.cpp:4258-4343`, specifically the unconditional `gen_typecast(ns, rhs, lhs.type())`, together with `goto_convertt::remove_assignment`, `goto_sideeffects.cpp:1714-1870`, which took the operation's type from `expr.op0()`. `regression/esbmc/compound_assign_narrow_overflow`, `..._explicit` (control) and `compound_assign_narrow_divzero`, all CORE | M8 triage | Done. The frontend records clang's `getComputationResultType()` on the side effect; `remove_assignment` performs the operation there and converts the result back on assignment. |
| **R24** | **Medium (spurious counterexample, default configuration)** — **confirmed with a reproducer** by M8 triage, §15 M8 (cont. 10); **FIXED**, §15 M8 (R24) | **`memset` does not constrain a struct's bitfield padding bits, so a type-punned read of the object is partly nondeterministic.** For `struct { int x : 12, y : 8; } s;`, `memset(&s, 0, sizeof s); s.x = -1; s.y = -1;` then reading `*(int *)&s` gives a value whose low 20 bits are correct — `(v & 0xFFFFF) == 0xFFFFF` verifies — but whose 12 padding bits are unconstrained: `(v >> 20) == 0` **fails**. gcc gives `0x000fffff` exactly, so the declared fields are laid out right and only the `memset`'s effect on the bits above them is lost. This is the direction an over-approximation produces (a false alarm, never a missed bug), and it is reachable with **no flags at all**, which is what separates it from the four flag-inadequacy entries triaged alongside it. Explains `github_732-1-1`, whose `sizeof(s) == 4` and `s.y == -1` assertions both hold and only whose type-punned assertion fails. | `regression/esbmc/bitfield_padding_memset`, `..._fields`, `..._fill` and `..._fail`, and `regression/esbmc/github_732-1-1` — all CORE, the first and last flipped from KNOWNBUG by the fix | M8 triage | Fixed: the optimised `memset` charged each member `type_byte_size()` bytes, which over-counts a bitfield, so a 4-byte struct's trailing member was written with zero bytes and kept its old value. `gen_value_by_byte` now declines any struct with a sub-byte member and leaves it to `__memset_impl`, whose byte-wise model gets the padding right. |
| **R25** | **High (false SUCCESSFUL, default configuration)** — found while root-causing R17, §15 M5 (R17 root cause); **FIXED**, §15 M5 (R25) | **The R17 vacuity is also reachable through a *symbolic* allocation size, and no flag is needed.** `size_t n = nondet_size(); __ESBMC_assume(n >= 0xFFFFFFFFFFFFFFF0UL); char *b = malloc(n); if (b) b[0] = 1; assert(0);` reported **`VERIFICATION SUCCESSFUL`** on default flags — the pointer is used, so the slicer keeps the allocation. The R17 fix could not see it: no constant is available at symex time. Worse than R17's shape, because the address-space constraint does not merely kill the path — `end == start + n` with `end >= start` silently *constrains the program variable `n`*, so **every** symbolic allocation quietly discarded its top 16 sizes, not just ones an assumption forced there. | `smt_memspace.cpp` `init_pointer_obj:409-421`; fixed in `symex_mem`. `regression/esbmc/symbolic_unrepresentable_malloc` and `no_slice_symbolic_unrepresentable_malloc` (CORE), `symbolic_malloc_bounds_preserved` (CORE, anti-vacuity), `force_malloc_success_unrepresentable` (KNOWNBUG, residual) | R17 root-causing | Fixed: give the object size zero on the branch where the request does not fit, so it is always layable, and return NULL there. Under `--force-malloc-success` the bound is stated as an assumption instead — branching to NULL reintroduces the case split that flag exists to remove, and cost 22 s → >200 s on `github_1352-*-32bit`. That leaves the residual pinned above. |
| **R31** | **High (false SUCCESSFUL, default configuration)** — found by extending the H-A6 census immediately after R29's fix closed it at 21/21, §15 M9 (H-A6 re-census) | **An `address_of` in front of the aggregate step defeats MPOR's access resolution.** `int **pp = &s.p; **pp = 1;` against a concurrent `g = 2` reports **SUCCESSFUL** by default and **FAILED** under `--no-por`, both under Bitwuzla and Z3. This is **not** punning: `&s.p` is a well-defined `int **`, so the false SUCCESSFUL is on ordinary C. The boundary is syntactic in R29's way — copying the pointer to a local first (`int *lp = *pp;`) restores detection — which places the gate in the resolution, not the value set. `record_aggregate_held_target` *is* entered (the inner `dereference2t` is not a `symbol2t`), so the loss is further down. `--show-symex-value-sets` pins it exactly: `c:@pp = { <s, 0, 8, struct S { signed int * p; }> }` names the **struct symbol**, with the suffix erased into a byte offset, while the entry that holds the answer — `c:@s.p = { <g, 0, 1, signed int> }` — is present and correct. Resolving `**pp` therefore needs the descriptor's constant offset mapped back to `.p` before the second lookup can find it. The information is not missing, only unaddressable. **The component this row first named was wrong**, and instructively: it read the local-copy boundary as placing the gate "in the resolution, not the value set" and pointed at `mpor_lock_array_key` as the precedent. The local copy works because symex's dereference pass has already rewritten `*pp` into the `member2t` `s.p` by the time `value_sett::assign` records `lp`, so the member arm keys `c:@s.p` directly; MPOR hands the value set a *synthetic* `dereference2t` it built itself, which never passed through that rewrite and so lands in the dereference arm, where the member survives only as a byte offset. The boundary separates *rewritten by symex* from *raw*, not MPOR from the value set, and the fault was in the value set on both sides. A struct-to-struct punning shape (`((struct B *)&a)->q`) prunes identically but is strict-aliasing UB and carries no soundness claim | `regression/esbmc-unix/mpor_aggregate_ptr_race_addrof` (KNOWNBUG → **CORE**) with `..._addrof_local` (CORE control), the pairing R29 was filed under, joined by `..._addrof_offset`, `..._addrof_nested`, `..._addrof_union`, `..._array_decay` and `..._addrof_merged`, one per arm of the descent and each pinned by its own mutant, plus `..._addrof_locked` (CORE, the passing direction) and `mpor_aggregate_ptr_zero_size_element` (CORE, pinning the `esize > 0` guard against a `BigInt` abort), both added when the coverage gate blocked the first cut | H-A6 | **Fixed**, §15 M9 (R31 fix): `get_value_set_rec`'s dereference arm now walks the descriptor's constant byte offset back into a field path and asks again under it, accumulating in bits as `member_offset_bits` does so the walk inverts the one that built the descriptor. The unrefined lookup stays, so the change only ever adds objects to a value set. The cheap alternative the first diagnosis suggested — a `simplify()` in `resolve_pointer_target` — was built and measured, and fixes **none** of the six shapes in either placement: there is no constant to fold, because the member was erased before the pointer's value set was ever written |
| **R32** | **High (false SUCCESSFUL, default configuration)** — found by probing the one gap R31's fix section had just declared witnessless, §15 M9 (R32) | **A symbolic offset erases the aggregate step exactly as a constant one did, and R31's walk has nothing to spell back out.** `int *a[2] = {&g, &g}; ap = &a[i];` with `i` nondeterministic and assumed in bounds, then `**ap = 1` against a concurrent `g = 2`, reports **SUCCESSFUL** by default under both Bitwuzla and Z3 and **FAILED** under `--no-por`. Replacing `a[i]` with `a[1]` reports FAILED, so the symbolic index is the whole discriminator. Well-defined C — the index is assumed in range — so unlike the punning shape this carries a full soundness claim. `--show-symex-value-sets`: `c:@ap = { <a, *, 8, signed int * [2]> }`, the `*` being the unset offset, against `c:@a[] = { <g, 0, 1, signed int> }` which holds the answer. R31's `offset_paths` requires `offset_is_set` and skips, so the unrefined lookup of `c:@a` misses `c:@a[]` and returns empty — read by every consumer as "points at nothing" | `regression/esbmc-unix/mpor_aggregate_ptr_race_symbolic_offset` and `..._symbolic_struct_member` (CORE, one per arm of the unknown route), `..._symbolic_offset_locked` (CORE, the passing direction) and `..._array_decay` the constant-index control | H-A6 | **Fixed**, §15 M9 (R32 fix): a second walk, `collect_typed_paths`, takes every path of the dereferenced type instead of the one an offset selects, and `offset_paths` dispatches on whether the descriptor carries an offset. No size is consulted there, there being no offset to place -- which incidentally keeps a target inside a variable-length element reachable, where the offset walk has to drop it. Monotone for the same reason R31's walk is, and measured at +1.1% worst case on a verdict-matched comparison. The array arm alone did not pin it: a mutant descending into member 0 only left all 21 tests passing |
| **R33** | **High (false SUCCESSFUL, default configuration)** — found by code review of R31's fix, not by the census, §15 M9 (R33) | **A constant member offset and a constant element offset would not compose, so the descriptor arrived with no offset at all.** `struct S { long pad; int *v[2]; }; int **pp = &s.v[1];` then `**pp = 1` against `g = 2` reported **SUCCESSFUL** by default, **FAILED** under `--no-por`. Each half works alone — `&s.v[0]` (base 8, index 0) and a member at a nonzero offset both detect the race — and only the composition failed, which is what makes it a distinct defect from R31 rather than another shape of it. The index arm of `get_reference_set_rec` added a constant element offset only when the base offset was **zero**, and otherwise fell to the unknown-offset branch and cleared `offset_is_set`; R31's walk then had nothing to spell back out. Reaching byte offset 16 by two members instead (`&s.v.b`) detects the race, which pins the route rather than the offset as the discriminator. The member arm one screen below already composed with `o.offset += offset_in_bytes` | `regression/esbmc-unix/mpor_aggregate_ptr_race_member_index` (CORE), with `..._addrof_offset` and `..._array_decay` the two halves that always worked | code review of R31 | **Fixed**: the index arm composes when the base offset is set (`o.offset += index_offset`) instead of requiring it to be zero. Identical on the old domain — `offset_is_zero()` already implied `offset_is_set`, and adding to a zero offset is assignment — so only the previously-abandoned case changes. Increases precision rather than widening: the descriptor gains a definite offset where it used to carry none |
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
**Closed, §15 M6.** H-C4 run (258/0 and 255/0). R11 answered and **superseded by
R18**: a confirmed default-configuration false SUCCESSFUL where POR prunes a race
reached through a nested dereference. A6.2 is refuted by that counterexample at
Tier B — the specced Tier-A model would have passed it. A6.1 and A6.3 discharged
by inspection. Both rows M6 carried forward are now closed: **R6** got its
witness and its fix (#6785, §15 M9), and **A6.4** — the active-row reset in
`calculate_mpor_constraints` — is discharged by the run-order invariant the
engine now checks in release (§15 M6 (A6.4)).

**M7 — End-to-end scenarios and regression pinning (1 wk).** H-C2, H-C3, H-C5,
H-C6, H-C7 wired as a scheduled CI job; H-B8. *Artefact:* the oracle job + a
per-oracle baseline of known divergences (each triaged to a filed issue or a
justified waiver — **an untriaged divergence is a blocker, not a baseline**).
**Closed, §15 M7.** H-C3 (1269/0), H-C5 (1360/0), H-C6 (0 violations, 44 of 163
tests exercising the relation), H-B8 (1358/3 → **R19**) and H-C7 (326 compared,
1 diverged) are run; `.github/workflows/symex-oracles.yml` wires all of them with
per-leg baselines, all of which are now triaged and cited.

**M8 — Previously-reported bugs and regression cases (0.5 wk, continuous).**
Convert every historical goto-symex issue with a reproducer into a Tier-A or
Tier-B case; start from the tree's own `KNOWNBUG` inventory. *Artefact:* a
`regression/esbmc/symex_regressions/` index mapping issue → harness.
**Closed, §15 M8.** All 27 goto-symex `KNOWNBUG`s surveyed and indexed; **9 of
27 never reach a verdict on this toolchain**, so their KNOWNBUG status is
uninformative. The index lives in §15 M8 rather than a new directory of copied
tests. R20 (#6544) and R21 (#6545) attribute five of the twelve unattributed
wrong-verdict entries. The Linux re-run (§15 M8 cont. 3) discharges the
"re-measure the masked ones" half: masking drops to 5/28, six tests rejoin the
inventory, and two of them produce **R22**. Triage of the resulting ten
(§15 M8 cont. 7-10) closes the inventory: **R23** (found through `github_162_fail`)
and **R24** (bitfield padding under type punning), both since fixed, with seven of
the eleven entries turning out to be wrong tests rather than defects — six of
those fixed and retired.

**M9 — The Tier-B remainder (0.5 wk).** H-B6 and H-B7, the two rows §7.2 never
scheduled under a milestone. **Closed, §15 M9.** I9 discharged on the real engine
with no defect found; the entry records why the obvious mutant (deleting the
union) is undetectable and the meaningful one (intersecting it) is caught. H-B7
then closed three of §7.3's seven rows and sharpened the rest, and H-A8 — the
row it left live — is closed by a third entry on a real `runtime_encoded_equationt`.
What remains is not backed by a live harness: H-A2's guard algebra is a
cross-document dependency, and H-A6 is refuted-and-fixed (R18, then R29, R31,
R32 and R33) rather than discharged — its census re-runs 22/22 from the tree,
but an enumeration is not the completeness the row claims, and this sentence
previously cited a 21/21 that a later round showed to be over-stated.

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
    ├── oracle_flag_parity.py                 H-C1, H-C2, H-C3, H-C4, H-C5, H-B8
    ├── oracle_unwind_monotonic.py            H-C6
    ├── oracle_claim_parity.py                H-C7
    ├── oracle_common.py                      shared: args, run, verdict, baseline
    ├── baselines/<leg>.txt                   triaged divergences, one per leg
    └── drift_check.py                        transcription-drift guard
.github/workflows/symex-oracles.yml           scheduled Tier-C job
```

`<area>` ∈ `{ssa, merge, mergequeue, slice, unwind, mpor, frame, eqctx,
lookup, refalias}` — one area per harness family, matching §7. `<nn> = 00` is
the M0 template.

H-C1, H-C2, H-C3, H-C5 and H-B8 are the *same* relation — verdict(A) ==
verdict(B) over a corpus — so §15 M5 replaced the planned shell scripts with one
parameterised `oracle_flag_parity.py`, invoked as `--b=--no-slice`,
`--b=--no-simplify`, `--a=--bitwuzla --b=--z3`,
`--b=--no-interval-symex-guard` and `--b=--smt-during-symex`. It builds each
argument list through `regression/testing_tool.py`'s `TestCase` rather than
re-parsing `test.desc`, so a sweep invokes each input exactly as `ctest` does.
H-C4 is two invocations of the same script (`--b=--no-por`,
`--b=--state-hashing`) over `regression/esbmc-unix`. H-C6 and H-C7 need their own
drivers: the first is monotonicity across bounds rather than a two-configuration
comparison and has to classify a FAILED by the violated property; the second
compares *per-claim* verdicts, where the two interfaces do not enumerate the same
property set (§15 M7).

`--baseline <file>` takes a list of already-triaged divergences: the script exits
non-zero only on a divergence *not* in the file, and prints `STALE-BASELINE` for
a listed test that starts agreeing again, so a fixed defect cannot keep its
exemption silently. Baseline entries must cite a finding id or issue — §11.3's
rule that an untriaged divergence is a blocker, not a baseline, is enforced by
review, not by the script.

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
| Tier C oracles | **scheduled**, `.github/workflows/symex-oracles.yml` — nightly `37 2 * * *` for C1/C2/C3, weekly `41 3 * * 0` for C4/C5/C6/B8 — plus `workflow_dispatch` | 120 min per leg, mirroring `sanitizers.yml`; `continue-on-error` during bring-up |
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
| **D8** | Tier-C oracle scripts + scheduled workflow | `scripts/verification/symex/`, `.github/workflows/symex-oracles.yml` | M5, M7 — **delivered except H-C7**, §15 M7 |
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

> **Re-measured 2026-08-05 — the table above is stale; see §15 M9 (G-remeasure).**
> **All seven probes now pass: G1–G7 are closed**, so §13.6's WI-1, WI-2 and
> WI-3 are all done. The blocker for including an ESBMC header is no longer a
> missing STL facility but **G9** below. Probe G7 at `--std c++23`, not c++20 —
> `std::unreachable` is a C++23 name and the OM gates it correctly.
>
> | ID | Facility | 2026-07-27 | 2026-08-05 |
> |---|---|---|---|
> | G1 | `<type_traits>` (4 probes) | absent | **closed** |
> | G2 | `<shared_mutex>` | header absent | **closed** — `src/cpp/library/shared_mutex` exists |
> | G3 | `std::initializer_list` as a template | absent | **closed** |
> | G4 | `iterator_traits::difference_type` | absent | **closed** |
> | G5 | `std::this_thread::yield` | absent | **closed** |
> | G6 | `<compare>` `strong_ordering` | absent | **closed** |
> | G7 | `std::unreachable` | absent | **closed** — #6631; needs `--std c++23` |
>
> **G9 — `std::map` with an incomplete `mapped_type`.** `irept` declares
> `typedef std::map<irep_idt, irept> named_subt` (`src/util/irep/irep.h:41`),
> naming `irept` as the `mapped_type` from inside its own definition. The OM's
> `<map>` instantiates the node eagerly and rejects it: `field has incomplete
> type 'mapped_type' (aka 'irept')`. This is the **sole** remaining error on
> `#include <goto-symex/renaming.h>`. The asymmetry with line 38's
> `std::vector<irept> subt`, which parses, is the standard's:
> [container.requirements.general] grants incomplete-type support to `vector`,
> `list` and `forward_list` **only** (N4510, adopted for C++17), so libstdc++
> accepting `named_subt` is a QoI extension the OM is not obliged to match.
> Closing G9 means either matching that extension in the OM's `map`, or changing
> `named_subt` — and the second is an ESBMC-wide change, not an OM one.

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
| ~~**WI-1**~~ | ~~`<shared_mutex>` operational model (G2)~~ | — | M0 | **Done** — closed in-tree; re-measured §15 M9 (G-remeasure) |
| ~~**WI-2**~~ | ~~`<type_traits>` completion (G1) + `<compare>` `strong_ordering` (G6) + `std::unreachable` (G7)~~ | — | M0–M1 | **Done** — G1 and G7 by #6631, G6 in-tree |
| ~~**WI-3**~~ | ~~`std::initializer_list` (G3), `iterator_traits::difference_type` (G4), `this_thread::yield` (G5), `aligned_storage[_t]`~~ | — | M1 | **Done** — `renaming.h` now stops only at G9 |
| **WI-4** | **Tier B′ pilot**: a reduced harness that `#include`s `renaming.h` and drives the real `level1t`. **Gate:** must parse *and* verify in < 60 s. If it does not, record the negative result in §13.3 and keep Tier A — do not force it. **Now blocked on G9 alone**, not on a missing header. | ~1 wk | M4 | Removes transcription drift for C1 |
| **WI-5** | E1 container reference/iterator invalidation modelling | ~2–3 wk | M6 | Stating R3/H-A9 on the real class; benefits all STL verification |
| **WI-6** | E2 native 2-safety / equivalence mode | unscoped | post-M7 | Promotes H-C1/H-C2 from sweep to proof |

**Critical path:** ~~WI-1 → WI-2 → WI-3~~ — retired; all three are done
(§15 M9 (G-remeasure)). What stands between here and WI-4 is **G9**, not this
chain. WI-4 is a gated experiment with an explicit
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
   the first is being removed. *(a) Parsing* — **re-measured 2026-08-05/06 and
   now down to one cause.** G1–G7 are closed (§15 M9 (G-remeasure)), so the
   operational model is no longer the obstacle; what remains is **G9**, `irept`'s
   `std::map<irep_idt, irept> named_subt` naming an incomplete `mapped_type`.
   Measured on the real target rather than projected: `--parse-tree-only` over
   `src/goto-symex/execution_state.cpp` emits exactly one distinct error,
   `field has incomplete type 'mapped_type' (aka 'irept')`. A backlog of eight
   has become a single decision (§13.2). *(b) Tractability* — the measurements
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
8. **Mode C (dead-code) proofs on ESBMC's own C++ sources.** `AGENTS.md`/
   `CLAUDE.md` require a C-Live proof for any patch to `src/**` that adds a
   branch, discharged by instrumenting the branch with `__ESBMC_unreachable()`
   and verifying the file. That is a corollary of item 1 and inherits its
   blocker: the file cannot be parsed, so the instrumentation cannot be
   verified. Confirmed on the R29 fix — the patched
   `src/goto-symex/execution_state.cpp` stops at G9 alone. **What stands in for
   it**, and what a report must say instead of claiming Mode C: an *empirical*
   reachability witness — an input that demonstrably drives the new branch and
   changes an observable. For R29's `dereference2t` arm that is three regression
   tests whose verdict flips (§15 M9 (R29 fix)). This is weaker than C-Live:
   it shows the branch is reachable on the inputs tried, not that it is
   reachable in general, and it offers nothing for **C-Dead**, where the
   obligation is to show a *removed* branch was unreachable — a negative no
   finite set of inputs can establish. C-Dead on `src/**` therefore rests
   entirely on the implicit discharge route (a cited issue or failing test
   proving the branch was live).

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
allocation.

**Root-caused and fixed on 2026-08-02, and the flag-pair framing above is a
symptom rather than the defect.** Three corrections to what M5 recorded. First,
the trigger is not the negative sign: `malloc(0xFFFFFFFFFFFFFFFCUL)` — a
positive literal — reproduces it, and under `--no-slice` *alone*. What M5 read
as a composition defect was `--no-simplify` disabling the pre-existing
negative-size guard (`do_simplify` is a no-op under that flag, so the guard
never saw a constant) while `--no-slice` kept the otherwise-dead allocation in
the equation. Second, the boundary is exact, and it is a layout limit rather
than a magnitude one: `1UL<<63` is fine, every size `>= 2^64 - 16` is vacuous.
`smt_memspace.cpp` (`init_pointer_obj`) lays each object out as
`[start, start + size]`, asserting `end == start + size` *and* `end >= start`,
while `start` is constrained past the NULL object at address 0 and aligned to
`max_alignment()` (16). For `size >= 2^64 - 16` that set has no model, so the
*whole formula* is UNSAT and every assertion in the program is discharged
vacuously. Those constraints are asserted unconditionally, so the nondet
malloc-failure branch does not rescue it.

Third, and the reason this outranks the rest of the M5 haul, **no flag is
needed**. The corpus pointed this out: verifying the fix flipped `github_1091`,
a KNOWNBUG since 2023 that §16 listed as unattributed in the missed-bug
direction. It runs on plain `--unwind 1` and allocates
`malloc(strlen(filename) - 4)`, which underflows to a huge `size_t` whenever
`strlen(filename) < 4`; the result is *used*, so the slicer has no reason to
drop it and the default configuration goes vacuous exactly as `--no-slice` did.

**That test is no longer the evidence, though it is how the evidence was
found.** #6592 landed on master in the meantime, promoted `github_1091` to CORE
at `--unwind 5`, and required an array-bounds violation — which the
compound-assignment fix produces on its own, so the test now passes with or
without the allocation fix and no longer discriminates. Re-measured on merged
master: at `--unwind 5` it FAILS either way, but at `--unwind 1` it is *still*
SUCCESSFUL without the allocation fix. The default-configuration claim therefore
stands, and `default_underflow_malloc` pins it directly instead of relying on a
corpus test whose flags move. R17 should be read as a default-configuration
false SUCCESSFUL that Tier C found
through a flag oracle, which is a stronger result for §7.4 than the composition
reading it was first given.

The fix (`symex_mem`) classifies the request on an unconditionally simplified
copy, so `--no-simplify` no longer blinds it, and fails any allocation the
address space cannot lay out — returning NULL, which is what real allocators do
for such a request. `no_simplify_no_slice_huge_malloc` flips KNOWNBUG → CORE,
and `no_slice_unrepresentable_malloc` pins the positive-literal form that the
entry above claimed did not reproduce.

**A residual survived as R25**, since fixed — see the next entry. The same
vacuity was reachable through a *symbolic* size
(`__ESBMC_assume(n >= 0xFFFFFFFFFFFFFFF0UL)` then `malloc(n)`), which a
constant-only symex check cannot see.

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

### M5 (R25) — 2026-08-02, R25 fixed

**Result: the symbolic form of R17 is fixed, it was a default-configuration
false SUCCESSFUL too, and the first fix that worked had to be thrown away for
costing 10x.**

R25 is worse than R17 in scope. R17 needed a constant the size could be folded
to; R25 needs nothing. Adding a use of the pointer is enough to stop the slicer
dropping the allocation, and then plain `esbmc file.c` reports SUCCESSFUL on a
reachable `assert(0)`. And the mechanism is not merely a dead path: because
`init_pointer_obj` asserts `end == start + n` and `end >= start` over the
*symbolic* `n`, **every** symbolic allocation in every program silently excluded
its top 16 sizes. A program whose bug lives only there was unprovable rather
than unproven.

**The fix.** Give the object size zero on the branch where the request does not
fit — a zero-size object always lays out, as `init_pointer_obj`'s own comment
notes — and return NULL on that branch, folding the condition into the existing
allocation guard.

**The measurement that changed the design.** The first version applied that
branch unconditionally, and `github_1352-fail-32bit` / `-success-32bit` went
from 22 s to over 200 s — a timeout. Both run `--force-malloc-success`, and that
flag's whole purpose is to delete the malloc-returns-NULL case split;
reintroducing it under a different name reinstated the cost across every
allocation in a loop nest. Under that flag the bound is now stated as an
assumption instead. The same executions are excluded as before the fix, so
`force_malloc_success_unrepresentable` stays KNOWNBUG — but they are excluded
*visibly*, in the equation, rather than as an emergent property of an
unsatisfiable layout constraint. Post-fix the pair runs in 23.8 s and 29.2 s.

**Anti-vacuity.** `symbolic_malloc_bounds_preserved` pins the direction the fix
could most plausibly have broken: with `10 <= n <= 100`, `b[n]` must still be
caught out of bounds. Clamping a size that *does* fit would silently weaken
every heap bounds check, and no other test in the sweep would have noticed.

Verified with Bitwuzla and Z3 on both directions, and against the 476-test
allocation subset (every regression source mentioning `malloc`/`calloc`/
`realloc`/`alloca`); the two residual failures there,
`esbmc-unix/03_boundedBuffer` and `esbmc-unix/github_5565_getopt_long_optarg`,
reproduce on master.

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

### M6 — 2026-07-30

**Result: R11 answered, and in answering it a confirmed soundness bug in the
default configuration — R18. POR drops a real race when the write goes through a
nested dereference. §10 said to budget for "there is a real gap"; there is.**

**H-C4 first, because it was one command.** `oracle_flag_parity.py` already takes
the flag pair, so the POR and state-hashing legs needed no new code:

| Leg | agreed | diverged | inconclusive | skipped |
|---|---|---|---|---|
| verdict(default) == verdict(`--no-por`) | 258 | **0** | 13 | 9 |
| verdict(default) == verdict(`--state-hashing`) | 255 | **0** | 9 | 16 |

Corpus `regression/esbmc-unix` CORE (280 of 355 dirs), 30 s cap. **This is weaker
evidence than it looks and must not be quoted as "POR is sound".** Zero
divergences over a corpus says nothing about mechanisms the corpus does not
exercise — R6 needs two states differing only in the L1 activation of a
*recursive* local, and nothing here establishes that any esbmc-unix input has
that shape. The R18 witness below makes the point concrete: it is a twelve-line
program that the 258-input sweep did not contain.

**R18, and how it was found.** R11 asked whether an `unknown` value-set entry
forces a conservative dependency. Reading `get_expr_globals` answers *no* — the
`dest` loop skips anything that is not an `object_descriptor2t` over a
`symbol2t`, with no fallback — but that is not the reachable defect. Writing
targeted racy programs and diffing default against `--no-por` produced this:

| writer body (main does `g = 2; seen = g;`) | default | `--no-por` |
|---|---|---|
| `g = 1;` | FAILED | FAILED |
| `*gp = 1;` | FAILED | FAILED |
| `int *q = *gpp; *q = 1;` | FAILED | FAILED |
| integer round-trip `(int *)(unsigned long)&g` | FAILED | FAILED |
| **`*(*gpp) = 1;`** | **SUCCESSFUL** | FAILED |
| **`*(*pp) = 1;`** (local double pointer) | **SUCCESSFUL** | FAILED |

`get_expr_globals` resolves **one** pointer level, so `*(*gpp) = 1` is recorded
against the intermediate `gp` rather than its target `g`. `main`'s direct write
records `g`, the keys do not alias, `check_mpor_dependency` reports independent,
and the interleaving is pruned. A decisive pair pins the mechanism as
wrong-key rather than nothing-recorded: with **both** threads using the nested
form the race is found again (the keys match each other), while
writer-nested/main-direct misses it. So the MPOR key depends on the syntactic
nesting depth of an access rather than on the object accessed.

**The harness §7.1 specified for H-A6 would not have found this.** A6.1–A6.4 are
properties of `check_mpor_dependency` — symmetry, completeness, read-read,
transitive closure — and a Tier-A model of that relation over nondet read/write
bitmasks satisfies all four while this bug is live, because the relation is
correct and the *keys handed to it* are wrong. Discharged instead at Tier B
against the real engine, where the defect is reachable. Second time in two
milestones that a specced Tier-A model would have verified the wrong component
(cf. §15 M5's `store_elided` blindness); the pattern is that a harness aimed at a
decision procedure must also police the data feeding it.

Pinned by `regression/esbmc-unix/mpor_nested_deref_race` (KNOWNBUG, observed
`VERIFICATION SUCCESSFUL`) and `..._nopor` (CORE, FAILED under `--no-por`). Not
fixed: resolving pointer chains to a fixed point, or making `mpor_keys_may_alias`
treat a pointer key as aliasing everything its value set reaches, both widen the
dependency relation and cost interleavings — #6480 already tuned this relation
for exactly that reason, so a fix needs the H-C4 sweep re-run as a cost gate.

**Still open in M6.** A6.1/A6.3/A6.4 (the relation's own algebraic properties)
have no harness; only A6.2 was exercised, and by counterexample. R6
(state-hashing on L0 names) remains untested for want of a corpus input with the
recursive-local shape — H-C4's clean state-hashing leg does **not** discharge it.
No `--no-mpor` flag exists, so `--no-por` cannot separate MPOR pruning from the
rest of POR; attributing R18 to MPOR rests on the code path, not on a flag.

### M6 (cont.) — 2026-07-30, M6 closed

**Result: A6.1 and A6.3 discharged by inspection, A6.4 left open with the
suspect operation named, and R6 advanced from a hypothesis about
`make_assignment` to a named unproven claim plus a stated witness requirement —
recorded as a negative result, since no witness was found.**

**A6.1 (symmetry) — discharged by inspection, no harness built.** Every clause of
`mpor_keys_may_alias` is symmetric in its arguments (`a == b`, then
`is_index2t(a) == is_index2t(b)`, then a base-symbol comparison), and
`check_mpor_dependency`'s three clauses map onto each other under j↔l: WW is
self-symmetric, and the "j reads what l wrote" and "j writes what l reads"
clauses exchange. So `dep(j,l) == dep(l,j)` holds structurally. A Tier-A model to
rediscover this would have cost more than reading it.

**A6.3 (read-read never forces a dependency) — discharged by inspection.** There
is no read-read clause; the omission is explicit and is the point of the
optimisation.

**A6.4 (transitive closure) — NOT discharged.** `calculate_mpor_constraints`
copies the previous chain, resets the active thread's row to −1, sets
`[active][active] = 1`, then takes a two-hop step: `new[j][active] = 1` when some
`l` has `dependency_chain[j][l] == 1` and `check_mpor_dependency(active, l)`. Two
observations, short of a proof. The `res == 0` path deliberately does **not**
overwrite, preserving an existing dependency — the conservative direction, so
harmless. The one operation that *removes* relations is the active-row reset, so
that is where an unsound step would live, and settling it needs the MPOR paper's
chain definition rather than inspection. This is the open half of H-A6.

**R6 — mechanism pinned, no witness, and the negative result is worth more than
the hypothesis was.** `generate_hash` combines `generate_l2_state_hash()` — an
ordered map of **L0 base name → `crc(value)`**, since
`state_hashing_level2t::make_assignment` keys on `to_symbol2t(lhs_sym).thename`,
which carries no L1 activation — with each thread's `source.pc->location_number`.
It does **not** include call-stack depth. Two states at the same pc, with the same
L0→value map, at different recursion depths therefore fingerprint identically and
are *not* bisimilar: they resume into different continuations. That makes the
comment at `reachability_tree.cpp:420` — "equal fingerprints are bisimilar and any
collision prunes soundly" — the precise unproven step, and false in the recursion
case. It belongs to the R9 family of unproven-direction claims, now with a
citable location rather than a suspicion.

Four targeted programs (constant-valued recursive local so the L0 entry matches
across depths; depths 2 and 4; the shared write at the recursion bottom) reported
FAILED under `--state-hashing` exactly as under the default, so **no
verdict-changing prune was produced**. The reason is structural, not a missing
trick: for pruning to change a verdict the collision must sit on the path to the
*only* buggy interleaving, and a two-thread race on one global admits many racy
interleavings, an early one of which is found before any duplicate state is
reached. A witness needs the bug reachable *exclusively* behind the colliding
state. R6 stays open with that requirement written down, in place of the original
"H-C4 parity sweep quantifies the current gap" — the clean sweep has already
shown it will not.

**M6 closed.** Delivered: the H-C4 parity report (both legs), an R11 verdict
superseded by R18 with a witness and a regression pin, and A6.1/A6.3 by
inspection. Carried forward: A6.4, R6, and the absent `--no-mpor` flag that would
let a sweep separate MPOR from the rest of POR. **R18 was subsequently fixed —
see M6 (fix).**

### M7 — 2026-07-30, M7 partial

**Result: three more oracle legs run, the scheduled job wired with a baseline
mechanism, and one new confirmed defect — R19, a per-property false `PASSED`.**

| Leg | agreed | diverged | inconclusive | skipped |
|---|---|---|---|---|
| **H-C3** bitwuzla vs z3 | 1269 | **0** | 72 | 90 |
| **H-C5** `--no-interval-symex-guard` | 1360 | **0** | 71 | 0 |
| **H-B8** `--smt-during-symex` | 1358 | **3** | 68 | 2 |
| **H-C6** unwind monotonicity | — | **0 violations** | 16 | — |

**R19 — the per-property report asserts a violable claim holds.** All three H-B8
divergences are `--multi-property` tests whose default leg matches the verdict
their `test.desc` expects, so the fault is in the `--smt-during-symex`
configuration. Minimised to seven lines: two non-trivial properties with the
violable one **first**. ESBMC prints `✓ PASSED` for the violable claim,
`Properties: 2 verified ✓ 2 passed`, then `VERIFICATION SUCCESSFUL`. Swapping the
assertions so the violable one is last restores FAILED, so the defect is
positional; and neither flag alone loses the counterexample, making it a
composition defect like R17. This is I13 as H-B8 hypothesised it — a per-claim
solve reusing a `runtime_encoded_equationt` whose context stack still carries the
previous claim's state — and it is worse than a verdict flip, because the report
actively claims the property was verified. Pinned by
`multi_property_smt_during_symex` (KNOWNBUG) and `..._last` (CORE), the second of
which pins the ordering dependence rather than the flag pair.

**H-B8 moved from Tier B to Tier C, and that is why it found anything.** §7.2
specced it as a unit test over "the same program with and without
`--smt-during-symex`". As one more argument to `oracle_flag_parity.py` it ran over
1363 real inputs instead of a handful of hand-written ones, and the three
divergences are all `--multi-property` inputs — a flag no hand-written Tier-B case
would have thought to combine. Fifth harness to be discharged somewhere other
than its specced tier.

**H-C6 needed its own driver and one non-obvious distinction.** Monotonicity —
FAILED at bound k must persist at every k' > k — is not a two-configuration
comparison, so `oracle_unwind_monotonic.py` runs a ladder of bounds. The
distinction that carries it: with unwinding assertions on (the default) a FAILED
at a small bound is usually the *unwinding assertion itself*, which correctly
disappears once the bound covers the loop. Classifying verdicts by the violated
property rather than by the FAILED line is what keeps the oracle from reporting a
divergence on every loop-bearing test in the corpus.

**Its headline number is 0 violations; its honest number is 44.** Of 163
loop-bearing tests only **44 produced a real counterexample at any bound**, so
the other 119 satisfy monotonicity vacuously. The script prints `exercised`
alongside `violations` for exactly this reason — a monotonicity oracle over
programs that never fail is a very fast way to prove nothing.

**The scheduled job exists.** `.github/workflows/symex-oracles.yml` wires seven
parity legs plus the unwind ladder, nightly for the cheap ones and weekly for the
rest, `actionlint`-clean. Two implementation notes worth keeping: the default
`scripts/build.sh` solver set has **no Z3**, so the solver-parity leg needs `-C`
(the competition set) or it would silently compare bitwuzla against itself; and
job-level `if` cannot see `matrix`, while job-level `env` can, so the
nightly/weekly gate is computed once into `RUN_LEG` rather than repeated per step.

**Baselines make the job runnable without being ignorable.** `--baseline <file>`
exits non-zero only on a divergence *not* already triaged, and prints
`STALE-BASELINE` when a listed test starts agreeing — so a fixed defect cannot
keep its exemption unnoticed. Populated for the two legs that diverge: eleven
entries for H-C2 (R16, R17) and three for H-B8 (R19), each annotated with its
finding. Verified end to end: the H-B8 leg reports `3 diverged, baselined 3 of 3`
and exits 0.

**Still open in M7.** **H-C7** (per-claim `--multi-property` versus individual
`--claim N` runs) is not written, and it is now the most interesting one left: R19
is an H-C7-shaped defect found accidentally by H-B8, which suggests the per-claim
comparison has more to say *without* `--smt-during-symex`. It also carries H-B3's
per-claim residue from M5. The workflow keeps `continue-on-error` until a full
Linux run confirms the baselines, since they were populated on macOS where 68-72
inputs reach no verdict.

### M7 (cont.) — 2026-07-30, M7 closed

**Result: H-C7 built and run. After five rounds of correcting the oracle it finds
exactly one real per-claim disagreement — R19 (#6540), which H-B8 had already
found. The other four it first reported were artefacts of the oracle, not defects
in ESBMC.**

| | |
|---|---|
| Relation | per-claim verdict under `--multi-property` == verdict of the matching `--claim k` run |
| **compared** | **326** (≥ 2 claims, within the 12-claim cap) |
| **diverged** | **1** — `github_1655`, i.e. R19 |
| skipped | 984 — 528 with fewer than 2 claims, 433 over the claim cap, 23 with no per-claim report |

**The first run reported 5, and 4 of those were mine.** They are recorded here
because the mistake is the interesting part: `--claim N` does **not** verify claim
N in isolation. The memory-safety checks stay in the formula, so a `--claim N` run
that reports FAILED may have violated something else entirely — on `github_192`,
claims 1, 3 and 5 all report FAILED on one unrelated `dereference failure` at
line 4, while the divergence was reported against line 18. Taking the run verdict
as the claim's verdict manufactured four disagreements. A FAILED now counts for a
claim only when the violated property sits at that claim's own location.

**Fixing that exposed a second flaw in the same oracle.** With inconclusive claims
discarded, the individual side reads "all PASSED" when it in fact knows nothing,
so the aggregate check fired on 24 tests in the opposite direction. It now
requires that every claim reached a conclusive verdict *and* that
`--multi-property`'s failures lie at enumerated claim locations — it also reports
memory-safety and unwinding-assertion properties that `--show-claims` never
lists, which the individual side can never match.

**What survives is one divergence, and it is not new.** `github_1655` carries
`--smt-during-symex` in its own flags, so H-C7 independently rediscovers R19 at
the claim level, naming the claim and the location. Baselined against #6540 in
`baselines/claim-parity.txt`; the leg is therefore green rather than permanently
red, without any untriaged exemption.

**The honest reading of H-C7 so far is that it has found nothing the flag-parity
legs did not.** Its argued advantage — catching a mis-reported claim that a
compensating failure hides from the run verdict — remains real but unexercised on
this corpus, and 433 of 1310 candidates are skipped by the claim cap, so the
argument is not yet tested where it would most likely pay off.

**Three oracle bugs of my own, each of which produced confident nonsense first.**
Worth recording because all three are the same failure mode — an oracle that
cannot tell "no disagreement" from "no comparison":

1. **Keying on the printed comment.** `--show-claims` puts the comment and the
   claim expression on separate lines; `--multi-property` concatenates them
   (`assertion` vs `assertion (_Bool)0`). Every generic-comment claim looked
   ABSENT. Fixed by keying on source location.
2. **Assuming the two interfaces enumerate the same properties.** They do not:
   `--multi-property` reports properties `--show-claims` never lists (unwinding
   assertions among them), and one location can carry three reported properties
   against one listed claim. So the comparison is now per-location *and* only
   where each side has exactly one entry, with the rest counted as not
   comparable; the aggregate check covers what that drops.
3. **Treating an empty per-claim report as "all passed".** Passing
   `--multi-property` to a test whose flags already contain it makes ESBMC emit
   no report at all, which scored as agreement and produced a false mismatch on
   the R19 companion test. Now `--multi-property` is stripped from the inherited
   flags and an empty report is a skip, not a pass.

Validated against a known answer rather than only on the corpus: the oracle flags
`multi_property_smt_during_symex` (R19's KNOWNBUG) at both the per-location level
— naming line 8, `FAILED vs PASSED` — and the aggregate level, and does **not**
flag its `_last` companion.

**M7 closed.** D8 is delivered: `oracle_flag_parity.py`, `oracle_unwind_monotonic.py`
and `oracle_claim_parity.py`, all wired in `.github/workflows/symex-oracles.yml`,
with every baseline entry citing a finding. Carried forward: the 433 tests over
the claim cap, which is a real coverage gap rather than a tuning choice — a test
with 38 claims costs 40 runs — and H-C7's unexercised advantage over the
run-verdict legs.

### M8 — 2026-07-30, M8 partial

**Result: the goto-symex `KNOWNBUG` inventory surveyed and indexed. A third of it
is uninformative on this toolchain — those tests pass without the documented bug
being exercised at all, which is the mechanism, not an accident.**

27 `KNOWNBUG` tests live in `regression/esbmc` (21) and `regression/esbmc-unix`
(6). Running each with its own flags and classifying what actually happened:

| Outcome | Count | What the KNOWNBUG status means |
|---|---|---|
| Wrong verdict | 16 | the documented defect, exercised — the useful inventory |
| **`ERROR: PARSING ERROR`** | **7** | never reaches symex |
| Crash (uncaught exception / SIGSEGV) | 2 | never reaches a verdict |
| No verdict / timeout | 2 | one is R14's `SYMEX_INVARIANT` stop, by design |

**`KNOWNBUG` passes when the output fails to match for *any* reason**, so a test
that no longer parses stays green and says nothing about whether its defect
survives. Nine of 27 (33 %) are in that state here. `fam_false_2` and
`fam_true_4` are the clearest case: both fail on `main() {` — implicit `int`,
which this Clang rejects — so neither reaches the flexible-array-member modelling
they were written for. The M4 (H-B1) entry identified this hazard for R14's own
pin; this measures it across the inventory. Caveat: the parse failures are
toolchain-dependent (macOS/Clang here), so on CI some may parse and rejoin the
useful 16 — the *masking* is what generalises, not the count.

**Index.** Four of the 16 are this plan's own pins, each carrying the finding it
documents; the rest are pre-existing and unattributed:

| Test | Observed | Finding |
|---|---|---|
| `double_assign_check_local_array` | invariant stop | **R14** (I10 violated on a real input) |
| `no_simplify_no_slice_huge_malloc` | SUCCESSFUL | **R17** (vacuous path, flag pair) |
| `mpor_nested_deref_race` | SUCCESSFUL | **R18** (POR drops a race) |
| `multi_property_smt_during_symex` | SUCCESSFUL | **R19** (per-property false PASSED) |
| `03_inf2`, `github_1091`, `github_1175_9`, `github_1175_11`, `github_159_postdecrement_fail`, `github_162_fail`, `github_1626-no-free`, `03_circular_reduce` | SUCCESSFUL, expected FAILED | unattributed — the missed-bug direction |
| `40_stack_64_inner_scope_true`, `github_2572_2`, `github_426_2`, `github_426_3`, `github_426_4` | FAILED, expected SUCCESSFUL | unattributed — spurious counterexample |
| `github_248` | UNKNOWN, expected SUCCESSFUL | unattributed |

**Delta since the snapshot (2026-08-02).** Two rows above are now stale, both
from the R17 fix. `no_simplify_no_slice_huge_malloc` is CORE, and `github_1091`
was promoted independently by #6592 — it was in the unattributed missed-bug row
and R17 does affect it (still SUCCESSFUL at `--unwind 1` without the fix), but
its current `--unwind 5` configuration no longer discriminates, so R17's own
claim is pinned by `default_underflow_malloc` instead. One pin replaces them:
`force_malloc_success_unrepresentable` (SUCCESSFUL), the residual R25's fix
leaves under `--force-malloc-success`. The inventory's point survives the churn
— the masking is what generalises, not the count.

**One hypothesis tested and rejected.** `03_circular_reduce` is a concurrency test
expecting FAILED that reports SUCCESSFUL, so R18 (POR pruning a racy
interleaving) was the obvious suspect. It reports SUCCESSFUL under
`--context-bound 2`, under `--no-por`, with both, and with neither — POR is not
the cause and this does not extend R18. Recorded so the next reader does not
re-run it.

**Artefact deviation.** §10 asked for a `regression/esbmc/symex_regressions/`
directory mapping issue → harness. Building it would mean copying 27 tests that
already exist, creating two copies to keep in step; and the four pins this plan
added already sit beside their subjects, which is the repo's own convention. The
index above is the artefact instead.

**Still open in M8.** The 12 unattributed wrong-verdict KNOWNBUGs are not yet
root-caused, and the 9 masked ones need re-running on Linux before their status
means anything. Neither is a Tier-A/Tier-B conversion in the §10 sense — the
inventory turned out to need triage first.

### M8 (cont.) — 2026-07-30

**Result: two of the twelve unattributed KNOWNBUGs root-caused to one new finding,
R20, with a one-line reproducer. A control run stopped the triage reaching the
opposite conclusion.**

**The control mattered more than the result.** `--show-claims` reports 0 claims for
`github_1175_9`, which reads as "the dereference check was never generated" — a
goto-conversion gap, out of §2.3's scope, case closed. It is wrong: a plain
`int *p = 0; return *p;` *also* reports 0 claims and still FAILS, because
dereference checks are raised during symex, not listed in the GOTO program. The
claim count says nothing here. Any triage of the remaining ten should run that
control first.

**R20.** Probing the boundary turned a 7-line test into one line:

| pointer | verdict |
|---|---|
| `(int *)0` | FAILED ✓ |
| `(int *)nondet_ulong()` | FAILED ✓ |
| `(int *)(unsigned long)&x` | SUCCESSFUL ✓ (a valid round-trip) |
| **`(int *)65`**, read | **SUCCESSFUL ✗** |
| **`(int *)65`**, write | **SUCCESSFUL ✗** |

Only a constant non-null integer address escapes the invalid-pointer obligation
that null and nondet addresses both receive. That single cause covers
`github_1175_9` (casts `'A'`) and `github_1175_11` (casts a constant-folded
`strlen("Hello")`), so two entries leave the unattributed list.

**And the obvious mechanism is refuted.** Constant propagation folding the cast
was the natural suspect; `--no-propagation` and `--no-simplify`, separately and
together, all still report SUCCESSFUL. Recorded as refuted rather than left as a
plausible-sounding guess — the real mechanism is not yet known.

Pinned by `regression/esbmc/deref_constant_int_address` (KNOWNBUG) and
`deref_nondet_int_address` (CORE), the second pinning the boundary so a fix that
merely stops checking nondet addresses cannot pass. Filed as #6544.

**Still open in M8.** Ten unattributed wrong-verdict KNOWNBUGs. Two carry a
caveat worth recording before someone spends time on them: `github_159_postdecrement_fail`
expects FAILED for a post-decrement that forms an out-of-bounds pointer without
dereferencing it, and `github_162_fail` expects an overflow report for a 3-bit
bitfield where the generated claim is `arithmetic overflow on add` over `int`
operands that genuinely cannot overflow. Both may be wrong tests rather than
defects, which is a distinct outcome from either "fixed" or "still broken".

### M8 (cont. 2) — 2026-07-30

**Result: three more KNOWNBUGs attributed to one cause, R21 — the exact
complement of R20. Seven of the original twelve remain unattributed.**

`github_426_2`, `github_426_3` and `github_426_4` all round-trip an address
through `uintptr_t` arithmetic and expect SUCCESSFUL; all three report FAILED.
Probing found the boundary is a single operation:

| address-derived integer expression | verdict |
|---|---|
| `u = (uintptr_t)&s` | SUCCESSFUL ✓ |
| `u += 4; u -= 4` | SUCCESSFUL ✓ (additive round-trip tracked) |
| `u = u * 1` | SUCCESSFUL ✓ (folds) |
| `o = 4; o *= 2; p = (int *)(o + v)` | SUCCESSFUL ✓ (pure-integer multiply) |
| **`u *= 2; u -= (uintptr_t)&s`** | **FAILED ✗** |

**`offsetof` is address-derived, which is why `github_426_2` fails despite
multiplying only an offset.** It expands to `(size_t)&((S *)0)->m`, so
`u = offsetof(struct S, y); u *= 2;` takes the same path as multiplying an
address; substituting a literal `4` for the `offsetof` makes the identical
program verify. That resolved the one case that looked like a counterexample to
the rule, rather than leaving it as an exception.

R21 is a *spurious counterexample*, the noisy direction, so it is P1 — and it sits
opposite R20, which accepts a computed address it should reject. Both live in
integer↔pointer conversion, from the two different directions.

Pinned by `ptr_int_multiply_roundtrip` (KNOWNBUG) and `ptr_int_additive_roundtrip`
(CORE), the second so a change that stops tracking additive round-trips cannot
make the pair pass. Filed as #6545, framed so that "document the limitation" is an
acceptable resolution.

**Attribution scoreboard for M8.** Of the twelve wrong-verdict KNOWNBUGs that were
unattributed after the survey: two are R20 (#6544), three are R21, and
`03_circular_reduce` was tested against R18 and rejected. Seven remain, two of
which (`github_159_postdecrement_fail`, `github_162_fail`) still look like wrong
tests rather than defects.

### M8 (cont. 3) — 2026-07-31, the Linux re-run

**Result: the masked third of the KNOWNBUG inventory re-measured on Linux, as M8
asked. Masking drops from 9/27 to 5/28, six tests rejoin the useful inventory,
five of them are wrong verdicts, and two of those five are one new finding —
R22.**

The M8 survey ran on macOS/Clang and flagged its own caveat: seven tests never
parsed, so their `KNOWNBUG` status said nothing about whether their defect
survived. Re-running all 28 with each test's own flags here:

| Outcome | Linux (28) | macOS (27) |
|---|---|---|
| Wrong verdict | 21 | 16 |
| `ERROR: PARSING ERROR` | 4 | 7 |
| Crash / no verdict | 1 (R14's `SYMEX_INVARIANT` stop, by design) | 2 crashes + 2 no-verdict |

The four still-unparsed are `fam_false_2`, `fam_true_4` (both `main() {` —
implicit `int`), `github_197` and `05_pfscan-1.0_01`. **The masking is
toolchain-dependent exactly as predicted**, which is the point worth keeping: a
`KNOWNBUG` verdict is only meaningful on a toolchain that parses the test, and no
single platform's survey settles the inventory.

**The six that were uninformative on macOS and reach a verdict here**, classified
against their own `test.desc`:

| Test | Expected | Linux | Direction |
|---|---|---|---|
| `03_wait_notify` | FAILED | SUCCESSFUL | missed bug — **attributed below** |
| `03_wait_notify2` | FAILED | SUCCESSFUL | missed bug — **attributed below** |
| `github_732-1-1` | SUCCESSFUL | FAILED | spurious counterexample |
| `github_1091` | FAILED | SUCCESSFUL | missed bug |
| `github_2513_6` | SUCCESSFUL | FAILED | spurious counterexample |
| `linking-7` | `ERROR` | FAILED | out of §2.3 scope (symbol linking, not symex) |

**Both `wait_notify` tests are attributed, and not to the obvious suspect.** They
share an `ecsc.h` whose `notify_event()` is
`{ __ESBMC_atomic_begin(); return 1; __ESBMC_atomic_end(); }` — the
`__ESBMC_atomic_end()` sits after a `return` and is unreachable, so the atomic
section is entered and never left and no further context switch is offered.
Balancing that one function makes both tests report FAILED, as their `test.desc`
expects. This is a defect in the tests' own header rather than in symex, but the
ESBMC-side observation transfers: **an unterminated atomic section silently
disables all remaining interleaving, with no diagnostic** — the same
false-SUCCESSFUL shape as R17, reached through a modelling error the tool does
not report. Worth a check at thread end; not filed as a finding here because the
input is at fault.

**R22 came out of the control, not the hypothesis.** The first guess was that the
leaked atomic explained a minimal probe too. It did not: the *no-atomic* control
`x = notify(); x = 2;` is equally SUCCESSFUL, while the inline `x = 1; x = 2;`
FAILS. That flipped the investigation onto the call, and the boundary is sharp —
splitting the call off the shared write, or putting any other shared write
between the two, restores the bug. So the return value lands in the equation and
only the scheduling point is missing. Full characterisation and the refuted
mechanisms in §9.2's R22 row.

**A partial mechanism, recorded as partial.** `analyze_assign` is called at
`RETURN` *after* `symex_return` falsifies the path guard, and `analyze_assign`
early-returns on a false guard — textually the same mistake #6558 fixed at
`symex_goto`. Reordering the two calls and rebuilding leaves the reproducer
SUCCESSFUL, so that is not the whole story and the fix is not landed. Recorded
as an unfinished trail rather than a plausible-sounding cause, per the standard
M8 (cont.) set for R20.

Pinned by `regression/esbmc-unix/symex_return_value_cswitch` (KNOWNBUG) and
`..._split` (CORE, dual-solver agreed), the second so a change that stops
generating interleaving points for ordinary shared writes cannot make the pair
pass.

**Still open in M8.** The seven unattributed wrong-verdict KNOWNBUGs from the
macOS survey, plus the three newly-revealed ones above (`github_732-1-1`,
`github_1091`, `github_2513_6`).

### M8 (cont. 4) — 2026-07-31, R22 mechanism split in two

**Result: R22 is two independent defects stacked, not one. The first is
confirmed and its fix is verified to work at its own level. The second is
*not* in the DFS — an intermediate reading that said so was refuted by measuring
frame identity instead of inferring it from frame counts — and is now pinned to
the state carried across a switch taken at a function-return boundary. Nothing
is committed as a fix, because half a mechanism is not a fix.**

M8 (cont. 3) left R22 with a plausible cause — `analyze_assign` running after
`symex_return` falsifies the guard — and the honest note that reordering the two
calls did not flip the reproducer. Instrumenting both halves settles which part
of that was right.

**Part 1 — confirmed, and the fix works at its own level.** Logging the `RETURN`
case directly:

| RETURN of `notify()` in `x = notify(); x = 2;` | `thread_last_writes` | `has_cswitch_point_occured()` |
|---|---|---|
| as shipped (`analyze_assign` after `symex_return`) | 0 | **false** |
| `analyze_assign` moved before `symex_return` | 1 | **true** |

So the guard-falsification argument is exactly right, and the one-line reorder
does create the interleaving point that is missing today. The verdict is
nevertheless still SUCCESSFUL — with `--no-por` and with `--context-bound 10`
too. A fix that is provably necessary and demonstrably insufficient is worth
recording as such rather than landing.

**Part 2 — where it is not.** With the point present, `--symex-trace` shows 8
interleavings and **none** places the observer between the two writes; the inline
control reaches exactly that schedule as its interleaving 4. One scheduler
observation holds up: `decide_ileave_direction` scans forward from
`active_thread + 1` and then *backward from `active_thread` itself*, so when no
higher-numbered thread is schedulable it re-selects the running thread — logged
as `decided=1 (active=1)` at the return boundary. Staying is legitimate; it
defers the alternative to a backtrack.

**The deferred alternative is taken, so the DFS is not where the schedule is
lost.** Logging each schedulability decision with the active thread's program
point identifies the frames directly, rather than inferring them from frame
counts:

```
DBG SCHED: tid=1 viable=true dfs=true active=1 activePC=248 tidPC=248   (choose to stay)
DBG SCHED: tid=1 viable=false ...        active=1 activePC=248          (backtrack: t1 explored)
DBG SCHED: tid=0 viable=true  dfs=true   active=1 activePC=248 tidPC=395 (switch to main)
```

`activePC=248` is `notify`'s `END_FUNCTION` — after the return-value write and
before `x = 2`. The DFS backtracks to exactly that frame and schedules the
observer from it. So the interleaving that should expose the write **is
generated**, and the violation is still not found.

That relocates the remaining defect from the scheduler to the **state carried
across the switch**. The distinguishing feature of that frame is that the
returning thread's guard is already false (`symex_return`) and its return state
is parked in `merge_state_map` awaiting `END_FUNCTION`; the inline control has no
such parked state at the corresponding point. Whether the observer resumed from
that frame reads the pre- or post-`x = 2` value of `x` is the next measurement.

**Refuted this round, each by a run rather than by argument.** Thread creation
order (`obs_first.c`); the observer being a spawned thread rather than `main`
(`main_obs.c` reproduces with two threads total, and its inline twin FAILS);
the observer thread not existing yet at the return (3 of the 4 `writes=1`
returns in the original probe occur with all 3 threads live); POR; and the
context bound. Constant propagation was already refuted in M8 (cont. 3) by the
split control.

**A methodological note worth keeping.** The DFS reading came from correlating
`exploration_frames.size()` with thread ids across a backtrack, which looked
conclusive and was not: frame counts do not identify frames. Logging the active
thread's `location_number` alongside each decision cost one rebuild and inverted
the conclusion. Any further triage in this area should identify frames by
program point, never by depth.

The pins from M8 (cont. 3) are unchanged and still red/green in the right
directions. The next step is to measure which value of `x` the observer reads
when resumed from the `activePC=248` frame, and — if it reads the post-`x = 2`
value — why the parked return state in `merge_state_map` lets a later write
overtake the switch.

### M8 (cont. 5) — 2026-07-31, R22 mechanism complete

**Result: R22 is fully explained, and its second half is #6558's defect at the
function-return boundary — the same `parent_guard` chaining fix, applied to
branches and not to returns. The fix shape follows from the mechanism.**

Picking up the measurement M8 (cont. 4) asked for: what the observer reads when
resumed from the `activePC=248` frame.

**It reads the right value.** Logging every claim after renaming and
simplification, the call form and the inline control each produce **exactly one**
`assertion x != 1` that folds to *false* — the observer does see `x == 1`. The
call form also produces three claims that fold to *true* against the inline
form's two, which is the extra schedule, not a lost one. So symex reaches the
violating state in both.

**Two hypotheses died on the way.** The claim is not lost to constant folding —
`claim()` drops a claim entirely when it simplifies to true, which would have
been a clean story, but the violating claim survives folding in both forms. Nor
is the *thread* guard contradictory: at the false claim `cur_state->guard` prints
as `constant_bool true`. Both were worth checking and neither is the answer.

**The vacuity is in the global guard.** `goto_symext::assertion` applies
`cur_state->global_guard` on top of `cur_state->guard`, and the fully-guarded
claim is byte-identical in the two forms — `not(execution_statet::\guard_exec)`.
What differs is the equation feeding it: 4 surviving assignments in the call
form against 6 inline. Same claim, same guard expression, different constraints
on `guard_exec`, and the call form's copy solves UNSAT.

**`execute_guard` is where it happens** (`execution_state.cpp:712-755`). When a
switch is taken *away from* a thread whose path guard is false, it emits an
assumption of that false guard into the equation and falsifies the incoming
thread's guard:

```cpp
parent_guard = threads_state[last_active_thread].guard.as_expr();
if (is_false(parent_guard) || is_cur_state_guard_false(parent_guard))
{
  if (active_thread != last_active_thread)
    target->assumption(guard2tc().as_expr(), parent_guard, ...);  // assume(false)
  cur_state->guard.make_false();
  return;
}
```

The in-code comment calls this "the only way to bail out of evaluating a
particular interleaving early right now". `symex_return` ends in
`cur_state->guard.make_false()`, so a switch at a return boundary hits this arm
every time: the interleaving that would expose the write is generated, entered,
and then assumed away.

**This is #6558 at a second boundary.** The `last_transition.branch` arm
immediately above exists because a constant-true goto killed the fall-through
guard and "poison[ed] the suffix with assume(false)" — the identical failure, at
the branch boundary, fixed by chaining the *pre-branch* guard instead of the
falsified one. Returns never got the same treatment. That makes R22's two halves
one story: `analyze_assign` at `RETURN` runs after the guard dies (so no switch
point is offered), and `execute_guard` treats a switch at that boundary as a
dead interleaving (so offering one is not enough).

**Fix shape, for the next iteration.** Record the pre-return guard on
`last_transition` at `RETURN` — the state `symex_return` parks in
`merge_state_map` already holds it — and chain it in `execute_guard` exactly as
the branch arm chains `last_transition.parent_guard`. Both halves must land
together: neither alone moves the verdict, which is why the two earlier
single-change attempts each looked like a failed fix.

### M8 (cont. 6) — 2026-07-31, R22 fixed

**Result: R22 is fixed, and the fix is smaller than the two-half diagnosis
suggested. A return parks its continuation exactly as a branch parks its
sibling arm, so the machinery #6558 built for branches generalises to returns
verbatim — once the return actually records what it parked.**

(cont. 5) prescribed recording a pre-return guard on `last_transition` and
chaining it in `execute_guard` "exactly as the branch arm chains
`last_transition.parent_guard`". Implementing that literally works, but writing
it exposed the redundancy: `execute_guard`'s branch arm keys off
`last_transition.branch`, which exists only because `symex_goto` calls
`record_branch_sibling` after pushing its snapshot. `symex_return` pushes a
snapshot the same way — `merge_state_map[end_of_function]` — and simply never
told anyone. So the fix is for `symex_return` to call the same hook, and both
existing branch arms then cover returns with no new conditions:

- `goto_symext::symex_return` calls the hook after `merge_state_list.emplace_back`.
- The `RETURN` case runs `analyze_assign` *before* `symex_return` and records
  `parent_guard` (the two changes (cont. 3) and (cont. 5) identified).
- Nothing else changes. The hook and its payload are renamed
  `record_branch_sibling` → `record_parked_path` and `branch_resultt` →
  `parked_patht`, since returns now set the field too and `branch` would lie.

**A third defect surfaced while fixing the first two, and the same change kills
it.** With only the (cont. 3) + (cont. 5) changes applied, the verdict is
correct but `preserve_last_paths` marks the *returning* thread `thread_ended`
at every return boundary it switches away from. The reason is structural: it
preserves the current path only when the guard is live (false at a return) and
otherwise only a recorded branch sibling (absent for a return), so `pp` comes
out empty and it takes its assume(0) branch. That silently truncates the
writer. Recording the parked path fills `pp`, so the thread survives *and* its
continuation is re-parked by `restore_last_paths` against post-switch level2
values — which is what #6571 established as necessary at branches, and is
equally necessary here.

**Measurements.** Verdicts A/B against a stashed master build:

| test | master | patched |
|---|---|---|
| `symex_return_value_cswitch` | SUCCESSFUL (the bug) | FAILED |
| `symex_return_value_cswitch_split` (control) | FAILED | FAILED |
| `symex_return_value_cswitch_resume` (new) | SUCCESSFUL (the bug) | FAILED |

The first flips KNOWNBUG → CORE and fires on the intended claim (`assertion
x != 1`, `observer`, thread 2). The debug counter for the truncation above goes
2 → 0.

`..._resume` is added here because the first two cannot see the third defect:
both detect the violation *at* the boundary, so a truncated writer still
reports FAILED. It closes that gap by requiring the writer to come back —
`x = notify(); x = 2; assert(z != 7);` against an observer that sets `z = 7`
only under `x == 1`. The observer can only take that branch at the return
boundary, and the writer only reads `z` after resuming from it, so the
assertion fires only if a switch is offered there *and* the returning thread
survives it.

**A probe that was vacuous, recorded because it nearly stuck.** The first
measurement of that counter used `--verbosity debug` and returned 0, which
looked like "the defect does not occur". `--verbosity` takes `N` or `module:N`;
a bare word silently enables nothing, and the whole run emitted 2 lines. Under
`--verbosity symex:9` the same run emits 63 lines and the counter reads 2. Any
`grep -c` over ESBMC log output is worthless until the channel is shown to emit
at all.

**Regression.** The full `esbmc-unix` label (557 tests) is clean apart from
`03_boundedBuffer` and `01_pthread60`, both of which are **pre-existing**
near-timeout THOROUGH tests rather than verdict changes: each produces the
expected `VERIFICATION FAILED` but exceeds the harness's hard 120 s cap on this
machine with or without the patch. Timed A/B against a stashed master build:

| test | master | patched |
|---|---|---|
| `03_boundedBuffer` | 1m56.9s | 1m53.1s |
| `01_pthread60` | 1m40.2s | 1m43.3s |

Adding interleaving points at returns was the obvious perf risk and it did not
materialise at a measurable scale.

### M8 (cont. 7) — 2026-07-31, the unattributed ten

**Result: five of the ten unattributed KNOWNBUGs resolved. One is a new
finding, R23, that is larger than the entry it came from; three are wrong
tests, two of which are now fixed and retired; one is narrowed to a specific
subsystem. All ten were first re-measured — none moved under R22's fix.**

**Re-measurement first.** R22's fix had just landed and two entries
(`03_inf2`, `03_circular_reduce`) are concurrency tests, so every entry was
re-run before triage. All ten still produce the same wrong verdict; none is a
stale record and none was fixed as a side effect.

**R23 — `github_162_fail`, and the plan's own note about it was wrong.** The
entry was carried as "may be a wrong test: the claim is `arithmetic overflow on
add` over `int` operands that genuinely cannot overflow". The operands cannot
overflow, but that is the *defect*, not an exoneration. The claim is

```
!overflow("+", (signed int)b.a, (signed int)((signed _ExtInt(3))a))
```

— the right operand is truncated to the bitfield width *before* the addition,
so an unbounded `int` becomes a value in −4…3 and the claim is unfalsifiable.
C11 6.5.16.2p3 makes `E1 op= E2` equivalent to `E1 = E1 op (E2)`, and ESBMC
disagrees with itself on the two forms. See §9.2's R23 row for the full
characterisation; it is a frontend defect, reaches `char`/`short`/struct
members as well as bitfields, and produces a **missed overflow** in one
direction and a **spurious division by zero** in the other (`char b = 100; int
a = 256; b /= a;`, which gcc + UBSan confirm is well defined and equals 0).

**Three wrong tests.** Each was settled by a control, not by reading:

| test | why it is wrong | disposition |
|---|---|---|
| `github_159_postdecrement_fail` | `(b--)->d.c` dereferences the *old*, in-bounds `&Q[0]`; the only UB is *forming* `Q-1`. `b--;` alone is SUCCESSFUL, `b--; *b` is FAILED, and `*(b--)` is SUCCESSFUL — ESBMC checks dereferences, not pointer formation, and no sibling in the 8-test family requires otherwise. | left KNOWNBUG: its intent needs a pointer-formation checker that does not exist, which is a maintainer's call, not a test edit |
| `github_1626-no-free` | the sibling `github_1626` catches a use-after-free; this variant comments the `free` out, leaving only the `printf("%s", *ptr)` defect, which needs the opt-in `--printf-check`. The test never passes it. | **fixed** — flag added, KNOWNBUG → CORE |
| `40_stack_64_inner_scope_true` | the family's own model (`40_stack_64_primitives_true`) counts an `int` as `4 * 8 = 32` units and a `char` as `8`; this test's comments say `int a; // 8` and it asks for `--stack-limit 16`. Two `int`s need 64, and the measured threshold is exactly 64 (63 FAILED / 64 SUCCESSFUL, and a `char` rewrite brackets 15/16). | **fixed** — limits corrected to 64/63 and the misleading comments with them, KNOWNBUG → CORE |

`40_stack_64_inner_scope_false` shared the bug in the opposite direction: it
asserts FAILED at limit 15 against a program that needs 64, so it passed
**vacuously** and would have kept passing however wrong the accounting became.
Its bound is now 63, one below the real threshold, which is what an anti-vacuity
twin is for (§6.1 r5).

**`github_732-1-1` narrowed, not closed.** Of its three assertions, `sizeof(s)
== 4` and `s.y == -1` both hold; only `*(int *)&s == 0x000fffff` fails. The
field *values* are right and the *memory image* is wrong, so this is bitfield
layout under type punning — unrelated to R23, which is about the type an
operation is performed in.

**Still open in M8.** Five unattributed: `03_inf2`, `03_circular_reduce`,
`github_2572_2`, `github_1091`, `github_2513_6`, plus `github_732-1-1` narrowed
as above and `github_248` (UNKNOWN under `--k-induction` on mutual infinite
recursion, which may be plain incompleteness rather than a defect).

### M8 (cont. 10) — 2026-07-31, the inventory closes: flags first, defect last

**Result: the remaining five checked flags-against-property before assuming a
defect, as (cont. 9) recommended. Four were the test asking for the wrong
thing; exactly one is a real defect, R24. All ten are now resolved.**

The lens has two halves, and both earned their keep:

- an **under-approximation** (a bound) can hide a bug but cannot invent one, so
  it is only a candidate for entries that expect FAILED and get SUCCESSFUL;
- an **over-approximation** can invent a counterexample but cannot hide one, so
  it is the candidate for the opposite direction.

That split immediately sorted the five, and each was then confirmed by
measurement rather than by the argument alone:

| test | flags asked | property needs | outcome |
|---|---|---|---|
| `github_1091` | `--unwind 1`, expects FAILED | **`--unwind 5`** — 4 is still SUCCESSFUL; the bug is a heap overflow in `strcpy` from `malloc(strlen(f) - 4)` (CWE-122), and the bound truncates the OM's `strlen`/`strcpy` loops | fixed, CORE, message pinned |
| `github_2513_6` | `--unwind 3`, expects SUCCESSFUL | **`--unwind 5`** — at 3 and 4 it fails its *own* unwinding assertions (loops 13 and 8), so it was asked to prove safety while bounded too tightly to discharge | fixed, CORE |
| `github_248` | `--k-induction --function b`, expects SUCCESSFUL | **UNKNOWN** — issue #248 was a *segfault*, not a verdict. It no longer crashes: it warns "k-induction does not support recursion yet" and returns UNKNOWN, which is the honest answer for unbounded mutual recursion. The test pinned the wrong property. | fixed, CORE, now pins the warning + UNKNOWN |
| `github_2572_2` | `--z3 --ir`, expects SUCCESSFUL | **`--ir-ieee`** — `--ir` is documented as *overapproximating*, so it cannot soundly prove IEEE-754 properties and produces a spurious counterexample on `0+f==f`, a true assertion (the source assumes `f` is neither NaN nor Inf). Its sibling `github_2572` proves the same property under `--ir-ieee`. | fixed, CORE |
| `github_732-1-1` | no flags | nothing a flag can supply | **R24**, below |

A trap avoided in that table: `github_2572_1` also runs `--z3 --ir` and expects
FAILED, which reads like "the family already pins `--ir` as failing". It does
not — that test's assertion is `0+f==-f`, which is simply **false**, so FAILED
is correct there under any solver mode. The family splits by *property truth*,
not by solver mode, and reading it the other way would have justified the wrong
correction to `github_2572_2`.

**R24 — the one real defect.** `github_732-1-1` has no flags, so neither
approximation direction can explain it, and it survives to be a defect. After
`memset(&s, 0, sizeof(s))` on a `struct { int x : 12, y : 8; }`, writing both
fields and reading the object back through `*(int *)&s` gives a value whose
**low 20 bits are correct** (`(v & 0xFFFFF) == 0xFFFFF` verifies) but whose
**12 padding bits are unconstrained** (`(v >> 20) == 0` fails). gcc gives
`0x000fffff` exactly. So the bitfield *layout* is right and the `memset`'s
zeroing of the bits above the declared fields is not visible to a type-punned
read. Pinned by `regression/esbmc/bitfield_padding_memset` (KNOWNBUG) and
`..._fields` (CORE control, so a fix cannot regress the low bits).

**M8's inventory is closed.** Of the ten unattributed wrong-verdict entries:
one produced **R23** (fixed), one is **R24** (fixed, see below), seven were wrong
tests — six fixed and retired, one (`github_159_postdecrement_fail`) left
KNOWNBUG because its intent needs a pointer-formation checker that does not
exist — and `github_248`, carried separately as the UNKNOWN entry, was also a
wrong test. **Seven of eleven entries were the test asking for the wrong
thing, not ESBMC answering wrongly**, which is the single most useful number
this milestone produced.

### M8 (R24) — 2026-08-03, R24 fixed

**Result: R24 is a byte-accounting error, not the modelling gap the entry
assumed, and the counterexample said so all along.**

The entry proposed teaching `memset` to constrain padding bits. That was the
wrong target. `memset`'s optimised path already writes every member; it just
runs out of bytes before reaching the last one. The counterexample names the
culprit directly — `s = { .x=0, .y=0, .anon_bit_field_pad#2=0, .anon_pad#3=255 }`
— three members zeroed and the fourth untouched, which is not what "padding is
unconstrained" would look like.

`struct { int x : 12, y : 8; }` lowers to four members; `gen_value_by_byte`'s
struct walk charges each one `type_byte_size()`, which rounds a sub-byte member
up to a whole byte:

| member | width | `type_byte_size` | bytes written | `bytes_left` after |
|---|---|---|---|---|
| `x` | 12 bits | 2 | 2 | 2 |
| `y` | 8 bits | 1 | 1 | 1 |
| `anon_bit_field_pad#2` | 4 bits | 1 | 1 | **0** |
| `anon_pad#3` | 8 bits | 1 | **0** | 0 |

The three bitfield members occupy 3 bytes but are charged 4, so the whole
4-byte budget is spent before `anon_pad#3`, and `gen_value_by_byte` returns it
unchanged — i.e. nondet. Nothing about padding *semantics* is wrong; the walk
simply cannot decompose a sub-byte layout into bytes.

**The fix** declines the struct rather than repairing the arithmetic: a struct
with any sub-byte member returns `expr2tc()`, which the caller already treats
as "bump to `__memset_impl`", whose byte-wise dereference model gets the
padding right. The existing guard at that spot was reaching for the same thing
and was dead — it tested `has_prefix(name, "bit_field_pad$")` while the
frontend mints `anon_bit_field_pad#`, so it never fired on any struct.

**Cost is not a concern, and the first measurement of it was wrong.** Forcing
the fallback with `--no-simplify` did not terminate in 10 minutes, which looked
like the fallback being unaffordable. That was the `--no-simplify` leg's own
timeout tail (R16), not `__memset_impl`: with the patch in and simplification
on, both pins verify in 0.46 s.

**Anti-vacuity.** `..._fail` asserts the object is `0x000FFFFE` and must stay
FAILED — the object really is `0x000FFFFF`, so a fix that made the read
unconstrained in the *other* direction would pass it. `..._fill` pins a
non-zero fill (`memset(&s, 0xFF, …)` giving `0xFFFFFFFF`), since a fix that
only special-cased zeroing would satisfy the other three pins. Both values are
what gcc produces. Verified pre-patch as FAILED and post-patch as SUCCESSFUL,
under Bitwuzla and Z3, with `github_732-1-1` flipping KNOWNBUG → CORE.

### M8 (cont. 9) — 2026-07-31, two more wrong tests, both under-approximations

**Result: `03_inf2` and `03_circular_reduce` are wrong tests, both fixed and
retired. Neither is a defect: each asks ESBMC for less than the property needs
and then expects the property to be found anyway. Seven of the ten are now
resolved.**

A correction to (cont. 7) first: it grouped `03_inf2` with `03_circular_reduce`
as "the two concurrency entries". `03_inf2` lives in `regression/esbmc/` and has
no threads at all — it is a `malloc` test. Only `03_circular_reduce` is
concurrent.

**`03_inf2` — the `github_1626-no-free` shape again.** Its two assertions are
*unreachable*: `st1 = st_alloc(a, b)` with `a > 0 && b > 0` takes the arm that
sets `t->z = NULL`, so `if (st1->z > 0)` is never taken. An
`__ESBMC_assert(0, …)` probe in each of the two bodies reports SUCCESSFUL, and
the same probe on the following `return` reports FAILED — so the probe fires
when it should and the bodies really are dead. What the program *does* have is
three unfreed `malloc`s, and `--memory-leak-check` reports
`forgotten memory: dynamic_1_array` (CWE-401), which is the FAILED the test
wants. Flag added, KNOWNBUG → CORE, with the leak message pinned so the test
cannot pass on some unrelated failure.

**`03_circular_reduce` — the context bound truncated the search by one switch.**
The loop variable shadows the global `i`, so `assert(i < 1)` fails exactly when
`receive` first becomes true on the second iteration, which needs
main → t1 → main → t1: **three** context switches. The test asked for
`--context-bound 2`:

| configuration | verdict |
|---|---|
| sequential simulation of that schedule | FAILED — the property is violable |
| `--context-bound 2` (as the test had it) | SUCCESSFUL |
| `--context-bound 3` | FAILED |
| `--no-por`, or no flags at all | FAILED |

So the bug is found by default and the entry never was a symex defect. Bound
corrected to 3 — the tight value, so the test still fails if a change loses a
switch — and KNOWNBUG → CORE. Worth noting for R12's file: this is
`--context-bound` behaving as designed (an under-approximation reported as
`VERIFICATION SUCCESSFUL`) biting a test inside this repo's own suite, which is
the same trap R12 records for `--no-unwinding-assertions`.

**Still open in M8.** Three unattributed — `github_2572_2`, `github_1091`,
`github_2513_6` — plus `github_732-1-1` (narrowed to bitfield layout under type
punning) and `github_248` (UNKNOWN under `--k-induction` on mutual infinite
recursion, likely incompleteness rather than a defect).

### M8 (cont. 8) — 2026-07-31, R23 fixed, and its attribution corrected

**Result: R23 is fixed. Fixing it also disproves half of what (cont. 7) said
about it — `github_162_fail` is a wrong test, not an entry R23 attributes.**

**The fix needs both ends.** (cont. 7) named the frontend's
`gen_typecast(ns, rhs, lhs.type())`, but that is only where the right operand
is narrowed. `goto_convertt::remove_assignment`, which turns the side effect
into `E1 = E1 op E2`, independently sets the operation's type from
`expr.op0().type()` — so fixing the frontend alone would still compute in E1's
type. The frontend now records clang's `getComputationResultType()` on the side
effect and casts the right operand to it; `remove_assignment` promotes E1 into
that type, builds the operation there, and converts the result back on
assignment. The `_Bool` arm is left first and the new arm made mutually
exclusive with it: `rhs.make_typecast` rewraps `rhs`, so a later
`rhs.op0()` would no longer be the operand the bool arm expects.

| probe | before | after |
|---|---|---|
| `char b = 3; int a = INT_MAX; b += a` (`--overflow-check`) | SUCCESSFUL | **FAILED** |
| same on a 3-bit bitfield | SUCCESSFUL | **FAILED** |
| `char b = 100; int a = 256; b /= a` | FAILED, "division by zero" | **SUCCESSFUL** |
| `b %= a` | FAILED | **SUCCESSFUL** |
| `b = b + a` (control) | FAILED | FAILED |
| `b = b / a` (control) | SUCCESSFUL | SUCCESSFUL |
| `int c; c += a` (control) | FAILED | FAILED |

The emitted claim for `char b; b += a` is now
`!overflow("+", (signed int)b, a)` — byte-identical to the explicit form's,
which is the property C11 6.5.16.2p3 actually asks for.

**And the correction.** (cont. 7) recorded R23 as attributing
`github_162_fail`. With the fix in, that test's claim becomes sound —
`!overflow("+", (signed int)b.a, a)` — and it *still* reports SUCCESSFUL,
correctly: its bitfield is a zero-initialised global, so the addition is
`0 + a` and cannot overflow whatever the operand types are. Initialising it
(`b.a = 3;`) makes the same program report FAILED. So the entry is a **wrong
test** — the fourth of the ten — and R23 is a genuine defect that was *found
through* it rather than one that explains it. Its expectation is corrected to
SUCCESSFUL rather than its program rewritten: the program is the historical
artefact, and `compound_assign_narrow_overflow` already pins the property the
test was reaching for. The tally in (cont. 7) is unchanged at five resolved;
only the reason for one of them moves.

### M7 (CI) — 2026-07-30, R19 pins withdrawn

**Result: the two R19 regression tests are not portable and are removed. The
reason is itself a finding — the per-claim solving order under
`--multi-property` is not stable across platforms.**

Windows CI failed on both, in opposite directions:

| test | macOS / local | Windows |
|---|---|---|
| violable claim **first** (KNOWNBUG, expects FAILED) | SUCCESSFUL — the bug | FAILED — no bug, so the KNOWNBUG "passes" its regex and the harness errors |
| violable claim **last** (CORE, expects FAILED) | FAILED — correct | SUCCESSFUL — the bug |

**R19's positional dependence inverts between platforms.** The cause is not the
solver: locally Bitwuzla and Z3 agree with each other. It is the order the claims
are *solved* in. Local Z3 4.16.0 solves the line-9 claim before the line-8 one;
Windows Z3 4.13.3 solves them in source order. The false `PASSED` follows the
order, so which of the two programs exhibits it flips with the platform.

That is a second-order observation worth keeping: **`--multi-property`'s per-claim
solving order is unstable across platforms and solver versions**, which belongs to
the R15 family of ordering instabilities and compounds R19 — a user comparing
per-claim output across machines can see different properties reported as passing.

**No portable pin exists.** An order-independent shape was attempted and does not
work: with *two* violable claims the defect does not appear at all (FAILED under
both solvers), so the trigger needs exactly one violable and one holding claim in
a particular solved order. `test.desc` matches output regexes and cannot express
"either verdict", so any pin encodes one platform's ordering. Keeping the tests
means permanently red Windows CI for a behaviour the suite cannot express.

The reproducer and the full characterisation stay in #6540 and in §9.2's R19 row;
only the two `regression/esbmc` directories are withdrawn.
### M6 (fix) — 2026-07-30, R18 fixed

**Result: the first defect this plan found *and* fixed. `get_expr_globals` now
follows the pointer chain, and the H-C4 oracle that motivated the fix validates
it.**

The one-level resolution recorded `*(*gpp) = 1` against the intermediate pointer
`gp` instead of its target `g`. Neither obvious repair works alone, which is the
part worth recording:

- **Stop at the first shared object** — does not fix it. `gp` is itself global,
  so the walk stops there and never reaches `g`.
- **Skip to the ultimate target** — loses a shared *intermediate*, so a thread
  writing the pointer itself stops being seen.

So the fix records **every** shared object along the chain: strictly additive,
which is the sound direction for a dependency relation. Objects deeper in the
chain are recorded on their own merit rather than behind the first symbol's
gate, because `int *lp = &g; int **lpp = &lp;` has no shared symbol until `g`.

| validation | result |
|---|---|
| R18 witnesses (nested-deref, local and global double pointer) | SUCCESSFUL → **FAILED** |
| controls (direct, single-deref, split-into-two-statements, integer round-trip) | unchanged |
| `unit/` | 600/600 |
| `regression/esbmc`, 1582 tests | **identical failure set**, 38 before and after |
| `regression/esbmc-unix`, 543 tests | 9 failures, all confirmed pre-existing |
| **H-C4 `--no-por`** | 258 → **259 agreed**, 0 diverged |
| **H-C4 `--state-hashing`** | 255 → **257 agreed**, 0 diverged |
| cost | 24.10 s vs 24.12 s on the concurrency suite |

**The cost gate mattered and did not bite.** §9.2's entry warned that widening the
dependency relation costs interleavings — #6480 tuned this same relation in the
opposite direction for exactly that reason — so the fix was gated on re-running
H-C4. Agreement *rose* with no new divergences, which is what repairing
over-pruning should look like, and the suite timing was unchanged.

**Mode C.** The patch adds branches, so C-Live applies. Each added branch is
exercised by the regression suite — the chain loop, the shared-object arm and the
deeper-targets loop all execute, or the witness would not flip. One guard added
defensively (`!is_pointer_type` on entry to the resolver) was **unreachable dead
instrumentation**; per the M4 (H-B1) precedent it is now a `SYMEX_INVARIANT`
stating the precondition instead of a dead arm. A formal Mode C run was **not**
performed — recorded as an outstanding gate rather than claimed.

`regression/esbmc-unix/mpor_nested_deref_race` moves KNOWNBUG → CORE.

### M8 (cont. 3) — 2026-07-30

**Result: both "probably wrong test" hypotheses tested. One was right, one was
wrong and hid a real defect — R22.**

I twice reported `github_162_fail` and `github_159_postdecrement_fail` as likely
*wrong tests*. Testing them split the pair.

**`github_162_fail` is a real defect, and my reasoning for dismissing it was
wrong.** I had argued the add could not overflow because the bitfield starts at
zero. It does not survive contact:

| program (`--overflow-check`) | verdict |
|---|---|
| `int x = 3; x += nondet_int();` | FAILED ✓ |
| `int x = 3; int a; x += a;` (uninitialised local) | FAILED ✓ — uninitialised locals *are* nondet |
| `struct { int a; } b = {3}; b.a += nondet_int();` | FAILED ✓ |
| `union { int a; } b = {3}; b.a += nondet_int();` | FAILED ✓ |
| **`struct { int a : 3; } b = {3}; b.a += nondet_int();`** | **SUCCESSFUL ✗** |
| **`union { int a : 3; }`**, and **`unsigned a : 3`** | **SUCCESSFUL ✗** |

The operand is explicitly nondet and the member starts at 3, so the addition
genuinely overflows. Only the *bitfield* loses the check — not the union, not the
sign, not the operand. Recorded as R22.

**`github_159_postdecrement_fail` is a wrong test.** ESBMC checks pointer
*dereferences*, not pointer *formation*: forming an out-of-bounds pointer
(`struct a *b = &Q[0]; b--;`) reports SUCCESSFUL, and dereferencing it reports
FAILED. The test's `(b--)->d.c` dereferences at the *old*, valid `b` and then
decrements without dereferencing, so nothing invalid is ever dereferenced.
Its expectation asks for a check outside ESBMC's model (C11 6.5.6p8 makes the
formation UB, but that is not what the tool checks). The useful action is to
re-scope or retire the test, not to "fix" anything.

**Method note.** Two dismissals in a row on the same pair, one of them wrong, is
the argument against triaging by reading. Both took three runs to settle. The
generalisable form is the one used throughout M8: build the neighbouring cases
and let the boundary say which component is at fault.

**Still open in M8.** Six unattributed wrong-verdict KNOWNBUGs, after R20 (2),
R21 (3), R22 (1), one rejected against R18 and one identified as a wrong test.

### M8 (cont. 4) — 2026-07-30

**Result: three more KNOWNBUGs classified, none of them a symex defect. Two fall
outside §2.3's scope and one rests on a premise the tool does not share. Three
remain.**

**`40_stack_64_inner_scope_true` — the premise, not the nested scope.** The test
declares two `int`s in nested scopes, comments them as 8 bytes each, and expects
`--stack-limit 16` to pass. Measuring what ESBMC actually charges:

| program (`--64`) | smallest passing `--stack-limit` |
|---|---|
| empty `main` | 8 (nothing charged) |
| `char a;` | ≤ 28 |
| `int a;` | **32** |
| `int a;` + nested `int a;` (the test) | **64** |
| three `int`s | > 80 |

So a 4-byte `int` costs 32, and the cost is linear in the number of locals.
Object size *is* tracked — a `char` is cheaper, an 80-byte array needs more than
128 — but the scale is inflated roughly eightfold. The test cannot pass at 16
whatever the nested-scope handling does, so it is not evidence of a scoping bug;
resolving it means deciding whether the per-object charge is intended. Left
unattributed rather than filed, since I did not read the accounting to find out.

**`github_1626-no-free` — C library model, not symex.** It expects
`printf("%s", *ptr)` on an empty struct to fail as `%s` scans for a terminator.
That requires modelling `printf`'s format-string semantics, which lives in the
operational model under `src/c2goto/library`, not in goto-symex. Out of §2.3's
scope.

**`github_248` — k-induction convergence, not symex.** Mutually recursive
`a()`/`b()` with an infinite loop and no assertions, under
`--k-induction --function b`; it returns UNKNOWN where the test wants
SUCCESSFUL. UNKNOWN is the honest answer from a non-converging inductive step,
and k-induction sits above the engine this plan verifies.

**Three remain unattributed:** `03_inf2`, `03_circular_reduce` (already tested
against R18 and rejected) and `github_2572_2` (`--z3 --ir`).

**Worth saying plainly about M8 as a whole.** Twelve wrong-verdict KNOWNBUGs went
in; four turned out to be real defects with reproducers (R20 ×2, R22 ×1, plus
R21 ×3 as an incompleteness), one was a wrong test, two are other subsystems, one
rests on an unshared premise, and one was tested against a candidate cause and
rejected. That distribution — a third genuine defects, a third mis-scoped or
mis-specified — is the useful output of the survey, and it is not visible from
the `KNOWNBUG` label alone.

### M8 (cont. 5) — 2026-07-30, M8 closed

**Result: the last KNOWNBUGs classified. Every wrong-verdict entry in the
goto-symex inventory now has a cause or a reason it is not one. Also a
correction: the population was 13, not the 12 quoted in earlier entries.**

**Count correction.** The survey found 16 wrong-verdict KNOWNBUGs, of which
**three** are this plan's own pins (`no_simplify_no_slice_huge_malloc`,
`mpor_nested_deref_race`, `multi_property_smt_during_symex`) — not four.
`double_assign_check_local_array` is a no-verdict entry, the `SYMEX_INVARIANT`
stop, and was miscounted with the pins. So 13 pre-existing entries needed
attribution, and the "12" in the M8 (cont.) through (cont. 4) entries is wrong.

**`03_inf2` — a wrong test.** Its two assertions sit inside
`if (st1 -> z > 0)`, and `st1` comes from `st_alloc(a, b)` with `a, b > 0`
assumed, which takes the branch setting `t -> z = NULL`. The guard is therefore
false and both assertions are unreachable, so SUCCESSFUL is right. Swapping the
call to `st_compact(st2, st1)` — `st2` is the allocation whose `z` is non-null —
makes it FAILED, confirming the assertions are live only in the other argument
order.

**`github_2572_2` — solver encoding, Tier D.** Under `--z3 --ir` it violates
`0 + f == f` (`IEEE_ADD((double)0, f) == f`, line 14) with NaN and infinity both
assumed away. The identity holds in IEEE-754 for every remaining value including
negative zero, so the counterexample is spurious — but it comes from the
integer/real encoding in `src/solvers`, which §14.3 places outside this plan.

**M8 closed.** Final disposition of the 13:

| outcome | count | entries |
|---|---|---|
| real defect, reproducer filed | 2 | R20 (#6544) ×2 |
| real defect, reproducer filed | 3 | R21 (#6545) ×3 |
| real defect, pinned unfiled | 1 | R22 |
| wrong test | 2 | `github_159_postdecrement_fail`, `03_inf2` |
| another subsystem | 3 | library model, k-induction, solver `--ir` |
| unshared premise | 1 | `40_stack_64_inner_scope_true` |
| real defect, pinned unfiled | 1 | R23 — `03_circular_reduce`, see M8 (cont. 6) |

Seven of 13 are genuine defects; five are mis-scoped, mis-specified or belong to
another subsystem. That ratio is the survey's real output: acting on "fix the
KNOWNBUGs" without triage would have spent roughly half the effort on tests that
should be rewritten or retired, or on subsystems this plan does not cover.

`03_circular_reduce` was the one loose end and is now R23, §15 M8 (cont. 6).

### M8 (cont. 6) — 2026-07-30

**Result: the last loose end is a new defect, R23 — and the more interesting one
of the pair, because no flag recovers it.**

`03_circular_reduce` was left unexplained after R18 was tested and rejected.
Reducing it produced a nine-line witness and a discriminator that names the
trigger exactly:

| branch body inside `if (receive)` | verdict |
|---|---|
| *(empty)* | FAILED ✓ |
| `other = 0;` — a different variable | FAILED ✓ |
| `receive = 1;` — same variable, guard-preserving value | FAILED ✓ |
| **`receive = 0;`** — same variable, guard-falsifying value | **SUCCESSFUL ✗** |

**The added write is not even executed on the failing schedule.** That schedule is
`i=0` reads 0 and skips the body, `main` writes 1, `i=1` reads 1 and the
assertion fires — reaching the assertion but not the write after it. So a
statement that provably does not run before the violation removes the
counterexample. The first row proves the schedule itself is explored.

**Every pruning suspect is ruled out.** `--no-por`, `--context-bound 8`,
`--state-hashing`, `--no-slice` and `--no-interval-symex-guard` all still report
SUCCESSFUL. The last was the natural suspect — the shape is exactly the
guard-variable pruning §7.4 aims H-C5 at, and `symex_goto.cpp:57-79` documents
that hazard — and it is refuted. Recorded with the mechanism unknown rather than
guessed, which is the third time in M8 that the obvious explanation did not
survive (constant propagation for R20, POR for this one, interval guards here).

**H-C5's relation would catch this; H-C5's corpus does not contain it.** The leg
swept 1360 inputs at 0 divergences. This nine-line program is not among them —
the same lesson M6 recorded for H-C4 and R18, now repeated: a clean corpus sweep
bounds nothing about programs the corpus lacks.

Pinned by `race_guard_self_clear` (KNOWNBUG) and `race_guard_other_write` (CORE),
the second so a change that stops detecting the ordinary case cannot make the
pair pass.

### M8 (cont. 7) — 2026-07-30

**Result: R23 narrowed from "some write kills it" to a three-condition signature,
and located in the interleaving set rather than the formula. Still not
root-caused.**

Splitting the guard variable from the written variable isolates the trigger:

| `t1` guards on | `t1` body writes | `main` writes | verdict |
|---|---|---|---|
| `receive` | `receive` | `receive = 1;` | **SUCCESSFUL ✗** |
| `flag` | `receive` | `flag = 1;` | FAILED ✓ |
| `receive` | `flag` | `receive = 1;` | FAILED ✓ |
| `receive` | `receive` | **`receive = 1; receive = 1;`** | **FAILED ✓** |

All three conditions are required: the guarded variable, the write to it inside
the branch, and a *single* write in `main`.

**The fourth row is the useful one.** Duplicating `main`'s write changes nothing
about what the program can do — the second assignment is idempotent — but it does
add a scheduling point, and it restores the counterexample. A defect that a
semantically-null edit repairs is a defect in *which interleavings are generated*,
not in how one is encoded. That also rules out the remaining encoding-side
explanations, on top of the flags already eliminated.

**And nothing upstream of symex differs.** Dumping `--goto-functions-only` for the
detected and missed variants gives instruction-for-instruction identical bodies
apart from the assignment target:

```
        IF !receive THEN GOTO 2
        ASSERT ... // assertion t1
        ASSIGN other=0;      <-- detected
        ASSIGN receive=0;    <-- missed
```

Same control flow, same claim (both programs generate 38 claims and report 4
properties), and in the missed variant the assertion claim is present and
reported `✓ PASSED`. So the claim is built and discharged against an interleaving
set that lacks the witness.

Root-causing this means reading how context-switch points are placed around a
write to a variable the same thread guards on. That is where the next attempt
should start, and it is left open rather than guessed at.

### M8 (cont. 8) — 2026-07-30

**Result: R23's gate identified. The assertion is provably violable, and the
interleaving cutoff that hides it is named — though the cutoff alone does not
explain the whole discriminator.**

`reachability_treet::get_next_formula` runs a thread until a context-switch point
and then calls `decide_ileave_direction`. Neither is gated by `--no-por`, which
is why disabling POR changed nothing. The blocking test is
`execution_statet::check_if_ileaves_blocked`:

```cpp
if (art1->main_thread_ended && !options.get_bool_option("deadlock-check") &&
    !options.get_bool_option("data-races-check"))
  // Don't generate further interleavings since __ESBMC_main thread has ended.
  return true;
```

`main` returns immediately after `receive = 1`, so every schedule that needs `t1`
to continue past main's exit stops being generated.

**Confirmed by bypassing the gate.** `--data-races-check` is one of the two flags
that disable it, and under it the program reports **FAILED on the assertion at
line 7** — the `assert(i < 1)` itself, not a race report. So the schedule is
reachable and the claim is violable; the default configuration simply does not
explore it. `--deadlock-check`, the other flag in the disjunction, does *not*
recover it, because it acquired its own post-main cutoff in `bb8366b002`.

**Stated honestly, this is not yet the complete story.** The `other = 0` variant
is detected under exactly the same gate and the same immediately-returning
`main`, so the cutoff cannot be the only factor — something about writing the
guarded variable inside its own branch decides whether the violation is found
before the cutoff bites. Both halves are needed for a fix, and only the first is
established.

**What this does settle** is that the counterexample is real rather than a
modelling artefact, and that the search for the second half belongs in the
interaction between cswitch-point placement and the main-ended cutoff — not in
POR, the interval domain, the slicer, state hashing or the encoding, all of which
earlier rounds eliminated.

### M8 (cont. 9) — 2026-07-30, correction

**The gate hypothesis recorded in (cont. 8) is wrong. Retracted here rather than
left in the log.**

(cont. 8) proposed `check_if_ileaves_blocked`'s `main_thread_ended` cutoff as
R23's mechanism, on the strength of `--data-races-check` recovering the
counterexample — that flag being one of the two that disable the cutoff.

**The prediction it implies fails.** If the cutoff were the cause, keeping `main`
alive past its write should recover the counterexample. It does not:

| `main` tail after `receive = 1;` | verdict |
|---|---|
| *(returns immediately)* | SUCCESSFUL |
| `dummy = 1;` | SUCCESSFUL |
| `dummy = 1; dummy = 2; dummy = 3;` | SUCCESSFUL |
| `for (int k = 0; k < 4; k++) dummy = k;` | SUCCESSFUL |

Three extra global writes, and a loop of them, give `main` both a longer lifetime
and more context-switch points, and change nothing.

**Why the earlier evidence was not conclusive.** `--data-races-check` does two
things at once: it disables the cutoff *and* instruments shared accesses, which
adds context-switch points inside `t1` around the very variable at issue. The run
could not distinguish them, and I attributed the recovery to the gate. The
control that would have caught this — vary the gate without varying the
instrumentation — is exactly the tail experiment above, and it belonged in
(cont. 8).

**What survives.** The counterexample is real: under `--data-races-check` the
violated property is the `assert(i < 1)` at line 7, not a race report. So R23 is
a genuine incompleteness in the default configuration and not a modelling
artefact. Everything else about the mechanism is open, with POR, the interval
domain, the slicer, state hashing, the encoding *and now* the main-ended cutoff
all eliminated.

**Method note.** Fourth mechanism proposed and refuted for M8's findings —
constant propagation for R20, POR and interval guards for R23, now this one. The
recurring error is accepting a flag that recovers a counterexample as evidence for
*why*, when the flag changes more than one thing. A flag is a valid probe only
when its effects are separable, and confirming that separation is the step to add.

### M8 (cont. 10) — 2026-07-30

**Result: R23's divergence point located by instrumenting the scheduler. The
detected and missed variants take identical scheduling decisions until one
point, where the missed variant switches away from `t1` after a single loop
iteration.**

There is no debug channel for the reachability tree — `src/goto-symex` logs only
`goto-trace`, `rename`, `slice` and `ssa` — so this needed temporary
instrumentation in `reachability_treet::get_next_formula`, logging every
`decide_ileave_direction` result as `from → to` with the source line. The patch
was local only and is reverted; it is described here so the measurement can be
repeated.

| decision | `other = 0` (detected) | `receive = 0` (missed) |
|---|---|---|
| 1–3 | identical | identical |
| 4 | `1→1` at line 7 | `1→1` at line 7 |
| 5 | `1→1` at line 6 | `1→1` at line 6 |
| **6** | `1→1` at line 7 — **second iteration** | **`1→0` at line 6 — switches away** |
| 7 | `1→1` at line 6 | `0→1` at line 10 |
| 8 | `1→0` | — |
| total decisions | 8 | 7 |

Line 6 is the `for` header and line 7 the guarded body, so in the detected
variant `t1` is allowed to run through *both* loop iterations, which is exactly
what the racy schedule needs — read 0 at `i=0`, `main` writes, read 1 at `i=1`.
In the missed variant the scheduler leaves `t1` after the first iteration and
never returns to it in a state where the second read can see the write.

**What this does and does not establish.** It locates *where* the exploration
diverges — the decision at the loop header after the first iteration — and
confirms the cause is the set of generated interleavings rather than the
encoding, which the duplicate-write result in (cont. 7) already implied. It does
**not** yet say why that decision differs, and the honest candidates are
`dfs_explore_thread`'s per-frame `mark_explored` bookkeeping and the extra
context-switch point the guarded write introduces. Distinguishing them means
logging scheduler-frame state alongside the decision, which is the next step.

Recorded rather than pushed further because the remaining question is a
scheduler-internals investigation, not a property this plan's harnesses can
settle.

### M8 (cont. 11) — 2026-07-30

**Result: R23 narrowed to context-switch-point generation, and the DFS
bookkeeping candidate eliminated. `t1`'s second loop iteration produces no
scheduling decision at all in the missed variant.**

(cont. 10) left two candidates: `dfs_explore_thread`'s per-frame
`mark_explored`, or the extra context-switch point the guarded write introduces.
Logging each candidate thread's viability *and* its frame-explored bit inside
`decide_ileave_direction` separates them. Both runs are identical for ten
decisions, then:

| | `other = 0` (detected) | `receive = 0` (missed) |
|---|---|---|
| decision 11 | `tid=1 viable=true already_explored=false → schedulable` at **line 7** | `tid=1 viable=false already_explored=false` at **line 245** |

**`already_explored` is `false` on both sides, so the DFS bookkeeping is not the
cause.** What differs is where the active thread *is*: in the detected variant
the scheduler is offered a decision at line 7, the guarded body on the second
iteration; in the missed variant `t1` has already run to completion and sits in
the pthread trampoline (`pthread_lib.c:245`).

Scheduling decisions are only taken at context-switch points, so `t1` ran its
**entire second loop iteration without registering one**. That is precisely why
`main`'s write cannot be interleaved between the two guard reads, and it explains
every earlier observation: the empty body and the different-variable body both
keep the second read's switch point, and duplicating `main`'s write supplies an
extra point from the other side instead.

**Where this leaves R23.** The remaining question is narrow and well-posed: why
does the second read of `receive` register a context-switch point when the branch
body writes `other`, but not when it writes `receive`? That is a question about
`analyze_read`/`analyze_assign` and the `vars_map` / `is_global` bookkeeping in
`get_expr_globals` — the same function R18 was fixed in — rather than about POR,
the DFS, the encoding or the solver, all now eliminated.

Instrumentation was local only and is reverted; both measurements are described
here so they can be repeated without rediscovering the approach.

### M8 (cont. 12) — 2026-07-30, R23 investigation halted

**Result: the divergence is visible at the level of individual accesses, but the
fifth mechanism hypothesis is also refuted. Halting the investigation here.**

Instrumenting `analyze_read` to log every access and its resolved globals gives
traces that are identical for fourteen accesses and then split exactly once:

| | `other = 0` (detected) | `receive = 0` (missed) |
|---|---|---|
| access 15 | `RD line=7 n=1 [c:@receive]` | `RD-SKIP line=7 guard_false=true` |

So in the missed variant the second guard read is not analysed at all, because
`analyze_read` returns early when the active state's guard is false
(`execution_state.cpp:811`). No global is recorded, no context-switch point is
registered, and `t1` runs to completion — which matches the scheduler trace in
(cont. 11) exactly.

**That is a symptom, not the cause.** Deleting the `guard.is_false()` clause from
`analyze_read` and rebuilding leaves the verdict **unchanged** (still
SUCCESSFUL). The read then registers, and the counterexample still does not
appear. So the false path guard at that point is a consequence of how the state
evolved, not the thing suppressing the interleaving.

**Halted.** Five hypotheses have now been proposed and refuted for this finding —
POR, interval-domain guard pruning, the `main_thread_ended` cutoff,
`dfs_explore_thread`'s frame bookkeeping, and now `analyze_read`'s guard-false
early return. Each round cost a build and a measurement, and the returns are
diminishing: what remains is understanding why the state guard is false on the
path that should be live, which is a symbolic-execution question about guard and
path management rather than about scheduling, and is better started fresh with
that framing than continued as a sixth probe from the outside.

**What R23 has, for whoever picks it up.** A nine-line reproducer; a four-row
discriminator isolating the trigger to writing the guard variable inside its own
branch; proof the counterexample is real (`--data-races-check` reports the
assertion itself); the exact divergent access and scheduling decision; and five
eliminated mechanisms. All of it is in #6558 apart from this last round.

### M9 (H-B6) — 2026-08-04, I9 discharged

H-B6 was the last Tier-B row never run. `unit/goto-symex/value_set_merge.test.cpp`
runs it on the real engine: three end-to-end programs whose joins give one global
pointer two targets, plus two cases exercising `value_sett::make_union` directly.
All five pass, and the assertions name the objects (`c:@a`, `c:@b`) rather than
counting them — a global pointer's map already holds its zero-initialiser, so a
cardinality of two is reached without any merge occurring, and an earlier
count-based version of these cases passed without ever inspecting a target.

**The verdict rests on separating two mutants, which is the substance of this
entry.** Deleting the `make_union` call from `merge_value_sets` leaves all five
cases green. Replacing it with an intersection fails three of them. Both mutants
were built and run; the second is I9's actual content, so I9 is discharged and
the surviving deletion mutant is not evidence against the harness.

Why deletion cannot be caught here, from instrumenting the call on the
`early_exit` program (an `if` arm leaving by its own `goto`, so the arms reach
the join by different routes): the union arm runs three times and reports
`changed == false` every time, and the `guard.is_false()` replacement arm above
it — the only arm that can drop entries — is never taken. Guarded assignment
*adds* to a pointer's object map rather than replacing it, and `cur_state`'s
value set is never rewound when a branch is abandoned, so both targets are
present before any join runs. The union is therefore redundant at every join
reachable at this tier, and is load-bearing only against a future change making
value sets path-sensitive. This also answers, negatively, the question the
harness's first draft left open — that a shape whose arms diverge at the join
would make the merge observable. `early_exit` is that shape, and it does not.

**Not covered, and deliberately.** `make_union`'s `keepnew` parameter decides
whether an entry present only in the source survives; `merge_value_sets` passes
`true`, but `value_set_domaint::merge` — the static analysis, outside
goto-symex — passes the caller's choice, and with `false` an entry that is
neither a `value_set::dynamic_object` nor `value_set::return_value` is dropped
(`value_set.cpp:133-149`). That is a documented asymmetry in a different
subsystem, not an I9 violation, and no case here constrains it.

R9's three "sound over-approximation" claims remain open: H-B6 checks that a
merge does not shrink the set, not that a *deliberate* narrowing elsewhere keeps
only `unknown`/`invalid` entries, as §14 already records.

### M9 (H-B7) — 2026-08-04, the assumption register audited

H-B7 is not a harness so much as a pass over §7.3, whose rule is that a row may
not be closed without a Tier-B discharge or a reviewed waiver. Seven rows; the
audit closes three, and the interesting part is that only one of them needed new
code.

**Already discharged, cited rather than rebuilt.** The I2 key-stability row
(H-A1/H-A9) is asserted by `renaming.test.cpp`'s "make_assignment publishes a
fresh increasing L2 index", which checks that the entry `coveredinbees` updates
is the one keyed by the caller's key across five successive publications — the
row was written before that test existed and was never revisited. H-A2's
overlap row was already marked "by construction".

**New: H-A4's shape row**, in `unit/goto-symex/assumption_discharge.test.cpp`.
The assumption reads "every `with2t` store the slicer elides has a `symbol2t`
source and constant index", which the guard at `slice.cpp:249-254` makes true of
whatever it admits — so the check that carries information is over the shapes it
*excludes*, which is what §7.3 asked for. Four cases: the assumption checked on
every elided store of a program where the elision demonstrably fires; a census
showing symbolic-index stores never qualify; a control showing constant-index
stores do; and the one worth keeping —

> **A struct member store must never reach that branch.** `symex_assign` spells
> it `s' == s WITH ["f" := v]` with a `constant_string2t` field
> (`symex_assign.cpp:958-970`), so `is_constant_int2t(update_field)` excludes it
> today. It matters because `index_reads`, the read-set the elision consults, is
> populated *only* from `index2t` reads (`slice.cpp:104-118`): a member read is a
> `member2t` and records nothing. A member store that qualified would find its
> field "never read" and be dropped as dead. The unsoundness would arrive
> through a change to how member updates are spelled — a change no one would
> file under "slicer" — which is exactly the kind of coupling a register row is
> for.

An anti-vacuity guard earned its place while writing this: the first version of
the elision program read the array through its return value, and every store was
removed wholesale by `ignore` without the branch firing at all, so
`REQUIRE(elided > 0)` failed rather than the case passing empty.

**Sharpened, not closed.** H-A6's row cited R11 as an open risk; R11 became
**R18** and was fixed by **#6550**, so the row is refuted-and-fixed rather than
discharged — pointer chains are now followed, but no harness asserts
completeness over every access shape, and the row stays open in that weaker
form. H-A2's guard-algebra row remains a cross-document dependency on the irep2
plan.

**The live residual is H-A8** — closed by the next entry — and the audit gives it
a consequence it did not have. The balance of `push_ctx`/`pop_ctx` rests on one explicit call —
`targ->push_ctx()` at `reachability_tree.cpp:339`, commented "Start with a depth
of 1" — pairing with `~dfs_execution_statet`'s pop, since the initial execution
state is constructed rather than cloned and so has no push of its own. Both are
conditional on `--smt-during-symex`. If that pairing ever breaks, the failure is
not a diagnostic: `runtime_encoded_equationt::pop_ctx` takes
`scoped_end_points.back()` unchecked, on a list its constructor leaves empty
(`symex_target_equation.cpp:527-537, 596-609`). Checking this at Tier B needs a
real `runtime_encoded_equationt` over a real `smt_convt` — §6.1's no-doubles rule
forbids a counting subclass standing in for the equation — which is the next
piece of work rather than a gap in this one.

### M9 (H-A8) — 2026-08-04, the last register row closes

`unit/goto-symex/context_stack.test.cpp` drives a real `runtime_encoded_equationt`
over a real `smt_convt` from `create_solver`, per §6.1: a counting subclass would
be a double, and would not exercise the stack at issue. Three cases, and the
first exists because the obvious way to write this test measures nothing.

**The template is not the equation.** `setup_for_new_explore` *clones* the target
it was given and pushes on the clone (`reachability_tree.cpp:330-339`,
`symex_target_equation.cpp:633-645`), so the object the caller constructed —
the one `bmc.cpp:143-152` builds and the one a test naturally holds — stays at
depth 0 for the whole run. The balance has to be read off the equation the
exploration returns in its `symex_resultt`. The first draft of this file
asserted on the template and reported an imbalance that was purely its own.

**The balance holds, and its shape is not what the row implied.** One push at
setup, one per clone, one pop per destruction; the initial state is destroyed
like any other, so an *exhausted* exploration lands on **0**, not on the setup
push. That push is exactly the partner for the initial state's own pop — the
state is constructed rather than cloned, so it has no push of its own. On
`TWO_WRITERS` the exploration runs 49 interleavings (the same count
`mpor.test.cpp` reports) and reaches depth 9 before returning to 0. A second
draft asserted the exploration ends at depth 1 and failed at 7, which is not an
imbalance either: it stopped at a 32-interleaving cap, so the remaining depth
was the live DFS stack. Only an exhausted exploration says anything, and the
case now requires exhaustion before it reads the depth.

**Mutation.** Deleting `targ->push_ctx()` fails the sequential case (depth 0
where 1 is required) and then **SIGSEGVs** — `pop_ctx` taking
`scoped_end_points.back()` on the empty list, which is the failure mode
§7.3's row was given on inspection in the previous entry and is now observed.
That is also why this file is allowed to fail by crashing: with the pairing
broken there is no diagnostic to produce.

With this row closed, §7.3 has no open assumption backed by a live harness:
H-A2's guard algebra remains a cross-document dependency on the irep2 plan, and
H-A6 stays refuted-and-fixed (R18/#6550) rather than discharged.

### M9 (R9) — 2026-08-04, two of three approximations pinned

R9 names three places where a comment argues an approximation is sound and
nothing checks it. Verifying the *arguments* is not on the table at Tier B; what
is, is stating each as a predicate over the produced equation, so a change that
reverses the direction fails here rather than in a verdict months later.
`unit/goto-symex/overapproximation.test.cpp`, four cases, both mutation-checked.

**Claim 2, the non-scalar uninterpreted-function fallback.** The comment says
the fallback "drops only the functional-congruence constraint … never adding
behaviour, and the body is still discarded". Two of those three are observable:
a pointer argument yields no `uninterpreted_func2t` anywhere in the equation
(the scalar program is the control — it yields two), and the discarded body's
write to a global never appears. The body check needed a correction that is
worth recording, because the first version of it was wrong in a way that reads
as a defect: asserting "no assignment to `side`" *fails*, since every global
carries its zero-initialiser. The check is `assignments_to(eq, "c:@side") == 1`
— the initialiser and nothing else. A count of two would be the discarded body
running.

**Claim 3, the function-pointer target filter.** Two directions, each with its
own case and its own mutant:

- An incompatible-arity candidate is dropped: `one`'s body appears in the
  equation, `two`'s does not. Disabling the filter (`if (false)`) fails exactly
  this case, so it is the filter doing the work and not the value set.
- A filter that would empty the list keeps it: with *every* candidate
  incompatible, the call must still dispatch. Deleting the `!compatible.empty()`
  guard fails exactly this case. The asymmetry is the point — a wrong-arity
  dispatch is a spurious counterexample, a dispatch to nothing at all is a
  missed one, and only the second direction is unsound.

**Programs that do not reach the branch.** The first draft gave both
function-pointer cases an array of function pointers indexed by a nondet, on the
assumption that the value set would list both elements. It does not: `p`'s entry
comes back without candidates, the call is skipped by the open-world path at
`symex_function.cpp:795-804`, and both cases pass or fail for reasons unrelated
to the filter. A direct `if (nondet) p = f; else p = g;` gives the two-candidate
set the claim is about. Worth remembering for any harness that needs a
multi-target function pointer.

**Claim 1 stays open, but scoped.** The value-set filter after a pointer havoc
(`symex_assign.cpp:554-576`) drops `unknown`/`invalid` entries from the restored
set. Its guard requires `inductive_step`, a nondet pointer side-effect,
`--add-symex-value-sets`, *and* `pc->inductive_step_instruction` — a flag only
the k-induction goto transform sets, so no `goto_factory` program reaches it and
Tier B cannot see the branch at all. It needs either a k-induction-aware
fixture or a Tier-C leg over the corpus that already uses the flag pair. Noting
also that this one is a *narrowing*, unlike the other two: it removes candidates
rather than constraints, so "never adding behaviour" is not the direction to
check — the question is whether the dropped `unknown` could have been the real
target, which is why it was the one left unproven.

### M9 (R9 cont.) — 2026-08-05, claim 1 pinned; the reachability claim was wrong

The entry above says Tier B "cannot see the branch at all". That is false, and
the correction is the cheap half of this one: `goto_k_induction` is a **free
function** (`goto_k_induction.h:20`) and `gotoalgorithms` is already linked into
every `unit/goto-symex` target. `symex_run::inductive_step_equation` runs
`remove_no_op` then `goto_k_induction` over a `goto_factory` program before
`setup_for_new_explore()`, with `inductive-step` and `add-symex-value-sets` set.
Instrumenting the site confirms arrival rather than inferring it. No Tier-C leg
was needed; the row was closed on inspection of the *option*, not of the call
graph.

**Getting a set worth filtering is the part that needs care.** The pre-havoc
map must mix a concrete candidate with a sink, which needs an *external*
function to supply the sink: `p = &a` on one arm, `p = ext()` on the other, with
the loop writing `p` so the transform havocs it. A first attempt whose pointer
had a single concrete target reached the branch with `pre map size=1` and
filtered nothing — passing for reasons unrelated to the claim.

**Two directions, two cases, three mutants.** The first case is the claim
itself: the sink is dropped and the concrete candidate is not. It is separated
by *both* one-sided mutants — keeping the sinks (`if (false && …)`) and keeping
only the sinks (inverting the test) each fail it.

The second case is the one the first entry did not anticipate, and it is where
the real asymmetry lives. `filtered` is installed only `if (!filtered.empty())`,
and on a program whose every entry is a sink that guard is the only thing
between `p` and an *empty* target set. Deleting it costs, on a four-iteration
loop, exactly one `dereference failure: invalid pointer` and one
`dereference failure: Incorrect alignment` — four checks become three, while all
40 assignments and every LHS name survive unchanged. A dereference simply stops
being verified, and nothing in the equation says so. That is the same
missed-bug direction as claim 3's function-pointer filter, and it makes the
narrowing's real risk concrete: not that a dropped `unknown` was the true
target, but that dropping *all* candidates removes the properties.

**A harness bug worth recording, because it inverted a verdict.** The sink
detector first scanned `step.rhs` only, and under that version the
keep-only-the-sinks mutant *passed* — the sink reaches the equation through
`lhs` and `guard` as well. A one-sided scan reads exactly like a discharged
claim. Relatedly the assertions name the property text rather than totalling
asserts: a bulk count cannot say *which* check went missing, and here that is
the whole content. A differential pin against a concrete-target twin was tried
first and abandoned — the twin carries 1 assertion to the sink program's 9, so
the two are not comparable.

With this, R9's three claims are all pinned and §7.3 has no row left open
behind a reachability argument.

### M9 (G-remeasure) — 2026-08-05, §13.2 re-run; WI-1, WI-2 and WI-3 are done

Appendix C says to re-run §13.2's figures before citing them. Doing so retires
§13.6's whole critical path: **all seven probes now pass**, so **WI-1, WI-2 and
WI-3 are closed** — carried as "not started" in every milestone entry since M0,
and closed by other people's work in the interim rather than by this plan.
`<shared_mutex>` (G2) is simply present in `src/cpp/library/`; G1 and G7 were
closed by **#6631**; G3, G4, G5 and G6 all parse. Nothing in §13.6's first three
rows remains to do, and no one had re-measured to notice.

**Two probe defects, both of which produced a wrong verdict before being
caught.** They are recorded because each one reads exactly like a finding about
ESBMC, and neither is.

1. *Grepping for `error:`.* The first sweep matched
   `runtime_error::~runtime_error` inside the `--parse-tree-only` AST dump and
   reported G4 and G5 as failing. The reliable criterion is `PARSING ERROR` or
   the exit code. Appendix C's `probe()` helper has this flaw.
2. *Probing a C++23 name at `--std c++20`.* This entry first recorded G7 as the
   one gap still open. It is not: `std::unreachable` is modelled at
   `src/cpp/library/utility:285`, correctly gated on `__cplusplus >= 202302L`,
   and closed together with G1 by **#6631** on 2026-08-02 — five days after M0
   measured them, so both were real gaps when the table was written. At
   `--std c++23` every probe passes. The lesson generalises: **the probe's
   `--std` must be at least the standard that introduced the name**, or the OM
   is blamed for the harness's setting.

**The payoff measurement, and the new blocker.** G2 was called "the first-order
blocker … it stops *any* file that reaches `irep_idt`". That is now testable
rather than projected: `#include <goto-symex/renaming.h>` gets to exactly **one**
error, and it is not a missing STL facility. `irept` declares
`typedef std::map<irep_idt, irept> named_subt` (`irep.h:41`) — a `std::map`
naming `irept` as its `mapped_type` from inside `irept`'s own definition — and
the OM's `map` instantiates the node eagerly: `field has incomplete type
'mapped_type' (aka 'irept')`. Recorded as **G9**.

G9 is worth stating precisely because the obvious reading ("the OM's `map` is
broken") is wrong. `std::vector<irept>` on the *previous line* parses, and the
difference is the standard's: [container.requirements.general] extends
incomplete-type support to `vector`, `list` and `forward_list` only (N4510,
adopted for C++17). `std::map` with an incomplete `mapped_type` is ill-formed,
so libstdc++ accepting `named_subt` is a QoI extension. Closing G9 therefore
means *choosing* to match that extension in the OM, or changing `named_subt`
itself — an ESBMC-wide change, not an operational-model one.

This does not move §13.3's conclusion: parsing one header was never the same as
verifying a translation unit, and the `immer`/`irep2` tractability wall stands
untouched. What it does move is the **Tier-B′ pilot (WI-4)** from "blocked on a
missing header" to "blocked on one identified, characterised incompatibility."

### M9 (R6) — 2026-08-05, the collision is observed; the witness is not

R6 has stood as "mechanism pinned, no witness" since M6: `current_hashes` is
keyed by the **L0** name, so two states differing only in the L1 activation of a
recursive local were *argued* to fingerprint identically. That argument is now
**observed** rather than read off the source.

Instrumenting `check_for_hash_collision` to record a per-thread call-stack-depth
signature alongside each hash, and running a program whose recursive local holds
the same value at every depth, prints exactly one line:

```
R6COLLIDE recorded_depths=3,5, current_depths=3,6,
```

Two states with equal fingerprints and **different call depths**, pruning one
against the other. That is R6's precondition, reproduced on a concrete input.
`generate_hash()` mixes the L2 value hashes with each thread's
`pc->location_number` and nothing else, so depth cannot enter it; the collision
is structural, not a hash accident.

**The verdict does not flip, and that is the honest state of R6.** Two
constructions were tried and both report `VERIFICATION FAILED` under
`--state-hashing` and without it:

- a recursive worker reached at two depths by a nondet branch, with an observer
  asserting on the post-unwind counter;
- the same with the observable narrowed to a one-step window that only the
  deeper unwind opens.

Pruning is real on both — symex drops from 178 assignments to 92 on the first —
but the assertion stays reachable through interleavings the prune does not
touch. That is the gap between *a collision occurs* and *the bug lives only
behind the collided state*, and it is the whole difficulty: the pruned edge
removes a schedule, not a sequential path, so a witness needs a property
violable **only** in a schedule that the collision removes. Narrowing the window
is not sufficient, because the surviving schedules re-open it.

R6 therefore moves from "mechanism pinned by inspection" to "mechanism observed,
soundness consequence still unwitnessed" — a smaller step than a witness, and
the two failed constructions are recorded so the next attempt does not repeat
them. Severity is unchanged and still bounded by `--state-hashing` being opt-in.

**2026-08-06 — the collision's shape, and why the witness resists.**
Re-instrumenting the prune to print the *whole* state signature — per-thread
call depth **and** program counter — rather than depth alone gives the pair
exactly:

```
R6PRUNE kept=[3@454 5@826]  pruned=[3@454 6@826]
```

Both states sit at the **same two program counters**. The only difference is one
extra frame on the worker. That is the collision R6 predicts, and it is now
pinned to the instruction rather than merely to a depth pair.

It also explains the failed constructions structurally, which the earlier
"narrowing the window is not sufficient" note did not. The kept and pruned
states resume into the *same code at the same pc*; they differ only in how many
times that continuation will repeat as the stack unwinds. So any property
distinguishing them can only be evaluated **after** the unwind completes — by
which point the surviving schedules have converged on the same values, and the
assertion is reachable through one of them. A witness therefore needs a property
that is observable *during* the unwind and whose truth depends on the remaining
frame count, which is a much narrower target than "make the bug live behind the
pruned state".

Still not witnessed, and no further speculative programs were tried: the two
recorded failures plus this structural reason are more useful to the next
attempt than a third variant that also does not flip.

### M9 (R10) — 2026-08-05, the low-severity row that was not benign

R10 read "no current default-construction site was found", which invited leaving
it. Writing the site the row hypothesises settles it: with the original
`name_record() = default`, a test that does nothing but default-construct two
records and compare them **traps** — `Trace/BPT trap: 5`, exit 133 — rather than
passing on plausible garbage. The row's "latent" is accurate only in that no
*shipped* path constructs one; the UB itself is immediate and observable.

**The fix is not just four initialisers.** `compare()` short-circuits on `hash`
before it looks at any field, so a default-constructed record needs a hash
*consistent with* its fields, not merely a defined one. Zeroing all five would
satisfy the compiler and still be wrong: a record built from an equivalent L0
symbol hashes to something non-zero, so the two would compare unequal while
being field-for-field identical. The fix therefore factors the computation into
a private `compute_hash()` that both constructors call, which makes "hash is a
pure function of the other four" a property of the class rather than of one
constructor.

That distinction is what the harness checks, and it is what separates the
mutants. Dropping the `compute_hash()` call from the default constructor — the
shape a reviewer would most plausibly accept — leaves `hash` at 0 while the
fields are correct, and fails on
`from_symbol.hash == a.hash` (`706246307815962 == 0`). The equality-only
assertions do not catch it: two default records still compare equal to each
other. Only the cross-check against a symbol-built record does.

Regression scope: 657 unit tests pass. `regression/esbmc-unix` has four
pre-existing macOS failures (`04_valgrind`, `error`, `error2`,
`unsupported_extensions`); the set is byte-identical with the change reverted
and re-applied, so none is attributable here.

### M9 (H-A6) — 2026-08-05, the census that found R29

H-A6's row has read "no harness asserts completeness for every access shape"
since M6, which is the kind of sentence that survives indefinitely because it
describes an absence. Enumerating the shapes takes about an hour and settles it.

**Method.** Twenty-one two-thread programs on the template `#6550` left behind:
a writer reaching a shared object through one access shape, `main` reaching the
same object directly, and an assertion the race violates. Each is run twice —
default (MPOR on) and `--no-por`. `--no-por` is the reference: the race is real
in every program, so any shape reporting SUCCESSFUL by default has had its
interleaving pruned. No oracle beyond the metamorphic pair is needed.

**Sixteen shapes pass and five do not.** Direct writes, single/double/triple
pointer chains, array elements (constant, symbolic, via pointer), struct
members, arrow, nested members, arrays of structs, member arrays, pointer
arithmetic and union members are all recorded correctly. The five that are not
share one property — **the pointer is held in an aggregate**:

| Shape | Write | MPOR | `--no-por` |
|---|---|---|---|
| struct member | `*(s.p) = 1` | SUCCESSFUL | FAILED |
| arrow | `*(sp->p) = 1` | SUCCESSFUL | FAILED |
| array element | `*(pa[0]) = 1` | SUCCESSFUL | FAILED |
| nested struct | `*(o.in.p) = 1` | SUCCESSFUL | FAILED |
| union member | `*(u.p) = 1` | SUCCESSFUL | FAILED |
| **local copy** | `int *lp = s.p; *lp = 1` | **FAILED** | FAILED |

The last row is what makes this precise rather than suggestive. It differs from
the first only by copying the pointer into a local before dereferencing it, and
that copy restores detection — so the gate is **syntactic**. `get_expr_globals`
resolves pointer chains only under `if (is_symbol2t(expr))`, and `s.p` is a
`member2t`. Recorded as **R29**, dual-solver confirmed under Bitwuzla and Z3.

**R29 is R18 with the same shape one level out.** #6550 taught the resolution to
follow a chain of *symbols*; nobody asked what happens when the pointer is not
spelled as a symbol at all. That is the value of enumerating rather than
reasoning: the fix's own regression test (`mpor_nested_deref_race`) passes, and
would have gone on passing.

Pinned as `regression/esbmc-unix/mpor_aggregate_ptr_race` (KNOWNBUG, stating the
verdict the fix must produce) with `..._local` as a CORE control, so the pair
fails if either the defect is fixed or the working shape regresses. Not fixed
here: generalising the resolution to `member2t`/`index2t` widens what MPOR
treats as conflicting, which needs its own soundness argument and a Mode C pass
rather than being appended to a census.

### M9 (R15) — 2026-08-06, determinism becomes literal

R15's remedy was already written down — reset both counters in
`setup_for_new_explore`, never in the `execution_statet` constructor, which the
reachability tree copies per interleaving and where a reset would mint
*colliding* names. Applying it as prescribed is the whole fix:
`execution_statet::reset_dynamic_counter()` and
`dereferencet::reset_object_counter()`, called once per exploration.

**Two counters need two mutants, and the first version of this only had one.**
Dropping the `dynamic_counter` reset fails the heap case immediately. Dropping
the `invalid_counter` reset failed *nothing*: `symex::invalid_object` is minted
only when a dereference cannot be resolved, and no case in the file created one.
The two resets were pinned by a single test that exercised one of them. The file
now carries a case whose pointer comes from an external function, so the
unresolvable dereference mints the object and the second reset is separated too.
This is the same lesson as M9 (R9)'s `rhs`-only scan: a harness that covers half
the mechanism reports a full discharge.

**The old pin is inverted, not deleted.** The case previously asserted
`first != second` — it pinned the leak, and its own comment said to delete it
once the counters reset. Asserting `first == second` instead keeps the coverage
and makes a regression *flip* the case rather than silently weaken the file. The
canonicalising comparator stays, because it localises a failure: a diff that
survives normalisation is a different defect from one that does not.

**A prediction that did not hold.** The row warned to "expect churn in
`test.desc` files whose expected output names a dynamic object; run the full
corpus before landing." No `test.desc` in the corpus matches `dynamic_[0-9]+` or
`invalid_object[0-9]+` — nothing pinned the numbering, so there was nothing to
churn. Worth recording because the warning would otherwise be re-inherited by
the next person to touch object naming.

Objective 7 said equations "must produce byte-identical equations", which M4
recorded as unachievable as stated. It is now achievable and asserted strictly;
the objective's wording is literal rather than aspirational. 658 unit tests
pass. Two interval-analysis cases timed out on the first `-j8` run and pass in
13.6 s isolated — machine load, not this change.

### M9 (R29 fix) — 2026-08-06, three shapes of five, and why the other two differ

The census entry above proposed generalising the resolution to
`member2t`/`index2t`. Instrumenting `analyze_assign` first is what made the fix
small: the missed write arrives as `dereference(member(s, p))` and the working
one as `dereference(symbol(lp))`, so the gate is on the *dereference operand*,
not on the assignment. And `resolve_pointer_target` was already general — its
only precondition is `is_pointer_type`, not `is_symbol2t` — so the fix is one
arm that hands it the operand the `is_symbol2t` path never sees. The operand
walk still runs afterwards: recording the aggregate *and* the target only makes
MPOR more conservative, which is the safe direction.

**Three of the five shapes close.** Array element (`*(pa[0])`), arrow
(`*(sp->p)`) and union member (`*(u.p)`) now report FAILED by default, matching
`--no-por`; pinned CORE as `mpor_aggregate_ptr_race_{array,arrow,union}`.

**The two that remain are a different defect, and the probe says so.** For
`*(s.p)` and `*(o.in.p)`, `resolve_pointer_target` returns **nil**:
`R29FIX ptr=member nil=1 to_global=0`, against
`R29FIX ptr=index nil=0 to_global=1` for the array shape that works.

*Corrected 2026-08-06 — the first version of this paragraph said "the value set
has no entry for a `member2t` over a struct symbol". That is false, and the
correction matters because it moves the defect to another subsystem.*
`--show-symex-value-sets` on the failing program prints
`c:@s.p = { <g, 0, 1, signed int> }`: the entry is there. Instrumenting
`resolve_pointer_target` at the lookup shows where it is lost —

```
STRUCT   R29RES in=member renamed=member dest=0
UNION    R29RES in=member renamed=member dest=1
```

Both stay a `member2t` through renaming, and `get_reference_set` returns **zero**
objects for the struct while returning one for the union spelling of the same
program. The two take different arms of `get_value_set_rec`'s `member2t` case
(`value_set.cpp:246-268`): a struct recurses once on the named field, a union
iterates every member name — and the single-field recursion is the one that
comes back empty. Why is a `src/pointer-analysis` question, which **§14 item 2
places outside this plan**: goto-symex *consumes* `value_sett`, and its
correctness is a separate obligation. Recorded here, not chased, and worth its
own issue against the pointer analysis rather than against MPOR.

*Chased and closed, 2026-08-12 — see §15 M9 (R29 residual). The arm-asymmetry
reading above is the symptom, not the cause: both arms recurse to the same
place, and the object is lost one level further down, in the constant-expression
case.*

Regression scope: 568/572 `esbmc-unix` pass, the four failures being the
pre-existing macOS set (`04_valgrind`, `error`, `error2`,
`unsupported_extensions`) confirmed identical before this change; 658 unit tests
pass. The new arm is reachable — the three flipped shapes exercise it — which
discharges C-Live's obligation informally; a formal Mode C pass has not been run.

### M9 (R4) — 2026-08-06, swept for a miss; none found

R4 says eight `*ns.lookup(...)` dereferences are unchecked and a miss is a
SIGSEGV mid-verification, naming `phi_function`'s site as the most exposed
because it filters only two name prefixes before looking up an arbitrary merged
variable. The row has stood since M0 with no witness either way, so the question
worth answering first is whether a miss is reachable at all.

**Method.** Instrument the three goto-symex sites (`symex_goto.cpp:433`,
`symex_function.cpp:159`, `symex_valid_object.cpp:47`) to *log* a null lookup
rather than dereference it, then run the corpus. Logging rather than asserting
matters: an assert would stop at the first miss and say nothing about how many
there are, and a crash is what the row already predicts.

**Result: 352 CORE inputs, zero misses** — 261 from `regression/esbmc`, 91 from
`regression/esbmc-unix`. The `phi_function` site in particular is reached
constantly and never with a name absent from the namespace.

**What this does and does not establish.** It does not make the dereferences
safe: the sites remain unchecked, `namespacet::lookup` still returns `nullptr`
on a miss, and `renaming.cpp:15-21` checks its own lookup, so the codebase is
inconsistent about it. What it does is move R4 from "a crash waiting to happen"
to "no input in the corpus reaches it", which changes the remedy's cost/benefit:
a guard at each site would be **defensive code with no reachable witness**, and
§14 item 8 has just recorded that C-Live cannot be discharged on `src/**` while
the empirical witness is the only substitute. A guard nobody can show reachable
is exactly the dead-code candidate the coverage rule says to report rather than
add. R4 therefore stays open as a *robustness* item, not a defect with a
reproducer, and any fix should say plainly that its branches are unwitnessed.

### M9 (R8) — 2026-08-06, stack lifetime is checked, just not here

R8 reads that `is_valid_object` returns false for every non-static, non-dynamic
symbol, that stack-object validity is therefore "not modelled", and that the net
effect on stack-lifetime bugs is a **missed-bug** direction. The first two
clauses are accurate. **The third is false**, and three programs settle it:

| program | shape | verdict |
|---|---|---|
| `u1.c` | `return &local`, then dereference | **FAILED** — `accessed expired variable pointer` |
| `u2.c` | local's address stored in a global, read after return | **FAILED** — same |
| `u3.c` | address of a block-scoped local read after the block | **FAILED** — same |

Stack lifetime *is* checked; it is checked somewhere else.
`dereference.cpp:2362` asks `dereference_callback.is_live_variable(symbol)`,
implemented at `symex_dereference.cpp:153` as a per-thread call-stack search over
**L1** names — precisely what the `#if 0` comment says it could not do with
global names. The disabled block is a *superseded* implementation, not a hole,
and the comment inviting someone to "re-enable to be able to check for
stack-var-out-of-scope problems" is an invitation to re-implement a working
check.

**The surviving call site is not reached either.** `is_valid_object` has exactly
one caller (`symex_valid_object.cpp:56`), lowering `valid_object2t` on a
non-dynamic symbol. Instrumented, the three programs above hit it **zero** times,
as do 101 CORE inputs from `regression/esbmc`.

Two consequences. R8's severity should fall: there is no missed-bug direction to
fix. But the `#if 0` block, its comment and `is_valid_object`'s stack arm are now
**dead-code candidates** — an unreachable-on-this-corpus function guarding a
disabled branch. Deleting them is a separate patch needing a **C-Dead** argument,
which §14 item 8 records has no empirical substitute: 101 inputs not reaching a
site is not a proof that nothing reaches it. That deletion should rest on the
implicit-discharge route or on a far wider sweep, and this entry is not licence
for it.

### M9 (side finding) — 2026-08-06, `__assert_rtn`'s argument order

Not a goto-symex defect, and recorded because the plan's own verification found
it and because it was silently costing every macOS run five CORE tests.

Running the full `regression/esbmc` suite for the first time on this branch —
1652 tests — left eight substantive failures. Five shared one symptom: the
`test.desc` expects `^  assertion 0$` and ESBMC printed `assertion main`. The
descs are byte-identical to master, and reverting this branch's six `src/` files
to the base commit reproduced all five, so they were **not** this work.

**The cause is an argument-order conflation.**
`builtin_functions.cpp:1354` handled `__assert_rtn` and `__assert_fail` in one
arm and read `arguments[0]` for both. Their signatures differ:

| libc | call | expression at |
|---|---|---|
| glibc `__assert_fail` | `(#e, file, line, __func__)` | argument **0** |
| Darwin `__assert_rtn` | `(__func__, file, line, #e)` | argument **3** |

Confirmed from the macOS SDK header, which expands `assert(e)` to
`__assert_rtn(__func__, __ASSERT_FILE_NAME, __LINE__, #e)`. The FreeBSD
`__assert` arm *immediately below* already handles this exact order by reading
`arguments[3]` — the Darwin spelling was grouped with the wrong sibling. Every
assertion counterexample on macOS therefore named the enclosing function where
the failing expression belonged, which is a diagnostic defect on every macOS
user's assertion, not only in tests.

One-line fix; the C suite goes from 10 failures to 5. The five that remain are
`cwe_dead_code_concurrent`, `cwe_dead_code_dead_store_sarif` and `github_2572_2`
— unrelated and unexamined — plus `github_4076_complex_deref{,_fail}`, which are
**empty untracked directories** left by another session and registered as tests
by a stale CMake configure, so they fail unconditionally and are not tests at
all.

No new regression test is added: the five tests this fixes already assert
`^  assertion 0$`, which is the fix's contract and is platform-independent —
they exercise the `__assert_fail` path on Linux and the corrected
`__assert_rtn` path on macOS.

### M9 (side finding 2) — 2026-08-06, `--ir-ieee` cannot prove `0*f == 0`

The second of the three substantive C failures left after the `__assert_rtn`
fix. Also not goto-symex; recorded for the same reason.

`regression/esbmc/github_2572_2` is CORE, runs `--z3 --ir-ieee`, and expects
SUCCESSFUL. It reports FAILED on `assert(0*f==0)` with a counterexample naming
`f = 1.797693e+308` — a finite value, so the program's `!isnan`/`!isinf`
assumptions hold and the property is true in IEEE: `0 * finite` is exactly `±0`.

**Isolated to one operation.** Seven single-assertion probes under
`--z3 --ir-ieee`:

| assertion | verdict |
|---|---|
| `0*f == 0` | **FAILED** |
| `1*f == f`, `f*1 == f` | SUCCESSFUL |
| `f+0 == f`, `f-0 == f`, `0.0+0.0+f == f` | SUCCESSFUL |
| `100.0+10 == 110` | SUCCESSFUL |

**Mechanism**, from `ir_ieee_convt::apply_ieee754_rne_enclosure`
(`src/solvers/smt/fp/ir_ieee_conv.cpp:161-215`): the enclosure widens each bound
by `eps_rel * |x| + eps_abs`, where `eps_abs` is the format's minimum subnormal
and is added **unconditionally**. An exact result of zero therefore receives the
enclosure `[-eps_abs, +eps_abs]` rather than `{0}`, and equality with `0` is no
longer implied. The relative term alone would collapse correctly at zero; the
absolute term is what prevents it.

Why the multiplicative identity survives and the annihilator does not is
consistent with this: `1*f` is constant-folded before the encoder sees it,
while `0*f` cannot be folded soundly — `0 * NaN` and `0 * Inf` are NaN, so the
simplifier is right to leave it alone, and the enclosure path then runs.

Direction is **incompleteness** (a spurious counterexample, P1), not
unsoundness: the enclosure is too wide, never too narrow. A fix would special-case
an exactly-zero enclosure, which needs its own argument that no rounding mode
produces a non-zero result from an exactly-zero real, and is not attempted here.

**Caveat.** This is measured on macOS only. The enclosure code is
platform-independent, but the test is CORE and presumably green in CI, so either
CI does not run it or something upstream differs by platform. Not resolved.

### M9 (corpus sweep closed) — 2026-08-06, 1652 tests, three failures, all explained

Closing the sweep the consolidated pass opened. The first full run of
`regression/esbmc` on this branch left ten failures with no attribution. Every
one is now accounted for, and the end state is **3 of 1652**:

| Failure class | Count | Resolution |
|---|---|---|
| `__assert_rtn` argument order | 5 | **Fixed** — §15 M9 (side finding) |
| Stale `out.sarif` in the source tree | 2 | **Not defects** — see below |
| `--ir-ieee` on `0*f == 0` | 1 | **Recorded, unfixed** — §15 M9 (side finding 2) |
| Phantom `github_4076_complex_deref{,_fail}` | 2 | **Not tests** — empty untracked directories |

**The two `cwe_dead_code_*` failures were stale artifacts, not defects.** Both
declare `CHECK_FILE`/`CHECK_JSON` on an `out.sarif` the run produces. Deleting
the untracked `out.sarif` files sitting in their source directories makes both
pass. Their origin is not this work — the harness runs ESBMC in a private
`tmp_dir` and resolves the checks against it, and three consecutive runs after
the deletion pass and leave **nothing** behind, so the tests do not poison
themselves. The files predate the sweep, most likely from another session on
this clone or an older binary.

Worth recording only because the failure is silent and misattributable: a
leftover output file in the source tree makes a passing test fail with a
diagnostic (`CHECK_JSON file not found`) that points at the *harness*, not at
the stale file. Anyone seeing it will suspect the CWD handling, as this entry's
author did for two iterations.

**The phantom pair is the same class of hazard.**
`regression/esbmc/github_4076_complex_deref{,_fail}` are empty, untracked
directories that a stale CMake configure registered as tests; they fail
unconditionally and contain no `test.desc`. Neither is in git.

What this leaves is one real, recorded, unfixed defect (`--ir-ieee`) and two
non-tests. The suite is otherwise clean on macOS for the first time in this
plan's history — which matters beyond tidiness, because every prior milestone's
"regression scope" line was measured against a corpus with five silently
mis-attributed failures in it.

### M9 (Mode C) — 2026-08-06, the self-verification obligation cannot be met

The entry above ends owing a Mode C pass: the R29 fix adds a branch to `src/**`,
and the repo rule requires C-Live for exactly that. Attempting it settles the
question for every such patch, so it is recorded here rather than in that entry.

**It cannot be run, and the reason is one line long.** C-Live means instrumenting
the new branch with `__ESBMC_unreachable()` and verifying the file. Verifying the
file means parsing it, and `--parse-tree-only` over the patched
`src/goto-symex/execution_state.cpp` emits exactly one distinct error:
`field has incomplete type 'mapped_type' (aka 'irept')` — **G9**. Measured on the
real target, not inferred from the `renaming.h` probe. Recorded as §14 item 8.

Two things follow that are worth separating.

**The obligation has a substitute for C-Live, and none for C-Dead.** An
*empirical* reachability witness — an input that drives the new branch and moves
an observable — is available and, for R29, is three regression tests whose
verdict flips. That is strictly weaker than C-Live: it shows reachability on the
inputs tried, not in general. **C-Dead has no such substitute**: its obligation
is that a *removed* branch was unreachable, a negative that no finite set of
inputs can establish. C-Dead on `src/**` therefore rests entirely on the
implicit-discharge route the rule already provides — a cited issue or failing
test proving the branch was live — and any report claiming otherwise is
overclaiming.

**The gap is now one decision wide, not a backlog.** §14 item 1(a) has said since
M0 that parsing is blocked by "G1–G8, starting with `<shared_mutex>`". Six of
those closed without this plan noticing (§15 M9 (G-remeasure)) and G7 was never
open at C++23. What is left is G9 alone, and G9 is not a missing facility — it is
a choice between matching libstdc++'s incomplete-`mapped_type` extension in the
OM's `map` and changing `named_subt` across ESBMC. Closing it would make Mode C
on `src/**` reachable for the first time, which is a larger prize than the
Tier-B′ pilot it was filed under.
### M9 (R16) — 2026-08-05, R16 re-measured: most of the list was stale, and the residue splits three ways

R16 was recorded at §15 M5 as ten `--no-simplify` divergences and never re-run.
Re-running it against a current binary is the whole of this entry, and it is
worth doing before triaging any single input, because seven of the ten no longer
diverge at all. Triaging a stale list is how a fixed defect gets a root-cause
essay written about it.

**Seven agree now.** `github_1174_{hex,lmod,oct,pass}`, `github_2341_3`,
`github_785-2` and `realloc13`. #6660, #6675 and #6676 all landed between M5 and
now, and each removed a modelling decision gated on `do_simplify` — which is the
shape the whole R16 list turned out to share, and is why they cleared several
entries at once rather than one each. I did not rebuild at the intervening
commits to attribute per test, so that is the likely account and not a measured
one; what is measured is that they agree.

**Three were never the simplifier's doing.** `github_2357_5` and `github_2566_1`
select `--ir`; `github_562`, which the original list did not contain, selects
`--fixedbv`. Neither encoding decides the C program — `--ir` reasons over
unbounded integers and reals, `--fixedbv` models floats as fixed-point — so
under them the verdict can turn on how much the simplifier folded in exact C
semantics *before* encoding. The same control settles all three: drop the
encoding flag, keep `--no-simplify`, and both legs agree again. That is decisive
in a way the reasoning alone is not, because it isolates the encoding rather
than the flag under test.

The three are instructive read individually. `github_2357_5` asserts
`(unsigned)-1 == UINT_MAX`; unsimplified, the counterexample prints the claim as
`casted_unsigned == (unsigned int)2147483647 * 2 + 1` — the unfolded `UINT_MAX`
macro, which `--ir` has no modular arithmetic to evaluate. `github_2566_1`
asserts a float ULP identity that is *true* under IEEE rounding and false over
the reals, so folding it with real float semantics is exactly what made the
default leg pass. `github_562` fails on `(int)3.75L == 3` under a 128-bit
fixed-point `long double`.

So the oracle now skips tests selecting `--ir`, `--ir-ieee` or `--fixedbv`,
reporting them as `abstract` and naming them, rather than comparing them. This
is a structural waiver rather than three baseline rows on purpose: the premise
fails for the whole class, so a future `--ir` test should be excluded without
anyone re-deriving the argument. It does cost coverage — it withdraws tests that
currently agree, which is why the count is printed.

**`github_252` is the only original entry left**, and it is the sound direction:
the default proves the property by induction at k=2, and without the simplifier
the forward condition no longer closes, so the run is UNKNOWN rather than wrong.
Incompleteness under a non-default flag; baselined.

**Two divergences the original list did not contain**, both on one input, and
they turn out to be one mechanism pointing in opposite directions — recorded as
**R28**. `--no-simplify` leaves `calloc`'s constant `total_size` unfolded, so the
trailing `memset` takes `__memset_impl`'s byte-wise loop instead of the folded
form, and the test's `--unwind 1` truncates it. `github_1257-memcleanup` adds
`--no-unwinding-assertions`, so the truncation becomes an `assume(false)` that
cuts every path and a real CWE-401 leak passes vacuously — FAILED → SUCCESSFUL,
in silence. `github_1257-memsafety` keeps unwinding assertions on and the same
truncation surfaces honestly as `unwinding assertion loop 4`.

**Corrections worth recording**, because both were wrong in ways that read as
results:

- The vacuity hypothesis looked *refuted* when raising the bound to `--unwind
  64` left the verdict SUCCESSFUL. It was not refuted, only under-tested: the
  memset is 800 bytes, so no path completes until the bound clears all of them.
  At `--unwind 801` the leak reappears and names `dynamic_2_array`. A bound that
  is merely larger is not a control; it has to be larger than the loop.
- A run that appeared to show the leak missed in the *default* configuration was
  an artifact of a shared flag list that already contained `--unwind 1
  --no-unwinding-assertions`. With unwinding assertions left on, truncation is
  reported rather than assumed away, so the silent form needs that flag and R28
  is scoped to the combination, not to the default.

The reduction is worth keeping either way: the discriminator is `calloc`, not
symbolic sizes or the leak checker. The same leak spelled `malloc(800)` is
caught under both legs, because nothing memsets it.

**Where H-C2 stands.** 1198 agreed, 3 diverged,
206 inconclusive, 55 skipped,
42 abstract. The inconclusive count is large and is mostly the
20 s per-leg timeout under a loaded machine rather than anything about the flag;
it bounds what this relation currently covers and is the obvious next thing to
reduce. **That attribution is wrong** — see M9 (H-C2 residue) below, which
tested it.

### M9 (H-C2 residue) — 2026-08-06, the residue is the flag, not the machine

The entry above closed by naming H-C2's 206 inconclusive results as the next
thing to reduce, and attributed them to "the 20 s per-leg timeout under a loaded
machine". Testing that attribution is the whole of this entry, and it does not
survive.

**The oracle was reporting two unlike things as one number.** `inconclusive`
counted a leg that reached no verdict together with a leg that ran out of time.
Reaching no verdict is a property of the *input* — a parse failure or an
unsupported construct reproduces on any machine — whereas a timeout is a
property of the *run*. Folded together, a stable exclusion cannot be told from a
load artefact, which is exactly the question this entry needed to answer, so
`classify` now splits them and `report` prints both.

**A budgeted serial second pass tells them apart.** The first pass runs
`--jobs` ESBMC pairs at once, so an input near the bound can lose to load; the
retry re-runs only the timed-out residue, alone and at a larger bound. Serial
because removing self-contention is the point, budgeted because the worst case
is hours, and it prints `settled N of M` so a budget buying nothing is visible
rather than assumed. On a 40-test prefix at `--jobs 8`, 12 timed out — every one
of them in the `--no-simplify` leg — and the four the budget reached at 90 s
serial settled **none**.

**Direct timing says why.** Load average was 20.5 on 14 cores when the sweep was
started, so the confound was real; it is simply not what produced the residue.
Timed one leg at a time, `00_memcpy_01` takes 1 s by default and over 400 s
under `--no-simplify`, `00_endianness_01` 0.6 s and over 240 s, `00_memcpy_02`
0.2 s and over 240 s. Two orders of magnitude is not a test that a loaded
machine nudged over a 20 s bound.

**The control settles it.** `--no-slice` over the same 40-test prefix, at the
same `--jobs 8`, on the same machine in the same session, times out on
**nothing** — against 12 for `--no-simplify`. Load was not re-measured at that
point, so "same load" is an assumption rather than a reading; what is measured
is that a machine supposedly slow enough to strand 12 tests in one leg stranded
zero in the other, minutes apart. The two prefixes are also not identical
inputs, since each leg skips the tests whose own flags name it, but they overlap
almost entirely.

So the residue is a **cost asymmetry in the `--no-simplify` leg**, and giving
the sweep more time does not reduce it — at 400× the default, a bound that
covered these inputs would be hours per test.

**And the asymmetry is not the solver.** `00_memcpy_02` reports
`Symex completed in: 0.000s (31 assignments)` by default; under `--no-simplify`
it never prints that line at all, stalling before a single VCC is generated. Run
it with `--unwind 8` and the reason is named: `__memcpy_impl`'s byte-wise loop
at `src/c2goto/library/string.c:284`, unwinding without end because the folded
`sizeof(int)` that bounded it is gone. **This is R28's mechanism with the bound
removed.** R28 pinned the same loss on `calloc`'s `__memset_impl` at
`string.c:304`, where `--unwind 1 --no-unwinding-assertions` turns the
truncation into a silent vacuous proof; with no `--unwind` at all the same lost
constant simply never terminates, and that is what a "timeout" in this leg is.

It is also more general than the libc models. `00_endianness_01` stalls in a
loop the *test itself* writes (`myMemcpy`, its own `main.c:12`) and
`00_memory_leak_02` in `__ESBMC_atexit_handler` (`stdlib.c:38`), so the rule is
not about `string.c`: **any loop whose trip count `do_simplify` folds to a
constant becomes unbounded once it is switched off.**

**Seven lines reproduce it**, with no libc model and no flag pair:

```c
#include <assert.h>
int main() {
  unsigned n = sizeof(int);
  unsigned s = 0;
  for (unsigned i = 0; i < n; i++) s++;
  assert(s == 4);
  return 0;
}
```

By default this is `Symex completed in: 0.002s (23 assignments)`, SUCCESSFUL,
and the log shows exactly four `Unwinding loop 3` lines — the true trip count.
Under `--no-simplify` it reaches **iteration 5064 in 25 s** and is still
unwinding, so this is divergence rather than slowness: `n` is a constant the
default folds into the exit condition, and without the fold the guard is never
decided. That the reproducer needs neither `calloc` nor `--no-unwinding-assertions`
is the point — R28 was found through a libc model under a flag pair, but neither
is load-bearing.

**The mechanism, in four lines of source.** `symex_goto` renames the branch
guard and calls `do_simplify(new_guard)` (`symex_goto.cpp:20`); `do_simplify` is
`if (!no_simplify) simplify(expr)` (`symex_assign.cpp:221`); and the next
statement decides the branch with `is_false(new_guard)` (`symex_goto.cpp:23`),
which is a **syntactic** test for a constant node. So whether symex can see that
a loop has exited depends on `simplify()` having already folded the renamed
guard to a literal. Turn the fold off and `i < n` stays a comparison node even
though both operands are constant-propagated constants, `is_false` never holds,
and the back-edge is taken forever.

The two escape hatches confirm it rather than mitigate it. `--smt-symex-guard`
asks the solver the same question (`symex_goto.cpp:30`), and
`--no-simplify --smt-symex-guard` on the reproducer stops at
`iteration 4` with `Symex completed in: 0.002s` and SUCCESSFUL — the guard was
always decidable, only the syntactic check failed. It is off by default. The
interval guard cannot help by construction: its comment states it prunes only
when the guard is provably *true* and never sets `new_guard_false`, precisely so
it cannot force entry into a loop.

**Two corrections, both found by writing the fix.** They matter because each
makes this entry's account narrower than the defect.

The first is the blast radius. This entry reaches the mechanism through a
`sizeof`-derived bound and reads as though loops whose bound needs *folding* are
the population. They are not: baselined against an unpatched binary,
`for (unsigned i = 0; i < 4; i++)` — a **literal** bound — also fails to
terminate under `--no-simplify`. Essentially no loop exits under the flag, which
is a much larger claim than the one made above and is the one the evidence
supports.

The second is that there are **two independent gates**, not one, and the entry
above describes only the first. Constant propagation records a value only when
the right-hand side already *is* a constant —
`const_value = constant_propagation(rhs) ? rhs : expr2tc()`
(`goto_symex_state.cpp:367`) — and under `--no-simplify` the rhs is never
folded, so `unsigned n = sizeof(int);` records nothing at all and no later fold
can recover it. That is why the `sizeof` reproducer still diverges once the
guard-fold gate is fixed, while literal and simple-variable bounds terminate.
Reading the guard fold as the whole mechanism is what made a partial fix look
like a complete one, and only the unpatched baseline separated them.

That makes the fix direction narrow, and worth stating even though this entry
does not take it: `--no-simplify` is documented as "Do not simplify any
expression" (`options.cpp:958`), but the fold at `symex_goto.cpp:20` is not an
encoding choice — it is how a control-flow decision is made. Conflating the two
is the defect, and either always folding the guard for the exit decision, or
having `--no-simplify` imply `--smt-symex-guard`, would remove it. That is the
account of the 206 — it subsumes them into R28 rather than leaving them as a
coverage gap, and it is the reason no bound the oracle can afford will reduce
the count. It also sharpens R16's conclusion from the other side: `do_simplify`
is load-bearing not merely for formula size but for *termination*.

### M9 (R30) — 2026-08-06, the same mechanism with no flags at all

The entry above ends on a mechanism, and a mechanism invites the obvious
question. If loop termination rests on `simplify()` having folded the guard, and
nothing else in the default configuration can decide a loop exit, then the
default configuration terminates *exactly* on the loops `simplify()` happens to
fold. That is a property of a fixed rule set, not a guarantee. One loop it does
not fold would move this from a `--no-simplify` story to a default-configuration
one, so that is what this entry goes looking for.

**Five candidates, one hit.** A concrete trip count held in an array element, a
struct field, a `double` narrowed to `unsigned`, and a union member all fold and
verify in under a second. A pointer difference does not:

```c
int a[5]; int *p = &a[0], *q = &a[4];
unsigned n = (unsigned)(q - p);
for (unsigned i = 0; i < n; i++) s++;
```

Plain `esbmc` on that reaches **iteration 867405 in 20 s** and is still
unwinding. Recorded as **R30**.

**Three controls say it is the guard fold and nothing else.** `assert(n == 4)`
with the loop deleted proves SUCCESSFUL, so the value is fully determined; the
same program with the bound written as the literal `4` proves SUCCESSFUL while
still asserting `n == 4`, so the arithmetic is not the obstacle; and
`--smt-symex-guard` on the unmodified reproducer stops at `iteration 4` in
0.004 s, so the guard is decidable and only the syntactic test fails. Together
they isolate the fold rather than the pointer arithmetic, the loop, or the
solver.

**What this does to R28.** R28 was written as a `--no-simplify` defect, and its
unbounded form as a consequence of switching the fold off. R30 reaches the same
non-termination with no flag set, so `--no-simplify` was never the cause — it
only widens the set of guards that fail to fold, from "what `simplify()` misses"
to "everything". The honest reading is one defect with two populations, and the
default-configuration one is the more serious even though it is much rarer.

Worth being exact about severity, because "hangs forever" reads worse than it
is. R30 produces **no verdict**, not a wrong one; nothing is proved that should
not be. What a user meets is a five-line program with a statically known bound
that never returns and never says why — a completeness and usability defect
rather than a soundness one.

**Filed and part-fixed.** R28 is **#6778**, R30 is **#6779**. **#6781** fixes
R28's guard-fold gate by deciding the branch on an unconditionally simplified
copy. Re-measured over 300 CORE tests against an unpatched build under
comparable load: 238 agreed / 49 non-terminating becomes 269 / 18, with no new
divergence and the `no-verdict` count unmoved at 13 either way -- the last of
those is the useful check, since that category is a property of the input and a
patch that shifted it would be doing something other than advertised. It does not touch the constant-propagation gate above, and by
construction cannot help R30, whose guard `simplify()` fails to fold even when it
runs.

**Where R30's fix would go.** The GOTO carries the bound as
`ASSIGN n=(unsigned int)(q - p)` — a bare `sub2t` over two pointers.
`sub2t::do_simplify` (`expr_simplifier.cpp:635`) already handles `x - 0`,
`0 - x`, `x - x` and `(base + X) - X`, but has no case for two `address_of`
expressions into one object. After constant propagation the operands are `&a[4]`
and `&a[0]`, so the missing fold is `&base[i] - &base[j]` → `i - j` for constant
`i`, `j` and a syntactically identical `base`.

That is not an optimisation guess. **C23 6.5.6p9** (N3220): "When two pointers
are subtracted, both shall point to elements of the same array object, or one
past the last element of the array object; the result is the difference of the
subscripts of the two array elements." The fold computes exactly what the
standard defines the result to *be*, and declining to fold when the bases differ
is conservative rather than incomplete, since the program is undefined there
regardless.

### M9 (R16 closed) — 2026-08-06, the last baseline entry was a symptom

Fixing #6778 makes a prediction worth checking rather than assuming: if the
guard fold is what a `--k-induction` forward condition needs in order to close
over a bounded loop, then R16's last surviving entry should stop diverging on
its own. It does. `github_252` reports SUCCESSFUL on both legs with #6781
applied, and diverges (SUCCESSFUL vs UNKNOWN) on a master build rebuilt for the
comparison; the guard fix alone is enough, with the pointer fold not involved.
So R16's residue was never a simplifier gap of its own — it was #6778 seen
through k-induction. Its baseline entry is removed, the plan's own rule being
that a fixed defect may not keep its exemption.

**R28's two entries could not be judged by that check** — and the reason was my
invocation, not the toolchain. Running `github_1257-memcleanup` and
`github_1257-memsafety` by hand gave `ERROR: PARSING ERROR` on a `memset`
builtin redeclaration, which this entry first recorded as "they do not parse on
macOS". They parse fine. Their `test.desc` carries `-D'__builtin_unreachable()'`
and `generate_run_argument_list` strips the quotes to one argv entry, whereas
`$(sed -n 3p test.desc)` in a shell word-splits the quotes into the argument and
produces a malformed `-D`. The full sweep below runs both and gets verdicts from
each.

That is exactly the hazard Appendix B states about reimplementing `test.desc`
parsing — "how a sweep ends up reporting divergences that are really invocation
differences" — reached by hand rather than in a script, which is the one place
the warning does not look like it applies. A first pass had also read the pair's
two empty results as *agreeing*, which the `no-verdict`/`timeout` split exists to
prevent. Two wrong readings of one pair, neither of them about ESBMC.

Taken as its own change rather than a tail-end edit, because the simplifier is
reached from everywhere and a wrong fold is a soundness bug rather than a missed
optimisation: **#6783**. The operands turn out to be bare `address_of` after
constant propagation, so no typecast peeling is involved — which matters here,
because peeling a cast can change the element size the subscripts are measured
in, and that is precisely how such a fold goes wrong. The reproducer completes
in 0.000 s where it previously passed iteration 867405.

Its regression pair asserts `q - p == 4` and `p - q == -4` alongside the loop
count, not merely that the run terminates: a fold that terminates with the wrong
difference is the failure mode worth pinning, and a termination-only test would
pass straight through it. Around 2300 tests were run against the change — unit,
pointer, cbmc, C++, floats, k-induction, and the core/unix/github slice — and
each of the seven failures was reproduced on a rebuilt master binary before
being set aside, since "it fails on master too" is worth nothing unless it is
checked rather than assumed.

---

### M9 (R6) — 2026-08-06, the witness, and why four attempts missed it

R6 sat at "mechanism pinned, no witness" after four targeted programs failed to
produce a verdict-changing prune. The reason they failed is in the finding's own
wording: it describes the collision as *recursive*, "two states that differ only
in the L1 activation of a recursive local", and a witness built to that
description has to arrange equal L0 values at two recursion depths — which is
hard, because the recursion parameter is itself an L0 value that differs.

Reading `generate_hash` rather than the finding gives a weaker condition. The
fingerprint is the L2 value map (keyed by L0 name) combined with each thread's
pc, and nothing else. Recursion is one way to reach one pc with one value map
and two continuations; **one function called from two sites is another**, and it
needs no recursion at all:

```c
void f(void) { int x = shared; (void)x; }
void *worker(void *p) { f(); f(); assert(0); return 0; }
```

With a second (idle) thread and `--unwind 4`, this reports FAILED by default and
**SUCCESSFUL under `--state-hashing`** — a reachable assertion silently pruned —
agreed by Bitwuzla and Z3.

The global read inside `f` is load-bearing and is what the four earlier attempts
lacked: `check_for_hash_collision` runs at a context-switch point
(`reachability_tree.cpp:732`), so the colliding state has to *be* one. Without a
shared access inside the callee the two calls never present the state for
comparison.

**Two controls.** A single call finds the bug (so `--state-hashing` is not
broken across the board), and writing `shared = 1` between the two calls also
finds it (so it is the equality of the visible state that does the damage, not
the repetition). Together they pin the miss to two occurrences of one state at
one pc with different continuations.

**The remedy this finding proposed would not have worked.** R6 suggested mixing
call-stack *depth* into the fingerprint. Implemented first, and the witness still
passed vacuously: both calls come from `worker`, so both states sit at equal
depth. What separates them is the return address, so **#6785** mixes each
frame's `calling_location` instead. Had the recorded remedy been trusted rather
than tested against a witness, the fix would have shipped without fixing
anything — which is the argument for finding the witness before the fix, not
after.

---

### M9 (R15 regression) — 2026-08-06, the fix broke `--incremental-bmc`

R15's fix reset `dynamic_counter` and `dereferencet::object_counter` in
`setup_for_new_explore`, on the reasoning that object names should depend on
(program, options) alone rather than on how many objects earlier explorations
minted. The reasoning holds; the placement does not, and five
`regression/python` tests turned red on the macOS CI leg because of it.

`setup_for_new_explore` is called from `bmct::run`, and `--incremental-bmc`
calls that once per k iteration while the symbol state persists across them. So
iteration k+1 re-mints object names iteration k already bound at a different
type, the two meet as operands in one formula, and Bitwuzla aborts:

```
Assertion failed: (a->sort->get_data_width() == b->sort->get_data_width()),
function mk_eq, file bitwuzla_conv.cpp, line 470
```

**Three builds settle it**, all DebugOpt, which is what the CI leg uses: the
branch aborts, master passes all five, and the branch with only R15 reverted
passes all five. The reset is the cause, not a coincidence of the branch.

**Why it never showed up locally.** A default RelWithDebInfo build carries
`-DNDEBUG`, so the assertion is compiled out and the mismatched `mk_eq` proceeds
in silence; the same tests pass. `--z3` does not complain either. This is R1's
observation arriving from the other direction — an invariant that exists only in
debug builds is not enforced in the shipped binary, and here that was the
difference between a red CI leg and a green local run.

**The methodological lesson is about the ownership check, not the bug.** The
first pass at attributing this compared the branch's `src/python-frontend` and
`regression/python` against master, found them byte-identical, and concluded the
branch could not be responsible. The inputs were Python; the defect was in
symex, which the branch does change. Matching the *language of the failing test*
against the changed files is not an ownership argument — the only thing that
settles it is building both revisions and running them, which took one rebuild
and would have replaced two wrong hypotheses (a 120 s timeout, a Python version
difference) that each looked plausible and cost more than the experiment would
have.

**What the fix has to satisfy is now exact.** The two requirements look like
they conflict at one call site, and they do not, because they differ in what
they share. `is_base_case_violated` and its siblings build a **fresh `bmct` per
k iteration but pass the same `context`** by reference
(`k_induction.cpp:704`), so the objects iteration k bound are still in the
symbol table when iteration k+1 re-mints their names. R15's determinism harness
is the opposite: `symex_run::equation` constructs its own `prog`, and therefore
its own `prog.context`, per instance, so the two runs the test compares share
nothing.

The rule is therefore **reset when the context is fresh, not when an exploration
begins** — the context is fresh exactly when the run is independent, which is
the distinction `setup_for_new_explore` cannot make on its own but the context
can make for it. Two ways to implement it:

- *Seed rather than reset*: start each counter from the number of such objects
  the context already holds. A fresh context gives 0, which is what the
  determinism test needs; a shared one continues monotonically, which is what
  `--incremental-bmc` needs. Self-keying, no extra state.
- *Scope the counters to the context*: move them off `static thread_local` so
  their lifetime is the context's. Cleaner, and larger.

A static pointer to the last-seen context is **not** a third option: a freed
context can be reallocated at the same address, and the reset would then be
skipped exactly when it is needed.

**Fixed** with the first shape: `setup_for_new_explore` resets only when the
context holds no `symex_dynamic::` symbol. All five inputs now match master's
verdicts, R15's two determinism tests still pass — the harness gives each run
its own `prog.context`, so the reset still fires for them — and
`regression/esbmc/incremental_bmc_object_names` pins the k-iteration case
directly rather than leaving the Python suite to notice.

Two wrong attempts preceded it, both cheap and both caught by running rather
than reasoning. Probing for `symex_dynamic::dynamic_1_value` never fired,
because `malloc` mints `dynamic_N_`**`array`** and only the struct path uses
`_value`; the probe therefore has to be the prefix, not any single name.
Keeping just the dynamic-counter reset still aborted, which is what established
that counter as sufficient on its own rather than the pair being jointly
responsible.

---

### M9 (H-C2 re-measured) — 2026-08-06, the residue was hiding nothing

With #6781 and #6783 both applied, H-C2 was re-run over the whole corpus. The
question it answers is not "does the relation still hold" but "what was behind
the timeouts", because a timed-out test is one the oracle **never compared** —
so every input the fixes rescued is an input this relation had never covered.

| | agreed | diverged | inconclusive | skipped | abstract |
|---|---|---|---|---|---|
| §15 M9 (R16), pre-fix | 1198 | 3 | 206 | 55 | 42 |
| this run, post-fix | **1299** | **2** | **120** (37 no-verdict + 83 timeout) | 55 | 42 |

101 tests moved from uncompared to agreeing, and the inconclusive count fell by
86. **No new divergence appeared.** The two that remain are the `github_1257`
pair that R28 already owns and the baseline already carries; `github_252`, R16's
last entry, now agrees and its baseline line is gone with #6781.

That is the useful result, and it is a negative one: the timeout residue was not
concealing a defect. H-C2's coverage claim can now be stated as measured rather
than bounded — the relation holds on 1299 of the 1421 inputs it is entitled to
compare, and the 120 it still cannot reach split into 37 that reach no verdict
(a property of those inputs) and 83 that remain over the bound.

Worth noting what a *whole-corpus* run cost to learn this: a little over an
hour. The 300-test prefix used earlier in M9 predicted the direction correctly
but would not have settled the question, because the interesting population —
tests that only became comparable once the fixes landed — is exactly the one a
prefix under-samples.

---

### M9 (H-C1 re-measured) — 2026-08-06, and what a clean leg does not prove

H-C1 had not been re-run since M5. Re-running it serves two purposes: the
slicer relation has never been checked against the inputs the fixes rescued
from timing out, and #6783's pointer fold is **unconditional** — unlike the
guard fold, it fires in every configuration — so a relation it might disturb is
worth exercising deliberately.

| | agreed | diverged | inconclusive | skipped | abstract |
|---|---|---|---|---|---|
| §15 M5 | 1328 | 0 | 67 | 35 | — |
| this run | **1364** | **0** | **52** (39 no-verdict + 13 timeout) | 39 | 63 |

Still zero divergences, over 36 more agreeing inputs than M5 saw.

**What that establishes, precisely.** It is evidence the pointer fold does not
disturb the slicer relation, and no more. It is *not* a check that the fold
preserves verdicts against master, and it cannot be: both legs of every oracle
here run the same binary, so a fold that changed a verdict identically in both
legs would leave every relation intact and every count unmoved. The oracles
compare configurations, not revisions. What actually pins the fold against
master is the regression suite — roughly 2300 tests, with each of the seven
failures reproduced on a rebuilt master binary before being set aside — plus
the fold's own pair asserting `q - p == 4` and `p - q == -4` rather than only
that the run terminates.

Worth stating because a clean sweep is easy to over-read: 1364 agreeing inputs
is a strong statement about `--no-slice` and says nothing whatever about
whether the default leg is right.

---

### M9 (state-hashing re-measured) — 2026-08-06, #6785 costs nothing measurable

#6785 mixes each frame's calling location into the state fingerprint, which
makes two states *less* likely to collide. The risk it carries is therefore not
unsoundness but over-precision: fewer prunes, more states explored, and in
principle a timeout or a verdict shift somewhere pruning had been load-bearing.
The state-hashing leg is the relation that would show it.

| | agreed | diverged | no-verdict | timeout |
|---|---|---|---|---|
| §15 M6 | 255 | 0 | — | — |
| this run, with #6785 | **1396** | **0** | 40 | **0** |

No divergence over a sample five times larger than M6's, and **zero timeouts** —
so the added precision did not strand a single input. The prune that R6's
witness showed to be unsound is gone without the pruning that remains costing
anything measurable here.

The same caveat as H-C1 applies and is worth repeating rather than assuming
read: both legs run one binary, so this establishes that the relation holds
under the new fingerprint, not that the fingerprint preserves verdicts against
master. What pins that is the regression suite plus
`state_hashing_callsite_sound_fail`, which was shown to fail on master and pass
with the fix — a test that only passes after the change is the one kind that
cannot be satisfied by doing nothing.

---

### M9 (POR re-measured) — 2026-08-06, and why it does not touch A6.4

A6.4 is the last finding in §9.2 with no verdict: `calculate_mpor_constraints`
resets the active thread's row to -1, and that reset is the one operation in the
chain update that *removes* relations, so an unsound prune would live there. The
POR leg is its empirical counterpart — if the reset dropped a dependency that
mattered, POR would prune an interleaving `--no-por` explores, and a bug behind
that interleaving would go missing.

| | agreed | diverged | no-verdict | timeout |
|---|---|---|---|---|
| §15 M6 | 258 | 0 | — | — |
| this run | **1409** | **0** | 39 | 2 |

**This does not discharge A6.4, and M9's own R6 entry is why.** R6 named a real
unsound prune in `--state-hashing`, and the H-C4 state-hashing leg was clean at
the time — 255 agreed, 0 diverged — for a structural reason that applies here
unchanged: a pruning defect only changes a *verdict* when the pruned state is on
the path to the **only** buggy interleaving. A corpus of concurrent tests
mostly admits many racy interleavings, an early one of which is found before any
prune matters. R6's witness had to be built to make the bug reachable
exclusively behind the collision, and no corpus test happened to have that
shape.

So 1409 agreeing inputs bounds how often A6.4 bites in practice and says
nothing about whether the reset is sound. Discharging it still needs either the
MPOR paper's chain definition — the citation is verified (Kahlon, Wang & Gupta,
CAV 2009, pp. 398-413, doi:10.1007/978-3-642-02658-4_31) but the text is
paywalled — or a witness constructed against the reset the way R6's was
constructed against the fingerprint. The second is now a known-workable
technique rather than a hope, which is the one thing this session changed about
A6.4's prospects.

---

### M6 (A6.4) — 2026-08-12, the reset is checked rather than argued

A6.4 has been the one row with no verdict since M6: `calculate_mpor_constraints`
resets the active thread's row to −1, that reset is the only operation in the
chain update which *removes* relations, and a removed relation is the direction
that costs interleavings. M9's POR leg (1409 agreed, 0 diverged) deliberately
did not close it, for the reason R6 had already demonstrated: a pruning defect
only moves a verdict when the pruned state is on the path to the *only* buggy
interleaving.

**The reset is sound, and the argument is one line once the invariant is written
down.** DCij asserts a dependency chain from Ti's last transition to Tj's, and a
chain follows execution order, so DCij = 1 requires Ti's last transition to
precede Tj's. The transition just taken is the newest in the run. Every entry in
the active thread's row bar the diagonal therefore asserts a chain leaving the
newest transition for an older one, which no run contains. The reset is not an
optimisation that might drop something — it is what keeps the matrix's own
meaning true, and it clears **exactly** the set the ordering forbids: all of row
`a` bar the diagonal, no more.

That is also the completeness half, which is what "removes relations" was really
asking. Nothing recoverable is lost, because the row is rebuilt forward: when a
later thread Tm runs, the column update writes DCjm from any DCjl already set,
so chains out of the active thread's *new* transition are recorded as they arise
rather than carried over. With the addition half already settled in M6 — the
two-hop step is MPOR's recurrence, and the `res == 0` non-overwrite only ever
keeps an extra 1, which loses reduction rather than interleavings — A6.4's
"preserves transitive closure" has a verdict in both directions.

**Written down, the invariant is a comparison, so it ships.** The engine now
carries `thread_last_transition`, the run-order ordinal of each thread's last
completed transition, advanced where the chain is advanced. Three checks run on
every transition, inside the loop already walking the threads, under
`SYMEX_INVARIANT` and so in the `-DNDEBUG` binary (R1, M3):

| Check | What it pins |
|---|---|
| `new_dep_chain[j][a] != 1 ∨ ord[j] < ord[a]` | no chain into the newest transition starts after it — covers the `res == 0` path, which keeps a 1 recorded against an *older* transition of the active thread |
| `new_dep_chain[a][j] != 1 ∨ ord[a] < ord[j]` | no chain leaves the newest transition — this is the reset |
| `(DCjj == 1) ⇔ ord[j] ≠ 0` | the diagonal means "has run", the precondition the un-run guard reads |

Only the active thread's row and column change meaning when it takes a
transition; every other entry keeps both its endpoints. So checking those two is
checking the inductive step, at O(T) per transition inside a body already doing
O(T²).

**It discriminates.** Deleting the reset — the one-line mutant A6.4 is about —
trips the row check on the first concurrent program tried,
`regression/esbmc/19_time_var_mutex_true-unreach-call` under its own flags:

```
ERROR: goto-symex invariant violated in calculate_mpor_constraints
  condition: new_dep_chain[active_thread][j] != 1 ||
             thread_last_transition[active_thread] < thread_last_transition[j]
  MPOR dependency chain leaves the newest transition
```

and fails `unit/goto-symex/mpor.test.cpp`, which re-checks the same property
from outside the engine on every interleaving of a two-writer program — `9 < 4`,
a chain recorded backwards. Both legs were green immediately before and after
the mutation, on the same tree.

**Anti-vacuity.** The unit test counts explorations in which some thread took a
second transition, since only there does the reset have anything to clear, and
requires that count to be non-zero: 1471 assertions over 49 interleavings. The
corpus leg is every CORE/THOROUGH test in
`regression/{esbmc,esbmc-unix,esbmc-unix2}` that creates a thread, run under its
own flags — **481 tests, 0 violations** (462 to a verdict, 18 to a 30 s cap, 2
without a source file). The cap does not hide anything: the check runs on every
transition from the first, so a capped run is a run that took thousands of
transitions without tripping it.

**What this does not claim.** The runtime check enforces the direction the reset
*establishes* — no chain pointing backwards. The other direction, that no true
chain is lost, has no runtime witness, because a dropped entry leaves no trace
in the matrix; it rests on the ordering argument above. The two are the same
fact seen twice: the entries cleared are precisely the entries the ordering
forbids.

**A6.4 closed.** With it, the row §15 M9 named as the last one carrying no
verdict has one.

---

### M9 (R29 residual) — 2026-08-12, the last two shapes, and a corrected cause

M9 left `*(s.p)` and `*(o.in.p)` reporting **false SUCCESSFUL by default** and
routed the cause out of this plan: `get_reference_set` returned zero objects for
the struct spelling and one for the union, which read as the `member2t` case
taking different arms. **That reading was wrong, and only instrumenting the path
showed it.** Both spellings recurse to the same place. The lookup the struct
performs even *succeeds*: `c:@s.p` is found, with one object in it —

```
R29KEY lookup='c:@s.p' rlevel=3 found=1 objs=1
R29MEMB single='p' src=symbol      dest=1
R29MEMB single='p' src=constant_struct dest=0   <- the call that feeds MPOR
```

There are two calls, not one. The second reaches the member case with a
`constant_struct2t` source — constant propagation has substituted the struct's
*value* — and recursing into it lands on `is_constant_expr` at
`value_set.cpp:375`, which returns having inserted nothing unless the expression
is a `constant_int2t` or a `constant_union2t` under a dereference. There is no
`constant_struct2t` sub-case, so the value set comes back empty, the write
resolves to no object, and MPOR calls the two transitions independent.

**The union works for a reason that makes the asymmetry exact rather than
coincidental**: `constant_union2t` *is* one of the two sub-cases, added for
unions, so the union spelling descends into its member and finds `g` while the
struct spelling falls off the end. The dedicated `is_constant_struct2t` arm at
`value_set.cpp:470` never runs — `is_constant_expr` shadows it — which is why
reading the code suggests the case is handled when it is not. A C-Dead
candidate, recorded here rather than deleted in the same patch.

**Fix:** the constant case descends into the member the suffix names, consuming
one component per level, so nesting works by the same rule as the flat case.
Both shapes now report FAILED, matching `--no-por`;
`mpor_aggregate_ptr_race` flips **KNOWNBUG → CORE**, and
`mpor_aggregate_ptr_race_nested` pins the two-level form that a single-level
descent would still lose.

Each test was mutation-checked rather than assumed, and one earned its place
only on the second attempt. A binary built from the unpatched value set reports
SUCCESSFUL on the flat test, which kills the whole-fix mutant by measurement.
`_nested` is driven through two descents (`consumed=2 remain='.p'`, then
`consumed=1 remain=''`), so a single-level descent loses it. `_prefix` exists
because `p` is a declared sibling of `p2`: the naive "first declared name that
is a prefix" reading resolves the write to the wrong global and the race
vanishes, and **that mutant is invisible to every other test in the set** — the
anonymous-member test cannot see it, because its struct declares one member.
Two candidate tests were written, mutation-checked, and deleted for asserting
nothing: a punned-dereference case that reports FAILED on the unpatched binary
too, and the SUCCESSFUL-expecting case discussed below.

**Two defects review found in that fix, both confirmed against `--no-por` and
both now closed.** They are recorded because each is a *false SUCCESSFUL* of
exactly the class this entry exists to remove, and the first was introduced by
the fix itself.

*Selecting the component by scanning for the next `.` or `[` is wrong.* Member
names are not C identifiers: clang names a C11 anonymous member
`struct Outer::(anonymous at main.c:8:3)`, whose text contains `.`, so the scan
cut the name apart, matched nothing, and the race was pruned again. The
discriminating evidence is two programs differing only in a `#line`-controlled
filename — a dotted name verifies SUCCESSFUL, a dot-free one FAILED. The
leading component is now the longest *declared* name the suffix continues on a
component boundary, which is delimiter-independent by construction; a tie
resolves to no match rather than to a guess.

*The constant-union arm never consumed its suffix.* `get_value_set_rec` pushes
`"." + init_field` for a `constant_union2t`, and the arm passed that straight
into the member's own type, where it named nothing — so `*(u.in.p)` was pruned
whether or not an outer struct wrapped it. Pre-existing, and reachable before
this entry, but the struct descent now feeds it suffixes it never saw. It
consumes the initialised member's component; a component naming a *different*
member is punning this analysis cannot follow and passes through unchanged, so
no precision is lost.

A third review finding is taken but closes nothing on its own: a lookup failure
now contributes `unknown` rather than nothing, per the contract at
`value_set.h:603-605`. That is the correct value-set semantics — empty asserts
"points at nothing" to every consumer — but `resolve_pointer_target`
(`execution_state.cpp:945-947`) discards any entry that is not an
`object_descriptor2t`, so MPOR still sees nil. **The invariant worth stating at
that boundary is that empty means unanalysable, not harmless**; defending it on
the consumer side would make the next R29-class residual cost precision instead
of soundness. Recorded, not done here.

**Every test here expects FAILED, and that is forced rather than lazy.** The
coverage gate asked for the missing half of the two-test rule — a program whose
race is *correctly* pruned, pinning that the descent does not over-approximate.
No such test can exist for MPOR. Two mutations were built to kill one: selecting
the shortest matching component instead of the longest, and selecting component
0 unconditionally. **Both left the verdict SUCCESSFUL.** The reason is
structural: the value set steers only which interleavings MPOR explores, while
the data the program moves comes from constant propagation, so a wrong or
over-wide set costs *interleavings* and never the verdict of a correct program.
Under-approximation hides a real race and shows up as SUCCESSFUL; over-
approximation shows up as run time. **Only the FAILED direction carries signal
here**, and a SUCCESSFUL-expecting test would have asserted nothing — it was
written, mutation-checked, and deleted. Worth remembering the next time this
plan is asked for a negative regression test on a pruning decision.

Regression scope: 598/603 `esbmc-unix` + `esbmc-unix2` pass at `-j8`. All five
failures are the 120 s cap, not verdict changes: each passes standalone —
`03_boundedBuffer` 93.0 s, `github_2513_1` 84.6 s, `github_595` 90.1 s,
`github_6480_deepening` 82.7 s, `01_pthread60` 107.7 s. Unit tests 643/643.
The widening this entry introduces was checked for cost rather than assumed
free: `01_pthread60` runs 104.95 s unpatched against 106.14 s patched, +1.1%,
and none of the five declares a constant aggregate holding a pointer, so none
reaches the new arms at all.

**What this does not close.** R29's row stays open as H-A6's completeness
obligation: this fixes the seven shapes now enumerated, and says nothing about
shapes it did not. That the census's own five missed both defects above is the
point — the enumeration is the weak step, not the fix. One further shape is
known and unexplained: struct-to-struct punning (`((struct B *)&a)->q`) never
reaches the constant-struct arm at all and reports SUCCESSFUL. It is strict-
aliasing UB, so no soundness claim is made, but it is the next probe R29's
completeness row needs.

---

### M9 (H-A6 re-census) — 2026-08-13, 21/21, and what that is not

The census of §15 M9 (H-A6) is the artefact that pinned R29, so it is also the
one that says whether R29 is gone. Re-run against the patched binary on the same
method — writer reaches `g` through one access shape, `main` reaches it
directly, `--no-por` is the reference — **all 21 shapes now agree**, against 16
of 21 when the census was written. The ten aggregate shapes were re-run under
**both Bitwuzla and Z3**, matching the dual-solver standard the original used;
no disagreement.

The census grew by the shapes this round's defects taught it to ask about:
anonymous member, struct-in-union, union-in-struct, three-level nesting, and
prefix-named siblings. Four of the five are shapes the original enumeration
would have had no reason to include, which is the honest reading of what the
first census was worth — it found five real defects and was blind to four more
that the *fix* for those five had to discover.

**H-A6 does not close, and the reason is not modesty.** The row claims
completeness over every access shape; 21 passing shapes is an enumeration over
the shapes someone thought of, and this round had already demonstrated twice
that the thinking-of is the weak step.

**Then extending the census by two shapes demonstrated it a third time, within
the hour.** The paragraph above originally ended "no known counterexample". It
was wrong when written:

| Shape | Write | MPOR | `--no-por` |
|---|---|---|---|
| address-of member | `int **pp = &s.p; **pp = 1` | SUCCESSFUL | FAILED |
| struct-to-struct pun | `*(((struct B *)&a)->q) = 1` | SUCCESSFUL | FAILED |
| **control** | `int *lp = *pp; *lp = 1` | **FAILED** | FAILED |

Both dual-solver confirmed. The second is strict-aliasing UB and carries no
soundness claim. **The first is not**: `&s.p` is a well-defined `int **`, and
this is a false SUCCESSFUL on ordinary C in the default configuration. It is
R29's mechanism one level further out — R18 followed chains of symbols, R29
followed the aggregate step, and this puts an `address_of` in front of the
aggregate step — and the boundary is syntactic in the same way, since the local
copy restores detection. Recorded as **R31** and pinned KNOWNBUG as
`mpor_aggregate_ptr_race_addrof`, with `..._addrof_local` as the CORE control,
the same pairing R29 was filed under. Not fixed here, for the reason the
original census gave for not fixing R29: widening what MPOR treats as
conflicting needs its own soundness argument, not an append to a census.

So the row is **refuted again** rather than "no known counterexample", and the
23-shape census is worth exactly what the 21-shape one was — the defects it
happened to enumerate. What is now empirically established is not that the
resolution is complete, but that **every extension of this census so far has
found something**, which is the strongest available argument that the row should
not be discharged by enumeration at all.

The next section extends it to 28 and finds, among other things, that the 21/21
claimed above was itself over-stated.

---

### M9 (R31 fix) — 2026-08-13, the offset spelled back out

**R31's row pinned the mechanism exactly and then named the wrong component.**
It read the local-copy boundary — `int *lp = *pp; *lp = 1;` detects the race
that `**pp = 1` does not — as placing the gate "in the resolution, not the value
set", and pointed at `mpor_lock_array_key` as the precedent a fix would follow.
The boundary is real; the inference from it was not. By the time
`value_sett::assign` records `lp`, symex's dereference pass has already rewritten
the right-hand side `*pp` into the `member2t` `s.p`, and the member arm keys
`c:@s.p` directly. `resolve_pointer_target` builds a `dereference2t` of its own
and hands it to `get_reference_set` raw, so it lands in the dereference arm,
where the member survives only as a byte offset. The local copy therefore never
separated MPOR from the value set — it separated *rewritten by symex* from
*raw*, and the fault was in the value set on both sides. Worth keeping as a
method note: "shape A works, shape A′ does not, so the fault is in the consumer"
holds only when the two shapes reach the consumer by the same route.

The first version of this section blamed the boundary on simplification instead
— `get_value_set` simplifies before descending (`value_set.cpp:193`) and
`*(&s.p)` folds to a member — which is true of the code and false of this
program. Two measurements refute it. Making the pointer unfoldable
(`pp = c ? &s.p : &t.p`, then the same local copy) still detects the race on an
unpatched binary, and `lp`'s value set is still the precise `{<g, 0, 1, int>}`
though `c:@pp` now holds two descriptors with the member erased in both — a
precision no fold could supply. Const propagation is beside the point as well:
`pp` is a global written before a thread starts. The rewrite, not the fold, is
what puts a `member2t` in front of the value set.

**The gap itself is one piece of bookkeeping.** An object descriptor names an
aggregate and a byte offset (`<s, 0, 8, struct S>`); value-set entries are keyed
one field path at a time (`c:@s.p`). The dereference arm of `get_value_set_rec`
looked the base object up under the caller's suffix and dropped the offset, so
the entry holding the answer was unaddressable — and empty, to every consumer,
asserts "points at nothing". `collect_offset_paths` walks the offset back into a
path and the arm asks again under each one. It accumulates in bits as
`member_offset_bits` does, so it is the exact inverse of the walk that built the
descriptor, and it yields a path only when the descent lands on the type being
dereferenced. The unrefined lookup stays, which makes the change monotone: it
can add objects to a value set, never remove one.

**Fixing it in MPOR would have fixed nothing, and that was measured rather than
argued.** The cheap alternative the first diagnosis suggests is a `simplify()` in
`resolve_pointer_target`, folding `*(&s.p)` as `get_value_set` would. It was
built on top of a reverted `value_set.cpp` and run, in both placements the
sentence admits — on the renamed pointer and on the dereference built from it —
and every one of the six race shapes stayed **SUCCESSFUL**, the pinned
reproducer included. The reason is the paragraph above: there is no `&s.p` in
the expression to fold, because the member was erased when `pp`'s value set was
written, not when it was read. A narrow fix aimed at the symptom the first
diagnosis described would have shipped as a no-op that the census then blamed on
the shapes.

The merged-pointer shape still earns a test of its own, for the weaker claim
that survives: `pp = c ? &s.p : &t.p; **pp = 1;` reaches the descent with two
descriptors and no constant anywhere, so it is the one shape that could not be
rescued by *any* folding, wherever placed. It detects the race under MPOR and
`--no-por` alike after the fix.

**The census grows to 28, and 21/21 was over-stated.** Re-run against a binary
carrying R29's fix but not this one, the shapes give **22/28**, not 26/28:
besides the two the section above recorded, `int **ap = a; **ap = 1;` — an array
element reached through a pointer *into* the array — was also a false
SUCCESSFUL, and the re-census had listed "array elements (constant, symbolic,
via pointer)" among its passing sixteen. Whatever spelling "via pointer" had
there, it was not this one. The 21/21 was true of the twenty-one programs run
and false of the claim they were taken to support.

| Shape | Write | pre-fix MPOR | post-fix MPOR |
|---|---|---|---|
| address-of member | `int **pp = &s.p; **pp = 1` | SUCCESSFUL | **FAILED** |
| … at a nonzero offset | `struct S { int *q, *p; }` | SUCCESSFUL | **FAILED** |
| … two levels down | `int **pp = &o.in.p` | SUCCESSFUL | **FAILED** |
| … union member | `int **pp = &u.p` | SUCCESSFUL | **FAILED** |
| array element via pointer | `int **ap = a + 1; **ap = 1` | SUCCESSFUL | **FAILED** |
| merged pointer † | `pp = c ? &s.p : &t.p` | SUCCESSFUL | **FAILED** |
| struct-to-struct pun | `*(((struct B *)&a)->q) = 1` | SUCCESSFUL | SUCCESSFUL |

† not one of the 28: it was written afterwards, to test the fix's *site* rather
than another arm of its descent, so the six pre-fix disagreements the counts
above give are the other six rows.

`--no-por` reports FAILED on every row of both columns. **27 of 28 shapes agree**
after the fix; the survivor is the strict-aliasing pun, which carries no
soundness claim and never reaches the descent at all.

**Each arm is pinned by exactly one test, and that was measured rather than
argued.** Four mutants were built and run — the array arm removed, a union laid
out end to end, a struct treated as overlaid, and the struct arm matching only
direct members instead of recursing:

| Mutant | Test that dies | Tests that survive |
|---|---|---|
| array arm removed | `_array_decay` | the other four |
| union laid out end to end | `_addrof_union` | the other four |
| struct treated as overlaid | `_addrof_offset` | the other four |
| direct members only, no recursion | `_addrof_nested` | the other four |
| whole patch removed | all six race tests | `_addrof_local`, `_addrof_locked` |
| `simplify()` in `resolve_pointer_target` instead | all six race tests | `_addrof_local`, `_addrof_locked` |
| `esize > 0` guard removed | `_zero_size_element` (aborts, no verdict) | the rest |

The four arm mutants ran against the five tests that existed then;
`_addrof_merged` was added afterwards and pins no arm of its own — it walks the
same struct arm `_addrof` does. What it pins is the *site*: it is the only shape
that survives every mutant of the narrow fix, because it carries no constant for
any fold to reach. The union case earns its row only because the pointer is
declared *second*: a union whose pointer comes first is reached by a
struct-shaped walk too, so the obvious spelling of that test would have covered
the arm without pinning it. `mpor_aggregate_ptr_race_addrof` flips KNOWNBUG →
CORE, and `..._addrof_local` stays the control that fails if the working shape
regresses.

**The coverage gate blocked this change twice, and both were fair.** Running it
on the diff returned BLOCK on two counts. First, all seven tests asserted
`VERIFICATION FAILED`: nothing pinned the *passing* direction, which for a patch
that only ever **adds** points-to targets is the direction most at risk —
nothing asserted that a correct program stays correct. `..._addrof_locked` fills
that in, the same aggregate-held pointer under a mutex, and it is the only test
here that walks the new descent and expects SUCCESSFUL. Second, the `esize > 0`
guard in `collect_offset_paths` had no test and a process abort behind it:
`type_byte_size` returns 0 for a zero-length array's element type, and without
the guard `offset % esize` reaches `error("Division by zero.")` in
`big-int/bigint.cpp`, which is `fprintf` + `abort()`.
`mpor_aggregate_ptr_zero_size_element` pins it, and the shape is fussier than it
looks: `BigInt` short-circuits equal operands before the divide, so `0 % 0`
returns 0 and only a **nonzero** byte offset into the zero-size element reaches
the abort. The test dies under the mutant by producing no verdict at all rather
than a wrong one.

Three sites remain uncovered and are triaged rather than tested. The
`array_size_excp` handler is defensive: three candidate witnesses were built and
measured, and none puts `collect_offset_paths` on the stack when the throw
happens, because a symbolic offset clears `offset_is_set` and skips the descent
before it starts — the throwing shapes and the entered shapes do not intersect.
The `offset == 0 && type == target` fast path in `offset_paths` does fire in
ordinary nested-deref code, but no mutant of it is observable: delete it and
`collect_offset_paths` hits the identical test one frame down, pushes `""`, and
the caller re-issues a call byte-identical to the unrefined one. It is a
de-duplication guard that saves a vector and a try block on the hottest path in
symex, kept for that and not for correctness. A test written to green it would
be theatre. One further note worth carrying: `..._addrof_local` executes *none*
of the new code, so it is a behavioural control and not a coverage one — it
should not be deleted as redundant with `..._addrof` on coverage grounds.

**Code review changed the code as well as the record.** Beyond **R33**, which it
found outright, the review closed three divergences between the new walk and the
one it claims to invert. `size_bits` resolves symbol types through `ns.follow`
before measuring and the inverse did not, so a typedef'd aggregate lost its path;
it now follows too. The exception handler cleared every path already collected,
which reverts that object to the pruning behaviour this change exists to remove
— it now stops the descent and keeps what it found, which is sound because a
path is only ever appended on an exact type match, and members after an unsized
one are unreachable by a constant offset regardless. And `struct_union_members`
and `struct_union_member_names` both return **by value**, so the walk was
heap-allocating two vectors and bumping N refcounts at every level of every
dereference in symex; it now binds const references off the concrete type. The
outer `try` and the duplicated base case in `offset_paths` went with them.

**Cost was measured, not assumed.** `01_pthread60`, the heaviest pthread case in
the corpus, runs 108.7 s / 114.7 s unpatched against 111.3 s / 112.8 s / 120.8 s
patched — each side landing inside the other's range, so no effect separable
from ±6 s of run-to-run variance. The fast path is why: `offset == 0 && type ==
target` returns before any walk, which is every dereference of a scalar through
a plain pointer.

**What this does not close.** The pun shape, as before. And a descriptor whose
offset is *not* constant (`offset_is_set` false) is skipped rather than
enumerated over every type-matching path. This paragraph first called that "a
conservative gap with no known witness", on the strength of one shape built to
try it — `pp = &arr[i].p` with symbolic `i`, recorded as detecting its race by
another route. **Both halves of that were wrong**, see §15 M9 (R32) and (R32
fix). A second shape witnessed the gap within the hour, and the shape offered as
evidence *against* it turns out to have been a false SUCCESSFUL itself, measured
against this very commit — so the round did not lack a witness, it held one and
misread it. It is a residual, tracked as **R32**, not a gap.
And H-A6 stays refuted on the same grounds as before, with two further data
points: the round that closed the census at 21/21 had itself missed a shape, and
the round that fixed R31 declared a witnessless gap that a single further probe
witnessed. The enumeration keeps being wrong about its own results, not merely
incomplete.

---

### M9 (R33) — 2026-08-13, the offset that never survived to be walked

R31's fix went to code review, and the review found a false `SUCCESSFUL` the
census had not: `struct S { long pad; int *v[2]; }; int **pp = &s.v[1];`. Both
halves of that address work in isolation — a member at a nonzero offset is
`..._addrof_offset`, an array element via a pointer is `..._array_decay`, and
both detect their race — and only the composition failed. A 2×2 over base and
index offsets isolates it to one branch:

| base offset | index offset | verdict |
|---|---|---|
| 0 | 8 | FAILED (correct) |
| 8 | 0 | FAILED (correct) |
| **8** | **8** | **SUCCESSFUL (false)** |
| byte offset 16 reached by two members | — | FAILED (correct) |

The last row is the control: same struct, same byte offset, same `int *` target,
different route. So the discriminator is how the offset was built, not what it
is.

The cause is one branch in the index arm of `get_reference_set_rec`, and it
predates R31 by a long way. It added a constant element offset only when the
base offset was **zero**, and otherwise fell through to the unknown-offset
branch, which clears `offset_is_set`. R31's walk requires a constant offset, so
it skipped, and the unrefined lookup missed as before. The member arm one screen
below had been composing correctly the whole time (`o.offset += offset_in_bytes`).

The fix is to compose whenever the base offset is set. It is identical to the
old code on the old domain — `offset_is_zero()` is defined as `offset_is_set &&
offset.is_zero()`, and `+=` onto a zero offset is assignment — so the only
behaviour that changes is the case that was abandoned. Note the direction: this
makes a descriptor *more* precise, replacing "offset unknown" with a definite
offset, which is the one change in this area that could be unsound if the
arithmetic were wrong. It is the same arithmetic the member arm already
performs, and `mpor_aggregate_ptr_race_member_index` pins the result.

Worth recording as a method point, since the plan keeps asking what the census
is worth: this defect sat in a two-line branch, on a shape more common than
several the census does cover, and twenty-eight enumerated shapes walked past
it. A reviewer reading the *code around* the fix found it in one pass. The
census generates shapes from a model of what could go wrong; reading the
neighbouring branch does not need that model to be right.

---

### M9 (R32) — 2026-08-13, the witnessless gap, witnessed

The section above closed by naming one thing R31's fix does not cover — a
descriptor whose offset is not constant — and judging it witnessless because the
one shape tried for it detected its race by another route. The next shape tried
witnessed it:

```c
int g = 0;
int *a[2] = {&g, &g};
int **ap;
int i = nondet_int();
__ESBMC_assume(i >= 0 && i < 2);
ap = &a[i];          /* symbolic index -> descriptor offset unset */
/* thread: **ap = 1;   main: g = 2; */
```

**`VERIFICATION SUCCESSFUL` by default under both Bitwuzla and Z3, `FAILED`
under `--no-por`.** The same program with `a[1]` in place of `a[i]` reports
FAILED, so the discriminator is the symbolic index and nothing else. This is
well-defined C — the index is assumed in bounds — so the false SUCCESSFUL is not
excused by UB the way the punning shape is.

`--show-symex-value-sets` shows the erasure in one line: `c:@ap = { <a, *, 8,
signed int * [2]> }`, where `*` is the unset offset, while the entry holding the
answer is `c:@a[] = { <g, 0, 1, signed int> }`. R31's walk needs a constant to
spell back out, finds none, and skips — leaving the unrefined lookup of `c:@a`
to miss `c:@a[]` and report an empty value set, which every consumer reads as
"points at nothing".

The fix is the branch R31's section already named and declined to build: when
the offset is unset, enumerate *every* type-matching path instead of the one
path an offset selects. For an array that is just `[]` regardless of index,
which is exactly how the value set keys it; for a struct or union it is every
member whose type matches. It stays monotone for the same reason R31's walk
does — paths are only ever added. Deliberately not built in the same change as
R31, so that R31's mutation evidence keeps referring to the code it was measured
against.

Recorded as **R32**, pinned by
`regression/esbmc-unix/mpor_aggregate_ptr_race_symbolic_offset`, with
`..._array_decay` as the constant-index control that must keep passing.

**The assertion leg is the one thing local runs cannot check, and the risk it
covers was measured another way.** CI builds `Debug`; a local `RelWithDebInfo`
tree defines `NDEBUG`, so no `assert` in `value_set.cpp` is exercised by any run
recorded in this document. That matters here for one specific reason: the new
code calls `get_value_set_rec` with the *same object* as the unrefined lookup
beside it but a **non-empty suffix**, and two arms of that function assert
`suffix == ""` — the `malloc`/`alloca`/`realloc` and `cpp_new` side-effect
cases. If an object could reach those arms, this change would turn a passing
assertion into a firing one, invisibly to every build used here.

A Debug rebuild was not available (18 GB build tree, and the disk had reached
100%), so the condition was instrumented instead: a `fprintf` at the entry of
`get_value_set_rec` on exactly `is_sideeffect2t(expr) && !suffix.empty()`. It
fires **zero times** across the 26 `mpor_aggregate` shapes, 500 core `esbmc/`
tests and 250 `esbmc-unix` tests. The zero is not vacuous — widening the same
probe to `is_symbol2t` fires 57 times on a single test, so the instrumentation
is live. A side effect never reaches the value set with a suffix, which is what
the arm's own comment says: SSA assignments have their side effects removed
before this code sees them. The obligation is discharged as **T1 for that
assertion**.

That left the other twenty-two assertions in the file, and the argument for
ignoring them was weaker than it looked: an assertion depending only on the
expression was already reachable with the same expression before this change,
but a new suffix does not stop at the symbol arm — it flows into
`get_constant_value_set`, which recurses into members with what remains of it.
So the whole file was re-armed rather than reasoned about. With `NDEBUG` on,
`assert` expands to nothing; redefining it *after all includes* as a
non-aborting `fprintf` re-arms every one of the 23 without touching a header,
which is what makes this safe where a `-UNDEBUG` rebuild is not — that one
changes inline bodies shared across translation units and corrupts the heap by
ODR. **No assertion in `value_set.cpp` is violated** across the 26
`mpor_aggregate` shapes, 500 core `esbmc/`, 300 `esbmc-unix` and 300
`esbmc-cpp/cpp` tests. The binary was checked to contain the probe's format
string and the stringified conditions first, so the zero is a measurement rather
than a macro that failed to expand.

**A residual neither R31 nor R32 closes: the inverse lives away from its
forward.** Both reviews raised this independently, and it is the most useful
thing either said about the shape of the code rather than its content.
`collect_offset_paths` inverts `member_offset_bits`
(`src/util/expr/type_byte_size.cpp`) for struct members and the index arm of
`get_reference_set_rec` for array elements, but sits in neither file. Every rule
the two directions must agree on — padding as explicit members, unions overlaid
at zero, bits for members and bytes for elements, `ns.follow` before measuring —
is therefore held in step by a comment. Two of the defects fixed here are that
disagreement made concrete: R31 is the inverse not existing at all, and the
`ns.follow` omission review caught was the forward walk resolving symbol types
where the inverse did not.

The suggested repair is to move the inverse next to its forward counterpart —
`member_paths_at_offset(type, offset, target, ns)` in `type_byte_size.{h,cpp}` —
which co-locates the conventions, makes the pair directly unit-testable in
`unit/` without standing up a `value_sett`, and offers the same inverse to
`dereferencet`, which faces the identical descriptor-to-field-path problem. That
last point is the one with teeth: both walks are file-`static` today, which is
why every piece of evidence for them in this document is an end-to-end MPOR race
rather than a Tier-B unit test, against this plan's own stated preference.

**Not done here, and the trigger for doing it has not cleanly fired.** The R32
review set the condition as "a third defect of this family". R33 is adjacent
rather than a third instance: it is a *forward*-direction bug, an offset that
was never composed, in the very `get_reference_set_rec` arm the inverse mirrors.
Counting it would be convenient rather than accurate. The honest position is
that the family stands at two, the coupling is real and now recorded, and the
move is a separate change that should not enlarge a patch already spanning four
defects. **Done as that separate change**, §15 M9 (R31/R32 residual) — not
because a third defect fired the trigger, but because the missing Tier-B
evidence did.

---

### M9 (R32 fix) — 2026-08-13, and the witness that was there all along

Fixed as the section above specified: `collect_offset_paths` gains an
`offset_known` flag, and on the unknown route it takes every path of the
dereferenced type instead of the one an offset selects. Sizes are not consulted
on that route at all — there is no offset to place, so nothing to measure — which
also keeps the `array_size_excp` and `esize == 0` handling on the constant route
where the tests for them live. The reproducer reports FAILED under Bitwuzla and
Z3 alike.

**The array arm alone did not pin it, and the mutant said so.** The first cut
shipped one test, `..._symbolic_offset`, whose descent is array → element with no
struct in it. Crippling the *struct* arm of the new route — descend into member
0 only — left all 21 tests passing. `..._symbolic_struct_member` (`&arr[i].p`,
pointer declared second) kills that mutant, and it has to be declared second for
the same reason `..._addrof_union` does: with the pointer first, member 0 is the
answer and a crippled walk finds it anyway.

| Mutant | Test that dies |
|---|---|
| unknown route removed entirely | `_symbolic_offset`, `_symbolic_struct_member` |
| unknown route descends into member 0 only | `_symbolic_struct_member` |

**The witness was in hand a round earlier and was misread.** R31's closing
paragraph justified "no known witness" with `pp = &arr[i].p`, recorded as
detecting its race by another route. Measured against the commit that paragraph
describes, that program reports **`VERIFICATION SUCCESSFUL`** — as does the
single-member spelling of it, and as does the `int *a[2]` shape that later became
R32. The round had a witness, ran it, and wrote down the opposite of the verdict.
That is a different failure from the census's usual one: not a shape nobody
thought of, but a shape that was tried and whose result was recorded wrong. It is
the fourth claim in this document's R31 material to be falsified by re-measuring
it, after the narrow-fix claim, the simplification story and the 21/21.

`..._symbolic_offset_locked` is the passing direction, and it matters more here
than elsewhere: the unknown route is the widest thing this walk does, so it is
the most likely to turn a correct program into a false alarm. It does not.

Cost is unchanged on the corpus. `01_pthread60` 102.8 s against 103.4 s before
this change, `ch13_10` 106.6 s against 106.6 s, `github_5868_string_conversions`
83.9 s against 82.9 s. The sweep was later widened to the core C suite as well:
**1737 `esbmc/`, 620 `esbmc-unix`, 775 `esbmc-cpp/cpp`, 307 `cbmc`, 164
`floats`** and a 550-test sample of `python/` (that suite does not fit the
five-minute cap, so it was sampled in two disjoint slices rather than claimed
whole). Every failure in those runs passes when re-run serially — they are `-j`
over-parallelisation timeouts — with one exception worth recording, because it
looks like the regression this change could plausibly have caused and is not
one. `regression/esbmc/github_4634` exceeds the harness's hard 120 s cap even
standalone. Measured against the pre-session binary it takes **134.3 s / 131.4
s**, and against the patched one **133.1 s / 133.7 s**: the same range, and over
the cap on both. It is a pre-existing failure, not a cost regression — and it is
the shape most at risk, being pointer-heavy and near the cap, so it was worth
the two builds to settle rather than assume.

**Review turned the flag into two functions, and refuted the cost worry
properly.** The `offset_known` flag threaded a *type-directed enumeration* —
which consults no size, catches no `array_size_excp` and ignores the offset —
through the offset walk, forking all three arms and leaving `offset_bits`,
`start_bits` and an `esize` initialiser live but unread on the unknown route.
Worse, `offset % esize` was a division by zero on that route, prevented only by
a ternary's short-circuit: the one line in the change a future reader could
break by hoisting it. Splitting `collect_typed_paths` out is shorter than the
flag was and deletes all of that. It also fixes a measured defect the flag
introduced: the two functions' empty-path conditions had drifted apart, so on
the unknown route `offset_paths` returned `""` and the caller repeated its own
unrefined lookup verbatim — instrumentation counted this **20 times per run** in
three C++ container tests, which is also the first evidence that this route's
blast radius reaches well past the MPOR suite. The dispatcher now drops the
empty path for both walks, on the *followed* type, which closes a second
instance of the same drift that predates R32.

The blow-up question was settled by measurement rather than by the corpus being
quiet. A naive pre/post comparison makes the patch look **2× faster**, purely
because the pre-patch binary misses the race and explores to exhaustion; matched
on verdict with mutex-protected variants, the widening costs **+1.0 %** at 800
same-typed members and **+1.1 %** at 1024 leaves of a nine-deep doubling struct.
The array arm collapses `T[1000000]` to a single `[]`, so extents never multiply
the walk, and any type large enough to make the walk costly has already made
symex and solving costlier still.

Two findings worth carrying. The unknown route is a small *soundness gain*
beyond R32: the offset walk drops paths when `type_byte_size` throws on a
variable-length element, and the size-free route does not, so a target held
inside a VLA is now reachable where it previously was not. And the widening
nudges the `--no-reachable-memory-leak` defect recorded as **#5400**, which uses
MAY-points-to to *exclude* objects from the leak set: a larger value set can
suppress a leak report there. That is the unsound direction, it is pre-existing
and out of scope here, but it is the one place where widening this analysis is
not free.

Review also produced seven further witnesses of R32 at shapes the tests do not
cover, all flipping SUCCESSFUL → FAILED across the patch, and one of them earned
a test: a struct whose matching member is reached only past members of
non-matching type. A walk that stopped at the first non-contributing member
would have survived every test above it. It does not survive
`..._symbolic_skip_mismatch` — and, measured, it does not survive
`..._symbolic_union` either, whose leading `long pad` is the same trap by
accident.

The size-free route's *reason for existing* went untested until the gate pointed
at it. Every test above reaches the typed walk through its member loop; none
reached the case the doc comment claims, a target inside an element of no
constant size. `..._vla_element` does: `int *arr[2][n]` with symbolic `n` yields
the path `[][]` and a FAILED verdict, where the offset walk throws out of
`type_byte_size` and returns nothing. It is pinned by its own mutant — giving
the typed walk a `type_byte_size` call that bails on the throw kills that test
**and no other** of the twenty-six.

---

### M9 (census re-run) — 2026-08-13, 22/22, and the two that are left

With R31, R32 and R33 all fixed, every `mpor_aggregate_ptr_*` shape in the
repository was re-run under the default configuration and under `--no-por`, and
the two verdicts compared directly rather than against a `test.desc`:
**22 of 22 agree**, the two SUCCESSFUL-expecting controls included. This is the
first round whose claim is reproducible from the tree — earlier rounds counted
programs that lived only in a scratch directory, which is how the 21/21 came to
be over-stated and how R32's witness came to be recorded with the wrong verdict.

Two shapes outside the repository still diverge, and both are undefined
behaviour, so neither carries a soundness claim:

| Shape | MPOR | `--no-por` |
|---|---|---|
| struct-to-struct pun, `*(((struct B *)&a)->q) = 1` | SUCCESSFUL | FAILED |
| `void **pp = (void **)&s.p; *(int *)*pp = 1` | SUCCESSFUL | FAILED |

Both read an object through an lvalue of a type C11 **6.5p7** does not permit —
`void *` is not compatible with `int *`, nor a qualified or sign variant of it,
nor a character type — so a conforming program reaches neither. The second is
worth recording anyway, because the cast is not the barrier on its own: the
*same* cast over a bare pointer (`int *p; void **pp = (void **)&p;`) detects its
race correctly. It is the combination of a cast with an aggregate descriptor
that misses, since the walk matches the dereferenced type exactly and a cast puts
the member out of its reach. Relaxing the match to same-width pointer types
would close it; nothing in scope requires that, and doing it on UB alone would
widen the analysis to serve a program the standard does not define.

So the honest statement of where H-A6 stands is: every shape this census
contains, and every shape in the repository, now agrees — and that is a
statement about 22 programs, not about pointer resolution. The row stays
refuted. Four claims in this material have now been falsified by re-measuring
them, one of them a recorded verdict that was simply wrong, which says more
about the method than any of the individual defects do.

---

### M9 (post-refactor sweep) — 2026-08-13, 621/621, and five failures that were not

The sweep recorded above predates the commit that split `collect_typed_paths`
and the aggregate literals out of their callers, so it could not speak for the
tree that shipped. Re-run on the full concurrency suite, **`esbmc-unix` is
621/621** with no wrong verdict anywhere.

It did not read that way at first. The suite reported **five** failures at
`-j4`, all at the 120 s harness cap, and two of them — `github_2513_1`
(`--add-symex-value-sets`, `--unlimited-k-steps`) and `03_boundedBuffer`
(`--context-bound 2 --no-slice`) — sit squarely in the code this branch widens,
which is the shape a cost regression would take. Re-run serially, every one
passes well inside the cap:

| Test | `-j4` | serial |
|---|---|---|
| `github_2513_1` | timeout | 81.9 s |
| `github_6480_deepening` | timeout | 83.1 s |
| `03_boundedBuffer` | timeout | 94.0 s |
| `01_pthread60` | timeout | 103.1 s |
| `github_595` | timeout | 107.2 s |

`01_pthread60`'s 103.1 s lands on the 102.8 s this document recorded for the
patched binary and the 103.4 s for the pre-patch one, which is the check that
turns "passes serially" into "unchanged by the patch" — a serial pass alone only
rules out contention, not cost.

**An earlier run of the same suite reported 305 of 621 failing, and none of them
were real.** `src/esbmc/esbmc` was relinked mid-run; during the link the file is
`-rw-rw-r--` and short of its final size, so `testing_tool.py` reports
`PermissionError` as a *test* error indistinguishable from a failure. The
signature separates cleanly — a sub-second failure is a relink, a 120 s failure
is the cap — and `.ninja_log` dates the relink. The trigger is worth recording
because it is self-inflicted by the suite: the build globs `regression/` with
`CONFIGURE_DEPENDS`, so artefacts a test run leaves behind make the *next*
`ninja` re-run CMake and relink, and even `ninja -n` runs the regeneration step
for real. Settle the build to completion before measuring anything.

---

### M9 (R31/R32 residual) — 2026-08-15, the inverse moves next to its forward

The residual both the R31 and the R32 review raised — *the inverse lives away
from its forward* — is closed. `collect_offset_paths` and `collect_typed_paths`
were file-`static` in `value_set.cpp` while the walks they invert are
`member_offset_bits` (`src/util/expr/type_byte_size.cpp`) and the index arm of
`get_reference_set_rec`. Every convention the two directions must agree on was
therefore held in step by a comment. Both now sit in `type_byte_size.cpp` as
`member_paths_at_offset` and `member_paths_of_type`, beside the forward walk;
`value_sett::offset_paths` keeps only the dispatch and the empty-path rule,
which is value-set-specific.

**The move is behaviour-preserving, and that is checked rather than asserted.**
Each moved body diffs identically against the version at `fd6f75d923` once the
`overlaid`/`members`/`names` triple is rewritten to the `aggregate_memberst`
helper that replaces it — the offset walk exactly, the typed walk modulo one
blank line. `esbmc-unix` is **621/621** and the 26 `mpor_aggregate_ptr_*` shapes
all pass. Three tests failed at `-j8` on the 120 s cap and pass serially, the
signature §15 M9 (post-refactor sweep) already records; `01_pthread60` at
**103.05 s** lands on the 103.1 s recorded there, which is what separates
"passes serially" from "unchanged by the patch".

**What the move buys is the evidence, not the tidiness.** Both walks being
file-`static` is why every piece of evidence for them in this document is an
end-to-end MPOR race, against this plan's own stated preference for Tier B.
`unit/util/type_byte_size.test.cpp` is that Tier-B test: 9 cases, 40 assertions,
no `value_sett` and no solver. It pins the forward/inverse round trip over every
member of a padded struct, nested descent, R33's member-plus-element
composition, the union overlay, `ns.follow` on a typedef'd member, the
offset-vs-typed discriminator that is R32, the variable-length element only the
typed walk reaches, and the empty path the dispatcher keys on.

**Nine mutants, all killed — but only four of them by the suite as first
written.** Review mutated the moved code independently and found five survivors,
each with a measured witness, and each a behaviour the walk is entitled to claim
only if it is checked:

| Mutant | Witness | Killed by |
|---|---|---|
| `ns.follow` dropped, offset walk | typedef'd member invisible | symbol-type case |
| union members accumulate like struct members | second member unreachable at 0 | union case |
| `offset % esize` dropped | element 1 loses its path | composition case |
| `[]` suffix dropped, typed walk | variable-length member unreachable | variable-length case |
| `ns.follow` dropped, **typed** walk | `.in.p` → `{}` | *added* — the original claim that one case killed both walks' `follow` was wrong; it killed the offset walk's |
| `offset == 0 &&` dropped | byte 12, mid-pointer, yields `.v[]` | *added* — this guard is the whole difference between an inverse and a "holds a target somewhere below" query |
| member bound `<` → `<=` | one-past-end of `v` wraps to `.v[]` | *added* — invisible until the array is moved off the end of the struct |
| unmeasurable member `return` → `continue` | offset 8 attributed to `.q` | *added* — invisible until the variable-length member is moved off the end |
| `esize > 0` guard dropped | **SIGABRT**, `BigInt` modulo by zero | *added* — a zero-sized element array; the guard was the only thing between the walk and an abort |
| array-arm `array_size_excp` catch swallowed | unmeasurable *element* yields a path | *added* by the coverage gate — the arm has its own catch, and a struct-wrapped VLA never reaches it |

Two of those five survived for the same reason: the shape that exposes them puts
the array, or the unmeasurable member, somewhere **other than last**, and every
case as first written put it last. That is the enumeration failure §15 M9
(census re-run) argues about, reproduced in miniature inside a unit test — the
cases were generated from the defects already known, so they inherited those
defects' shape. A third, the array-arm catch, hid behind the same reflex in a
different guise: a VLA was always reached *through a struct member*, where the
member loop's catch fires first, so the arm's own catch was never entered. The
coverage gate found that one by asking which lines the tests execute, which is a
question neither reading nor mutation-by-hypothesis had put.

The `return`-on-unmeasurable-member rule is now justified rather than asserted:
C11 **6.7.2.1p18** puts a flexible array member last and **6.7.2.1p9** bars a
variably modified member outright, so no later member is reachable by a constant
offset. It matters which way the rule errs — dropping paths under-approximates a
may-points-to set, which is the unsound direction, so the argument for it has to
be a standard citation and not a plausibility.

**One recorded claim was wrong, in the direction that matters.** The first cut
of the round-trip case built `struct { char c1; int *p; char c2; }` unpadded and
asserted `member_offset(s, "p") == 8`. It is **1**: `member_offset_bits` sums the
widths of the preceding members and makes no alignment adjustment, because the
frontend hands padding over as explicit members. The inverse agreed with the
forward on that struct too — only the hard-coded offsets were wrong — but the
case would have documented a layout ESBMC does not use. It now builds the struct
the way `add_padding` does and pins `type_byte_size(s) == 24` alongside the two
offsets, so the forward walk is pinned rather than trusted.

The offer this makes to `dereferencet`, which faces the same
descriptor-to-field-path problem, is left unexercised: nothing in scope needs it,
and a second caller should arrive with its own test rather than be anticipated.

---

### M9 (byte_extract census) — 2026-08-15, the probe that found no defect because it found no code

R31, R32 and R33 were each an arm of the value set failing to compose an offset,
so the next probe went to the arm that composes least.
`get_reference_set_rec`'s **`byte_extract2t` arm** (`value_set.cpp:1338-1353` at
`fd6f75d923`) neither recurses into `source_value` nor composes an offset — it
calls `insert(dest, extract.source_value, o)` directly with the extract's own
offset, where the member arm does `get_reference_set(memb.source_value, …)` and
then `o.offset += offset_in_bytes`, and the index arm has done the same since
R33. On shape alone it is R33 before its fix.

**It is never entered.** Counters compiled into four arms of the function and
swept over **1,197** sources from `regression/esbmc-unix`, `regression/esbmc`
and `regression/esbmc-cpp/cpp`:

| Arm | Calls | Files |
|---|---|---|
| `get_reference_set_rec` entry | 152,478 | 790 of 1,197 |
| index | 15,538 | 598 |
| member | 470 | 37 |
| **`byte_extract`** | **0** | **0** |
| `assign_rec`'s `byte_extract` lhs arm (`:1721`) | **0** | **0** |

Twelve hand-written shapes aimed at the arm — `char *` arithmetic into a struct,
misaligned punned writes, union punning, an array element read as `short`, a
`struct` overlaid on `long buf[]`, `memcpy` into a local buffer, and the C++
`reinterpret_cast` spelling of the first — add **0** more.

The entry and index counters are the control that makes the zero a measurement
rather than dead instrumentation, and the first attempt at this sweep was
exactly that mistake: run under `--goto-functions-only`, which exits before
symex ever starts, it reported 0 for *every* arm across 427 files — a clean,
confident, entirely vacuous result. The control is not a formality.

**The one route that looked like it would feed the arm provably does not.** An
assignment whose lhs is a `byte_extract` is handled at `value_set.cpp:1721` by
`assign_rec(to_byte_extract2t(lhs).source_value, …)`, recursing on
`source_value` directly and bypassing `get_reference_set_rec` altogether. That
is the *same* non-composing shortcut the dead arm takes, which is the part worth
keeping: both places that handle a byte-extract lvalue decline to compose, and
the live one gets away with it by reaching its base through recursion instead.

**Verdict: no defect, and the hypothesis is unwitnessable rather than refuted.**
If the arm does mis-compose, nothing measured here can show it, because nothing
reaches it. It is the strongest dead-code candidate in the function — and it
**stays**. §14 item 8 is explicit that C-Dead cannot be discharged on `src/**`:
the file does not parse (G9), and a corpus sweep is evidence of non-coverage,
never a proof of unreachability. Deleting a branch on the strength of 1,197
files of silence is precisely the "deletion drops live behaviour" regression the
rule exists to prevent. Recorded so the next reader does not spend the same
afternoon rediscovering it.

---

### M9 (dereferencet agreement) — 2026-08-15, the other half of the coupling, checked

R31's material argued that the descriptor-to-field-path problem is solved twice
in ESBMC — once by the value set, once by `dereferencet` — and that the pair
agreeing is held by nothing but convention. The value set's half is now pinned
by `unit/util/type_byte_size.test.cpp`. This entry checks the other half against
the *same shapes that broke it*, at verdict level: write through a punned
pointer, read the sub-object back by its declared path, assert they agree.

**Ten shapes, all correct**, each with both an inverted-assertion twin and an
`assert(0)` reachability twin:

| Shape | Byte | Result |
|---|---|---|
| `struct S { long pad; int *v[2]; }`, `&s.v[1]` — R33's composition | 16 | correct |
| the same, read direction | 16 | correct |
| member offset alone, `&s.v[0]` | 8 | correct |
| element offset alone, no pad, `&s.v[1]` | 8 | correct |
| symbolic index, `&s.v[i]`, `i` nondet and assumed in range — R32's shape | — | correct |
| symbolic offset by `char *` arithmetic, `(char *)&s + 8 + i * 8` | — | correct |
| `struct Outer { long pad; struct Inner in[2]; }`, `&o.in[1].q[1]` | 48 | correct |
| the same at `&o.in[0].q[1]` | 24 | correct |
| union member through a pun | 8 | correct |

Every one reads an `int *` object through an `int **`, which is the object's own
type, so C11 **6.5p7** is satisfied and none of this is a strict-aliasing test —
unlike the two shapes §15 M9 (census re-run) had to set aside.

**So the coupling is real but not currently broken.** That is worth stating
precisely: it is not evidence that `dereferencet` cannot drift from the value
set, only that on the shapes that actually broke one, the other is right. The
pair is pinned going forward by
`regression/esbmc/deref_punned_member_index{,_fail}`.

**A false finding, and the harness bug behind it.** The first run of this probe
reported `assert(0)` proving `SUCCESSFUL` on two straight-line programs — a
false-SUCCESSFUL of the most serious kind, and it was reproduced on master
before being believed, which is the right instinct applied to a wrong result.
It was neither. The variants were generated with `sed 's/== &g)/!= \&g)/'`, and
those two inputs had been written compactly as `s.v[0]==&g`, with no spaces. The
pattern never matched, `sed` reported nothing, and the "mutant" was a byte-copy
of the original — so the *same passing program* was run three times and read as
three results. The rewritten harness selects its variant with `-D` at compile
time (`PROBE_NEG`, `PROBE_REACH`), which cannot silently fail to apply: a
misspelt macro changes no behaviour and the twin visibly stops failing.

The lesson generalises past `sed`. §15 M8's rule was that a detector which
cannot fail teaches nothing; this adds that **a mutation which cannot be
observed to have applied is not a mutation**. Both anti-vacuity twins here exist
to catch the ordinary vacuity — an unreachable assertion — and neither would
have caught this, because the fault was upstream of the tool entirely.

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
