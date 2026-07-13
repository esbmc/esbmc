# IREP2 Formal Verification Plan

**Subsystem:** `src/irep2` — ESBMC's typed, intrusively reference-counted,
copy-on-write internal representation for expressions (`expr2t`) and types
(`type2t`).
**Verifier:** ESBMC itself (bounded model checking + k-induction), supported by
Catch2 property/differential unit tests for the parts that are outside ESBMC's
practical reach.
**Status:** Plan / not yet executed. Every harness below is a *proposal*; nothing
here asserts a proof has been discharged.
**Audience:** An engineer who will implement the harnesses and run the
verification tasks directly from this document.

> **Framing.** ESBMC is a verification tool, so soundness of *its own* IR is
> load-bearing: a latent bug in `irep2` (a wrong `crc`, a non-total ordering, a
> mis-computed width, a dropped guard conjunct) can silently corrupt every
> downstream verdict ESBMC produces. This plan treats `irep2` as safety-critical
> and adopts a conservative stance: **assume subtle bugs exist until formally
> disproven.**

---

## 1. Verification objectives

1. **Memory safety of the container machinery.** The hand-rolled intrusive
   refcount (`irep_container`, `irep2t::refcount`) and copy-on-write `detach()`
   must never double-free, use-after-free, leak, or free a still-referenced
   node — including under the self-aliasing assignment the code explicitly
   guards against (`irep_container::operator=`, `irep2.h:190-225`).
2. **Algorithmic correctness of the value-semantics operations.** `crc`
   (hashing), `cmp`/`operator==` (structural equality), `lt`/`operator<`
   (ordering), `clone`, `with_type`, and operand/subtype traversal must be
   mutually consistent and satisfy their mathematical contracts (equality is an
   equivalence relation; `<` is a strict weak order; `a == b ⇒ crc(a) == crc(b)`).
3. **Absence of ESBMC-checkable undefined behaviour** in every arithmetic,
   indexing, and bit-manipulation kernel: no overflow producing a wrong result,
   no division by zero, no out-of-bounds read/write, no use of uninitialised
   data, no invalid pointer dereference.
4. **Representation-invariant preservation.** Every constructor establishes, and
   every mutator preserves, the per-kind invariants (`fields` covers storage,
   array-size normalisation, guard canonical-chain ↔ `guard_list`
   correspondence, `crc_val == 0 ⇔ not-yet-computed`).
5. **Soundness of the guard set-algebra.** `operator-=` / `operator|=` on
   `guard2tc` must compute path conditions that are *logically* correct: the
   optimised O(Δ) prefix walk must agree with the naive semantics in every case,
   because a wrong result is an unsound path condition (a whole class of missed
   or spurious counterexamples).
6. **Regression durability.** Each discharged property is pinned by a
   check-in-able test (Catch2 case and/or an ESBMC `regression/esbmc/…` harness)
   that would fail if the property were re-broken.

---

## 2. Scope and the reach of ESBMC on this codebase

`src/irep2` is ~8.8k lines of modern C++ (plus an 832-line README) — C++20/23:
`std::apply`, `std::unreachable`, fold expressions, `if constexpr`,
concepts-style traits — linked against `immer`, `bigint`, `fmt`, Boost, and
`std::atomic`. ESBMC's C++
frontend cannot practically ingest the whole subsystem end-to-end (template
metaprogramming over `K::fields`, `immer`'s persistent trie, `fmt` internals).
Pretending otherwise would produce vacuous "proofs." The plan therefore splits
targets into three honest tiers.

| Tier | What it covers | Verification technique | Why |
|------|----------------|------------------------|-----|
| **A — ESBMC-verifiable kernels** | Self-contained algorithmic cores: byte/CRC loops, width arithmetic, endianness reconstruction, index math, refcount lifecycle, guard prefix walk, `as_ulong`/`as_long` bounds | **Standalone reduced C/C++ harnesses** verified by ESBMC (`--overflow-check --unsigned-overflow-check --memory-leak-check --ub-shift-check …`) with nondeterministic inputs. Kernel logic is transcribed faithfully (or `#include`d where it compiles standalone) and driven by `nondet_*()`. | These are ordinary imperative code once lifted out of the template layer; ESBMC excels here and gives a deterministic witness on failure. |
| **B — Property / differential harnesses (in-tree)** | Algebraic laws that need the *real* classes (equality/ordering/CRC consistency, COW value semantics, guard-algebra logical equivalence) | **Catch2 tests in `unit/irep2/`** driving the actual `expr2tc`/`type2tc`/`guard2tc` API, bounded-exhaustive or randomised; the *logical-equivalence* checks additionally shell out to ESBMC/an SMT check on small terms. | The template dispatch cannot be reduced without losing fidelity, but the observable contracts can be exercised against the genuine implementation. |
| **C — Not effectively ESBMC-verifiable** | `std::atomic` memory-ordering / data-race freedom of the refcount and CRC cache; `immer` internals; `fmt`/`BigInt` library internals; the compile-time `fields_cover_class` machinery | **Sanitizers (TSan/ASan/UBSan) + hand proof + `static_assert`.** Documented, not claimed as ESBMC-proven. | ESBMC's weak-memory support does not cover the release/acquire linearisation argument these rely on; library internals are out of scope. See §13. |

**Harness locations.**
- Tier A standalone harnesses: `regression/esbmc/irep2_<kernel>_<n>/` (one
  passing + one failing variant per §CLAUDE.md "two regression tests per PR"),
  each with a `test.desc`.
- Tier B property tests: `unit/irep2/*.test.cpp` (new files, wired via
  `unit/irep2/CMakeLists.txt` `new_unit_test(...)`).
- Tier C: notes in this document + a `MALLOC_PERTURB_`/TSan CI recipe.

**ESBMC binary for Tier A:** `build/src/esbmc/esbmc` from a current `master`
build. Record the exact commit hash in the verdict log (§10/§11) when the
harnesses are run, so results are reproducible against a pinned artefact.
Default solver per repo convention is Bitwuzla; dead-code / soundness gates
require **dual-solver agreement (Bitwuzla + Z3)**.

---

## 3. Architecture overview

```
                       irep2t  (base: atomic refcount, atomic crc_val, dbg writer stamp)
                      /      \
                 type2t      expr2t   (type_id/expr_id enum + switch dispatch)
                    |            |
        <kind>_type2t...   <kind>2t...   (122 expr kinds, 15 type kinds; each: fields tuple + field_names)
                    \            /
              irep_container<T>  ==  type2tc / expr2tc   (COW smart pointer, value semantics)
                         |
                   guard2tc : expr2tc  (+ guard_seq guard_list — immer::vector of conjuncts)
```

**Core mechanisms and their source of truth**

> **Anchor convention.** Every `file:line` reference names the enclosing
> symbol (`Class::member`, function, or field) alongside the line, so an anchor
> stays resolvable via `grep` even after a ±1 line drift. Treat the symbol name
> as authoritative and the line number as a hint; re-`grep` before relying on a
> line.

| Mechanism | File / lines | One-line contract |
|-----------|--------------|-------------------|
| Intrusive atomic refcount | `irep2t::refcount` `irep2.h:495`; `irep_container` ctors / `irep_container::release` `irep2.h:158-425` | ctor `fetch_add(relaxed)`; `release` `fetch_sub(release)`, delete on `1→0` behind an acquire fence. |
| Copy-on-write | `irep_container::detach` `irep2.h:287-298`; non-const `get()` `irep2.h:251-264` | non-const access clones when `refcount > 1`, then invalidates `crc_val`. |
| Self-alias-safe assignment | `operator=` `irep2.h:190-225` | snapshot `ref.ptr_` and bump **before** `release()`, so `x = to_array_type(x).subtype` cannot UAF. |
| Single-allocation factory | `make_irep` `irep2.h:450-456` | one `new T(...)`, adopt at refcount 1 via private tag ctor. |
| CRC cache | `crc_val` (`irep2.h:700, 913`), reader `crc()` `irep2.h:319-329` | `0 ⇒ not computed`; acquire-load hit / release-store publish. |
| Iterative CRC traversal | `irep2_crc.cpp:258-298` | explicit-stack postorder (no recursion → no stack overflow on 50k-deep trees); per-node memoise. |
| BigInt CRC ingestion | `irep2_crc.cpp:30-60` | sign byte + magnitude dump into a 256-byte stack buffer, growing on the heap when larger. |
| Switch-on-id dispatch | `irep2_type.cpp:232-345`, `irep2_expr.cpp:410-648` | X-macro over `*_kinds.inc` routes each op to `generic_*<K>`; `end_*_id` case + `-Wswitch` forces exhaustiveness; `std::unreachable()` tail. |
| Generic field walk | `irep2_dispatch.h:195-367` | `std::apply` over `K::fields` implements cmp/lt/tostring/get_sub_expr/foreach uniformly. |
| Compile-time layout invariant | `fields_cover_class` `irep2.h:1109-1119`, `assert_kind_invariants` `irep2.h:1132-1149` | `Σ derived field sizes == sizeof(K) − sizeof(base) − excluded`, modulo trailing padding. |
| Width computation | `irep2_type.cpp:111-345` | per-kind `get_width`; throws `symbolic_type_excp` / `array_size_excp` for non-static widths. |
| Guard set-algebra | `irep2_guard.cpp:334-649` | `operator-=` / `operator|=` maintain the path condition; O(Δ) prefix walk cross-checked by an O(N) scan in debug. |
| Guard conjunct sequence | `guard_seq.h` | `immer::vector` with **non-atomic** refcount (thread-confined); O(1) copy, O(log₃₂ N) index. |

---

## 4. Component analysis

### 4.1 Classes, data structures, and algorithms

**C1. `irep_container<T>` (COW smart pointer).** Members: single `T *ptr_`.
Operations: copy/move ctor, copy/move assign, `detach`, `release`, const/non-const
`get`/`operator*`/`operator->`, `reset`, `swap`, `crc`, `simplify`, ordered
comparison friends. Owns the refcount lifecycle.

**C2. `irep2t` base + `type2t`/`expr2t`.** `refcount` (`atomic<unsigned>`),
`crc_val` (`atomic<size_t>`), debug `writer_thread` (`atomic<uintptr_t>`).
`expr2t` additionally holds an immutable `type` and `expr_id`; `type2t` holds
`type_id`. Both expose `crc/cmp/lt/clone/tostring/pretty` and (expr side)
`get_sub_expr/get_num_sub_exprs/with_type/simplify/foreach_operand`.

**C3. Concrete kinds (122 expr + 15 type — counts derived from `expr_kinds.inc` / `type_kinds.inc`).** Each is a flat struct of `expr2tc`/
`type2tc`/scalar fields plus `static constexpr fields` and `field_names[]`.
Non-trivial per-kind logic: `constant_int2t::as_ulong/as_long`
(`irep2_expr.cpp:101-112`); `constant_string2t` byte reconstruction
(`irep2_expr.cpp:227-291`); `array_type2t`/`vector_type2t`/`struct_type2t`/
`union_type2t::get_width` (`irep2_type.cpp:127-219`); `with2t::assert_consistency`
(`irep2_expr.cpp:330-358`); `array_type2t` constructor size-normalisation
(`irep2_type.h:274-298`).

**C4. Generic dispatch layer (`irep2_dispatch.h`, `irep2_crc.cpp`).**
`generic_cmp/lt/tostring/get_sub_expr/get_num_sub_exprs/foreach_*` and the
per-field `do_type_cmp/do_type_lt/do_get_sub_expr/do_count_sub_exprs/
call_expr_delegate`. CRC iterative traversal + `feed_bigint`.

**C5. Utilities (`irep2_utils.{h,cpp}`).** `gen_zero`/`gen_one` (recursive value
constructors), `conjunction`/`disjunction`, `make_not`,
`distribute_vector_operation`, `get_symbols`, `make_cmp_value`, `do_type_lt`
overloads for vectors/`BigInt`/containers, family accessors
(`struct_union_members`, `struct_union_get_component_number`, …).

**C6. Guard machinery (`irep2_guard.{h,cpp}`, `guard_seq.h`).** `guard2tc`
(set-of-conjuncts wrapper over `expr2tc`), the `-=`/`|=`/`==` set-algebra,
`common_pointer_prefix_size`, `cached_prefix_expr(_at)`, `add`/`add_leaf`/
`build_guard_expr`, and the `immer`-backed `guard_seq`.

### 4.2 Representation invariants (the properties to assert)

| ID | Invariant | Where established/relied on | Enforcement today |
|----|-----------|-----------------------------|-------------------|
| **I1** | `refcount` equals the number of live `irep_container`s pointing at the node; `1→0` deletes exactly once. | `irep2.h` container lifecycle | runtime only (no assert) |
| **I2** | After a non-const `get()`/`detach()` on a shared node, the caller's container is the *sole* owner (refcount 1) of a structurally-equal but distinct object. | `detach` `irep2.h:287-298` | tested (`irep2.test.cpp` COW scenario) |
| **I3** | `crc_val == 0 ⇔ CRC not yet computed`; a non-zero cache equals the freshly recomputed CRC. | `crc()` `irep2.h:319-329`, `irep2_crc.cpp` | runtime only |
| **I4** | `a == b ⇒ a.crc() == b.crc()` (no false CRC splits) and `a != b` for structurally different `a,b` with high probability (no silent collisions in tested corpus). | `cmp` + `crc` | partially tested |
| **I5** | `cmp`/`operator==` is an equivalence relation; `lt`/`operator<` is a strict weak order consistent with `cmp` (`lt==0 ⇔ cmp`). | dispatch + `do_type_lt` | partially tested (recent OOB fix) |
| **I6** | `fields` tuple covers all identity-bearing storage of each kind (nothing silently excluded from cmp/crc/hash). | `fields_cover_class` `irep2.h:1109` | `static_assert` (compile time) ✔ |
| **I7** | `array_type2t.size_is_infinite ⇒ array_size` is nil; a non-`constant_int` `array_size` denotes a dynamically-sized array; `get_width` throws rather than returning a wrong number for non-static widths. | `irep2_type.cpp:127-147`, `irep2_type.h:263-298` | runtime throw (array); **asymmetric for vector — see R3** |
| **I8** | Guard canonical form: the cached `expr2tc` base is the left-leaning and-chain of `guard_list`, oldest-first, element-for-element. | `add_leaf`/`build_guard_expr` `irep2_guard.cpp` | debug `assert` cross-check (`common_pointer_prefix_size_scan`) |
| **I9** | `guard_list` is duplicate-free (set semantics); `x && x ≡ x`. | `guard2tc` doc `irep2_guard.h:22-29` | by convention, **not enforced** |
| **I10** | `as_ulong` precondition: value is non-negative **and ≤ UINT64_MAX**; `as_long`: within `int64` range. | `irep2_expr.cpp:101-112` | **only the sign half is asserted — see R2** |

### 4.3 Safety-criticality ranking

| Rank | Component | Rationale |
|------|-----------|-----------|
| **P0 (critical)** | C1 refcount/COW lifecycle; C6 guard set-algebra; I7/I10 width & `as_ulong` arithmetic | A defect here is *silent and global*: memory corruption (P0-mem) or an unsound path condition / wrong width feeding SMT encoding (P0-soundness). |
| **P1 (high)** | C4 CRC + cmp/lt; I4/I5 | Wrong hashing/ordering breaks hash-consing, `std::set`/`unordered_map` keys, and dedup — corrupting results non-locally and non-deterministically. |
| **P2 (medium)** | C3 per-kind kernels (`constant_string` reconstruction, `gen_zero` recursion), C5 utilities | Localised; wrong value or OOB, but usually on a specific construct. |
| **P3 (low)** | pretty-printing (`tostring`, `type_to_string`), diagnostics | Cosmetic / diagnostic; a `snprintf` truncation is not a soundness issue. |

---

## 5. Target properties (ESBMC-supported)

Every Tier-A harness is run with the full property set enabled. Mapping of the
task's required checks to ESBMC flags:

| Property | ESBMC flag(s) | Primary targets in `irep2` |
|----------|---------------|----------------------------|
| Memory safety (bounds, invalid deref, UAF, double-free, leak) | `--memory-leak-check` (pointer/bounds/deref checks are **on by default**) | C1 refcount/`release`/`detach`; `do_get_sub_expr<vector>` index math (`irep2_utils.cpp:338-355`); `constant_string_access::operator[]` (`irep2_expr.cpp:255-265`) |
| Arithmetic overflow / underflow | `--overflow-check --unsigned-overflow-check` | `num_elems * sub_width` (`irep2_type.cpp:146,159`); struct width accumulation (`:188-191`); `hash_combine` shifts; `feed_bigint` `heap_buf.size()*2`; refcount `fetch_add` wrap |
| Division / modulo by zero | **on by default** (disable via `--no-div-by-zero-check`) | `struct_union_get_component_number`, any width-ratio math in string reconstruction (`w`), `distribute_vector_operation` lane count |
| Invalid pointer dereference | **on by default** | `dynamic_cast` result use in `array/vector::get_width` (`irep2_type.cpp:141-144,155-157`); `object_descriptor2t::get_root_object` walk (`irep2_expr.cpp:360-373`) |
| Bounds violations | array-bounds **on by default** (disable via `--no-bounds-check`) | `guard_seq::operator[]`/`front`/`back` (`guard_seq.h:77-88`); `do_get_sub_expr<vector>` (`ptr = &item[idx-it]`); `field_names[idx]` indexing in `generic_tostring` |
| Uninitialised data | no dedicated flag — model explicitly with `nondet_*()` "read-before-write" | fields left nil; `constant_string_access` members; harness models "read before write" |
| Failed assertions | default (`assert` becomes a VCC) | every `assert(...)` in the kernels (e.g. `assert(!value.is_negative())`, `assert(const_elem_size != nullptr)`, `assert(0 < w && w <= 4)`) |
| Undefined behaviour (shift, conversion, lifetime) | `--ub-shift-check` (shifts) + `--overflow-check` (conversions) | shift amounts in `constant_string_access` and `hash_combine`; signed↔unsigned in width math; `size_t` underflow `idx - it` |
| Floating-point NaN | `--nan-check` | float value constructors / `ieee_floatt` round-trips (adjacent; H-B2 corpus) |

Additional non-functional properties proven with dedicated modes:
- **Termination / no unbounded loop:** `--k-induction` (with convergence) on the
  CRC traversal and guard `suffix`/`prefix` rebuild loops; full-unwind where the
  bound is a nondet-but-bounded input.
- **Reachability of new/removed branches (dead-code gates):** Mode C
  (C-Live/C-Dead) per repo `CLAUDE.md` whenever a harness-motivated *fix* adds
  or deletes a branch (e.g. the R3 vector-width guard).

---

## 6. Verification strategy: modular decomposition

To contain state-space explosion, verify **one kernel per harness** with the
smallest faithful model, then compose.

**Principles**
1. **Lift, don't inline the world.** For each Tier-A kernel, copy the exact
   arithmetic/indexing logic into a `.c`/`.cpp` harness and drive it with
   `nondet_*()` inputs constrained by `__ESBMC_assume(...)` to the documented
   precondition. Prove the postcondition/absence-of-UB. This isolates the
   property from `immer`/`fmt`/atomics.
2. **Bound the shape, not the values.** Where a structure is recursive
   (`gen_zero`, CRC traversal, guard chains), bound *depth/length* with a small
   nondet cap (`__ESBMC_assume(n <= K)`), verify for the bound, then argue the
   inductive step separately with `--k-induction`.
3. **Stubs for the environment.** Model `BigInt` as a bounded integer wrapper,
   `irep_idt` as an integer handle, `expr2tc` as an id+refcount pair — only as
   much fidelity as the property needs (the esbmc-verifier "stub-shadowing"
   pattern).
4. **Differential oracles for Tier B.** For laws that must hold on the *real*
   classes (ordering totality, `cmp⇔crc`), enumerate small terms and assert the
   law across all pairs; for guard logical-equivalence, emit the two path
   conditions and check `¬(A ⇔ B)` UNSAT via a small SMT query.
5. **One passing + one failing harness per property** (repo convention): the
   failing variant perturbs the code (e.g. removes the overflow guard) to prove
   the harness actually *detects* the defect — an anti-vacuity check.

**Modularity map (minimises cross-coupling):**

```
Refcount lifecycle (H-A1)         ──┐
COW detach / clone (H-A2)           ├─ independent; model expr2tc as {id, *refcount}
Self-alias assignment (H-A3)      ──┘
CRC combine + BigInt feed (H-A4)  ── independent; integer model of hash_combine + byte buffer
Width arithmetic (H-A5)           ── independent; unsigned math only
as_ulong/as_long bounds (H-A6)    ── independent; BigInt→u64 model
String byte reconstruction (H-A7) ── independent; byte array + endianness
Sub-expr index math (H-A8)        ── independent; vector index accumulation
Guard prefix walk (H-A9)          ── depends on canonical-chain model only
gen_zero/gen_one recursion (H-A10)── bounded-depth type model
--- Tier B (real classes) ---
Equality/ordering laws (H-B1)
CRC↔cmp consistency (H-B2)
COW value semantics (H-B3)
Guard algebra logical equiv (H-B4)
with_type / dispatch totality (H-B5)
```

---

## 7. Proposed harnesses

Each entry gives: **target**, **preconditions** (`assume`), **postconditions /
assertions**, **ESBMC invocation**, and the **anti-vacuity (failing) variant**.
Code is a faithful sketch; the implementer transcribes the cited source exactly.

> **Sequencing of the R1/R2/R3 harnesses (CI must stay green).** H-A5 (R1),
> H-A6 (R2) and the H-A5 deref variant (R3) are expected to **FAIL on the
> current source** — they encode invariants the code does not yet enforce. A
> `CORE` harness asserting `^VERIFICATION SUCCESSFUL$` would therefore make CI
> red the moment it lands. Two acceptable landing patterns, in order of
> preference:
> 1. **Fix-and-prove in one PR** — the R-fix (`irep2_type.cpp` /
>    `irep2_expr.cpp`) and the passing `CORE` harness land together, so the
>    harness is green from its first commit and pins the fix as a regression.
>    This is the required pattern for R1/R2/R3.
> 2. **`KNOWNBUG` first** — if a finding must be recorded before its fix is
>    ready, land the harness as `KNOWNBUG` (documents the reproducer without
>    failing CI), then flip it to `CORE` in the fix PR.
>
> Never land an R-harness as `CORE`-SUCCESSFUL ahead of its fix.

### Tier A — ESBMC-verifiable kernels

#### H-A1 — Refcount conservation & single-free (P0-mem)

**Target:** `irep_container` ctor/`release` (`irep2.h:158-425`). Property: across
any sequence of copy/move/reset/destroy on containers sharing a node, the node is
deleted exactly once, never while a container still points at it, and refcount
never underflows.

```c
// model: one node, N container "slots"; refcount is the ground truth.
// A container can only *copy* the node from another live container (copy
// ctor bumps the source's node refcount) — you cannot adopt a freed node.
// Slot 0 is seeded live with refcount 1 to model the make_irep that created
// the node; the loop then only copies-from-live or drops.
unsigned refcount = 1;      // node created by make_irep (slot 0 holds it)
int deleted = 0;            // free counter
_Bool slot_live[N] = {1};   // slot_live[0]=1, rest 0

// copy the node from a live source `src` into dead slot `dst` (copy ctor)
void copy(int src, int dst){
  if(!slot_live[src] || slot_live[dst]) return;  // source must be live
  slot_live[dst]=1;
  refcount++;                            // fetch_add on the live source's node
}
void drop(int s){                                                         // release()
  if(!slot_live[s]) return;
  slot_live[s]=0;
  unsigned prev = refcount--;            // fetch_sub
  assert(prev >= 1);                     // no underflow  (I1)
  if(prev==1){ deleted++; assert(deleted==1); }   // exactly one free
}
int main(){
  for(int step=0; step<K; step++){
    int a=nondet_int(), b=nondet_int();
    __ESBMC_assume(0<=a && a<N && 0<=b && b<N);
    if(nondet_bool()) copy(a,b); else drop(a);
    // invariant: live slot count == refcount
    unsigned live=0; for(int i=0;i<N;i++) live+=slot_live[i];
    assert(live==refcount);              // I1 — refcount = #live handles
    assert(!(deleted>0) || refcount==0); // freed ⇒ nobody points at it
  }
  return 0;
}
```

Because `copy` requires a live source, once the last handle is dropped
(`refcount==0`, `deleted==1`) no further `copy` can resurrect the node — this
matches container semantics and removes the spurious "resurrect a freed node"
counterexample a source-agnostic `adopt` would report.

**Invocation:** `esbmc H_A1.c --overflow-check --unsigned-overflow-check
--k-induction` (unbounded step loop; require convergence). Also full-unwind for
`K = N+2`.
**Failing variant:** change `copy` to skip the `refcount++` (models a missed
`fetch_add` on the copy ctor) → the `live==refcount` invariant must fail, proving
the harness catches a leak/early-free.

#### H-A2 — COW detach clones iff shared, invalidates CRC (P0-mem, I2/I3)

**Target:** `detach` (`irep2.h:287-298`) + non-const `get()` (`:251-264`).
Property: `detach` allocates a new object **iff** `refcount > 1`; afterwards the
mutating container is sole owner and its `crc_val` is 0.

```c
// model each node as {refcount, crc_val, id}; clone → fresh id, refcount 1.
// assert: after detach, (shared ⇒ id changed) ∧ (¬shared ⇒ id unchanged)
//         ∧ crc_val==0 ∧ this.refcount==1
```
**Invocation:** `--overflow-check --memory-leak-check`.
**Failing variant:** make `detach` clone unconditionally → `¬shared ⇒ id
unchanged` fails (regression guard against the "detach always copies" perf bug
the header warns about at `irep2.h:245-250`).

#### H-A3 — Self-aliasing assignment is UAF-safe (P0-mem)

**Target:** `operator=` (`irep2.h:190-225`) — the documented `x =
to_array_type(x).subtype` hazard. Model a container whose assigned-from reference
*aliases a member of the node the LHS solely owns*; verify the snapshot-and-bump
ordering prevents reading freed storage.

```c
// node holds an embedded child handle; LHS is sole owner (refcount 1).
// RHS = &node->child (aliases into the soon-to-be-freed node).
// Correct order: bump child.refcount; release node (may free); ptr_ = child.
// Assert: child is never accessed after node is freed AND child not freed.
```
**Invocation:** `--memory-leak-check` + ASan cross-check on a compiled version.
**Failing variant:** reorder to `release()` before snapshot → ESBMC reports a
use-after-free / invalid deref, reproducing the AppleClang crash the comment
cites.

#### H-A4 — `hash_combine` + `feed_bigint` CRC ingestion (P1, no overflow/OOB)

**Target:** `esbmct::hash_combine` (`irep2.h:110-114`) and `feed_bigint` /
`do_type_crc(BigInt)` (`irep2_crc.cpp:30-60`). Properties: (a) the byte-feeding
loop never reads out of the buffer; (b) the heap-grow loop terminates; (c) sign
is always mixed (regression for the tested "negation didn't change CRC" bug).

```c
// model BigInt.dump(buf,len) as: returns true iff len >= need; writes need bytes.
// need is nondet in [0, BIG]. stack buf = 256.
unsigned char stack_buf[256];
size_t need = nondet_size(); __ESBMC_assume(need <= 4096);
size_t len = 256; unsigned char *buf = stack_buf;
_Bool ok = (len >= need);
if(!ok){
  size_t hlen = 256*2;
  while(hlen < need) { assert(hlen <= SIZE_MAX/2); hlen*=2; }  // (b) no overflow, terminates
  // heap buffer of hlen …
}
// (a): every index i<need is < allocated length. (c): sign byte fed before magnitude.
```
**Invocation:** `--overflow-check --unsigned-overflow-check --k-induction`
(grow loop) `--memory-leak-check`.
**Failing variant:** seed `hlen` at `need-1` without the doubling loop → OOB
write flagged.

#### H-A5 — Width arithmetic overflow (P0-soundness) — **exposes R1**

**Target:** `array_type2t::get_width` `num_elems * sub_width`
(`irep2_type.cpp:146`), `vector_type2t::get_width` (`:159`), `struct` sum
(`:188-191`), `union` max (`:199-201`).

```c
#include <stdint.h>
// Use a fixed-width accumulator (uint64_t), NOT `unsigned long`: on ILP32
// `unsigned long` is 32-bit and equals `unsigned int`, so the check would be
// vacuous. The production `num_elems` is itself `unsigned long` (as_ulong),
// so on ILP32 R1 manifests as an *overflow* of the product before truncation;
// on LP64 it is a truncation on the narrowing to `unsigned int`. The harness
// must cover both by making num_elems a full-width uint64_t.
unsigned int width_array(uint64_t num_elems, unsigned int sub_width){
  // FAITHFUL: return (unsigned int)(num_elems * sub_width);
  uint64_t prod = num_elems * (uint64_t)sub_width;
  assert(num_elems == 0 || prod / num_elems == sub_width); // no 64-bit overflow (ILP32 path)
  assert(prod <= UINT_MAX);                                // no truncation (LP64 path)  — R1
  return (unsigned int)prod;
}
unsigned int width_struct(const unsigned int *ws, int n){
  uint64_t w=0;
  for(int i=0;i<n;i++){ w += ws[i]; assert(w <= UINT_MAX); } // accumulation overflow — R1
  return (unsigned int)w;
}
```
Drive with `num_elems`, `sub_width`, member widths nondet within realistic
caps and *also* at adversarial extremes. Run the harness under **both** the
32-bit (`--32`) and 64-bit (`--64`, the default) machine-word models so the
ILP32 overflow and the LP64 truncation are each exercised.
**Invocation:** `--overflow-check --unsigned-overflow-check`.
**Expected outcome:** **FAILS on the current code** (no guard exists) — this is a
finding (R1), not a passing proof. The passing harness is the *post-fix* version
with a checked-width path (throw `array_size_excp` or saturate) added to
`irep2_type.cpp`; that fix triggers a **Mode C C-Live** obligation for the new
branch. Sequencing (CI): the R1 fix and this harness must land in the **same
PR** — see the sequencing note under §7 preamble.

#### H-A6 — `as_ulong` / `as_long` truncation (P0-soundness) — **exposes R2**

**Target:** `constant_int2t::as_ulong` (`irep2_expr.cpp:101-106`), `as_long`
(`:108-112`). Today only `assert(!value.is_negative())` guards `as_ulong`; a
magnitude `> UINT64_MAX` is silently truncated by `to_uint64()` — the code even
carries the `XXXjmorse` TODO.

```c
#include <stdint.h>
// model value as a bounded big integer (two-limb) → u64.
uint64_t as_ulong(bigval v){
  assert(!is_negative(v));
  assert(fits_u64(v));   // R2: MISSING in production
  return to_u64(v);
}
```
**Invocation:** `--overflow-check`. **Expected:** fails without the
`fits_u64` guard, demonstrating the missing invariant I10; the passing harness
pins the guarded version. Same **fix-and-prove-in-one-PR** sequencing as H-A5
(see §7 preamble) — do not land as `CORE`-SUCCESSFUL before the R2 fix.

#### H-A7 — `constant_string` byte reconstruction (P2-mem/bounds)

**Target:** `constant_string_access` (`irep2_expr.cpp:227-291`). Properties: for
`w ∈ {1,2,4}`, `s.length() % w == 0`, index `w*i+j` stays in `[0, s.length())`
for the char-bearing branch (`i<m`), and the shift `8*(le?j:w-1-j)` is `< 32`; the
returned char reconstruction is endianness-correct (differential vs. a reference).

```c
uint32_t reconstruct(const unsigned char*s, size_t slen, unsigned w, _Bool le,
                     size_t i, size_t n, size_t m){
  __ESBMC_assume(w==1||w==2||w==4);
  __ESBMC_assume(slen % w == 0); __ESBMC_assume(m == slen/w);
  if(i>=n) return 0;                        // sentinel path
  uint32_t c=0;
  if(i<m) for(unsigned j=0;j<w;j++){
    size_t idx = w*i + j; assert(idx < slen);      // bounds
    unsigned sh = 8*(le? j : w-1-j); assert(sh < 32);
    c |= (uint32_t)s[idx] << sh;
  }
  return c;
}
```
**Invocation:** `--overflow-check --ub-shift-check` (array-bounds is on by
default). Add a differential check: big-endian reconstruction of 2 bytes equals
`(s[0]<<8)|s[1]`.
**Failing variant:** widen the width assumption to `w ∈ {1..8}` (i.e.
`__ESBMC_assume(1<=w && w<=8)`) — this admits `w ∈ {5,6,8}` where `sh = 8*(w-1)`
reaches 40..56, so the `sh < 32` shift-UB assertion fails **for the intended
reason**. Do *not* simply drop the `w<=4` bound down to `w>=0`: that admits
`w==0`, which makes the harness's own `slen % w` / `slen/w` preconditions divide
by zero, so the variant would fail for the wrong reason (a precondition div-0
rather than the shift UB under test). Keep `w>=1`.

#### H-A8 — Sub-expression index accumulation (P1-bounds)

**Target:** `do_get_sub_expr<std::vector<expr2tc>>` (`irep2_utils.cpp:338-355`)
and `generic_get_sub_expr` fold (`irep2_dispatch.h:253-264`). Property: the
running accumulator `it` never exceeds `idx` when the vector branch is entered
(so `idx - it` cannot underflow `size_t`), and `&item[idx-it]` is in-bounds
(array-bounds check is on by default).

```c
// model K fields as segment lengths seg[]; walk left→right accumulating `it`.
size_t it=0; const_ptr result=NULL;
for(int f=0; f<F && !result; f++){
  if(idx < it + seg[f]){ assert(idx >= it); result=&elem(f, idx-it); }  // no underflow
  else it += seg[f];
}
// idx past total ⇒ result==NULL   (matches irep2.test.cpp contract)
```
**Invocation:** `--overflow-check` (array-bounds on by default).
**Failing variant:** start the scan at a nonzero `it` (mis-modelled prior field)
→ underflow flagged.

#### H-A9 — Guard shared-prefix walk agrees with the naive scan (P0-soundness)

**Target:** `common_pointer_prefix_size` (`irep2_guard.cpp:64-108`) vs. the debug
reference `common_pointer_prefix_size_scan` (`:35-50`). This is the crux of guard
soundness (I8). Model two guards as arrays of conjunct-ids that share a
left-leaning chain; verify the O(Δ) descent returns exactly the element-scan
prefix length **for all shapes** (single-leaf, strict prefix, no common prefix,
empty), under the canonical-chain invariant.

```c
// chain[g][k] = id of k-th conjunct (oldest first); same_pointer ⇔ same id
//               AND same suffix (canonical-chain invariant, modelled explicitly).
size_t walk(...);   // faithful port of the da/db descent + lockstep
size_t scan(...);   // faithful port of the reference loop
assert(walk(g1,g2) == scan(g1,g2));   // I8 — must hold for every nondet pair
```
Bound `n1,n2 ≤ K`; then `--k-induction` for the unbounded-length argument.
**Invocation:** dual-solver (Bitwuzla + Z3), `--overflow-check`.
**Failing variant:** break the canonical-chain modelling (let a shared id have a
divergent suffix) → the equality fails, showing the walk's correctness *depends
on* I8 (which is only a debug assert in production — see R4).

#### H-A10 — `gen_zero`/`gen_one` recursion & aggregate size loops (P2)

**Target:** `gen_zero` (`irep2_utils.cpp:63-143`), `gen_one` (`:145-176`).
Properties: recursion depth is bounded by the (bounded) type nesting; the
`for(i=0;i<s.as_long();…)` element loops (`:88,105`) use a **non-negative,
bounded** count; the `union` branch's `assert(!members.empty())` precondition
holds on the modelled inputs. Flags the latent risk that `as_long()` on a huge or
negative array size produces a bogus / unbounded loop bound.
**Invocation:** `--overflow-check --k-induction`. **Failing variant:** allow a
nondet negative `as_long()` result → loop-bound / signed-overflow issue surfaces.

### Tier B — property / differential harnesses (real classes, `unit/irep2/`)

#### H-B1 — Ordering is a strict weak order; equality is an equivalence (I5)

Enumerate a fixed small corpus of `expr2tc`/`type2tc` (mix of kinds, widths,
BigInt signs, unequal-length vectors). For every triple `(a,b,c)` assert:
irreflexivity `!(a<a)`; asymmetry `a<b ⇒ !(b<a)`; transitivity of `<` and of the
incomparability equivalence; `lt(a,b)==0 ⇔ (a==b)`; `operator==` reflexive /
symmetric / transitive. Extends the existing single-pair checks in
`irep2.test.cpp:135-309` to a full relational sweep. (Guards against a future
`do_type_lt` overload reintroducing a non-total order like the fixed
unequal-length-vector OOB.)

#### H-B2 — CRC ↔ cmp consistency, collision sampling (I4)

For the same corpus plus randomised trees: assert `a==b ⇒ a.crc()==b.crc()`
(mandatory) and record the empirical collision rate for `a!=b` (report, don't
assert a hard bound). Include the deep-tree determinism check (cold==warm) from
`irep2.test.cpp:115-128` and the oversized/negative BigInt cases (`:203-254`).

#### H-B3 — COW value semantics on the real container (I2)

Extend `irep2.test.cpp:396-439`: after mutating one of two aliased handles,
assert (i) the other handle's *value* is unchanged, (ii) pointers diverge, (iii)
no double-free (run the case under ASan in CI). Add a move-assignment aliasing
case mirroring H-A3 on the genuine `expr2tc`.

#### H-B4 — Guard algebra logical equivalence (P0-soundness)

For small nondet conjunct sets, build `g1`, `g2`; compute `g1 -= g2` and `g1 |=
g2`; then check with an SMT query that `as_expr()` of the result is **structurally
identical (or logically equivalent) to a naive reference implementation**, not to
a hand-written logical formula:

- **`-=` oracle — conjunct set difference, *not* `g1 ∧ ¬(shared)`.**
  `operator-=` (`irep2_guard.cpp:334`) computes
  `result = ⋀( conjuncts(g1) \ conjuncts(g2) )` — the conjunction of the
  conjuncts of `g1` that are absent from `g2` (order-independent set
  difference). This is a *syntactic* operation and is **not** logically
  equivalent to `g1 ∧ ¬(g1∧g2 shared part)`; specifying the oracle as
  `g1 ∧ ¬(shared)` verifies the wrong law and would spuriously flag correct
  code. The reference must be the naive O(N) loop: `for c in g1.guard_list: if
  c ∉ set(g2.guard_list): push c`, then conjunct. Assert the optimised result's
  `guard_list` equals the reference's as a set (and `as_expr()` matches under
  the canonical left-leaning chain).
- **`|=` oracle — `g1 ∨ g2`.** Here the logical form *is* correct: assert
  `as_expr(g1 |= g2) ⇔ (as_expr(g1) ∨ as_expr(g2))` is valid (its negation
  UNSAT).

Because the operators are *heavily* optimised (`irep2_guard.cpp:334-649`), this
differential check against the naive definition is the only credible correctness
argument. Cross-check `operator==` (`:657-681`) reflexivity/symmetry and that the
crc-fast-path (equal guards, one with a cached crc) never returns a wrong answer.

#### H-B5 — `with_type` / dispatch totality (P2-robustness)

For every expr kind, assert either `supports_with_type_v` and a successful
round-trip (`e.with_type(e->type)` structurally equal to `e`), or that the kind
is on the documented unsupported list and `with_type` aborts deliberately
(`irep2_expr.cpp:538-568`). Guards against a new kind silently falling into the
`abort()` path. Pair with an exhaustiveness probe: constructing one node of each
kind and exercising `crc/cmp/clone/tostring/get_num_sub_exprs` (forces
`assert_kind_invariants<K>` for every K and smoke-tests every dispatcher case).
The sweep must derive the expected kind count from the `expr_kinds.inc` /
`type_kinds.inc` manifests (an X-macro count, e.g. a `constexpr` fold over the
same `.inc`) rather than a hard-coded literal, so it can never drift out of sync
when a kind is added or removed.

---

## 8. Data-structure integrity verification

Mapping the task's required integrity checks to the harnesses above:

| Integrity concern | Harness(es) | What is proven |
|-------------------|-------------|----------------|
| **Representation invariants** (I1–I10) | H-A1, H-A5, H-A6, H-A9; H-B1, H-B2 | refcount = live handles; width normalisation; `as_ulong` domain; canonical guard chain; ordering/equality laws; CRC⇔cmp |
| **Ownership & lifetime** | H-A1, H-A2, H-A3; H-B3 | exactly-once free; sole-owner-after-detach; self-alias assignment UAF-safety |
| **Aliasing & sharing** | H-A2, H-A3; H-B3 | COW isolates mutation; shared readers unaffected; move/copy assignment aliasing |
| **Copy / move** | H-A1, H-A2, H-A3; H-B3 | value semantics preserved; no leak/double-free across copy/move |
| **Comparison** | H-A9; H-B1 | strict weak order; equivalence; `lt==0 ⇔ ==` |
| **Hashing** | H-A4; H-B2 | no OOB/overflow in ingestion; sign mixed; `==⇒equal crc`; determinism |
| **Serialisation** (`tostring`/`pretty`/`type_to_string`) | new P3 Catch2 case | no `snprintf` truncation UB; round-trip readability; every enum has a printable name (guards the `assert(0 && "Unrecognized…")` tails in `irep2_utils.cpp:449-565`) |
| **Traversal** (`foreach_operand`, `get_sub_expr`, CRC postorder) | H-A8; H-B2; existing `irep2.test.cpp:529-577` | count = children; index nullable past end; deep-tree traversal terminates without stack overflow |

**Note on serialisation:** `irep2` has *no* binary (de)serialiser — persistence is
via `util/migrate.{h,cpp}` to/from legacy `irept`. The migrate round-trip is
already unit-tested (`unit/python-frontend/irep2_type_roundtrip_test.cpp`,
`unit/util/migrate.test.cpp`); this plan treats round-trip fidelity as an
adjacent obligation and recommends extending those with the H-B5 per-kind sweep
rather than duplicating them here.

---

## 9. Coverage matrix

Rows = components (§4.1); columns = required properties. Cell = harness ID(s)
covering that (component, property). `—` = not applicable; `TSan/§13` = deferred
to Tier C.

| Component ＼ Property | Mem-safety | Overflow | Div-0 | Invalid deref | Bounds | Uninit | Assert | UB/other |
|---|---|---|---|---|---|---|---|---|
| **C1** container/refcount/COW | H-A1,H-A2,H-A3,H-B3 | H-A1 | — | H-A3 | — | H-A2 | H-A1 | TSan/§13 |
| **C2** `irep2t`/base + crc cache | H-B3 | H-A4 | — | — | — | H-A2 | H-B5 | TSan/§13 |
| **C3a** width (`get_width`) | — | **H-A5** | — | H-A5 (dyn_cast) | — | — | H-A5 | H-A5 |
| **C3b** `as_ulong`/`as_long` | — | **H-A6** | — | — | — | — | H-A6 | H-A6 |
| **C3c** `constant_string` recon | H-A7 | H-A7 | H-A7 (`%w`) | — | H-A7 | — | H-A7 | H-A7 (shift) |
| **C3d** `with2t`/`with_type` | — | — | — | — | — | — | H-B5 | H-B5 |
| **C4a** cmp / lt | — | — | — | H-B1 | H-B1 | — | — | — |
| **C4b** CRC traversal + combine | H-A4 | H-A4 | — | — | H-A4 | — | H-A4 | H-A4 |
| **C4c** get_sub_expr / foreach | H-A8 | H-A8 | — | — | H-A8 | — | — | H-A8 (underflow) |
| **C5a** gen_zero/gen_one | — | H-A10 | — | — | H-A10 | — | H-A10 | H-A10 |
| **C5b** do_type_lt overloads | — | — | — | H-B1 | H-B1 | — | — | — |
| **C6a** guard prefix walk | — | H-A9 | — | — | — | — | H-A9 | — |
| **C6b** guard `-=`/`\|=`/`==` | H-B4 | — | — | — | H-B4 | — | H-B4 | H-B4 (soundness) |
| **C6c** guard_seq (immer) | TSan/§13 | — | — | — | H-B4 | — | — | §13 |
| **Serialisation** tostring/pretty | P3 case | — | — | P3 case | P3 case | — | P3 case | P3 case |

**Coverage summary:** every P0/P1 component has at least one Tier-A or Tier-B
harness against memory-safety and its dominant arithmetic property. Concurrency
(atomic ordering, `immer` non-atomic refcount under the thread-confinement
assumption) is explicitly **deferred to Tier C** with a documented rationale
(§13), not silently claimed.

---

## 10. Risk assessment (concrete findings and hypotheses)

Ranked by (soundness/safety impact × likelihood). Each is a *hypothesis to be
confirmed or refuted by the cited harness* — consistent with the "assume subtle
bugs exist" mandate. File:line references are to the current tree.

| ID | Severity | Finding / hypothesis | Evidence | Harness | Recommended action |
|----|----------|----------------------|----------|---------|--------------------|
| **R1** | **High (soundness)** | `array/vector/struct::get_width` multiply/accumulate widths into `unsigned int` with **no overflow check**; a large array or wide element silently truncates the width feeding SMT encoding. | `irep2_type.cpp:146,159,188-191` | H-A5 | Add a checked path (throw `array_size_excp`/saturate + diagnostic) or widen to `uint64`; gate the added branch with Mode C **C-Live**. |
| **R2** | **High (soundness)** | `as_ulong()` asserts only non-negativity; a `BigInt > UINT64_MAX` is **silently truncated** by `to_uint64()`. The `XXXjmorse` TODO documents the missing check. Callers (e.g. `array_size…as_ulong()`) then use the truncated value. | `irep2_expr.cpp:101-112` | H-A6 | Add `assert(value <= UINT64_MAX)` / a checked accessor returning `optional`; audit call sites for truncation reliance. |
| **R3** | **Medium (safety)** | `vector_type2t::get_width` does **not** guard against a non-`constant_int` / infinite size before `dynamic_cast` — it relies on `assert(const_elem_size != nullptr)`, which is compiled out under `NDEBUG` → **null-pointer deref in release** for a symbolic-sized vector. `array_type2t::get_width` *does* throw for this case (asymmetry). | `irep2_type.cpp:149-160` vs `:127-147` | H-A5 (deref variant) | Mirror the array guard in the vector path (throw `dyn_sized_array_excp`). Removing the now-dead `assert` after adding the throw is a Mode C **C-Dead** obligation. |
| **R4** | **Medium (soundness, defence-in-depth)** | Guard `common_pointer_prefix_size` correctness depends on invariant I8 (canonical chain ↔ `guard_list`), which is only checked by a **debug-only** `assert(result == …_scan(…))` (`irep2_guard.cpp:106`). In release, a hash-cons aliasing corruption would yield a wrong prefix → **unsound path condition** with no tripwire. | `irep2_guard.cpp:64-108`; comment `:101-107` | H-A9, H-B4 | Keep the O(N) scan as a cheap release-mode `expensive-asserts` opt-in, or add a lightweight structural check; at minimum pin H-A9/H-B4 as permanent regressions before deleting the scan. |
| **R5** | **Low–Medium** | `gen_zero`/`gen_one` element loops bound on `as_long()` (`irep2_type` → `constant_int`); a huge or (defensively) negative size yields an unbounded loop / OOM or `int` overflow. No upper bound asserted. | `irep2_utils.cpp:88,105` | H-A10 | Assert `0 <= n <= CAP` before the loop; surface oversized aggregates as a diagnostic rather than looping. |
| **R6** | **Low (theoretical)** | `refcount` is `atomic<unsigned int>`; sharing a node > `UINT_MAX` times wraps `fetch_add` → premature free. Unreachable in practice but unbounded in principle. | `irep2.h:495,161,177` | H-A1 (extend with a saturation assert) | Document the bound; optional `assert(prev < UINT_MAX)` in debug. |
| **R7** | **Low** | Enum→string `type_to_string` overloads `abort()` on an unrecognised value (`assert(0 && "Unrecognized…")`); a newly-added enumerator without a case aborts the process mid-run instead of degrading. | `irep2_utils.cpp:449-565`,`502-546` | P3 serialisation case | Exhaustiveness Catch2 test over each enum; consider returning `"<unknown>"`. |
| **R8** | **Info (concurrency, deferred)** | The whole atomic-refcount / CRC-cache / writer-stamp scheme's data-race freedom rests on the single-writer contract and specific `memory_order` choices; `guard_seq` uses a **non-atomic** immer refcount justified by "symex is not parallelised" (`guard_seq.h:34-43`). Not ESBMC-verifiable here. | `irep2.h:1-29`, `guard_seq.h:34-43` | §13 (TSan) | TSan CI job under `--parallel-solving`; re-audit if symex is ever parallelised. |

**None of R1–R8 is confirmed as a live end-to-end bug by this document** — they
are the prioritised targets. R1, R2, R3 are the highest-value first runs because
each is a *demonstrable missing invariant in the source*, not merely a
hypothetical.

---

## 11. Recommended verification order & milestones

Ordered by risk × tractability. Each milestone yields check-in-able artefacts.

**Milestone M0 — Infrastructure (0.5 wk).**
Stand up `regression/esbmc/irep2_*` harness skeleton + `test.desc` template;
confirm `esbmc` runs the property flag-set on a trivial harness; add a
`unit/irep2/invariants.test.cpp` scaffold. *Artefact:* one green + one red
smoke harness proving the pipeline detects an injected bug.

**Milestone M1 — Arithmetic soundness (highest ROI) (1 wk).**
H-A5 (R1), H-A6 (R2), H-A3 (self-alias). Expect H-A5/H-A6 to **fail on current
code** → file findings, implement the checked-width / checked-`as_ulong` fixes,
discharge Mode C for the added branches, re-run to green. *Artefacts:* R1/R2 fix
PRs with C-Live proofs + Phase-2 contract regressions.

**Milestone M2 — Memory-safety core (1 wk).**
H-A1 (refcount), H-A2 (COW), H-A7 (string recon), H-A8 (index). *Artefacts:*
passing harnesses + ASan-compiled H-B3.

**Milestone M3 — Vector-width guard + CRC (0.5 wk).**
H-A5 deref variant (R3 fix, C-Dead for the removed `assert`), H-A4 (CRC/BigInt).
*Artefacts:* R3 fix PR; CRC ingestion proof.

**Milestone M4 — Guard soundness (1 wk).**
H-A9 (prefix walk vs. scan), H-B4 (algebra logical equivalence). Highest
soundness stakes; run dual-solver. *Artefacts:* guard-algebra differential suite;
R4 mitigation decision.

**Milestone M5 — Relational & totality laws (0.5 wk).**
H-B1 (order/equality), H-B2 (CRC↔cmp), H-B5 (`with_type`/per-kind sweep), P3
serialisation case (R7). *Artefacts:* relational property suite over all kinds.

**Milestone M6 — Concurrency (Tier C) (0.5 wk).**
TSan CI recipe under `--parallel-solving`; hand-proof note for the release/acquire
argument; R8 re-audit checklist. *Artefacts:* CI job + `docs` note.

Total ≈ 5 engineer-weeks. M1–M2 alone retire the two highest-severity findings.

**Expected verification artefacts (deliverables of execution):**
- `regression/esbmc/irep2_{width,as_ulong,refcount,cow,selfalias,string_recon,
  subexpr_index,crc_bigint,gen_zero,guard_prefix}_{ok,fail}/` — Tier-A harnesses,
  each `CORE` with a `test.desc`.
- `unit/irep2/{ordering,crc_consistency,cow_semantics,guard_algebra,
  kind_sweep,serialisation}.test.cpp` — Tier-B suites, wired in
  `unit/irep2/CMakeLists.txt`.
- Fix PRs for R1/R2/R3 (each with Mode C proof + contract regression) and R5/R7.
- A TSan CI job (Tier C) + this document updated with per-harness verdicts.

---

## 12. Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | This plan | `docs/irep2-verification-plan.md` |
| 2 | Tier-A ESBMC harnesses (10 kernels × {ok,fail}) | `regression/esbmc/irep2_*/` |
| 3 | Tier-B Catch2 property suites (6 files) | `unit/irep2/*.test.cpp` |
| 4 | Findings + fixes for R1–R3 (and R5/R7) | code PRs, each with Mode-C proof where a branch changes |
| 5 | Tier-C concurrency recipe + rationale | CI config + §13 |
| 6 | Verdict log (append per-harness result back into this doc) | `docs/irep2-verification-plan.md` §10/§11 |

---

## 13. Code that currently cannot be effectively verified — and why

Stated plainly, to avoid over-claiming:

1. **Atomic memory-ordering / data-race freedom** of `refcount`, `crc_val`, and
   the debug `writer_thread` stamp (`irep2.h:158-546`). The correctness argument
   is a release/acquire *linearisation* over concurrent container drops plus a
   single-writer discipline. ESBMC's weak-memory reasoning does not cover this
   composed argument at the fidelity needed; **use TSan** under
   `--parallel-solving` plus the hand-proof already sketched in the header
   preamble. *Mitigation:* Tier-C CI job; re-audit if symex is ever parallelised
   (the load-bearing assumption for `guard_seq`'s non-atomic refcount,
   `guard_seq.h:34-43`).
2. **`immer::vector` internals** (`guard_seq`). The persistent RRB/bitmapped-trie
   implementation is third-party and template-dense; we verify only the *guard
   contract layered on top* (H-B4, bounds via `guard_seq.h:77-88`), trusting the
   library's own test suite for the trie.
3. **`BigInt`, `fmt`, `ieee_floatt`, `fixedbvt` library internals.** Out of
   scope; modelled as bounded stubs in Tier A (H-A4/H-A6). We verify irep2's
   *use* of them, not their implementation.
4. **The compile-time `fields_cover_class` / `assert_kind_invariants`
   metaprogramming** (`irep2.h:1030-1149`). This is already a *sound
   `static_assert`* discharged by the compiler for every kind at build time — a
   stronger guarantee than BMC for what it covers (storage coverage). No runtime
   harness needed; H-B5 merely *forces instantiation* of every case.
5. **`std::unreachable()` dispatcher tails** (`irep2_expr.cpp`,
   `irep2_type.cpp`). Reachability is excluded by the exhaustive `-Wswitch` +
   `end_*_id` case, which is a compile-time argument; a BMC "proof" of
   unreachability would be circular. We rely on the compiler's exhaustiveness
   check and note it explicitly.

---

## Appendix A — Methodological basis

The plan applies established verification practice without over-claiming novelty:

- **Design-by-contract.** Every harness is framed as precondition (`__ESBMC_assume`)
  → operation → postcondition/invariant (`assert`), the classic contract triple.
  The representation invariants in §4.2 are the class invariants a DbC discipline
  would attach to each type; several (I3, I8, I10) are *documented in comments but
  unenforced in release* — the plan promotes them to checked obligations. (Cf. the
  contract-checking model used by ACSL/Frama-C and by ESBMC's own function-contract
  mode.)
- **Anti-vacuity via mutation.** Each property ships a *failing* companion harness
  that perturbs the code so the check is shown to have discriminating power — the
  same idea as mutation testing, adapted to model checking.
- **Differential verification.** Where a heavily-optimised routine (guard
  set-algebra, prefix walk) has a simple reference definition, we verify
  *equivalence to the reference* (H-A9, H-B4) rather than re-deriving correctness
  — the standard optimised-vs-naive oracle strategy.
- **Bounded proof + inductive step.** Recursive/iterative structures are verified
  at a small bound with full unwinding, then generalised with `--k-induction`
  requiring convergence, per the repo's dead-code/soundness gates.
- **Dual-solver agreement.** Soundness-critical runs (guard algebra, width fixes)
  require Bitwuzla **and** Z3 to agree, matching the project's Mode-C requirement.

## Appendix B — `test.desc` format for Tier-A harnesses

`test.desc` has **no comment syntax**: line 1 is the mode
(`CORE`/`KNOWNBUG`/`FUTURE`/`THOROUGH`), line 2 the source file, line 3 the
ESBMC flags, and **every line from 4 onward is consumed verbatim as an expected
output regex** by `regression/testing_tool.py`. Do not put explanatory comments
in the file — a stray `#…` becomes a regex that never matches. Flags are also
**per-harness**, not universal: only the loop-bearing kernels use
`--k-induction`; the arithmetic kernels (H-A5/H-A6/H-A7) are plain BMC.

Passing harness (loop-bearing kernel, e.g. H-A1 refcount):
```
CORE
irep2_refcount_ok.c
--overflow-check --unsigned-overflow-check --memory-leak-check --k-induction
^VERIFICATION SUCCESSFUL$
```

Passing harness (straight-line arithmetic kernel, e.g. H-A5/H-A6/H-A7 — no
`--k-induction`):
```
CORE
irep2_width_ok.c
--overflow-check --unsigned-overflow-check
^VERIFICATION SUCCESSFUL$
```

Failing / anti-vacuity variant (line 3 flags identical to its passing twin; only
the expected-verdict regex changes):
```
CORE
irep2_width_fail.c
--overflow-check --unsigned-overflow-check
^VERIFICATION FAILED$
```

Reference invocation for local iteration (a fail variant witnessing R1):
```sh
build/src/esbmc/esbmc regression/esbmc/irep2_width_fail/irep2_width_fail.c \
  --overflow-check --unsigned-overflow-check
# expect a counterexample witnessing R1 (width truncation) on unfixed code.
```

Cross-check a Tier-A kernel against the source before transcription:
```sh
grep -n "get_width" src/irep2/irep2_type.cpp     # confirm the arithmetic line
```

---

## Execution log (per-harness verdicts)

Appended as each harness is implemented and run, per §12 deliverable #6. Records
the harness location, technique, verdict, and the landing PR so the plan doubles
as a progress tracker.

| Milestone | Harness | Location | Technique | Verdict | Status |
|---|---|---|---|---|---|
| **M0** | H-A1 refcount conservation, single-free & self-alias UAF-safety (I1) | `unit/irep2/refcount.test.cpp`, `refcount_ops.h`, `refcount.fuzz.cpp` | **Tier B** — Catch2 property tests + a **nondeterministic-input** operation driver, both over the **real** `irep_container<T>`; reads the actual `irep2t::refcount` atomic after copy / move / assign / detach | 5 cases, 91 assertions PASS on `master` @ 4ba1903130; libFuzzer run 2,000,000 execs, no violation | **Done** (PR #6024). Verifies the genuine implementation, not a model, per §2 Tier B. The `run_ops` driver decodes a byte stream (libFuzzer, or fixed-seed in the unit test) into a sequence of container ops and checks refcount conservation after each; the libFuzzer target (`-DENABLE_FUZZER=On`) adds ASan UAF/double-free coverage. Built under the `Sanitizer` build type (ASan) the fixed cases also witness UAF freedom. Also pins **H-A2** (COW detach clones iff shared) and **H-A3** (self-aliasing assignment UAF-safety) as Tier-B cases on the real container. |
| **M1** | H-A6 `as_ulong`/`as_long` >64-bit truncation (R2, I10) | `unit/irep2/const_int.test.cpp`; fix in `irep2_expr.cpp:101-118` | **Tier B** — real `constant_int2t`; `as_ulong` now asserts `value.is_uint64()`, `as_long` asserts `value.is_int64()` before the `to_uint64/to_int64` narrowing | PASS on real classes | **Done** (PR #6027, fix-and-prove-in-one-PR). |
| **M1** | H-A5 `get_width` >32-bit width overflow (R1) | `unit/irep2/get_width.test.cpp`; fix in `irep2_type.cpp:129-215` | **Tier B** — real `array/vector/struct::get_width`; 64-bit-accumulate + reject a product/sum that truncates on the narrow to `unsigned int` | PASS; truncation observed only under NDEBUG | **Done** (PR #6029). |
| **M2** | H-A7 `constant_string` byte reconstruction | `unit/irep2/const_string.test.cpp` | **Tier B** — real `constant_string_access`; bounds/shift/endianness | PASS | **Done** (PR #6030). |
| **M2** | H-A8 `get_sub_expr` index accumulation | `unit/irep2/subexpr_index.test.cpp` | **Tier B** — real dispatch; no `size_t` underflow, in-bounds, null past end | PASS | **Done** (PR #6032). |
| **M3** | H-A4 `hash_combine` + `feed_bigint` CRC ingestion | `unit/irep2/crc_bigint.test.cpp` | **Tier B** — real BigInt CRC heap-grow path; no OOB, sign mixed | PASS | **Done** (PR #6033). |
| **M3** | H-A5 deref variant — `vector_type2t::get_width` symbolic-size guard (R3) | `unit/irep2/get_width.test.cpp`; fix in `irep2_type.cpp:156-171` | **Tier B** — real `vector_type2t`; a non-`constant_int` size now throws `dyn_sized_array_excp` instead of null-deref'ing the failed `dynamic_cast` (assert compiled out under NDEBUG). Anti-vacuity: unfixed code **SIGSEGVs** on the same input under RelWithDebInfo | PASS (fixed); SIGSEGV (unfixed) | **Done** (this PR, fix-and-prove-in-one-PR). Adds only the guard branch (**C-Live**, discharged by the reachability witness — the test input reaches it and the pre-fix crash proves the path executes); no assert removed, so no C-Dead. |

**Approach note.** H-A1 is realised as a **Tier-B** harness (real classes) rather
than a Tier-A standalone C model: verifying `irep2`'s *actual* C++ is the goal, and
the ownership/refcount contract is directly observable on the real
`irep_container` (the `refcount` atomic is a public member). Tier-A standalone
C harnesses remain the right tool only for the self-contained arithmetic kernels
(H-A5/H-A6 width & `as_ulong`) that *can* be lifted out of the template layer
faithfully.

**Nondeterministic input — fuzzing vs. ESBMC (empirical).** Both explore with
nondet input, but reach the real classes differently:
- **ESBMC on the real classes: not possible.** A driver that `#include`s the real
  `irep2` headers fails to parse at the *first* stdlib header — ESBMC's bundled
  C++ operational models do not provide `<atomic>` (nor `immer`, `fmt`), and there
  is no model for `std::atomic::fetch_add` or `immer`'s persistent trie. Confirmed
  by direct trial. So nondet-input + ESBMC applies only to the **lifted arithmetic
  kernels** (Tier A: H-A5/H-A6/H-A7), where it gives bounded-exhaustive proofs a
  fuzzer only samples.
- **Fuzzing on the real classes: the right nondet tool here.** `refcount.fuzz.cpp`
  drives the genuine `irep_container` with libFuzzer bytes under ASan. The same
  `run_ops` driver is replayed deterministically by the unit test so the property
  is pinned in normal CI even with fuzzing off. NB the fuzz target must be built
  **internally consistent** w.r.t. `NDEBUG`: the `#ifndef NDEBUG writer_thread`
  member changes `sizeof(irep2t)`, so mixing an asserts-on object with an
  `NDEBUG` library is an ODR/layout mismatch (the repo build never does this).
  The oracle uses `abort()`, not `assert()`, so it stays live under `NDEBUG`.

**Progress summary.** Tier-A kernels are complete: H-A1/H-A2/H-A3 (refcount, COW
detach, self-alias — PR #6024), H-A4 (CRC/BigInt — #6033), H-A5/R1 (width overflow
— #6029), H-A5-deref/R3 (vector-width guard — this PR), H-A6/R2 (`as_ulong` — #6027),
H-A7 (string recon — #6030), H-A8 (sub-expr index — #6032). All three source-level
findings R1/R2/R3 are fixed-and-pinned. Remaining Tier-A: H-A9 (guard prefix walk),
H-A10 (`gen_zero`/`gen_one` recursion).

**Next task:** M4 — H-A9 guard shared-prefix walk (`common_pointer_prefix_size`
`irep2_guard.cpp:64-108`) agrees with the naive element-scan reference, then H-B4
(guard algebra logical equivalence). Highest soundness stakes; run dual-solver
(Bitwuzla + Z3). H-A9 is a lifted Tier-A kernel (conjunct-id arrays, no atomics),
so ESBMC-on-a-standalone-model is viable here — unlike the refcount/COW harnesses
which had to be Tier-B. See R4 (§10): the walk's correctness depends on the
debug-only canonical-chain assert, so the harness is the load-bearing regression.

