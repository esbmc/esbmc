# Scope — the V.1k (b) IREP2-native Python adjuster

**Program:** Part V of `docs/roadmap/irep2-migration.md` (IREP2-native frontend→goto, #4715).
**Question this scopes:** is the whole-body "resolve-then-build" adjuster the right
next step to close the V.3 residue, and if so, what exactly does it own?
**Status (updated 2026-07-23):** §5.1's Spike-1 conclusion ("adjuster
unnecessary; finish inline") held *for Goal A* — **converter-construction 100%
is now complete**: every arithmetic/relational/member/index node the Python
converter returns is built IREP2-native, the last legacy `plus_exprt` (pointer-
source indexing) drained in PR #6323. So §5.1's Goal-A recommendation was
carried out. But the adjuster was **not** retired: the §5a contingency was taken
for a *different* reason than Goal A — the whole-body `python_adjust` pass
(`--python-irep2-adjust`, default off) was built out (S1/S2 type+aggregate
completion #5985/#5988, exception-id/finally/call-rewrite flip-prep
#5992/#5995/#5996, S4 width reconcile #5999) toward the **flip** that replaces
`clang_cpp_adjust` on the Python path (Goal B / §5a B.5), which the sibling
W1-loc keystone (`docs/roadmap/spike-v1k-w1loc.md`) does not cover. Remaining flip
blockers: "bases" carriage, S3 member/index at scale, S5 arg casts (the
adjust()-error-return blocker is discharged, `python_language.cpp:325`). **Owner:**
TBD. **Refs:** #4715, #5055, #6323; sibling doc `docs/roadmap/spike-v1k-w1loc.md`.

> **Correction to §5.1 / §5a below.** The "adjuster is unnecessary — retire the
> dead pass" wording in §5.1 is scoped to **Goal A only** and is now realised by
> the inline drains completing (#6323). It must *not* be read as a
> recommendation to delete `python_adjust.{h,cpp}`: that pass is live,
> flag-gated infrastructure for the clang_cpp_adjust-replacement flip (§5a was
> taken for that purpose). The "NOT taken" annotation on §5a is stale.

---

## 1. Two goals that the doc sometimes conflates

Separate them before scoping, because they have different blockers and different sizes.

- **Goal A — converter-construction 100%.** Every expression the Python converter
  produces is built IREP2-native, *before* the `clang_cpp_adjust` pass, with no
  legacy `*_exprt` at the returned seam. Blocked by the **V.3 residue** (F-P11 +
  width-hazard). **This is what the adjuster is proposed to close, and the only
  goal in scope here.**
- **Goal B — literal 100% (the §V.1 acceptance bar).** The function *body* handed
  to `goto_convert` is IREP2 with no `migrate_*` back-hop. Blocked by **W1-loc**:
  IREP2 value operands carry no source location, so the body round-trip is
  load-bearing (`restore_value_locations`, `goto_convert_functions.cpp`).
  **Out of scope here** — it is a cross-frontend concern with its own spike
  (`spike-v1k-w1loc.md`). The adjuster does *not* reach Goal B.

## 2. Current state — the infra already exists and is inert

B.0–B.2 landed (`src/python-frontend/python_adjust.{h,cpp}`, ~160 lines,
flag-gated behind `--python-irep2-adjust`, default off):

- It walks each **code** symbol's `get_value2()`, recurses operands inner-to-outer,
  and resolves a transient `symbol_type2t` `member2t`/`index2t` **source** to its
  followed `struct`/`union`/`array`/`vector` via `ns.follow` (`resolve_source`).
- It is **dead-but-tested**: it runs *after* `clang_cpp_adjust`, so it currently
  resolves nothing (the converter still emits post-adjust-resolvable bodies). B.2
  validation showed flag-on vs flag-off parity over the 20-test fixture: 0
  divergences. B.3 (flip the converter to emit transient sources pre-adjust) is
  recorded as "not separable from B.5".

It handles exactly **one** of the two residue mechanisms (member/index source
following). It does **not** do implicit-arithmetic-conversion (width
reconciliation), and it has **never resolved a live site** — every real drain
to date bypassed it (§4).

## 3. The residue, re-grounded (current line numbers) — smaller and simpler than framed

Prior status notes named *three* residue classes. Re-checking the tree collapses
them to **two**, and both are already being drained inline:

1. **F-P11 general-operand** — an *enclosing* node built legacy because migrating
   its operand would trip the (relaxed but real) `member2t`/`index2t` resolved-source
   assert on a recursively-produced sub-expression:
   - the `and`/`or` (BoolOp) node — `converter_binop.cpp:413`
     (`get_logical_operator_expr`; its list-truthiness operand is *already* IREP2).
   - the `isnone` node — `converter_compare.cpp:522`.
   - the `isinstance` node — `builtins.cpp:444`, `python_typechecking.cpp:201`.
   - assorted arithmetic in `build_binary_expression` (`converter_binop.cpp:2172+`).
2. **Width-hazard** — a `-`/`+`/`<` whose operands have concrete-but-mismatched
   widths that `sub2t`/`lessthan2t` reject at construction (legacy tolerates it;
   `clang_cpp_adjust` reconciles downstream). Only **4** legacy arith
   construction sites remain, in 3 files: `list_access.cpp:704/709`
   (`signed_add`/`signed_sub`), `converter_compare.cpp:344` (`plus_exprt`
   pointer arithmetic), `numpy_call_expr.cpp:2637` (`minus_exprt` complex
   negation in `make_complex`). (`python_set.cpp` was already drained to
   `sub2tc` with inline width reconciliation — the exemplar for the rest.)

**Correction to the record — the "isinstance/isnone custom-node retain boundary" is
stale.** `isinstance2t`/`isnone2t`/`hasattr2t` are real IREP2 kinds with **both**
forward (`migrate.cpp:600`,`:1839`, since #3289) and back (`:3661`,`:3677`)
`migrate_expr` arms. `migrate_expr` *can* lower them from a legacy source; the only
thing that can block these sites is an **F-P11 operand** — so they are class 1, not
a fourth class. No `migrate_expr` extension is needed.

## 4. The central finding — the adjuster has not been needed

Every F-P11 / width-hazard site drained so far was closed **inline**, at its own
construction site, with the two mechanisms `clang_cpp_adjust` uses — *not* by the
whole-body pass:

| Drain | Mechanism used | Whole-body adjuster? |
|---|---|---|
| member/index over symbol source (#5710) | inline `ns.follow` (as `build_member_expr_from_class` already did) | no |
| `python_math` mod/floor-div width (#5725) | inline `c_implicit_typecast_arithmetic` (`reconcile_operand`) | no |
| `python_set` `size-1` width (`python_set.cpp:180`) | inline `c_implicit_typecast_arithmetic` → `sub2tc` | no |
| complex int→double, tuple/struct literals (#5727, this session) | migrate-forward round-trip | no |

The `python_adjust` pass resolves member/index sources *post-conversion*, but the
converter's own inline `ns.follow` resolves them *at construction* — so the pass
has stayed inert. This is the same lesson the record already logged twice (the
#5727 "ungateable guard" and the twice-overstated "drained" claims): **the wall is
milder than framed.**

## 5. Recommendation — size the true residue before building the big thing

Do **not** greenlight the whole-body adjuster (B.3–B.5, a large flag-gated,
verdict-parity-gated, cross-cutting project) on the current evidence. Its
necessity is unproven and the trend says it may be unnecessary.

**Spike-1 (bounded, ~1–2 days, the gate):** enumerate the *reachable* residue and
test inline-drainability.
1. Over the whole `regression/python` + model-`.py` corpus, enumerate the enclosing
   sites in §3 that actually receive an operand which (a) migrates to an unresolved
   `member2t`/`index2t` source, or (b) has a mismatched concrete width. (Instrument
   the construction points; count and bucket by site.)
2. For each reachable site, attempt the inline drain: resolve the operand source
   with `ns.follow` (as `build_member_expr_from_class`) and/or reconcile widths with
   the extracted `reconcile_operand` helper, then build the node IREP2-native and
   `migrate_expr_back`. Gate each on **byte-identical GOTO** (A/B) + the suite.
3. Classify the outcome:
   - **If every reachable site drains inline** → converter-construction 100% is a
     short inline task (finish the 6 arith sites + and/or/isnone/isinstance);
     **retire the `python_adjust` pass** (or keep it as documented dead infra).
     The adjuster project is closed as unnecessary. *(Most likely, on current evidence.)*
   - **If some site's operand is genuinely unavailable at its enclosing site**
     (a recursively-produced member/index the enclosing site cannot resolve because
     the inner build deferred it) → that set, and *only* that set, is the adjuster's
     real scope. Proceed to §5a with the residue enumerated, not forecast.

**Acceptance for Spike-1:** a table of reachable residue sites, each marked
`inline-drained` / `needs-whole-body`, with the A/B result. That table decides the
project.

### 5.1 Spike-1 result (executed 2026-07-03) — the adjuster is unnecessary

Spike-1 was run. **Verdict: every residual site drains inline; the whole-body
adjuster is not needed; retire the dead `python_adjust` pass.**

*Method.* An F-P11 / width-hazard violation aborts at **GOTO-build time**, not
solve time, so a `DebugOpt` (asserts-on) build run with `--goto-functions-only`
over the corpus surfaces every abort without solving. Baseline (unmodified,
asserts-on) over 859 and/or+isinstance+isNone test sources: 843 OK; the only 6
aborts are the pre-existing `--irep2-bodies` / `--python-irep2-adjust` flag tests
run without their flags (a harness artifact) — excluded as the clean reference.

*Experiments (each a throwaway flip of the enclosing node to the IREP2 drain,
`migrate_expr` operands → build the `2t` node → `migrate_expr_back`, with a stderr
marker to prove the path was exercised):*

| Exp | Site flipped | Tests | New aborts vs baseline | Drain fired |
|---|---|---|---|---|
| E1 | pure-boolean `and`/`or` (`converter_binop.cpp:530`) → `and2tc`/`or2tc` | 622 | **0** | yes, 7–14×/test |
| E2 | `isnone` (`converter_compare.cpp:522`) → `isnone2tc` | 264 | **0** | yes, 5–8×/test |
| E3 | `isinstance` (`builtins.cpp:444`) → `isinstance2tc` | 264 | **0** | yes, 1–2×/test |

E1 is the doc's own "known-hard" site — the 2026-06-22 status recorded a prior
attempt "reproducing an `index2t` abort even on pure-boolean `(a>0) and (b>0)`". It
**does not reproduce** on current master, including on the 64 tests with and/or over
`self.attr` / `[subscript]` operands (the exact F-P11 risk shape). #5710's inline
member/index resolution means the operands arriving at these enclosing nodes are
already resolved, so `migrate_expr` no longer hits the source assert. The arith
residue (§3.2) is synthetic/concrete-operand or the proven width-reconcile pattern,
so it is inline-drainable by construction. **The F-P11 general-operand wall, as a
live blocker, is gone.**

*Remaining work to converter-construction 100% (Goal A) — all inline, no adjuster:*
finish the drains E1–E3 exercised plus the 4 arith sites (§3.2), each its own
byte-identical-GOTO-gated one-site PR (the same idiom as the struct-literal drains).
Then **retire `python_adjust.{h,cpp}` and the `--python-irep2-adjust` flag** as dead
infra (or leave them documented-dead). Goal B (W1-loc) is unaffected and still open.

### 5a. Contingency — if the adjuster is justified (NOT taken — see §5.1)

Only the `needs-whole-body` set matters; scope the pass to it. Then the doc's
B.3–B.5 stand, tightened:
- **B.3** flip *those* sites to emit transient `symbol_type2t` sources pre-adjust;
  `python_adjust` (already landed) resolves them. Add the missing
  implicit-arithmetic-conversion arm to `python_adjust` for the width-hazard members.
- **B.4** exit assertion: every `member2t`/`index2t` source resolved post-adjust;
  fold in W3 (`#cpp_type`/`#member_name` IREP2-native carriage).
- **B.5** flip `--python-irep2-adjust` default-on, then drop the legacy
  `clang_cpp_adjust` hop from the Python path.
- **Gates (every commit):** full unit suite + `regression/{python,esbmc,esbmc-cpp,floats}`
  verdict parity, dual-solver (Bitwuzla+Z3), asserts build; the 20-test acceptance
  fixture green. RV: the pass must reproduce `clang_cpp_adjust`'s dataclass/inference
  completion exactly — reuse the converter's inferred `tag-` types, do not re-derive.

## 6. Non-goals / boundaries

- **W1-loc / Goal B** (`spike-v1k-w1loc.md`) — the larger, cross-frontend keystone;
  the adjuster does not touch it.
- **On-disk goto-binary format** — unchanged; a permanent `RETAIN_BOUNDARY`.
- **Other frontends** — the relaxed construction assert is a proven no-op for
  C/C++/CUDA/Solidity (all migrate at goto-convert, post-adjust); keep any new
  work Python-scoped.
- **`goto_convert` body seam (W1)** — a `RETAIN_BOUNDARY` per the V.4 outcome; off
  this critical path.

## 7. One-line summary

The residue is two mechanisms (member/index following, width reconciliation), both
already drained inline site-by-site with no whole-body pass; Spike-1 (§5.1) confirmed
every remaining site — including the doc's "known-hard" and/or node — drains inline
with zero asserts-on aborts, so **the adjuster is unnecessary: finish inline and
retire the dead pass.**

---

## Appendix — flip hop-off re-census (2026-07-23)

With S1–S4 and the flip-prep fixes landed (#5985/#5988/#5992/#5995/#5996/#5999),
the `python_adjust` "flip" (skip `clang_cpp_adjust` on the Python path, run
`python_adjust` alone) was re-measured to update the stale 2026-07-10 census.
Method: an `ESBMC_PY_SKIP_LEGACY_ADJUST` env gate around the `clang_cpp_adjust`
call in `python_language.cpp` (throwaway, reverted), comparing the hop-off
verdict to the normal (hop-on) verdict over 150 `regression/python` tests.

**Result: 132/150 MATCH (88%), 0 wrong verdicts, 16 no-verdict, 2 skipped
(KNOWNBUG/FUTURE).** The **zero wrong verdicts** is the important number — the
adjuster is *sound* where it produces a verdict; the gap is entirely
crash/hang (no silent unsoundness).

Blocker distribution of the 16 no-verdict cases:

- **~6 × `type2t::symbolic_type_excp`** — `builtin2`, `builtin2_fail`, `cast`,
  `cast_fail`, `casting-chr-func`, `casting-chr-var-multibyte`. This is the S3
  gap the earlier F-A1 drill localised: an `empty_type2t` is constructed
  **downstream** in goto-convert/symex (not left by `python_adjust` at adjust
  time — the bodies are clean when `adjust()` returns), so the fix is *not* a new
  `python_adjust` arm. Round-2 recipe (still the next step): instrument the four
  `get_width` throw sites (`irep2_type.cpp:168-223`) to name the constructing
  consumer, fix there. This is the dominant, deepest remaining flip blocker.
- **~10 × hang / unsupported / timeout** — `abs-fail`, `assign-fail`,
  `boolop-short-circuit{,-fail}`, `boolop-len-or`, `bytes_fromhex_invalid_fail`,
  `bytes_range_error_fail`, `branch_coverage-fail` (coverage-mode), `builtin2`
  neighbours, `builtin_all_nonliteral` (reached solving), `bytearray_unsupported`
  (frontend feature gap). These are heterogeneous and lower-priority than the
  `symbolic_type_excp` cluster; several may resolve once the S3 downstream fix
  lands (an unresolved width can also manifest as a non-terminating symex).

**Takeaway.** The flip is close (88%) and sound (0 wrong verdicts); the single
highest-value remaining task is the S3 `symbolic_type_excp` / downstream
`empty_type2t` fix, which is a consumer-side (goto-convert/symex) investigation,
not an adjuster arm. The full flip (default-on, delete the `clang_cpp_adjust`
hop) remains a dedicated multi-PR effort gated on dual-solver verdict parity.

### Round-2 drill on the `symbolic_type_excp` cluster (2026-07-23)

Localised the dominant blocker (builtin2 = `chr(ord('A'))`). A backtrace at
`empty_type2t::get_width` (throwaway) names the consumer: it is thrown from
`dereferencet::construct_from_array` (`dereference.cpp:1216`,
`deref_size = type->get_width()`) during `make_return_assignment` of
`test_chr_ord` — symex dereferences the returned `chr()` value with an **empty
target type**.

The GOTO delta is a single missing cast in the return expression:

- hop-on:  `RETURN: (signed int)(back_to_char[0]) == 65 && …`
- hop-off: `RETURN: back_to_char[0] == 65 && …`

`back_to_char` is `signed char[0]`; clang_cpp_adjust promotes the `char` element
to `int` for the comparison, and without that promotion symex derives an empty
deref type on the `char[0]` read.

**Refuted hypothesis (negative result, do not retry):** a `python_adjust` arm
mirroring `clang_c_adjust::adjust_expr_rel` — reconcile the two relational
operands with `gen_typecast_arithmetic` — does **not** insert this cast. Traced
with a debug print: the arm fires on every relational node, but
`gen_typecast_arithmetic` produces no promotion for the builtin2 return equality
because the operands the adjuster sees already carry matching types. So the
hop-on `(signed int)` cast is **not** emitted by `adjust_expr_rel`; it comes from
a different `clang_cpp_adjust` path (candidate: `adjust_index` / the char-array
element read, `clang_c_adjust_expr.cpp`). That path is the next drill target.

**Second lesson (mechanism trap):** a wholesale `migrate_expr_back` +
`migrate_expr` round-trip of a node **reverts** any resolved `member2t`/`index2t`
source in its subtree back to a by-name `symbol_type2t` (the exit invariant then
rejects it, e.g. `__ESBMC_list_sort`). A flip arm that needs to rewrite a node
must wrap/rebuild operands **in place** and never round-trip an
already-resolved subtree.

### Round-3 drill + scope call (2026-07-23)

Probed the structure of the failing builtin2 return comparison. It is **not** a
simple `equality2t(index2t, const)` — no equality with an `index2t`, a
`dereference2t`, or an empty-typed operand reaches `adjust_expr` (the equalities
that do are pointer/null comparisons). `back_to_char` is a **zero-length
`signed char[0]`** modelled as a pointer, so `back_to_char[0]` lowers to a
nested pointer-access sub-expression, and the `empty_type2t` sits on that
nested node — deeper than a relational operand. The next drill needs to dump the
**full expression** `dereferencet::construct_from_array` is dereferencing
(instrument `dereference.cpp:1216` to print the operand, not just crash on
`get_width`), then trace which node carries the empty subtype back through
goto-convert to the adjusted body.

**Scope call.** Two drill rounds have shown this blocker is not a localized
adjuster arm: it is a nested pointer/array element-type resolution that spans the
converter, `python_adjust`, goto-convert and symex, on a **default-off** flip
path. It is the "multi-quarter" adjuster-flip work the record has always flagged,
not a loop-sized slice. Recommendation: the `symbolic_type_excp` cluster (and the
flip generally) should be a **focused, single-owner effort**, sequenced after the
already-built, review-clean W1-loc / destructor-arc / V.3 PRs are merged — not
pursued further by incremental autonomous drills, which now yield findings
without a shippable fix.

### Round-4 drill — root cause pinned (2026-07-23)

Instrumented the crash site (`dereferencet::construct_from_array`,
`dereference.cpp:1216`) to dump the empty node. The **deref *result* type is
empty**: symex dereferences a `char[5]` array
(`symex_dynamic::alloca::dynamic_1_array`, subtype `signedbv` width 8) but the
read's target type is `empty_type2t`. So `back_to_char[0]` is a `dereference2t`
whose **result type is left empty** by the converter; `clang_cpp_adjust` resolves
it to the char element type (after which the comparison promotes char→int), and
`python_adjust` does neither.

**Concrete root and fix direction.** `python_adjust` resolves member/index
*source* types (`resolve_source`) but has **no arm that resolves an empty
*result* type** on an `index2t`/`dereference2t`, and no `dereference2t` arm at
all. The fix is two-part, in order: (1) resolve the empty result type of the
element access to the source's element type (`char`); (2) then the char→int
promotion the comparison needs becomes expressible (round-2's relational arm
no-opped precisely because the operand type was empty — zero width — so
`gen_typecast_arithmetic` had nothing to promote). Both must be byte-identical to
`clang_cpp_adjust`'s output and gated on the hop-off corpus. This is the S3
"member/index at scale" work, now pinned to a specific missing capability
(result-type resolution for element accesses), still a focused-owner slice rather
than a loop drill.

### Census methodology correction + true parity (2026-07-24)

The iter-L/P hop-off census (88% → 92%) **overcounted no-verdicts** due to a
harness bug: it read `test.desc` line 3 as ESBMC flags unconditionally, but a
flagless test (the common `CORE` / `main.py` / `^expected$` three-line form) has
its **first expected-output regex** on line 3, not a flags line. Passing that
regex as an argument made ESBMC abort with "failed to figure out type of file
^...", producing a spurious no-verdict. Fix: treat line 3 as flags only when it
starts with `-`, else empty.

Re-running with the corrected harness **and the deref-result fix (#6340)**:

- Of the "10 remaining no-verdict" cases from iter P, **9 were pure harness
  artifacts** — with correct flags they match hop-on exactly (abs-fail,
  boolop-short-circuit{,-fail}, branch_coverage-fail, builtin_all_nonliteral,
  bytes_fromhex_invalid_fail, bytes_range_error_fail, assign-fail,
  bytearray_unsupported). Only **`boolop-len-or`** is a genuine hop-off issue,
  and it is *past* adjust: it reaches the solver ("Caching time…") and then
  hangs/times out under hop-off while hop-on solves quickly — a symex/solver
  divergence (a harder formula from a shape difference), **not** a
  type-resolution crash.
- A corrected 62-test alphabetical slice (`a*`/`b*`) scores **61 match / 1 diff
  (`boolop-len-or`)** — ~98%.

**Revised picture.** The `python_adjust` flip is materially closer than the raw
census suggested: the dominant crash cluster (`symbolic_type_excp`) is fixed
(#6340), and most of the residual "blockers" were measurement noise. The genuine
remainder is small and different in kind (a solver hang, plus the cosmetic
char→int promotion gap that does not affect verdicts).

### Definitive corpus-wide census (2026-07-24)

Ran the corrected, correctly-flagged harness (`fullcensus.sh`: line-3-is-flags
only when it starts with `-`; hop-on vs hop-off verdict of the #6340 binary) two
ways over `regression/python` (4367 `main.py` tests total):

- **Sequential a–b prefix:** 228 tests, **228 match / 0 diff** (the earlier
  `boolop-len-or` diff did not recur — it is a load-dependent timeout flake at the
  solver, not a hop-off type-resolution failure).
- **Unbiased strided sample** (every 15th test across the whole alphabet, so it
  spans `c`–`z` the sequential run never reached): 291 tests, **290 match / 0
  diff** (one test produced no verdict under either mode — a genuine timeout in
  both — so it is parity, not a divergence).

Across ~520 sampled tests spanning the full corpus, **zero verdict divergences
and zero wrong verdicts**. The full 4367-test sweep is impractical inline (hours
under the per-test timeout cap) and belongs in CI/nightly, but the strided sample
is unbiased and decisive: with the deref-result fix (#6340) the `python_adjust`
hop-off is at **verdict parity with `clang_cpp_adjust` corpus-wide**. The only
known non-parity case, `boolop-len-or`, is an intermittent solver-time flake
(reaches "Caching time…" then times out under load), not a type-resolution
defect. The flip's remaining work is therefore not correctness but the two
cross-cutting engineering items already scoped: making `python_adjust` the sole
adjuster on the Python path, and (optional, cosmetic) GOTO byte-identity.

### gap-2 (char→int promotion) — byte-identity not worth pursuing (2026-07-24)

The deref-result fix (#6340) gives the flip **verdict-parity** on the
`symbolic_type_excp` cluster but not GOTO byte-identity: hop-on additionally
promotes the resolved char element to int in the enclosing comparison
(`(signed int)(back_to_char[0]) == 65`). Attempted to close that "gap 2" with a
`python_adjust` relational arm mirroring `clang_c_adjust::adjust_expr_rel`
(`gen_typecast_arithmetic` on the two operands, wrapping in place). Two negative
results:

1. A type-inequality wrap test **over-wraps** (`== (signed int)65` vs clang's
   `== 65`): `migrate_type` does not round-trip every type attribute, so an
   untouched operand's migrated type compares unequal to its own. Fixed by
   wrapping only when `gen_typecast_arithmetic` actually inserted a typecast on
   the legacy copy (`l.id() == "typecast"`) — the builtin2 return line then
   matches clang exactly.
2. But the arm runs on **every** relational node in the program, and even with
   the corrected wrap it diverges **corpus-wide** from clang's `adjust_expr_rel`
   (~7500 diff lines on builtin2 alone) — the migrate-based operand
   reconciliation does not reproduce clang's promotions node-for-node across the
   OM bodies.

**Conclusion:** byte-identity for these cases is a hard, corpus-wide
reconciliation problem, **not needed** for the flip (whose acceptance gate is
dual-solver *verdict* parity, already met by #6340), and **cosmetic** (verdict
matches). Deferred/abandoned. The deref fix stands on its own; do not gate the
flip on this promotion.

### Remaining hop-off gaps — precise map (2026-07-24, post-#6340 + #6348)

With the deref-result arm (#6340), the `if2t` bool-cast arm (#6348), and the
`--python-irep2-adjust-only` flag in place, a clean census (mypy noise and
expected uncaught-exceptions filtered out; genuine ESBMC-error signatures only)
over a 546-test strided sample (every 8th `regression/python` test) gives:

- **520 / 546 verdict-and-error parity (~95%)** between hop-off and the legacy
  `clang_cpp_adjust` path.

The residual ~5% is now categorised precisely — reproduced signatures, not the
old vague "F-A" buckets:

| Family | Count (in sample) | Reproduced signature | Root cause | Tractability |
|---|---|---|---|---|
| **S3 unresolved-by-name** | 7 | `python_adjust: symbol '…' retains N unresolved by-name (symbol_type2t) node(s) after adjust` | **✅ 7/7 fixed (#6369, #6372).** Not the dict-key-type infra this row first assumed — the firing check was the *operand-count* invariant: an Optional (`int \| None` = `{is_none, anon_pad$, value}`) built resolved-but-underpadded. #6369 pads a resolved-struct literal (6/7); the last case then hit an argument struct mismatch because `adjust_type` had no `code_type` arm, so a struct embedded in a function signature was never padded — #6372 recurses into `code_type`. | ~~hard~~ **two arms** |
| **index-over-pointer** | 3 | `std::runtime_error: Unexpected index type in computer_pointer_offset` (type_byte_size.cpp:338) | an `index2t` whose `source_value` is a `char*` (a Python string / decayed-array parameter). `clang_c_adjust::adjust_index` rewrites `p[i]` → `*(p+i)`; `python_adjust` does not. A naive mirror (`dereference2tc(elem, add2tc(ptr, idx))` — the exact `build_dereference` shape) removes this crash but surfaces a **deeper** `irep2_cast_error` in `goto_symex_statet::fixup_renamed_type` (a function-argument rename where `orig_type` is pointer but the renamed value is not), so the fix needs symex-side type-tracking care, not just the adjust arm. | **hard** (symex rename interaction) |
| **wrong/absent verdict** | ~4–6 | legacy `SUCCESSFUL` → hop-off `FAILED` (sqrt3, math_gamma_noninteger, github_3690, github_6258) or hop-off no-verdict (complex_pow_float_exponent, div6, jpl, list25, string18, ternary_operator4 — several are the crashes above manifesting as an empty verdict) | SMT-level: mostly the two crash families above surfacing as no-verdict; the true wrong-verdicts cluster on math/complex functions and need per-case triage | **mixed** |

**Notes.**
- The heterogeneous ternary case the earlier reviewer flagged (`"a" or 0`,
  pointer-vs-int operands) is **at parity** — both paths report the same verdict
  (a shared pre-existing mixed-type BoolOp imprecision, not a hop-off gap).
- `string22_fail` is hop-off-*better*: legacy gives no verdict, hop-off reports
  `FAILED` correctly.
- Method reminder (recurring trap): the raw census over-counts by ~40× if mypy
  `error: … [tag]` lines and expected `uncaught exception: …` in `*_fail` tests
  are not filtered — grep the genuine ESBMC signatures (`must be boolean`,
  `terminating due to`, `symbolic_type_excp`, `retains … unresolved by-name`,
  `Unexpected index type`) present in hop-off **but not** legacy.

**#6323 does not address index-over-pointer.** #6323 ("Build pointer-source
index arithmetic natively in IREP2") changed the *converter*'s `build_index`
pointer branch (`plus_exprt` → `add2tc`, `python_expr_builder.cpp`). Re-checked
after it merged: the three index-over-pointer cases still crash under hop-off
with the same `Unexpected index type in computer_pointer_offset` — the offending
`index2t` reaches `python_adjust` from a site *other* than `build_index` (a
survived string-element read), so a converter-side fix does not cover it.

**Root cause pinned — a missing array→pointer decay, not an index problem.**
Instrumenting `goto_symex_statet::fixup_renamed_type` at the deeper
`irep2_cast_error` (the crash the naive index arm trades into) shows the node
with `orig_type = pointer` but a renamed value of `constant_array` (a `signedbv[8]`
of size 1 — the empty string `""`). So a `char*` variable (e.g. `word`) holds a
**bare array value that was never decayed to `address_of(array)`**. `clang_c_adjust`
performs this array→pointer decay when an array value meets a pointer context;
`python_adjust` does not, so the pointer variable carries an array value and any
pointer use of it (the `*(p+i)` the index arm builds, but also plain pointer
arithmetic) mismatches at symex rename. The index-over-pointer crash is therefore
a *symptom*: the real gap is the missing decay, which is broader than indexing and
must be fixed at the assignment/typecast seam (where `word = ""` should become
`word = &""[0]`-style), mirroring clang, before the index arm can be sound.

**✅ FIXED — PR #6363.** Both arms landed together in `python_adjust::adjust_expr`:
(1) an array→pointer decay on a `code_assign2t` with a pointer target and array
source (`word = ""` → `word = &array[0]`, with the `address_of` subtype the
target *pointee* so the value is exactly the target type), and (2) the
`p[i]`→`*(p+i)` index rewrite. The decay is the load-bearing half — with `word`
now holding a real pointer, the index arm renames cleanly (no `irep2_cast_error`).
Census: MATCH 520 → 525 (the 3 index-over-pointer cases plus `string18` and
`string22_fail`), zero regressions; C-Live on both arms; a matched hop-off test
pair added. So of the three residual families above, **index-over-pointer is
resolved**; S3 unresolved-by-name and the math/complex verdicts remain.

**Direction — update.** Every family in this map that was first labelled "hard /
needs its own infra" turned out, on inspection, to be a one- or two-arm mirror of
what `clang_c_adjust` already does: the deref-result (#6340), the `if2t` bool cast
(#6348), the array→pointer decay + pointer indexing (#6363), the resolved-struct
padding (#6369), the `code_type` signature padding (#6372), and the index
typecast (#6373). **Drill the actual invariant or error before believing the
label** — the census signatures over-abstract the cause. The hop-off is now at
~97.5% parity with zero crashes in sample.

### The assignment-conversion trap — do not mirror `adjust_assign` alone (2026-07-24)

The one remaining false alarm worth naming is `precedence2` (legacy SUCCESSFUL,
hop-off FAILED). Its violated property is `assert x == 7` where `x` is *double*
(Python retyped it at an earlier `x /= 3`), so `x |= 7` is a bitwise op on a
double. Two things differ from legacy:

1. the index typecast — fixed in #6373;
2. the assignment conversion: legacy emits `x = (double)((signed long)((signed
   int)x) | 7)`, the hop-off emits the integer result with no conversion back to
   the target's type.

`clang_c_adjust::adjust_assign` (`clang_c_adjust_code.cpp`) is exactly
`gen_typecast(ns, code.op1(), code.op0().type())`, so mirroring it in
`python_adjust` looks like a two-line fix. **It is not, and it is unsafe.** Both a
blanket `typecast2tc(target->type, source)` *and* a faithful `gen_typecast` call
on the legacy view fix `precedence2` — and both make `neural-net_fail`
(`--fixedbv`) report **SUCCESSFUL where the legacy path correctly reports
FAILED**, i.e. they *mask a real bug*. Isolated by removing the arm and
re-testing.

The reason is *where*, not *how*: `adjust_assign` runs **after
`adjust_operands(code)`**, which recursively applies `gen_typecast_arithmetic` to
the right-hand side's binary operations. `python_adjust`'s recursion does not do
that operand-level arithmetic reconciliation, so converting only at the
assignment seam changes the stored value. The assignment conversion is therefore
only sound **coupled with** operand-level arithmetic reconciliation (the S4
width-reconcile work) — they must land together, and a missed bug is worse than a
false alarm, so neither half should be shipped alone.

**Sizing the coupled fix.** The operand half is not a thin
`gen_typecast_arithmetic` call. `clang_c_adjust::adjust_expr_binary_arithmetic`
(`clang_c_adjust_expr.cpp:428`) is ~114 lines and additionally performs
complex-number promotion (`{val, 0}` for a non-complex operand), maps `+`/`-`/`*`
to `ieee_add`/`ieee_sub`/`ieee_mul` for `floatbv` element types, and binds
side-effecting operands into a `code_blockt` before reconciling. Mirroring it
faithfully in `python_adjust` — which is what soundness requires, per the trap
above — is a **multi-PR effort in its own right**, not an adjuster arm. It should
be scoped and sequenced deliberately (operand reconciliation first, with its own
parity gate, then the assignment conversion on top), not attempted as a
single slice. Until then `precedence2` stays a known, documented false alarm.

**State of the hop-off after this round.** ~97.5% verdict parity on a 365-test
strided sample with **zero crashes**. Everything that was a *structural* gap has
landed (#6340, #6348, #6363, #6369, #6372, #6373). What remains is one coupled
arithmetic-conversion effort (above) plus six pre-existing per-case divergences —
two false alarms (`cmath_polar_rect_semantics_success_07`, `github_3690`) and
four no-verdict cases (`github_3012_3_fail`, `higher-order3`,
`int_to_bytes_kwargs_fail`, `ternary_string_fail`,
`github_4796_object_handle_eq_fail`) — each needing its own triage rather than a
shared mechanism.

### Per-case triage round 1 — `higher-order3` was a missing function→pointer decay (2026-07-25)

Triaging the no-verdict list one case at a time immediately paid off: three of
the five turned out to have *distinct* signatures, and the first is another
one-arm structural mirror, not a per-case oddity.

| Case | Reproduced signature under hop-off |
|---|---|
| `higher-order3` | `ERROR: Unexpected type in int/ptr typecast` (`smt_casts.cpp:228`), plus `No target candidate for function call *times3` |
| `github_3012_3_fail` | `bitwuzla: … term with unexpected sort at index 0` |
| `int_to_bytes_kwargs_fail` | `ERROR: shr` |
| `ternary_string_fail`, `github_4796_object_handle_eq_fail` | silent no-verdict |

**✅ FIXED — `higher-order3`.** The GOTO diff against legacy pins it in one line:
legacy emits `RETURN: &make_multiplier@F@mul`, the hop-off emits
`RETURN: make_multiplier@F@mul` — a bare code-typed *function designator*. The
caller then holds `typecast(mul, double (*)(void *))`, which
`convert_typecast_to_ints` cannot encode (its `from` is neither sbv/ubv/fbv/fpbv
nor bool) and which symex cannot resolve to a call target.
`clang_c_adjust::adjust_symbol_expr` (`clang_c_adjust_expr.cpp:246-253`) decays
*every* code-typed symbol reference to an implicit `&f` ("special case: this is
sugar for &f", C11 6.3.2.1p4); `python_adjust` did not. The fix is a
`code_return2t` arm mirroring it at the one seam Python reaches this from —
deliberately narrower than clang's universal symbol decay, because the callee
position of a call is already owned by `wrap_function_pointer_callee` and a
blanket decay would fight it. Same family as the array→pointer decay (#6363):
**the third "per-case" bucket entry to dissolve into a missing clang mirror.**

**Note on closure capture.** The matched test pair deliberately returns a
*non-capturing* nested `def`. `higher-order3`'s own `make_multiplier(3)` /
`times3(4)` free-variable capture is wrong on **both** paths (an
`assert times3(4) == 12` fails under legacy too) — a pre-existing frontend gap,
at parity, and out of scope here.

**Direction, restated.** The per-case list is not a homogeneous bucket. Reproduce
each signature separately before assuming a shared cause: two of the five are
solver-level (`bitwuzla` sort, `shr`) and two are still silent. The remaining
four are the next slices.

### Per-case triage round 2 — `int_to_bytes_kwargs_fail` was an unmigrable `shr` (2026-07-25)

`ERROR: shr` is **not** solver-level, as round 1 guessed from the message
prefix — the full output ends `migrate expr failed`. The Python frontend builds
the `int.to_bytes()` byte-extraction shift as a legacy `exprt("shr", …)`
(`str_conv.cpp:486`), but `shr` is a **pre-adjust placeholder**:
`clang_c_adjust::adjust_expr_shifts` (`clang_c_adjust_expr.cpp:325-360`) resolves
it to `lshr`/`ashr` on op0's signedness (C11 6.5.7), and `migrate_expr` only has
arms for the *resolved* forms (`migrate.cpp:3240`, `:3392`). With
`clang_cpp_adjust` gone, the raw `shr` reaches migration and aborts before any
verdict.

Unlike round 1's fix this is **converter-side**, so it changes the default path
too — and `lshr` skips `adjust_expr_shifts` entirely (it is gated on `shl`/`shr`
at `clang_c_adjust_expr.cpp:139`), losing the `gen_typecast_arithmetic` that
would have run on both operands. That is only safe if the promotion was a no-op.
It was, and this was **measured, not argued**: the default-path
`--goto-functions-only` output over all 14 `to_bytes`/`from_bytes` tests is
byte-identical pre- and post-patch (modulo timing lines), because `value` is
unsignedbv-by-construction at that point and both operands already share its
type.

**Census sweep — `shr` was the only such id.** Enumerating every legacy `exprt`
id the Python frontend constructs and checking each against `migrate.cpp` gives
one genuine gap. `if` / `not` / `typecast` are handled via the `exprt::` id
constants (a literal-string grep misses them) and `_init_undefined` is an
already-guarded sentinel (`python_exception_handler.cpp:317`). So this class of
hop-off abort is now **closed**, not merely sampled.

Two of five triaged; `github_3012_3_fail` (bitwuzla sort), `ternary_string_fail`
and `github_4796_object_handle_eq_fail` (silent) remain.

### Per-case triage round 3 — the branch/loop condition was never cast to bool (2026-07-25)

`github_3012_3_fail`'s `bitwuzla: … term with unexpected sort at index 0` is
again not what the message suggests. Its hop-off and legacy VCCs are nearly
identical; the one structural difference is the guard:

```
legacy:  {-4} goto_symex::guard…#1 == (x&0#1 != 0)   {-5} !goto_symex::guard…#1
hop-off: {-4} !x&0#1                                  {1} … => x&0#1
```

The hop-off applies `!` and `=>` to a **signedbv**. `clang_c_adjust` casts every
branch and loop condition (`adjust_ifthenelse`, `adjust_while`, `adjust_for` —
all `gen_typecast_bool`, `clang_c_adjust_code.cpp:106-128`); `python_adjust` had
only the `if2t` *ternary-expression* arm (#6348), never the statement-level ones.
Python's `if n:` on a plain int is the common case, so the raw bitvector reached
the solver. Fixed by mirroring all four (`code_ifthenelse2t`, `code_while2t`,
`code_dowhile2t`, `code_for2t`).

**This one arm closed six divergences, not one.** Besides `github_3012_3_fail`
it fixed `github_4796_object_handle_eq_fail` (the last of the round-1 "silent"
pair but one) and four cases the earlier census had filed under *wrong/absent
verdict* — `div6`, `filter_loop`, `github_3841_6`, `jpl` — all of which were
no-verdict for this same reason. The "wrong/absent verdict — needs per-case
triage, mixed tractability" row in the table above was therefore substantially
**one missing mirror**, not a set of independent SMT-level oddities.

**A wider census confirms the recorded parity figure.** A stride-6 run (733 of
4 401 `regression/python` tests, executed 12-way parallel against a *pinned
binary snapshot* so concurrent rebuilds cannot perturb it) found **18
divergences — 97.5% parity**, matching the "~97.5% on a 365-test strided sample"
recorded above. The wider sample did not move the *rate*; what it changed is the
**composition**, by naming the full divergence set rather than a handful.

Two of the 18 are method artifacts: `python_irep2_adjust_only_boolop_or` and
`python_irep2_adjust_only_optional` already carry `--python-irep2-adjust-only` in
their own `test.desc`, so both census columns are hop-off runs and the empty cell
is the known load-dependent solver timeout — they pass under `ctest`. Twelve are
genuine.

After this arm, parity on the same sample is **98.1%** (14 divergences), and the
whole "no verdict because the guard was a bitvector" family is gone.

**Still open after round 3** (excluding the two artifacts):

| Kind | Cases |
|---|---|
| no verdict | `ternary_string_fail`, `github_2934_2`, `github_3078_fail`, `github_3337_2_fail`, `github_4784_isnone_short_circuit_fail`, `none3`, `optional6` |
| legacy SUCCESSFUL → hop-off FAILED | `cmath_polar_rect_semantics_success_07`, `github_3690`, `github_4745_pep604_class_attr`, `math13`, `math_edge_frexp_success`, `sqrt5` |

The wrong-verdict group is the higher-risk one — a hop-off `FAILED` against a
legacy `SUCCESSFUL` is either a false alarm or a real bug the legacy path masks,
and only per-case triage distinguishes them. Note it now clusters visibly on
math/float (`math13`, `math_edge_frexp_success`, `sqrt5`,
`cmath_polar_rect_semantics_success_07`), which points at the parked S4
arithmetic-conversion work rather than at six independent causes — but that is a
hypothesis from the names, not a triaged finding.

### Per-case triage round 4 — the math cluster was `sqrt`, not S4 (2026-07-25)

The round-3 hypothesis above was **wrong**: the math/float cluster has nothing to
do with arithmetic conversion. `sqrt5`'s counterexample shows `result = -NAN`,
and the GOTO diff isolates it to one instruction:

```
legacy:  ASSIGN result = ieee_sqrt((double)x, __ESBMC_rounding_mode)
hop-off: FUNCTION_CALL: result = sqrt((double)x)
```

`python_math::handle_sqrt` emits a call to `c:@F@sqrt` whenever the argument is
not a foldable constant; `clang_c_adjust` rewrites that call to the `ieee_sqrt`
intrinsic (`clang_c_adjust_expr.cpp:1414-1423`), and `python_adjust` did not, so
the hop-off ran the library model, which returns NaN. Fixed by mirroring the
lowering — closing `sqrt5`, `math13`, and
`cmath_polar_rect_semantics_success_07`, the last of which this document had
recorded as a standing *false alarm*. It was not; it was this.

**Method note — match the field the legacy guard matches.** A first attempt
compared `symbol2t::thename` (the full identifier, `c:@F@sqrt`) against
`"sqrt"`, and the arm silently never fired. The legacy guard uses the symbol's
**base** name (`to_symbol_expr(f_op).name()`) and applies the `py:` exclusion to
the *identifier*. IREP2's `symbol2t` carries only the identifier, so the base
name must be recovered (segment after the last `@`). A non-firing arm looks
exactly like a wrong hypothesis — check the arm fires before discarding the
theory.

**Also landed: `&array` → `&array[0]` at the node level.** `clang_c_adjust::adjust_address_of`
(`clang_c_adjust_expr.cpp:743-754`) decays unconditionally; #6363 added only the
assignment-seam form, which never reaches an `address_of` nested inside an
aggregate literal. The OM raise sites build `{ .message = &"math domain error" }`
with a `char*` member, so the literal carried a `char(*)[N]`. **This flips no
verdict in the sampled corpus** — it closes a structural parity gap (the hop-off
GOTO now matches legacy byte-for-byte at those sites), and is recorded as such
rather than as a divergence fix.

**Remaining after round 4** — a stride-12 census (366 tests) leaves five genuine
divergences: `github_3078_fail`, `github_4784_isnone_short_circuit_fail` (no
verdict), `github_3690`, `math_edge_frexp_success`,
`github_4745_pep604_class_attr` (legacy SUCCESSFUL → hop-off FAILED), plus
`ternary_string_fail`, `github_2934_2`, `none3`, `optional6` from the wider
stride-6 run. `math_edge_frexp_success` is `frexp`, so the "one missing intrinsic
lowering per math builtin" shape may repeat — check `clang_c_adjust`'s builtin
list before assuming anything deeper.

**Census artifact, restated.** Any test whose own `test.desc` already carries
`--python-irep2-adjust-only` (`python_irep2_adjust_only_*`) appears as a
divergence in this census, because both of its columns are hop-off runs and the
40 s cap bites under 12-way parallelism. They pass under `ctest`. Filter them
out before counting.

### Per-case triage round 5 — the intrinsic-lowering class is closed; frexp is S4 (2026-07-25)

Rather than triage the remaining math cases one at a time, enumerate the class.
`clang_c_adjust` lowers exactly thirteen library calls to SMT intrinsics —
`fabs`, `finite`, `fma`, `huge_val`, `inf`, `isfinite`, `isinf`, `isnan`,
`isnormal`, `nan`, `nearbyint`, `signbit`, `sqrt`. Intersect that with the names
the Python frontend actually emits as `c:@F@…` calls (`acos … tanh`, `trunc`,
plus the `__python_*` string helpers) and the answer is **two**: `sqrt` (round 4)
and `fabs`. The other eleven are never reached as calls from this frontend, so
an arm for any of them would be dead instrumentation.

**`fabs` landed.** `clang_c_adjust_expr.cpp:1239-1245` rewrites it to the `abs`
intrinsic; a probe shows legacy `RETURN: abs(x)` against hop-off
`FUNCTION_CALL: fabs(x)`. Verdicts happened to agree (the `fabs` model is
faithful, unlike `sqrt`'s), so this is a **parity** fix with the divergence risk
removed rather than a verdict flip. With it, **the intrinsic-lowering class is
closed** — not sampled.

**`math_edge_frexp_success` is not an intrinsic gap at all.** `frexp` appears in
neither list. Its GOTO diff is one instruction:

```
legacy:  ASSIGN e = (double)ESBMC_unpack_temp….element_1;
hop-off: ASSIGN e =         ESBMC_unpack_temp….element_1;
```

`e` is a `double` tuple-unpack target and `element_1` is the integer exponent, so
this is precisely `clang_c_adjust::adjust_assign`'s conversion — **the
assignment-conversion trap documented above**. It must *not* be fixed with a
standalone `adjust_assign` mirror: both the blanket and the faithful
`gen_typecast` version make `neural-net_fail` report SUCCESSFUL where legacy
correctly reports FAILED, i.e. they mask a real bug. It is S4 work, and it is now
pinned to a concrete second reproducer (`precedence2` was the first).

**Round-3's hypothesis, settled.** The guess that the math/float cluster pointed
at S4 was *partly* right and mostly wrong: `sqrt5`, `math13` and
`cmath_polar_rect_semantics_success_07` were a missing intrinsic lowering, while
`math_edge_frexp_success` genuinely is S4. The lesson stands — the cluster's
*name* carried no information; only the per-case GOTO diff did.

### Per-case triage round 6 — the ternary pair was a *migrate-synthesised* cast (2026-07-26)

The last two no-verdict cases from round 3's table, `ternary_string_fail` and
`github_3337_2_fail`, are one cause. Both assign a string-valued ternary
(`s: str = "" if b else "foo"`), and the GOTO diff is one instruction:

```
legacy:  ASSIGN s = b ? &{ 0 }[0] : &{ 102, 111, 111, 0 }[0];
hop-off: ASSIGN s = b ? (signed char *){ 0 } : (signed char *){ 102, 111, 111, 0 };
```

A pointer-typed array constant, which the SMT layer rejects
(`ERROR: Unexpected type in int/ptr typecast`); `ternary_string_fail` instead
runs `strcmp` off the end of the bogus pointer and unwinds until the heap is
exhausted. Fixed by an `is_typecast2t` arm that decays an array operand to
`&arr[0]`, mirroring `c_typecastt::do_typecast`'s array case
(`c_typecast.cpp:877-905`) — the same decay the `address_of` (#6395) and
assignment-seam (#6363) arms already perform, now at the cast node itself.

**The novel part: the offending node has no converter site.** Every earlier
round fixed something the converter emitted and `clang_c_adjust` then rewrote.
Here the raw cast is synthesised *during migration*: `migrate_expr`'s ternary arm
coerces a branch whose type id diverges from the result type
(`migrate.cpp:1001`), and the Python converter's ternary genuinely has array
branches under a `char*` result type. On the legacy path `adjust_if` decays the
branches *before* migration, so that arm never fires and the cast never exists.
**Consequence for future triage: a hop-off-only node may have been built by
`migrate_expr`, not by the converter — grep migrate before hunting for a
converter site.**

**Negative result — do not mirror `adjust_if`'s branch conversion.** The obvious
fix is the other half of `clang_c_adjust::adjust_if`
(`clang_c_adjust_expr.cpp:1689-1693`): convert both branches to the result type.
It is **dead code** in `python_adjust` today — probed with a stderr marker over
40 ternary-bearing tests, **0 firings**.

*Be precise about why, because the obvious explanation is wrong.* `migrate_expr`
coerces on **`type_id` inequality only** (`migrate.cpp:1001`) — exactly what
`if2t`'s constructor asserts (`irep2_expr.h:809-810`) — whereas `adjust_if`
tests **full type inequality**. Migration therefore does *not* equalise the
branch types in general: a same-kind/different-width pair (`signedbv 8` vs
`signedbv 64`) is coerced by neither, and would reach the SMT backend as an
`ite` over differently-sorted terms. That residual class is simply unobserved on
the Python path — probes with `ord(s[0]) if b else 300`, `s[0] if b else 300`,
`5 if b else len("foo")` and an untyped `5 if b else "foo"` all either matched
types or routed to the frontend's nondet ternary-result fallback. The mirror
stays out under the C-Live bar, not because the case is unreachable in
principle; the fix this round needed belongs at the cast node anyway.

### Per-case triage round 7 — the instance-pointer pair (2026-07-26)

`github_4784_isnone_short_circuit_fail` (hop-off: `irep2_cast_error:
to_pointer_type() called on type whose type_id is struct`) needed **two**
mirrors, and neither alone is enough — measured, by disabling each in turn:

```
legacy:  ASSIGN cur=&(*head);           ASSIGN …=ISNONE(cur->nxt, 0);
hop-off: ASSIGN cur=*head;              ASSIGN …=ISNONE(<raw member irep>, 0);
```

1. **`member2t` over a pointer source.** `clang_c_adjust::adjust_member`
   (`clang_c_adjust_expr.cpp:307-313`) wraps a pointer base in a dereference, so
   `p.field` becomes `p->field`. `python_adjust` resolved a *symbol-typed*
   source but left a *pointer* source alone, so symex read a member off a
   pointer and the expression printer fell back to dumping the raw irep.
2. **struct value assigned to a pointer target.** Binding an instance parameter
   (`cur = head`, `head` a `pointer→tag-Node`) lowers to `cur = *head` — a
   struct value. `c_typecastt::implicit_typecast_followed`'s struct arm
   (`c_typecast.cpp:729-740`) takes its address; legacy emits `cur = &(*head)`.
   This is the struct sibling of the array→pointer decay (#6363) at the same
   seam.

With only (1), the run still aborts — the same crash, one layer down; with only
(2), it aborts differently (`struct_union_member_names() called on incompatible
type (type_id = pointer)`) and the member node stays malformed. **Do not accept
the first arm that changes the error message as the fix**; re-run with each half
disabled to establish which are load-bearing.

The `adjust_member` *array*-base arm (`base.type().is_array()` → `base[0]`) is
**not** mirrored. None of the shapes triaged here produced an array-typed member
base — Python attribute access is over a class instance, i.e. a symbol or an
instance pointer — and adding the arm without first proving it reachable would
be dead instrumentation. Probe before adding it, not after.

### Per-case triage round 8 — `not` over a non-Boolean operand (2026-07-27)

Re-running the round-3 open list against the round-7 binary shrinks it from nine
cases to three: `github_3078_fail`, `github_2934_2` and `optional6` came to
parity with the instance-pointer mirrors (#6428) without being triaged
individually — a reminder to **re-measure the list before triaging it**, since a
structural mirror routinely closes cases filed under unrelated headings.

**✅ FIXED — `none3`.** Its hop-off run aborted at
`Assertion failed: (a->sort->id == SMT_SORT_BOOL), function mk_not`. The
`python_user_main` diff is two instructions, both a missing outer cast:

```
legacy:  ASSERT !((_Bool)((_Bool)x ? (_Bool *)1 : x))
hop-off: ASSERT !(        (_Bool)x ? (_Bool *)1 : x )
```

`x` is `None`, so `x and True` lowers to a pointer-typed short-circuit select and
`not (…)` negates a non-Boolean. `clang_c_adjust::adjust_expr_unary_boolean`
(`clang_c_adjust_expr.cpp:1530-1538`) casts `not`'s operand with
`gen_typecast_bool`; `python_adjust` had the *statement*-level condition casts
(round 3) and the `if2t` ternary cast (#6348) but nothing for `not2t`. Fixed by
mirroring the unary arm.

**The binary half of that clang arm stays out — measured, not assumed.**
`adjust_expr_binary_boolean` casts both operands of `and`/`or`
(`clang_c_adjust_expr.cpp:1540-1548`). A probe firing on any `and2t`/`or2t` with
a non-Boolean operand recorded **0 firings across 40 boolean-op-bearing
`regression/python` tests**: Python's `and`/`or` desugars to a ternary select
(the `if2t` #6348 already covers), and the `and2t`/`or2t` nodes the frontend does
build come from comparisons, which are Boolean by construction. Mirroring it
would be dead instrumentation under the C-Live bar.

**`github_4745_pep604_class_attr` is a padding bug, and it is *not* a re-pad by
`adjust_type`.** Its hop-off `dereference failure: Object accessed with illegal
offset` traces to one type in the GOTO dump — the `int | None` attribute's
`tag-Optional_signedbv` is padded **twice**:

```
legacy:  { _Bool is_none; unsigned _ExtInt(56) anon_pad$1; signed long int value; }
hop-off: { _Bool is_none; unsigned _ExtInt(48) anon_pad$1;
           unsigned _ExtInt(56) anon_pad$1; unsigned short int anon_pad$3;
           signed long int value; }
```

The offsets are exactly what re-running `add_padding` over an already-padded
struct whose pad member has lost its `#is_padding` flag produces (the 56-bit pad
aligned as a real 7-byte field: `1 → +6 → 7`, then `value` at `14 → +2 → 16`).
That makes `adjust_type`'s struct arm the obvious suspect — but instrumenting its
`add_padding` call **acquits it**: over this test the arm fires 322 times, and
every one of the 5 firings on `tag-Optional_signedbv` reads `{is_none, value}`
(both unpadded) and writes `{is_none, anon_pad$1, value}`. It never once receives
an already-padded struct, so it cannot be the second pad. `Box` itself is padded
once, `{x, flag}` → unchanged.

Two consequences for the next drill. (1) The second pad is applied **downstream
of `python_adjust`** — find that site before touching the adjuster. (2) The five
separate firings are themselves a finding: the type-symbol write-back is not
making the padded form visible to later sites, so each one re-pads its own copy
from scratch. That is the likely mechanism by which one copy reaches a
downstream padding pass with `#is_padding` already lost.

**Method warning — this negative result was wrong once.** An earlier run of the
same probe reported *zero* firings and was recorded here as "not this arm". The
binary was stale: the `make esbmc` that was supposed to build the probe inherited
a `cd` to the source root and failed silently. Re-check that a probe binary
actually contains the probe (a non-zero firing count on *some* input) before
believing a zero.

`github_3690` (a dict-of-lambda call returning `1.0000000000000002` instead of
`1.0`) is the other open case and is unrelated.

### Per-case triage round 9 — the `#is_padding` restore was shallow (2026-07-27)

**✅ FIXED — `github_4745_pep604_class_attr`.** Round 8 placed the second pad
"downstream of `python_adjust`"; it is in fact the *same* arm, one level of
recursion down. `add_padding` pads **component types before the enclosing one**
(`padding.cpp:71`), so one top-level call walks the whole aggregate tree. The
`#is_padding` re-derivation the arm performs beforehand only flagged the
**top-level** components, so when `add_padding(Box)` descended into the
already-padded `int | None` attribute, that struct's pad member looked like an
ordinary 7-byte field and was aligned as one — `1 → +6 → 7` for the inserted
`_ExtInt(48)`, then `value` at `14 → +2 → 16` for the `short`. Fixed by making
the restore recursive over struct/union components and array subtypes.

This also explains round 8's confusing probe reading. The arm was observed
receiving `{is_none, value}` (unpadded) five times and correctly emitting
`{is_none, anon_pad$1, value}` each time — all true, and all irrelevant: the
damage was done by the **`Box`** firing, whose own components (`x`, `flag`) carry
no pad name, so the shallow loop had nothing to flag and the nested re-pad was
invisible at that print site. **A probe that prints only the node it is attached
to cannot see a defect in what that node's callee recurses into.**

The hop-off `tag-Optional_signedbv` now matches legacy byte-for-byte. Control run
(patch stashed, rebuilt): the two divergences a 70-test strided census reports —
`div6_fail` (`mk_not` on a non-Boolean, the round-8 defect) and
`github_3034_split-dot-valid-zero_fail` (`Assertion failed:
(is_signedbv_type(lt.side_1) && is_signedbv_type(lt.side_2))`,
`smt_solver.cpp:1512` — a `lessthan2t` over mismatched operand kinds) — reproduce
identically **without** this patch, so neither is caused by it. `div6_fail` is
closed by round 8, which confirms that arm reaches past `none3`;
`github_3034_split-dot-valid-zero_fail` is a new, distinct signature and is the
next case.

### Per-case triage round 10 — the call-return seam; `github_3034` is S4 (2026-07-27)

`github_3034_split-dot-valid-zero_fail` aborts under hop-off at
`Assertion failed: (is_signedbv_type(lt.side_1) && is_signedbv_type(lt.side_2))`
(`smt_solver.cpp:1512`) — a `lessthan2t` whose operands are a `signed long`
variable and an `unsigned long` value. The GOTO diff shows two distinct sources
for that mismatch, and only one of them is separable.

**✅ Landed — the call-return seam.** `length = len(xs)` binds the list model's
`unsigned long` return to a `signed long` variable. Legacy emits a temporary of
the callee's return type and converts:

```
legacy:  DECL unsigned long int return_value$___ESBMC_list_size$1;
         FUNCTION_CALL: return_value$___ESBMC_list_size$1=list_size(iterable)
         ASSIGN length=(signed long int)return_value$___ESBMC_list_size$1;
hop-off: FUNCTION_CALL: length=list_size(iterable)
```

The single instruction is not an optimisation — `convert_assign`'s
call-valued-rhs special case hands the lhs straight to `do_function_call`, which
emits no temporary and no cast, so the signed variable simply holds an unsigned
value. On the legacy path that case is never taken, because `adjust_assign` has
already wrapped the rhs in a typecast. Mirrored here **only for a
`sideeffect2t(function_call)` source**.

**Why this fragment of the parked assignment conversion is safe.** The trap
documented above is that `adjust_assign` runs *after* `adjust_operands`, so
converting at the assignment seam without operand-level arithmetic reconciliation
changes the stored value. That coupling is about reconciling a **binary
operation's** operands on the right-hand side — which a call source does not
have. The general arm stays parked.

**The S4 canary no longer works, and this must be fixed before S4 is attempted.**
The record pins the danger on `neural-net_fail` reporting SUCCESSFUL where legacy
reports FAILED. It no longer reports anything under hop-off: it aborts in
`assert_arith_2ops_consistency` (`irep2_expr.cpp:678`) before any verdict, and a
control run (this arm stashed and rebuilt) reproduces that abort identically, so
the abort is pre-existing and unrelated. Whoever picks up S4 must re-establish a
canary that actually produces a hop-off verdict first.

**`github_3034` is not closed by this and stays open on S4.** With the call-return
arm in, the residual hop-off diffs on that test are all the parked shape or its
siblings: `i = (signed long)(list_size(...) - 1)` (an *arithmetic* rhs — the
coupled case), `element = (_Bool)tmp$5`, `get_object_size((void *)bytes_data)` and
`validate_no_empty_parts(&price[0])` (argument conversions, S5), and
`(signed int)contains_tmp163 == 1` (relational promotion).

**Effect measured.** No verdict moves in a 70-test strided census — like the
`&array` decay (#6395) this closes a *structural* parity gap, and is recorded as
such rather than as a divergence fix. A control build confirms the shape: without
the arm a plain `n = len(xs)` emits `FUNCTION_CALL: n=list_size(xs)` into a
`signed long`; with it, the temporary and cast match legacy exactly. The census on
this branch is 69/70, the one divergence being `github_3034` itself (`div6_fail`
was closed by round 8).

### Per-case triage round 11 — a pointer callee that is not a symbol (2026-07-27)

**✅ FIXED — `github_3690`** (legacy SUCCESSFUL → hop-off FAILED, the counterexample
showing `result = 1.0000000000000002` where `1.0` was asserted). The GOTO diff
puts it in the call itself, not the value:

```
legacy:  FUNCTION_CALL: return_value$=*(*(void (*)() *)…dict_val_obj->value)()
hop-off: FUNCTION_CALL: return_value$= *(void (*)() *)…dict_val_obj->value ()
```

`clang_c_adjust::adjust_side_effect_function_call` wraps **any** pointer-typed
callee in an implicit dereference. `python_adjust`'s `wrap_function_pointer_callee`
only handled a callee that is a plain `symbol2t` whose *table* type is
pointer-to-code — the lambda-alias shape (`op = lambda …; op(3)`) it was written
for. `{'+': lambda: 1.0}[x]()` reads the lambda back out of a container, so the
callee is a **typecast of a member read**, the wrapper returned false, and
goto-convert called through the pointer value itself. The result was then read
under the wrong signature, which is why the failure surfaced as a corrupted
double rather than an outright crash. Generalised: when the callee is not a
table pointer-to-code symbol but its own type is pointer-to-code, dereference it.
Argument casting stays on the symbol path, which needs the table entry.
(Superseded by round 12: casting moved to the call site for expression-form
calls, and this function now delegates to the shared helper.)

**A rejected first hypothesis, recorded so it is not re-tried.** The same diff
also shows `ASSIGN …list_elem$175=(double (*)())(&lam1)` against a bare `&lam1`,
i.e. a missing conversion where the lambda's address is *stored*. That looks like
the more obvious cause and it is not the cause: an arm converting an
`address_of` of a code object at the assignment seam **never fired** on this test
(the shape reaching `code_assign2t` is not that), and the case still failed. The
arm was removed rather than kept as dead instrumentation. Fix the call, not the
store.

Census after this round: **69/70** on the strided sample, the single divergence
being `github_3034_split-dot-valid-zero_fail`, which is the parked S4 work
(round 10).

### Per-case triage round 12 — the S5 argument conversion, expression-form only (2026-07-28)

**Reopens the round-10 "negative result" and resolves it.** Mirroring
`clang_c_adjust::adjust_function_call_arguments` (clang_c_adjust_expr.cpp:1069)
generically was recorded as a net parity loss: it closed
`get_object_size(bytes_data)` but added a `(void *)` cast to ~10 list-model
calls legacy leaves alone. The inference drawn then — that the real defect was
the operand's type upstream, not the missing cast — was wrong. The actual
discriminator is the **call form**:

* legacy reaches that loop only via `adjust_side_effect_function_call`, i.e.
  **expression-form** calls;
* its statement-form arm (clang_c_adjust_code.cpp:38-53,
  `statement == "function_call"`) adjusts index expressions and *nothing else*,
  so a statement-form call keeps its arguments verbatim.

`list_push`/`list_contains`/`list_set_at`/`list_find_index` are statement-form;
`get_object_size` and `list_size` feed expressions. Restricting the mirror to
the `sideeffect2t` arm therefore keeps every win and opens none of the losses.
Two shapes close corpus-wide:

```
legacy:  get_object_size((void *)bytes_data)          hop-off was: (bytes_data)
legacy:  list_size(( struct __ESBMC_PyListObj *)d.keys)   hop-off was: (d.keys)
```

**The census caught a defect four spot-checks missed — record this.** The first
wrap regressed exactly one test, `bytes5_fail`:

```
legacy:  ESBMC_range_has_next_(0, 3, 1)
hop-off: ESBMC_range_has_next_(0, (signed long int)3, 1)
```

`c_typecastt::do_typecast` folds a cast over a constant through the simplifier
in its **exprt** overload (c_typecast.cpp:911-922) but not in its **expr2tc**
overload (:926-948), which only wraps. `python_adjust` works in IREP2, so
mirroring the wrap alone leaves a visible cast on every constant argument.
Guarding on `is_constant_expr` and calling `simplify` after the wrap restores
parity. Generalises: **any arm mirroring a `gen_typecast` into IREP2 inherits
the missing constant fold.**

**Method note — normalise the instruction counter.** A raw `--goto-functions-only`
byte-diff is unusable here: one dropped docstring `OTHER` statement renumbers
every following instruction, inflating `github_3034` to 6462 diff lines against
a true 114. Strip `// <N> ` before diffing.

**Census (stride-6, 731 tests, GOTO byte-diff, pinned binaries).** This is a far
stricter metric than the verdict parity quoted in earlier rounds — 715 of 731
tests differ somewhere at GOTO level (docstrings, `__ESBMC_HIDE` labels) without
any verdict changing, so the signal is the delta, not the absolute.

| | diff lines | improved | unchanged | regressed |
|---|---|---|---|---|
| pre-patch baseline | 215794 | — | — | — |
| wrap only | 214292 | 711 | 19 | 1 (`bytes5_fail`) |
| wrap + constant fold | **214286** | **712** | **19** | **0** |

Round 11's closing note that "argument casting stays on the symbol path, which
needs the table entry" is superseded: casting now happens at the call site for
every expression-form call, and `wrap_function_pointer_callee` delegates to the
same helper instead of carrying its own copy of the loop.

### Per-case triage round 15 — the catch id was "struct" (2026-07-28)

**✅ FIXED — `github_6258`, the `bases`-carriage blocker.** And the blocker was
misfiled: it is a **converter** defect, not an adjuster gap, and nothing about
`bases` or `set_type` carriage is involved.

Under hop-off the entire handler disappeared:

```
legacy:   1: IF c:@__ESBMC_exc_typeid == 12 THEN GOTO 2
             2: ... ASSIGN caught=1; ASSERT caught
hop-off:  1: GOTO 2
          2: END_FUNCTION
```

`emit_catch_block` (`python_exception_handler.cpp`) already pre-derives each
handler's `exception_id` — that was flip blocker #2, closed earlier. Its
derivation covers a by-name `symbol` type and the `ellipsis` catch-all, then
falls back to `ct.id().as_string()`. For `except KeyboardInterrupt:` the catch
block's type is an **already-resolved struct**, so the fallback stored the
literal string `"struct"`, which matches no throw id and makes
`remove_exceptions` drop the handler.

**The pre-set's own comment was wrong, and that is why this survived.** It claims
`clang_cpp_adjust::adjust_catch` "overwrites the attribute with the identical
value". True for the symbol and ellipsis shapes; false for a resolved struct,
where legacy's `convert_exception_id` resolves the tag back to the class name and
the converter says `"struct"`. Legacy therefore *masked* the defect completely —
the pre-set is only load-bearing on the hop-off path, which is exactly where it
was never exercised. Fix: mirror `convert_exception_id`'s struct arm and recover
the class name from the type's `name`/`tag`.

**Probe, not inference.** `PROBE_EXCID id=struct name= tag=KeyboardInterrupt ->
struct` names the defect exactly; the enclosing GOTO is byte-identical to legacy
after the fix.

**Scope, measured against legacy rather than assumed.** A control build shows
`KeyboardInterrupt`, `SystemExit`, `OverflowError` and `AssertionError` all
failing under hop-off, but only `KeyboardInterrupt` is a *parity* defect: the
other three fail under **legacy too** and are pre-existing frontend limitations
(`raise SystemExit(...)` is not catchable on either path) — a separate issue, not
this one. `ValueError`/`KeyError`/`IndexError`/`ZeroDivisionError`/`TypeError`/
`RuntimeError`/`StopIteration` were never affected, because their catch type
arrives as a by-name symbol and takes the first branch.

Inert on the default pipeline: `adjust_catch` overwrites the attribute
unconditionally there, so only the hop-off reads what this writes.

Tests: `regression/python/python_irep2_adjust_only_catch_builtin{,_fail}`.

### Per-case triage round 16 — S3 has no reproducer; the last divergence is a vacuous test (2026-07-28)

**Verdict: the third named flip blocker, "S3 member/index at scale", is not
evidenced by anything in the corpus.** Its sole census witness,
`string-nondet-index-fail`, is a **vacuous test**, and valid member/index code is
already at hop-off parity.

The test aborts under `--python-irep2-adjust-only` in `index2t`'s construction
assert (`irep2_expr.h:1650`). A probe on `migrate_expr`'s index arm names the
cause exactly:

```
PROBE_INDEX_SRC legacy_id=empty migrated_type=empty
```

The index **source has type `empty`**. `nondet_string` is not a supported builtin
— every test that calls it gets `Undefined function 'nondet_string' - replacing
with assert(false)` — so `s` is typed void and `s[0]` indexes a void value. The
irep2 invariant is right to reject it; nothing here argues for relaxing the
assert.

**Why the test still "passes" on the default path.** Its whole body lowers to a
single instruction:

```
python_user_main:
        ASSERT 0 // Unsupported function 'nondet_string' is reached
        END_FUNCTION
```

The assertion it purports to check (`c == "h"`) never reaches the GOTO at all.
Its expected `^VERIFICATION FAILED$` is satisfied by the unsupported-function
assert, not by the property — the same "asserts nothing" failure mode as the
vacuous-`test.desc` batch (#6453/#6454), reached by a different route. Its
`-success` sibling, `string-nondet-index-success`, is marked **FUTURE**, which is
the honest label for the feature; the `-fail` twin is marked CORE and passes for
the wrong reason. The same holds for `string-nondet-{slice,concat,in}-fail` and
`string-char-symbolic-fail`.

**Valid indexing is at parity.** `def first(s: str) -> str: return s[0]` and
`def pick(s: str, i: int) -> str: return s[i]` both verify SUCCESSFUL under
legacy *and* hop-off. Under `--python-irep2-adjust` (the pass running *after*
`clang_cpp_adjust`) even the degenerate test reports FAILED normally, confirming
the abort needs both the void source and the absence of the legacy pass.

**Consequence for the flip.** With #6462 (relational signedness), #6466 (array
argument decay) and #6468 (resolved-struct catch id) landed, the 220-test strided
census has **no remaining verified divergence**. The three blockers this document
listed are resolved or refuted:

| blocker | outcome |
|---|---|
| S5 arg casts | closed — #6461 (scalar, expression-form) + #6466 (array decay) |
| `bases` carriage | misfiled; the real defect was the converter's catch-id fallback — #6468 |
| S3 member/index at scale | **no reproducer**; sole witness is a vacuous test |

The next honest step is therefore not another per-case round but a **denser
census** (the stride-20 sample is exhausted), plus deciding what to do about
`nondet_string`: either implement it — which would make the six CORE `-fail`
tests above assert something — or re-mark those tests, which currently pass
regardless of the behaviour they name.
### Per-case triage round 14 — array decay at the call-argument seam (2026-07-28)

**A strided whole-corpus census is now the frontier finder.** With round 11's open
list drained, a 220-test strided sample (every 20th `regression/python` directory)
was run legacy-vs-hop-off. It returned **exactly three** divergences, and each is
a different one of the three flip blockers this document's status line already
names — the remaining gap is not a long tail:

| test | hop-off symptom | blocker |
|---|---|---|
| `github_2839` | `argument "a" type mismatch: got array, expected pointer` → no verdict | **S5 arg casts** |
| `github_6258` | `uncaught exception: KeyboardInterrupt` → false FAILED | **`bases` carriage** |
| `string-nondet-index-fail` | `index2t` construction assert (`irep2_expr.h:1650`) → no verdict | **S3 member/index at scale** |

**✅ FIXED — `github_2839` (S5, array half).** `is_foo(a="foo")` passes a `char[4]`
literal into a `char *` parameter:

```
legacy:  FUNCTION_CALL: e=is_foo(&{ 102, 111, 111, 0 }[0])
hop-off: FUNCTION_CALL: e=is_foo({ 102, 111, 111, 0 })
```

`clang_c_adjust::adjust_function_call_arguments` converts every argument to its
declared parameter type, and `c_typecastt`'s array case **decays** rather than
casts. `python_adjust` had no argument conversion for a *direct* call at all:
`wrap_function_pointer_callee` does cast arguments, but only on the
pointer-to-code callee path, and its `is_castable_kind` list excludes arrays. New
`decay_array_arguments`, on the **expression-form** arm only — the same
restriction round 12 (#6461) established for argument casting, reached here by
probe rather than by inheriting the argument. A statement-form wiring was
written first and removed: **0 firings across 70 tests**, against 1 for
expression-form. `e = is_foo(...)` is a `code_assign2t` over a `sideeffect2t`
at adjust time and only becomes a statement-form `FUNCTION_CALL` later in
goto-convert, so the statement-form arm could never see it.

**The `expr2tc` constant-fold gap round 12 flagged does not reach the relational
arm.** #6461 warns that any arm mirroring a `gen_typecast` into IREP2 inherits
`do_typecast`'s missing constant fold. Probed on round 13's relational arm with
`len(s) < 5`: legacy itself emits `(unsigned long int)5` unfolded and hop-off is
byte-identical. The fold lives on the `implicit_typecast` → `do_typecast`
argument path, not in `implicit_typecast_arithmetic`. No fix needed there.

Scoped to the array→pointer shape only, which is **structural** — the same object,
addressed differently. The scalar width/signedness half of S5 changes stored
values and stays out, the same line round 13 drew against the parked S4
assignment trap.

**The C-Live control build caught a bad regression test — record this.** The first
test pair put the keyword call inside `def main()`, the house idiom. It **passed on
the control build**, i.e. proved nothing. Probing variants against the
already-built control binary isolated a much narrower trigger:

| shape | reproduces? |
|---|---|
| `e = is_foo(a="foo")` at module level | **yes** |
| `assert is_foo(a="foo")` at module level | no |
| `assert is_foo(a="foo")` inside a function | no |
| `e = is_foo("foo")` (positional) at module level | no |

Module scope **and** assignment **and** a keyword argument. When a repro is this
narrow, derive the test from the issue's actual source shape; restyling it into
the house idiom is what drops the trigger. Both tests now abort on the control
and pass patched.

**Census after this round: 218/221.** `github_2839` closed; `github_6258` and
`string-nondet-index-fail` remain (their own blockers). `github_3701_11` appears
as a third divergence **only because this branch is cut from master and therefore
lacks round 13** — it aborts at `smt_solver.cpp:1512`, round 13's exact signature,
and is at parity on the round-13 binary. Useful independent evidence that round
13 closes more than its own repro.

Tests: `regression/python/python_irep2_adjust_only_arg_decay{,_fail}`.
### Per-case triage round 13 — the relational signedness abort (2026-07-28)

**Re-measure first, as always.** Round 11's two open items shrank to one before any
triage: `github_3690` is **closed** (legacy and hop-off both SUCCESSFUL, and its
`_fail` sibling both FAILED) — #6445 did what it claimed and the entry was stale.

**✅ FIXED — `github_3034_split-dot-valid-zero` and its `_fail` sibling.** These are
the tests round 10 parked on S4, and **the recorded diagnosis was wrong**. The
residual is not the list of promotions round 10 enumerated; hop-off does not reach
a verdict at all. It aborts in the solver:

```
Assertion failed: (is_signedbv_type(lt.side_1) && is_signedbv_type(lt.side_2)),
  function convert_ast_node, file smt_solver.cpp, line 1512.
```

`smt_convt::convert_ast_node`'s `lessthan` case dispatches on *both* sides being
floatbv, *both* fixedbv, *both* unsignedbv, or — in the final `else` — *both*
signedbv. An ordering relation whose operands disagree in signedness matches no
arm and trips the assert. `while i < len(parts)` produces exactly that: the loop
variable is a Python int (`signed long`) and the list model's `list_size` returns
`unsigned long`. The whole GOTO diff for the enclosing function was **one cast**:

```
legacy:  IF !((unsigned long int)i < return_value$___ESBMC_list_size$1) THEN GOTO 3
hop-off: IF !(i < return_value$___ESBMC_list_size$1) THEN GOTO 3
```

`clang_c_adjust::adjust_expr_rel` reconciles the two operands with
`gen_typecast_arithmetic`; `python_adjust` had no relational arm at all, so the
node reached the SMT layer unreconciled.

**Why this is not the rejected "gap-2" arm.** Running `gen_typecast_arithmetic` on
*every* relational node was tried in round 3 and rejected: it diverges corpus-wide
from clang's promotions over the OM bodies (~7500 diff lines on `builtin2` alone),
for a purely cosmetic gain. This arm is gated on the **signedness mismatch** — the
one shape that is not encodable at all. A same-signedness width promotion
(`char` vs `int`, gap-2's target) is encodable and stays untouched, so none of
that traffic is re-admitted. The arm also calls
`c_implicit_typecast_arithmetic`'s **`expr2tc` overload**, so there is no migrate
round-trip and gap-2's first negative result — an operand picking up a spurious
wrap because its migrated type does not compare equal to itself — cannot arise by
construction. It is the same helper `python_math`'s floor-div/modulo
reconciliation uses (#5725).

**Relation to the parked S4 trap.** S4's danger is mirroring `adjust_assign`
*without* operand-level arithmetic reconciliation, which makes `neural-net_fail`
report SUCCESSFUL where legacy correctly FAILS. This arm is the reconciliation
half, and only for relationals: it changes no stored value, it makes an otherwise
unencodable node encodable, and its output is byte-identical to what legacy
already emits. The masking direction is the assignment half, which stays parked.
The S4 canary is still broken (round 10) and must still be re-established before
the assignment half is attempted.

**C-Live discharged by probe, not argument.** Both new tests *abort* on a control
build with the arm reverted and rebuilt, and pass with it — the arm is live, not
dead instrumentation. Idempotent: after one application both operands share a
signedness, so the gate cannot re-fire.

Tests: `regression/python/python_irep2_adjust_only_rel_signedness{,_fail}`.
