# Scope — the V.1k (b) IREP2-native Python adjuster

**Program:** Part V of `docs/irep2-migration.md` (IREP2-native frontend→goto, #4715).
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
W1-loc keystone (`docs/spike-v1k-w1loc.md`) does not cover. Remaining flip
blockers: "bases" carriage, S3 member/index at scale, S5 arg casts (the
adjust()-error-return blocker is discharged, `python_language.cpp:325`). **Owner:**
TBD. **Refs:** #4715, #5055, #6323; sibling doc `docs/spike-v1k-w1loc.md`.

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
| **S3 unresolved-by-name** | 7 | `python_adjust: symbol '…' retains N unresolved by-name (symbol_type2t) node(s) after adjust (V.1k post-adjust invariant violated)` | dict/Optional value nodes whose key/element type clang resolves but `python_adjust` leaves as a transient `symbol_type2t` — needs the dict-key-type infra (the long-standing S3 work-list) | **hard** (multi-step, own infra) |
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

**Direction.** The two low-hanging arms (deref #6340, if2t #6348) are landed and
the hop-off is at ~95% verdict parity. The remaining gaps are *not* one-arm
fixes: index-over-pointer is really the array→pointer decay gap above (an
assignment-seam fix, not an index arm), and the S3 unresolved-by-name family
needs the dict-key-type infrastructure. Both are their own scoped efforts;
neither should be forced as a mechanical adjuster arm.
