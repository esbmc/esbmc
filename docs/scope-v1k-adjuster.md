# Scope — the V.1k (b) IREP2-native Python adjuster

**Program:** Part V of `docs/irep2-migration.md` (IREP2-native frontend→goto, #4715).
**Question this scopes:** is the whole-body "resolve-then-build" adjuster the right
next step to close the V.3 residue, and if so, what exactly does it own?
**Status:** Spike-1 executed 2026-07-03 — **adjuster ruled unnecessary** (§5.1);
converter-construction 100% is a small inline task. **Owner:** TBD. **Refs:**
#4715, #5055; sibling doc `docs/spike-v1k-w1loc.md` (the *other*, larger
keystone — see §6).

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
