# Roadmap — all frontends → IREP2-native construction

> **Status: forward plan, opened 2026-08-03. This document reverses standing
> decisions by owner direction.**
> Part I's B1 ("frontends stay legacy"), Part III §14's Solidity close-out, and
> Part V's V.1a/V.2/V.6 closures all concluded that migrating frontend
> *construction* to IREP2 was not worth its cost. The project goal is now full
> frontend migration, so those conclusions are reopened as **decisions, not
> discoveries** — the arguments behind them were sound and are answered here
> (§5), not ignored.
>
> Parent record: `irep2-migration.md`. Sibling scopes:
> `scope-coupled-arith-assign-conversion.md`, `scope-v2-w3-attribute-carriage.md`,
> `scope-v1k-adjuster.md`.

## 1. The goal, as a measurable bar

Per frontend `F` in {clang-c, clang-cpp, python, solidity, jimple}:

| # | Bar | Command |
|---|---|---|
| B-1 | Legacy type mentions → ~0, modulo enumerated boundary glue | `git grep -P '\b([A-Za-z_]*(exprt\|typet\|codet)\|irept)\b' -- src/F-frontend` |
| B-2 | Symbol-table writes carry IREP2 values only | `grep -rn 'set_type(\|set_value(' src/F-frontend \| grep -vc 2tc` → 0 |
| B-3 | Bodies reach `goto_convert` with no `migrate_*` back-hop | native dispatcher coverage = 100 %, round-trip deleted |
| B-4 | No `#`-attribute escape hatch into a shared pass | W3 removed, not merely seamed |

B-1/B-2 are frontend-local. **B-3 and B-4 are shared** — they are one repo-wide
job each, not five. That asymmetry is the whole shape of this program: do the
two shared jobs once, then the five frontends become largely mechanical.

## 2. Baseline census (measured 2026-08-03, `f14cd73ff8`)

| frontend | legacy mentions | IREP2 (`*2tc`) | LOC | construction state |
|---|---:|---:|---:|---|
| python | 5 547 | 806 | 79 282 | partially native (Part V) |
| clang-c | 971 | 49 | 13 783 | legacy + isolated IREP2 |
| clang-cpp | 626 | **0** | 7 394 | fully legacy |
| solidity | 1 420 | **0** | 23 589 | fully legacy |
| jimple | 176 | **0** | 3 259 | fully legacy |

The three zeros are real: those frontends construct no IREP2 node at all and
rely entirely on `migrate_expr`/`migrate_type` at the symbol-table seam. Note
jimple's 176 is small because the frontend is small, not because it is nearly
done — normalise by LOC before reading these as effort.

## 3. The W1 wall is stale — this is the finding that makes the program tractable

`irep2-migration.md` §V.1's wall table states:

> **W1 (P1)** — IREP2 has the flat goto-level code kinds … but **no structured
> CF kinds** (`ifthenelse`/`while`/`for`/`switch`/`break`/`continue`/`label`).

**That is no longer true and has not been since #5265** ("V.4.0: add structured
control-flow code kinds"). `src/irep2/expr_kinds.inc:129-139` defines
`code_ifthenelse2t`, `code_while2t`, `code_dowhile2t`, `code_for2t`,
`code_switch2t`, `code_break2t`, `code_continue2t`, `code_label2t`,
`code_switch_case2t`, `code_assert2t`, `code_assume2t`. They have forward and
back `migrate` arms (`src/util/irep/migrate.cpp`), and `goto_convert` consumes
them natively via `convert_native_rec`
(`goto-programs/goto_convert_functions.cpp:254`), with a **per-function**
fallback to the legacy round-trip for unsupported shapes — currently ≈78.7 % of
functions native on the `esbmc-cpp` census.

The consequence is large. Every prior "frontends stay legacy" argument rested
partly on W1: a frontend *could not* build a body natively because the
representation for `if`/`while`/`switch` did not exist. It exists. What remains
of W1 is not a representation gap but **dispatcher coverage** — a finite,
enumerable list of shapes `convert_native_rec` returns `false` on. That is
ordinary work with a measurable completion criterion, not an architectural wall.

**Action:** §V.1's W1 row must be corrected in the parent document (§9 below).
Any planning that still treats W1 as "IREP2 cannot represent structured CF" is
working from a stale premise.

## 4. The four walls, re-grounded at `f14cd73ff8`

| Wall | Original statement | State now | Remaining work |
|---|---|---|---|
| W1 | no structured CF kinds in IREP2 | **dissolved as stated** (§3) | drive `convert_native_rec` to 100 %, then delete the round-trip |
| W2 | `member2t`/`index2t` assert a resolved source | **dissolved as a construction blocker**, with a standing obligation | the relaxation is repo-wide, but it is conditional — see below |
| W3 | `#cpp_type`/`#member_name`/`#cformat` read off legacy nodes by shared passes | **seamed, not removed** (Option D, #6569/#6570) | carriage — §5, the one genuine design problem left |
| W4 | counterexample printer consumes the attributes | untouched, deferred | falls out of W3 if W3 is solved by a typed field |

Two of four walls are gone, one is reduced to a coverage exercise, and one is a
real design problem. That is a materially better starting position than the
Part V close-out implies.

**W2's obligation, stated because it is a per-frontend cost.** The relaxation
(`irep2_expr.h:1549-1570`) permits a `symbol_id` source **only as a transient
pre-resolution state**, and the comment is explicit that "the adjuster MUST
follow it to a struct before symex" — the strong invariant is re-enforced
post-adjust, not dropped. So the *representation* is repo-wide, but the
*obligation* is not: every frontend that builds `member2t`/`index2t` before type
resolution needs a resolve pass of its own, or must reuse one. Python's is
`python_adjust`. Budget one per frontend in Phases 5-9, and note that the assert
is `#ifndef NDEBUG` — an unresolved source will pass a `RelWithDebInfo` build
silently and fail in symex, which is gate G4's reason for existing.

## 5. The one real wall — value-carried metadata (W3 and Solidity's Q-S1)

### 5.1 Why the recorded closures are right about the mechanism

Part III §14 (Q-S1) and `scope-v2-w3-attribute-carriage.md` §3 independently
established the same fact, and it is not in dispute: **these attributes are read
off transient type values with no symbol in scope.** `cpp_expr2string.cpp:138`
reads the type currently being printed, reached recursively through
`convert(src.subtype())`. `solidity_convert_decl.cpp` reads `get_sol_type(t)`
off a local `typet t` before any symbol exists. A side table keyed by symbol id
— §V.2's prescribed design — cannot serve either. That finding stands.

The options considered, and their recorded verdicts:

| Option | Carries with the value? | Recorded verdict |
|---|---|---|
| A — symbol-keyed side table | no | not viable (Q-S1) |
| B — value-bundled wrapper over `type2tc` | yes | viable, rejected on cost; recreates attribute flexibility |
| C — extend `type2t` with a spelling field | yes | rejected: "pushes presentation concerns into the verifier IR" |
| D — encapsulate writers, leave carriage legacy | n/a | **taken**; does not move B-4 |

### 5.2 Option F — a closed typed field, which is not Option C

Both rejections of B and C rest on the same objection: *do not reinstate the
open, string-keyed attribute flexibility IREP2 abolished.* That objection is
correct against a generic map. It does **not** apply to a closed enum, and two
measurements say a closed enum suffices:

- **Solidity's classification is already a closed `enum class`** —
  `SolType` at `solidity-frontend/solidity_grammar.h:484`. It is stringified
  only to cross the `irept` boundary. Restoring it to a typed field removes a
  serialization step rather than adding an escape hatch.
- **`#cpp_type`'s value domain is the C type-keyword set** — the writers emit
  `"bool"`, `"signed_char"`, `"unsigned_char"`, `"void"` and a `c_type` variable
  drawn from the same finite vocabulary. It is an enum wearing a string.

So the third option the record never separated out:

> **Option F.** Add a **closed, typed, optional** classification field to the
> specific `type2t` kinds that need it — not a generic attribute map, not a
> wrapper struct. `enum class c_spelling` on the integer/float kinds;
> `enum class sol_class` on the kinds Solidity tags. Absent by default, ignored
> by every solver backend, exhaustively switchable.

**Why this is legitimate where C was not.** C was framed as pushing
*presentation* concerns into the verifier IR, and for `cpp_expr2string` and
`goto2c/expr2c` that framing is right. But it is incomplete:
`clang_cpp_adjust_expr.cpp:582` uses `#cpp_type` to **build exception type ids
for catch matching**. That is semantics, not presentation — a C++ program's
observable behaviour depends on it. Metadata that catch-matching depends on
belongs in the typed IR by the migration's own governing rule that verification
correctness outranks implementation convenience. The presentation readers then
ride along for free.

**The honest price**, which must not be glossed: IREP2's type system stops being
purely structural. Two types identical in width and signedness may now differ in
a spelling field. Every equality, hashing and canonicalisation path over
`type2t` must decide whether the field participates — and the answer is almost
certainly **no** (it must not, or `long` and `long long` stop unifying in the
solver and verdicts change). That asymmetry is a sharp edge and needs an
explicit invariant plus a test that pins it.

**Option F is a spike before it is a plan** (Phase 0 below). If the equality
asymmetry proves unmanageable, fall back to Option B for Solidity only and
accept that B-4 closes for C/C++ but not Solidity.

## 6. Phased program

Ordered by dependency. Phases 1 and 2 are shared and unlock everything else;
5-9 are per-frontend and parallelisable once 1 and 2 land.

### Phase 0 — Option F spike (gate for the whole B-4 half)
Prototype `enum class c_spelling` on `signedbv_type2t`/`unsignedbv_type2t`.
Answer three questions and stop: does the field participate in `type2t`
equality/hash (expected: no)? Does exception catch-matching still work off the
typed field? Does the whole `esbmc-cpp` suite hold verdict **and
counterexample-text** parity? Deliverable: a go/no-go with measurements.
*Accept:* a recorded answer either way. A no-go re-routes §5.2 to Option B, it
does not stall the program — Phases 1, 3-9 are independent of B-4.

### Phase 1 — drive `convert_native_rec` to 100 % (closes W1, shared)
Instrument the dispatcher to log every `(kind, shape)` it declines, run the full
corpus, and work the resulting histogram down. This is the highest-value phase:
it is mechanical, measurable, benefits all five frontends at once, and its
completion criterion is a number reaching zero.
*Accept:* 0 fallbacks corpus-wide; then delete the round-trip and the fallback
path in the same PR that proves it unreachable.

### Phase 2 — W3 carriage removal (closes B-4, shared)
Only if Phase 0 says go. Land `c_spelling`/`sol_class` as typed fields, repoint
the four readers, delete the `irept` accessors. Option D already gave one
repoint-point per attribute, so this is now a small diff at a single seam — the
work Option D was explicitly designed to make cheap.

### Phase 3 — finish the Python flip
Phases 2-3 of `scope-coupled-arith-assign-conversion.md`, plus the two ownerless
mechanisms that block it (§9.4's second mechanism; the array-typecast class of
§14). Each needs its own scope doc. Python is the pathfinder: it is the only
frontend with construction experience, so its remaining defects are the ones the
other four will hit.

### Phase 4 — extract the reusable construction kit
Before touching a second frontend, factor what Python learned into shared
helpers: the width-reconciliation idiom (`c_implicit_typecast_arithmetic` on
`expr2tc`), the resolved-source `ns.follow` pattern, the operand-surgery recipe.
Without this, four frontends re-derive the same lessons at four times the cost.

### Phases 5-9 — per-frontend migration, in this order
**5. jimple** (176 mentions, 3 259 LOC) — smallest surface, no operational-model
complication, lowest blast radius. The pathfinder for the kit.
**6. clang-c** (971, already 49 IREP2) — has a partial head start.
**7. clang-cpp** (626) — small but highest semantic density; owns catch-matching
and `clang_cpp_adjust`, which every other frontend's output passes through.
**8. solidity** (1 420) — gated on Phase 2's `sol_class` outcome. If Phase 0 said
no-go, this is where the program's scope is formally cut.
**9. python** (5 547) — largest, and deliberately last so it inherits every
lesson, exactly as §V.1 reframed it.

Each of 5-9 opens its own `scope-<frontend>-irep2.md` at start, following the
established scope-doc pattern: census, phased decomposition, gates, risks.

## 7. Gates (every phase)

| # | Gate |
|---|---|
| G1 | Verdict parity on the affected suites, dual-solver (Bitwuzla + Z3) |
| G2 | Counterexample **text** parity — asserted verbatim by `test.desc` regexes; W3/W4 readers make this load-bearing |
| G3 | `--goto-functions-only` A/B, normalised per the recorded harness rules (§8) |
| G4 | Asserts-on (`DebugOpt`) build — `assert_*_consistency` is `#ifndef NDEBUG` and a `RelWithDebInfo` build compiles it out; ill-formed IR passed every local gate once already |
| G5 | `esbmc-solidity` rides **Linux CI** — macOS has no `solc` and stubbed `sol64` models |

**Inherited harness rules — non-optional.** Every census on this track has been
invalidated at least once by harness artifacts. Reuse them verbatim:

1. Normalise the whole temp-path segment (`s@/esbmc[-._][^/ ]*@/TMPD@g`), not
   individual prefixes — four spellings exist; partial normalisation reports
   ~90 % false divergence.
2. Strip timing lines (`completed in:|time:|Runtime|Elapsed`) — 46 false
   divergences out of 106 from `0.000s` vs `0.001s` alone.
3. Exclude or serialize `--k-induction-parallel` tests — UNSTABLE against
   themselves.
4. Skip tests whose `test.desc` already passes the flag under test — boost
   throws `multiple_occurrences`.
5. Minimum-size guard (`< 200 bytes → SKIP`) — two collapsed error lines
   otherwise count as a match.
6. Sample dense and unbiased; stride-20 missed a 0.5 % defect rate entirely.
7. **When a divergence survives, run the baseline against itself before
   attributing it to the patch.** That control is what settled `cpp_sum_class`.
8. **Probe the invariant you depend on, not a proxy for it**, and prove the probe
   fires on known-bad input before trusting a zero. A probe measuring a narrower
   condition than its invariant reported "0 firings" for a violation that CI then
   caught.

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Option F's spelling field leaks into `type2t` equality and silently changes verdicts | Phase 0 answers this first; explicit invariant + a pinning test |
| R2 | Five frontends re-derive Python's lessons independently | Phase 4 exists solely to prevent this; do not start Phase 5 before it |
| R3 | Partial migration is tracked as progress | B-1..B-4 are all-or-nothing per frontend, as V.5's deferral argument established: migrating 10 of ~159 kinds removes **zero** back-hops |
| R4 | Solidity's metadata proves genuinely open, not enum-shaped | Cut scope to four frontends; record it, do not let it stall the rest |
| R5 | The program stalls mid-way, leaving frontends in a worse hybrid state than pure legacy | Each phase is independently valuable and independently revertible; Phase 1 alone is a net win even if nothing else lands |

## 9. What this reverses, and required parent-document edits

Three recorded conclusions are superseded **by decision**:

- **B1** (Part I, "frontends stay legacy") — reopened.
- **Part III §14** ("the Solidity frontend stays legacy by design") — reopened,
  conditional on Phase 0.
- **Part V's V.1a/V.2/V.6 closures** — reopened; V.2's blocker is now Option F,
  not the refuted Option A.

And one is superseded **by fact**, independent of the goal change:

- **§V.1's W1 row is stale** (§3). It should be corrected regardless of whether
  this program is prosecuted, because it misstates the tree.

## 10. Honest sizing

No total estimate is given, deliberately — the Part V record shows what happens
when a multi-phase program is sized before its keystone spike returns. What can
be sized:

| item | estimate | confidence |
|---|---|---|
| Phase 0 spike | days | high — bounded, one question |
| Phase 1 (dispatcher to 100 %) | weeks | medium — histogram is measurable but its tail is unknown until instrumented |
| Phase 2 (W3 removal, post-Option D) | days | high — one seam, four readers |
| Phases 5-9 | unknown until Phase 4 | low — Phase 5 (jimple) is the calibration run |

**The recommended first action is Phase 1, not Phase 0.** It is independent of
the Option F question, benefits all five frontends whether or not the rest of
the program proceeds, and its output — the declined-shape histogram — is the
single most informative artefact available about how far frontend construction
actually is from native.

## 11. Phase 1 — what it measured (2026-08-03)

§10 called the declined-shape histogram "the single most informative artefact
available". It was produced; this section records it, the eight patches it
drove, and the two bounds the work discovered. §1-§10 are the forward plan,
unchanged.

### 11.1 The census, and the cascade correction

`convert_native_rec`'s `return false` sites were instrumented with a
`__LINE__`-tagged logger and the `esbmc-cpp/cpp` suite run under
`--goto-functions-only` (sound here: the dispatcher runs before symex, and it
takes each test off its solve timeout).

**The first ranking was wrong, and the correction is the reusable part.**
Ranking by raw count puts `code_block` first at 13 799 — but a block declines
only because a child did. Sites split into:

- **cascade** — `code_block`, `code_while`, `code_dowhile`, `code_switch`,
  `code_switch_case`, and 3 of 4 `code_for` sites. They clear for free when the
  genuine causes do, and fixing them directly is wasted work.
- **genuine** — everything else.

Classify before picking a target: a site is cascade if its guard tests the
result of a nested `convert_native_rec`. Ranking cascade sites by count sent
one iteration at `code_while` before the handler was read; `code_while` already
lowers side-effecting conditions natively and has no genuine cause of its own.

### 11.2 Result

| | declines (stride-4 `esbmc-cpp`) |
|---|---:|
| baseline | 28 243 |
| after the eight patches below | **1 324** |

**−95.3 %.** Of the 1 324 remaining, all but ~160 were cascade; the last genuine
leaf was the scope-leak check, which the final two patches target. All eight
branches merge into one another with **zero conflicts**.

| PR | site | share of its kind |
|---|---|---|
| #6668 | `code_return`, side-effect value | 100 % |
| #6671 | `code_ifthenelse`, side-effecting guard | 76 % |
| #6672 | `code_decl`, static/code-typed/array | 100 % |
| #6674 | `code_expression`, top-level ternary or nil | 100 % |
| #6677 | `code_function_call`, bodyless callee | 100 % |
| #6678 | `code_assign`, generic rhs guard | 100 % |
| #6679 | `code_ifthenelse`, leaked scope-exit state | the last genuine leaf |
| #6681 | `code_while`/`code_for`, body did not convert | cascade collapse |

Five of the eight censuses landed **100 % on a single guard**. Census before
patching: the distribution is far more concentrated than reading the handler
suggests.

### 11.3 The delegation pattern, and its two bounds

Not one fix reproduced a construct natively. Every one routed the decline to
the legacy converter that already owns that statement's lowering
(`convert_return`, `convert_ifthenelse`, `convert_decl`, `convert_expression`,
`convert_function_call`, `convert_assign`) — the route the try/catch handler
established. The soundness argument is uniform: **on the fallback path that
same function converted the statement anyway**, so the emitted instructions are
unchanged; what changes is that the *rest of the function* stays native.

The pattern has two bounds, both found by the A/B gate and neither visible to
the regression suite:

1. **Delegating after a partial native attempt drifts temp numbering.** The
   abandoned attempt has already allocated from the shared `tmp_symbol`
   counter, so the delegated conversion numbers its temps one past it — 20 of
   51 sampled tests diverged on `tmp$N`. Fix: snapshot `tmp_symbol.counter`,
   `context.mark()` and `targets` before the attempt and restore before
   delegating, exactly as `convert_function` does on a whole-function fallback.
2. **The snapshot must precede *everything* the handler lowers, not just the
   body.** A side-effecting loop condition allocates temps through
   `generate_conditional_branch`; a snapshot taken after it leaves those in
   place and 17 of 51 tests still diverged. Taken at the top of the handler,
   0 diverge.

Delegations added *before* any native attempt (#6668, #6672, #6674, #6677,
#6678) need neither.

### 11.4 Gates that earned their place

- **`--goto-functions-only` A/B against `--no-irep2-native-body`** — compares
  the native path with the pure-legacy path on the *same* binary, so it needs
  no control build. This is what caught both bounds above. Byte-identical GOTO
  is a stronger claim than verdict parity: no `test.desc` asserts temp names,
  so all three near-misses would have passed the whole suite.
- **Prove the A/B discriminates** before trusting a zero: under the census env
  var the native arm must log declines and the legacy arm none. A test with no
  declines proves nothing either way.
- **Probe that a regression test reaches the fixed site.** Three candidate
  tests fired the delegation zero times — clang decomposes `x = y++` and
  `p = new int(9)` before goto-convert, and wraps a single-statement branch in
  an implicit block. The shapes that reach these sites arrive through the
  container operational models, so the tests that pin them use `std::vector`.
  A passing test is not evidence it exercises the patch.

### 11.5 What remains

- The `esbmc-cpp` corpus is drained to the assert-fold sites (~4 declines).
  **Phase 1's "0 fallbacks corpus-wide" is not yet claimable**: only
  `esbmc-cpp` has been censused. The C, Python, Solidity and Jimple suites
  exercise different frontends and may reach sites this corpus never did.
- Only then does deleting the round-trip and the fallback path become the
  measurable next step §"Phase 1" describes.

## 16. Option F spike — Phase 0 answered from the tree (2026-08-04)

§6 Phase 0 gates the whole B-4 half of this program on prototyping Option F and
answering three questions. §5.2 called the equality asymmetry "a sharp edge" and
the phase's first question. **Two of the three are answered by reading the tree,
and the answer is favourable enough that the prototype is smaller than sized.**

### 16.1 Does the field participate in equality and hashing?

**No — provided it is omitted from the kind's `fields` tuple, and there is a
compile-time-checked mechanism for saying so.**

Every IREP2 kind declares e.g.

```cpp
static constexpr auto fields = std::make_tuple(&signedbv_type2t::width);
```

and `cmp`/`crc`/`hash`/`tostring` are generated over exactly that tuple
(`irep2.h:1050-1075`). A member absent from it does not enter value identity.

`fields_cover_class<K>()` would normally reject a missed member at compile time
— but the codebase already provides the escape for deliberate exclusions:

```cpp
static constexpr std::size_t excluded_field_bytes = sizeof(locationt);
```

`irep2.h:1077-1088` documents the rationale, and it is **the same rationale
Option F needs**:

> Source locations must travel with the statement for `goto_convert`, but must
> not enter value identity.

Substitute "spelling" for "source location" and that is Option F. The mechanism
is in use at eight sites in `irep2_expr.h` (the V.4 structured-CF kinds,
`code_block2t`'s `end_location`, `if2t`'s ternary position, the loop kinds'
`pragma_unroll_count`).

### 16.2 Is the spelling lost when two types compare equal?

**No — IREP2 does not intern or hash-cons types.** A grep for a type cache /
interning / hash-consing in `irep2_type.h` and `irep2.cpp` returns nothing; the
`fields`-derived hash exists for hashing containers, not for deduplicating
nodes. So two `signedbv_type2t`s of the same width with different spellings are
distinct objects that merely compare equal — each keeps its own spelling.

This was the risk §5.2 raised implicitly ("two types identical in width and
signedness may now differ in a spelling field") and it does not materialise.

### 16.3 What is still to be prototyped

Question 3 — verdict **and counterexample-text** parity over `esbmc-cpp` — still
requires the prototype and a run. Nothing above substitutes for it. But the two
design risks that made Option F look speculative are retired:

| §5.2 concern | status |
|---|---|
| the field leaks into `type2t` equality and changes verdicts | **retired** — omit from `fields`, declare `excluded_field_bytes` |
| `long` vs `long long` stop unifying in the solver | **retired** — same mechanism; they remain equal and unhashed apart |
| spelling lost to canonicalisation | **retired** — no interning |
| presentation concerns in the verifier IR | unchanged, and answered on merit in §5.2: `clang_cpp_adjust_expr` uses `#cpp_type` for **exception catch-matching**, which is semantics, not presentation |

**Revised sizing for Phase 0:** the spike is now "add one excluded field to two
kinds, repoint one reader, run the suite" rather than "discover whether the type
system can tolerate this at all". §10's "days, high confidence" stands, and the
no-go branch it hedged against is much less likely.
