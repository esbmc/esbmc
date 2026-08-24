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
- ~~**`#cpp_type`'s value domain is the C type-keyword set** — the writers emit
  `"bool"`, `"signed_char"`, `"unsigned_char"`, `"void"` and a `c_type` variable
  drawn from the same finite vocabulary. It is an enum wearing a string.~~
  **Refuted 2026-08-08, see §32.1.** The `c_type` variable is drawn from LLVM's
  builtin-type list, not from a type-keyword vocabulary: 83 distinct values, 56
  of them ARM SVE names. It is not an enum wearing a string.

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

> **The spike came back the other way round — see §36.** The equality asymmetry
> this paragraph hedges against is *not* the problem (§16 retired it). The
> domain is. So the fallback splits the opposite way to what is written here:
> Option F fits **Solidity**, whose classification is genuinely a closed enum,
> and not C/C++, whose spelling domain is open.

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

1. Normalise every temp *name*, not just temp paths
   (`s@esbmc[-._][A-Za-z0-9._-]*@TMPD@g`) — several spellings exist and partial
   normalisation reports ~90 % false divergence. Do not anchor on a leading
   slash: `--gcc-nested-functions` synthesises `esbmc-nested.<hash>.c`, which
   appears inside a symbol id with no path separator (§20.3).
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

## 15. Solidity and Jimple are not measurable here; Python re-censused (2026-08-04)

§13.5 left three suites open. Two cannot be measured on this machine, and the
third was re-measured after the §14 investigation produced a working fix.

### 15.1 Solidity and Jimple — blocked, not zero

| suite | attempt | outcome |
|---|---|---|
| `esbmc-solidity` | tests ship pre-generated `.solast`, so `solc`'s absence is not itself fatal | **conversion fails** — `ERROR: \`' is not a goto-binary`. Nothing is converted, so a decline census measures nothing |
| `jimple` | — | **frontend not built**: `ERROR: frontend for Jimple was not built on this version of ESBMC` |

A first pass reported "14 Solidity tests, 0 declines". **That figure is void** —
zero because nothing ran, not because nothing declines. It is recorded here
because it is exactly the failure mode §12.3's methodology note and §14.2's rule
are about: a census must show the thing under test executed before its zero
means anything. Both suites need Linux CI (Solidity) or a build with
`-DENABLE_JIMPLE_FRONTEND=On` (Jimple).

### 15.2 Python, re-censused with the §14 fix

The docstring-location fix (#6695) applied, same instrumentation, the five tests
the earlier samples covered:

| | before (§13) | after #6695 |
|---|---:|---:|
| declines per test | ~75 | **~27** |
| `code_expression` (unlocated statement) | 93 / 225 | **0** |

**The dominant site is gone entirely.** What remains, ranked:

| site | count | class |
|---|---:|---|
| `code_block` | 67 | cascade |
| `code_ifthenelse` — then-branch scope-exit leak | 30 | genuine |
| `code_ifthenelse` — else-branch scope-exit leak | 25 | genuine |
| `code_ifthenelse` — then-branch cascade | 10 | cascade |
| `code_assert` | 2 | genuine |

### 15.3 The remaining Python residue is already fixed, pending merge

The two genuine `code_ifthenelse` sites — 55 of the 57 genuine declines — are
**precisely what PR #6679 addresses**: it delegates a branch that leaks
scope-exit state instead of failing the walk, with the `tmp_symbol`/`context`/
`targets` rollback that made it byte-identical. #6679 is open at the time of
writing; the rest of the dispatcher series has merged.

So the Python picture after #6695 and #6679 together should be dominated by
cascade alone, with `code_assert` the only genuine site left in this sample.
**That is a prediction, not a measurement** — it needs re-running once #6679
lands.

### 15.4 Phase 1 exit criterion

| suite | status |
|---|---|
| `esbmc-cpp` | drained; residue is the assert-fold |
| `esbmc` (C) | drained; 4 declines / 60 tests |
| `python` | dominant site fixed (#6695); residue predicted to clear with #6679 |
| `esbmc-solidity` | **not measurable here** — needs Linux CI |
| `jimple` | **not measurable here** — frontend not built |

"0 fallbacks corpus-wide" remains unclaimable, but for a different reason than
in §13: the C-family and Python causes are addressed or identified, and what
blocks the claim now is **measurement access to two frontends**, not unknown
defects.

**Superseded by §18 (2026-08-05).** Both frontends *are* measurable on an
ordinary Linux box — Solidity needs no `solc`, Jimple only a build flag. What
measuring them found replaces the access blocker with a sharper one: Solidity
reaches zero declines and still fails the byte-identity A/B on every test
sampled.
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
## 13. The Python suite censused — and it is not like C (2026-08-04)

§12.3 named Python as the notable remaining gap: the largest frontend and the
converter furthest from the C/C++ path. It was measured, and §11.5's warning
that the other frontends "may reach sites this corpus never did" is **confirmed
in the strongest form so far**.

### 13.1 Result

Two independent stride samples over `regression/python`, all eight dispatcher
patches applied, each test replaying its own `test.desc` flags (§12.3's
methodology note). ~7 tests, **480 declines**:

| site | count | class |
|---|---:|---|
| `code_block` (cascade) | 249 | cascade |
| **`code_expression` — statement with no usable location** | **200** | **genuine, dominant** |
| `code_ifthenelse` — lone-`assert(false)` fold | 30 | genuine, shared with C/C++ |
| `code_assert` | 1 | genuine |

**~75 declines per test**, against **4 declines across 60 tests** for C. Python
is not close to drained; C effectively is.

### 13.2 The dominant site

```cpp
// The OTHER carries the statement location directly; without a usable one
// the legacy path would instead locate it at an enclosing block.
if (expr_stmt.location.is_nil() || expr_stmt.location.get_file().empty())
  return false;
```

An expression statement with **no usable source location** declines outright.
That is rare in C and C++, where nearly every statement comes from a source
line — and common in Python, whose converter emits synthetic statements
(operational-model calls, desugared constructs) carrying no location.

This is the clearest vindication of §11.5's refusal to extrapolate from one
suite: eight patches tuned on C++ drove that corpus down 95 % and left C at
essentially zero, while Python's single largest cause was never touched because
C++ never produced it.

### 13.3 Candidate fix, not yet attempted

The sibling branch immediately above it already threads `inherited` down for
exactly this problem, and the handler has `effective_location(expr_stmt.location,
inherited)` available. Using it here instead of declining is the obvious
candidate.

It is **not** a free change: it assigns a location where the legacy path would
have used the enclosing block's, so it must be gated on the byte-identical
`--goto-functions-only` A/B rather than verdict parity — location fidelity is
the whole subject of the W1-loc work, and `restore_value_locations` exists
because of it.

### 13.4 Sample size

~7 tests across two independent stride samples, consistent between them. Small,
and the reason is recorded rather than hidden: each Python test spawns the
parser subprocess and the dev machine was contended throughout. The *ranking* is
unambiguous at this size — one site is 200 of 231 genuine declines — but the
absolute per-test figure should be re-measured on a quiet machine before it is
quoted as a corpus rate.

### 13.5 Phase 1 exit criterion

| suite | censused | result |
|---|---|---|
| `esbmc-cpp` | yes | 28 243 → ~1 324, residue = assert-fold |
| `esbmc` (C) | yes | 4 declines / 60 tests |
| `python` | **yes, here** | ~75 declines/test; one dominant unfixed site |
| `esbmc-solidity` | no | macOS-blocked; rides Linux CI |
| `jimple` | no | — |

Phase 1 is **not** near its exit criterion. C and C++ are drained; Python has a
large, single, well-localised cause that no existing patch addresses.

## 14. §13.3's candidate fix is refuted (2026-08-04)

§13.3 proposed using `effective_location(expr_stmt.location, inherited)` instead
of declining, on the reasoning that the sibling branch already threads
`inherited` for exactly this problem. **It was implemented and it never fires.**

### 14.1 The measurement

A probe placed inside the new branch — printing only when the statement's *own*
location is unusable, i.e. exactly the case the change exists to serve — was run
on `casting31`, one of the tests the §13 census recorded at ~75 declines:

| build | firings |
|---|---|
| master + the change | **0** |
| all eight dispatcher patches + the change | **0** |

Zero in both. So `effective_location` returns something equally unusable:
**these statements have no usable location anywhere in their ancestry**, not
merely none of their own. The guard still declines, and the change is dead code.

### 14.2 The gate that nearly passed it

The change was A/B'd first and came back **byte-identical on six tests,
including all three of the decline-heavy ones**. That looked like a clean
behaviour-preservation result. It was vacuous: the output is identical because
the code never ran.

This is the same trap recorded at §11.4 and hit repeatedly on this track — *a
passing gate is not evidence unless the thing under test is shown to execute*.
Byte-identity is especially prone to it, because a no-op scores perfectly.
**Probe that the change fires before, not after, running the A/B.**

### 14.3 What this means for the Python residue

The dominant Python site is not a location-plumbing gap. Whatever emits these
statements gives them no location and places them where no enclosing statement
has one either. So the fix must either:

1. give the synthetic statements a location at the point the Python converter
   emits them — the OM-call and desugaring sites; or
2. reproduce what the legacy path does for a wholly unlocated OTHER, which the
   §13.2 comment says is to locate it at an enclosing *block* — a construct the
   dispatcher does not track, and which `inherited` evidently is not.

Option 1 is the more promising and is frontend work, not dispatcher work.
Neither has been attempted.

`fix/native-expr-inherited-location` (#6692) should be closed unmerged: it is
inert by measurement.
## 12. The C suite censused (2026-08-04)

§11.5 recorded that "0 fallbacks corpus-wide" was **not** claimable because only
`esbmc-cpp` had been measured, and that the other frontends "may reach sites
this corpus never did". The C suite has now been measured, with all eight
dispatcher patches applied.

### 12.1 Result

**60 `regression/esbmc` tests, 4 declines in total.**

| site | count | class |
|---|---:|---|
| `code_block` | 2 | cascade from the two below |
| `code_label` — the `--error-label` shape | 1 | genuine, flag-specific |
| `code_ifthenelse` — the lone-`assert(false)` fold | 1 | genuine |

Against a 28 243-decline `esbmc-cpp` baseline before the patches, C lands at
essentially zero. **The patches were developed entirely against C++ and drain C
too** — expected, since `convert_native_rec` is frontend-agnostic, but worth
measuring rather than assuming, which is what §11.5 refused to do.

### 12.2 The residue is the same class in both suites

- **the assert-fold** — `generate_ifthenelse` folds a branch that reduces to a
  lone `assert(false)` into the guard; the dispatcher declines rather than
  reproduce the fold. Present in both suites.
- **`--error-label`** — `convert_label` turns a matching label into an
  `ASSERT(false)` carrying property metadata. Fires only under that flag, so it
  is invisible to any census that does not replay `test.desc` flags. It never
  appeared in the C++ corpus.

Both are candidates for the same statement-local delegation the eight patches
use; neither is a representation gap.

### 12.3 What the Phase 1 exit criterion still needs

| suite | censused | result |
|---|---|---|
| `esbmc-cpp` | yes | 28 243 → ~1 324, residue = assert-fold |
| `esbmc` (C) | yes | 4 declines / 60 tests |
| `python` | **yes, here** | ~75 declines/test; one dominant unfixed site |
| `esbmc-solidity` | no | macOS-blocked; rides Linux CI |
| `jimple` | no | — |

Phase 1 is **not** near its exit criterion. C and C++ are drained; Python has a
large, single, well-localised cause that no existing patch addresses.
| `esbmc` (C) | **yes, here** | 4 declines / 60 tests |
| `python` | no | — |
| `esbmc-solidity` | no | macOS-blocked (no `solc`); rides Linux CI |
| `jimple` | no | — |

Python is the notable gap: the largest frontend, and the one whose converter
differs most from the C/C++ path. "0 fallbacks corpus-wide" cannot be claimed
until all four are measured — and per §11.5 that claim is the precondition for
deleting the round-trip and the fallback path.

**Methodology note.** Replay each test's own `test.desc` flags. The
`--error-label` site is invisible otherwise, and it is one of only two genuine
sites C has left.


## 17. Post-series census — §15.3's prediction confirmed (2026-08-04)

§15.3 predicted that with #6679 merged, Python's two `code_ifthenelse`
scope-leak sites — 55 of its 57 genuine declines at the time — would clear, and
flagged it explicitly as a prediction rather than a measurement. The whole
dispatcher series is now on master. Measured.

### 17.1 Result

Five Python tests, current master, all eight dispatcher patches merged:

| site | count | class |
|---|---:|---|
| `code_block` | 197 | cascade |
| **`code_expression` — statement with no usable location** | **155** | genuine |
| `code_ifthenelse` — lone-`assert(false)` fold | 25 | genuine |
| `code_assert` | 2 | genuine |

**Both scope-leak sites are gone.** The only `code_ifthenelse` decline left is
the assert-fold — the same residue C and C++ carry (§12.2). §15.3's prediction
holds.

### 17.2 The dominant cause is fixed but unmerged

`code_expression` at 155 is the unlocated-statement site §13 identified, and
**#6695 takes it to zero** — measured at 31 → 0 on `casting31`. It is open at
the time of writing. So the top Python decline cause on master today is
addressed by a pending PR, not by undiagnosed work.

Projected residue once #6695 lands: the assert-fold and `code_assert`, plus
cascade. That is the same shape C and C++ already reached, and it would mean
**all three C-family/Python suites are drained to the same two narrow sites.**

### 17.3 Caveat on comparing censuses

Numbers from different configurations are not directly comparable. Fixing one
site changes what is *reachable*, so a later site's count can rise even as the
program improves — more statements convert natively, so more branches are
attempted and more of them get the chance to decline. Compare like with like:
same tests, same patch set, and prefer the per-site breakdown over the total.

## 18. Solidity and Jimple censused — and a defect the decline metric cannot see (2026-08-05)

§15.4 lists both frontends as **"not measurable here"** and §11.5 makes that the
reason "0 fallbacks corpus-wide" stays unclaimable. Both are measurable, on an
ordinary Linux dev box, and both have now been measured.

### 18.1 Why they were thought unmeasurable, and why that was wrong

| frontend | §15 reason | actual |
|---|---|---|
| `esbmc-solidity` | needs `solc`, so "rides Linux CI" | **`solc` is not needed.** Every test ships a committed `contract.solast` — `test.desc` line 2 names the AST, not the `.sol`. `--sol contract.sol` supplies source mapping only |
| `jimple` | "frontend not built" | a build configured with `-DENABLE_JIMPLE_FRONTEND=On` runs the suite; no JDK invocation is involved at verify time, the `.jimple` is the input |

The blocker was a property of the machine §15 was written on, not of the
suites. Neither suite needs CI.

### 18.2 Jimple — 15 tests, 24 declines, one genuine site

Whole suite, each test replaying its own `test.desc` flags:

| site | count | class |
|---|---:|---|
| `cpp_throw` | 12 | **genuine** |
| `code_block` | 12 | cascade from the above |

**1.6 declines per test** — the C profile (§12.1), not the pre-#6695 Python one.

The genuine site is new: a **bare** `code_cpp_throw2t` *statement*.
`convert_native_rec`'s `code_expression` arm already delegates a throw to the
legacy `convert()` when it arrives wrapped in an expression statement
(`goto_convert_functions.cpp:448-456`) — that is what #6295 added. The
Jimple frontend emits the throw as a statement in its own right, which no arm
claims, so it reaches the unsupported-kind fallback and takes the whole
function with it. §11.5 predicted exactly this: "may reach sites this corpus
never did".

The fix is the same statement-local delegation the eight merged patches use.
Not attempted here.

### 18.3 Solidity — 0 declines, and that is the misleading part

26-test stride sample: **0 declines**. Every function body converts natively.
By the metric §12/§13/§17 use, Solidity is the most drained frontend in the
tree.

It is also the only one that is **not byte-identical**.

### 18.4 The A/B gate fails 13/13 on Solidity

§11.4 names the `--goto-functions-only` A/B against `--no-irep2-native-body` as
the gate that "earned its place", and warns that byte-identity is strictly
stronger than verdict parity. That gate has evidently never been run against
this frontend. It fails on **every Solidity test sampled** — 13/13 in one
stride sample, and 8/8 re-run on a clean (uninstrumented) binary — at 24 to
148 divergent lines per test, after normalising the two known noise sources
(the GOTO timing lines and the per-run `/tmp/esbmc*` path).

The controls are clean, so this is Solidity-shaped, not a normalisation
artefact: **C 10/10 identical, Python 5/5, Jimple 15/15**, same sweep, same
binary.

Every divergent instruction is a **RETURN**:

```
native:   // 2902 no location        legacy:   // 2902
          RETURN: 1                            RETURN: 1
```

The native arm assigns the statement's own IREP2 location
(`goto_convert_functions.cpp:646-654`, `r->location = ret.location`). Where
that location is nil, the instruction is nil-located; the round-trip's
`convert_return` reads the location off the migrated `codet`, which is
empty-but-present. Both are "unlocated" to a reader, but `is_nil()`
distinguishes them, and the dispatcher's contract is byte-identity, not
approximate agreement.

Two things make this worth fixing rather than waiving:

1. It is **on master today**, on the default path — the native dispatcher is
   on unless `--no-irep2-native-body` is passed.
2. The return arm is the one arm that does *not* route its location through
   `effective_location` (contrast lines 278, 336, 429, 453), which is why it
   is the only shape that diverges.

Jimple is byte-identical 15/15, but trivially so: a decline falls back to the
round-trip, and the round-trip is the reference. Solidity is the only frontend
that takes the native path everywhere and disagrees with it.

### 18.5 What this does to the Phase 1 exit criterion

| suite | declines | byte-identical |
|---|---|---|
| `esbmc-cpp` | drained; residue = assert-fold | gated per patch (§11.4) |
| `esbmc` (C) | 4 / 60 tests | gated per patch |
| `python` | dominant site fixed by #6695 (merged) | not swept |
| `esbmc-solidity` | **0 / 26** | **fails 13/13** |
| `jimple` | 24 / 15 tests, one genuine site | **15/15** |

The measurement-access blocker §15.4 recorded is **gone**; all five suites are
measurable here. What replaces it is sharper: *"0 fallbacks corpus-wide" is not
the exit criterion it was taken to be.* A frontend can reach zero declines and
still not reproduce the round-trip. The criterion needs both clauses, and the
A/B sweep needs to run per frontend, not per patch — a patch developed against
`esbmc-cpp` is gated against `esbmc-cpp`, and Solidity is what that misses.

### 18.6 Reproduction

Both numbers come from temporary instrumentation — one `fprintf` at each of
`convert_native_rec`'s 21 `return false` sites, printing `get_expr_id(code2)`,
which is exactly the site name §12/§13/§17's tables use. There is no census
switch in the tree; §11.4's "census env var" describes a build that was never
merged. The byte-identity sweep needs no instrumentation at all — it is the
A/B, normalised. Normalise `esbmc*` broadly: the C driver's temp dir is
`esbmc.<hash>` and Solidity's is `esbmc_solidity_temp-<hash>`, and a pattern
that catches one and not the other reports a false divergence. Match the name
wherever it appears, not only after a `/` — see §7 rule 1 and §20.3.

## 19. §18.4's defect fixed, and what the whole-suite sweep found underneath (2026-08-05)

§18.4 sampled 13 Solidity tests. Sweeping all 520 changes both the fix and the
picture of what is left.

### 19.1 The RETURN divergence was two defects, not one

The nil-location diagnosis holds, and the fix is the one §18.4 implies: route
the RETURN and its end-of-function GOTO through a materialised-empty location
rather than the nil `location2t`, because `convert_return` reads its location
off the round-tripped `codet` through the **non-const** `exprt::location()`,
which materialises an empty — and so not nil — `#location`. The native decl arm
already open-coded that step; both now share one helper.

Fixing it exposed a second: `convert_return`'s `else` arm logs *"function
should not return value"*, and the native arm had no counterpart, so the
diagnostic was silently dropped. §18.4 could not see this — C and C++ reject
the shape, and on Solidity the RETURN divergence masked it. The native arm now
delegates that shape, as it already does for the four `convert_return`
rewrites.

### 19.2 Whole-suite result: 502 / 510

| | before | after |
|---|---|---|
| `esbmc-solidity` A/B | fails every test sampled (13/13) | **502 identical, 8 divergent** (10 tests skip: no source file) |
| controls | C 10/10, Python 5/5, Jimple 15/15 | C 80/80, Python 25/25, Jimple 15/15, `esbmc-cpp/destructors` 14/14 |

### 19.3 The eight residuals are a different defect — and it is not stable

Every residual is **location-only**: zero instruction-text differences across
all eight. The divergent locations are synthetic, always one line past the end
of the contract (`swc_107_1`: 57-line file, native 59 vs legacy 58;
`doftcoin_1`: 104 lines, native 105 vs legacy 106 — note the direction
reverses), and they land on the generated scope-exit run — `DEAD`,
`END_FUNCTION`, and whatever else inherits from it.

The sharper finding is that **the legacy path is not run-to-run
deterministic**. Ten repeats of `erc20_1` on the *same* binary, *same* flags,
`--no-irep2-native-body` throughout, produce two distinct outputs — 8 runs at
`line 96`, 2 at `line 97`. That is why the divergent set moves between sweeps:
`erc20_1`, `swc_107_1` and `whole_contract_1` appeared in one sweep and not the
next, and re-running them 5×5 shows native and legacy agreeing. Only the eight
above diverge stably.

Two consequences:

1. **A single A/B run is not a verdict** on a Solidity test. Re-run a
   divergence before believing it; §18.4's 13/13 was safe only because the
   RETURN defect was universal and large.
2. A nondeterministic synthetic location is a defect in its own right,
   independent of the dispatcher — it is reachable with the native path off.

Neither is fixed here; they are filed as #6759 (the stable divergence) and
#6760 (the nondeterminism). Both predate the dispatcher: the fix in §19.1 can
only turn a nil location into a blank one, so it cannot produce a line-number
difference, and it cannot introduce nondeterminism.

### 19.4 Reproduction

Same A/B as §18.6, no instrumentation. For the residuals, run each side five
times and compare the *sets* of output hashes, not one run against one run —
`native_variants`, `legacy_variants`, and whether the two sets intersect is the
measurement that separates a stable divergence from a nondeterministic one.

## 20. §18.2's Jimple site closed — and what its mutants say about the gates (2026-08-05)

§18.2 recorded Jimple's one genuine decline site: a **bare**
`code_cpp_throw2t` *statement*, 12 declines across the 15-test suite, which no
dispatcher arm claimed and which therefore took each containing function into
a whole-function fallback. It is fixed by the same statement-local delegation
the eight merged patches use.

### 20.1 Result

| | declines | jimple suite | A/B byte-identity |
|---|---:|---|---|
| before | **12** (12 of 15 tests, 1 each) | 15/15 pass | 15/15 identical, but *trivially* — §18.4 |
| after | **0** | 17/17 pass | **15/15 identical, non-trivially** |

The A/B number is unchanged and that is the point: before the fix those twelve
functions were converted by the round-trip, so the native arm was compared
against itself. They now take the native path end-to-end and still agree
byte-for-byte. Jimple is the first frontend to reach **both** clauses of the
criterion §18.5 says the exit needs — zero declines *and* byte-identity.

The delegation needs no `tmp_symbol`/`context`/`targets` snapshot: like #6668,
#6672, #6674, #6677 and #6678 it runs *before* any native attempt on the
statement, so §11.3's two bounds do not apply.

The two tests added here do **not** execute the new arm — they pass
`--no-irep2-native-body`, which short-circuits `try_convert_body_native`
before the dispatcher is entered. They are the *legacy half* of an A/B pair
whose native half already exists (`github_4715_irep2_bodies_jimple_01{,_fail}`
carry byte-identical inputs on the default path), so what they make durable is
native/legacy verdict agreement on a throw-bearing body. Coverage of the arm
itself comes from the twelve pre-existing tests, measured by the census, not
from anything this patch adds.

### 20.2 The mutants, and which gate caught which

Three mutants were run rather than assumed, because §11.4 and §14.2 both turn
on the difference between a gate passing and a gate discriminating:

| mutant | what it breaks | caught by |
|---|---|---|
| M1 — arm absent (`return false`) | nothing observable; falls back | **only the decline census** |
| M2 — arm present, conversion dropped | the throw disappears | **6 regression tests** + A/B 12/15 |
| M3 — `restore_value_locations` given a nil stamp | nothing observable | **nothing** |

M1 is the honest limit of this patch class: the delegation is
behaviour-preserving by construction, so *no verdict test can distinguish an
arm that exists from one that does not.* Only the census can. Do not ask a
`test.desc` to pin arm presence; ask it to pin arm correctness, which M2 shows
it does.

Read M2's kill list off ctest's full output, not its tail: the first run of
this table said five, because `tail -8` cut the head of the failure list. The
sixth is `github_4715_irep2_bodies_jimple_01_fail`, whose input is
byte-identical to `kt-hello-false` and whose `--irep2-bodies` flag is a
documented no-op, so it takes the native path like the other five. Six of the
twelve throw-bearing tests expect FAILED, and all six flip.

M3 is a live coverage note rather than dead code. `restore_value_locations`
iterates operands, and Jimple's throw has none — `jimple_statement.cpp:368`
builds `codet("cpp-throw")` with the thrown value deliberately unattached
(a frontend TODO). The call is required for byte-identity the moment a value
*is* attached, and the C++ expression-statement arm relies on it today, so it
stays; it is simply unexercised by any corpus that currently reaches this arm.

### 20.3 A normalisation gap the C control exposed

The C control sweep reported one divergence, `gcc_nested_func_06`, which was
neither: `--gcc-nested-functions` synthesises a per-run temp **file name**,
`esbmc-nested.<hash>.c`, which appears *without a leading slash* inside a
symbol id (`c:esbmc-nested.4b6b-67a7.c@F@...`). §18.6's pattern anchors on
`/esbmc*` and misses it.

The control settled it, exactly as §7 rule 7 prescribes: the legacy arm run
against **itself** three times produced three distinct hashes. Widen the
pattern to `s@esbmc[-._][A-Za-z0-9._-]*@TMPD@g` — no leading slash — and the
test is identical on both arms. C is then **138/138** on a stride-12 sample.

This is the same class as §19.3's Solidity finding, arrived at from the other
direction: there a synthetic *location* varied run to run, here a synthetic
*file name* does. Treat any per-run artefact as noise until a self-control
says otherwise.

### 20.4 Phase 1 exit criterion

| suite | declines | byte-identical |
|---|---|---|
| `esbmc-cpp` | drained; residue = assert-fold | gated per patch (§11.4) |
| `esbmc` (C) | 4 / 60 tests | **138/138** (stride-12, §20.3 normalisation) |
| `python` | dominant site fixed by #6695 (merged) | not swept |
| `esbmc-solidity` | 0 / 26 | 502/510; 8 residuals = #6759, #6760 |
| `jimple` | **0 / 15** | **15/15** |

What is left before the round-trip can be deleted is now short and named: the
assert-fold in C/C++/Python, a Python A/B sweep that has never been run, and
Solidity's eight residuals — which #6759 and #6760 already establish are *not*
dispatcher defects.

## 21. The Python A/B sweep, run — and the one defect it found (2026-08-08)

§20.4 listed the Python sweep as the one clause never measured. It is run here,
at stride 15 over `regression/python` (303 of 4 539 tests), same A/B as §18.6.

### 21.1 Result

| | divergent | identical |
|---|---:|---:|
| before | **19** / 303 | 284 |
| after | **0** | **303 / 303** |
| controls (C stride-12, C++ stride-12) | 1 / 202, pre-existing (§21.4) | 201 |

A further 16 tests diverged only under §18.6's normalisation and are not
counted above; §21.3 records what they needed.

### 21.2 One defect, and it is a hole in a stated premise

All 19 are the same site, and they are location-only. The native `ASSIGN` arm
stores `code2` verbatim on the reasoning — written into the code as a comment —
that "migrate_expr drops the operand locations, so none of
restore_value_locations' stamping survives in the stored code." That premise is
false for exactly one kind: `if2t` carries a location field
(`irep2_expr.h:786`) and `migrate.cpp:1006` round-trips it. So a ternary nested
in a side-effect-free right-hand side keeps its stamped location on the legacy
path and loses it natively.

Python is where it shows because Python is where the shape occurs: floor
division lowers to an arithmetic expression with an unlocated `if2t` correction
term, so every `//` inside an assignment hits it. C and C++ do not — the clang
frontends stamp sub-expression locations at parse time, so the ternary already
has one and the legacy stamping is a no-op.

The fix is the IREP2 half of `restore_value_locations`: stamp the statement's
effective location onto location-less `if2t` operands before storing `code2`.

**The premise is written into four arms, not one.** The A/B sample only reached
the `ASSIGN` one; review found the same sentence — and the same divergence —
at the three other sites that store `code2` verbatim, each reproduced against
the patched binary before being fixed:

| arm | Python shape that reaches it |
|---|---|
| `code_assign2t` → ASSIGN | `y = (x + n // x) // 2` |
| `code_return2t` → RETURN | `return (x + n // x) // 2` |
| `code_expression2t` → OTHER | `(x + n // x) // 2` as a bare statement |
| `code_function_call2t` → FUNCTION_CALL | `g((x + n // x) // 2)` |

`code_decl2t` was probed and does not diverge: a decl with an initializer
delegates on `has_sideeffect` before reaching a verbatim store. Each of the four
arms has already excluded code-typed operands by the time it emits, so unlike
`restore_value_locations` the IREP2 walk never has to re-root on a nested
statement — an invariant the helper's comment now states, because it is a
coupling across a function boundary rather than a local property.

### 21.3 The sweep needed three normalisations §18.6 does not name

Each was settled by the §7 rule 7 self-control — the legacy arm against itself —
not by inspection:

| artefact | why it varies | normalisation |
|---|---|---|
| `GOTO program processing time: N.NNNs` | wall clock | `time: Ts` |
| `ESBMC_unpack_temp_<n>` | temp name derived from an address | `_N` |
| `ASSIGN __file__={ 47, 118, ... }` | the astgen temp dir, **as decimal character codes** | collapse the initialiser |

The third is the one to remember: `__file__` holds the per-run temp directory
encoded byte-by-byte, so no amount of widening §20.3's `esbmc[-._]…` text
pattern can reach it. A per-run artefact need not be legible as text.

### 21.4 Two findings the sweep produced that this patch does not fix

1. **A C divergence that is not location-only.** `esbmc/cwe_uninit_array_vla`
   (`--uninitialised-vars-check --incremental-bmc`) renders a VLA bound as `n`
   natively and `tmp$1` under the round-trip — the first *instruction-text*
   divergence recorded in any suite; every prior residue was a location. It
   reproduces with the patch reverted, so it predates it. Not filed yet.
2. **The A/B sees `if2t::location` only through the tree-dump fallback.** The
   dump renders a `#location` block only where `from_expr` cannot print the
   expression; on a printable one the field is invisible. Python surfaced this
   defect only because its `xor` node forces the fallback, which is also why the
   303-test sample reached one of the four affected arms and not the other
   three. Treat the sweep's coverage of this field as partial.

### 21.5 Mutants

| mutant | what it breaks | caught by |
|---|---|---|
| M1 — stamping call removed | the ternary's location | `…ternary_loc_01` (ASSIGN) and `…_03` (RETURN/OTHER/CALL); and the A/B, 19/303 |
| M2 — stamp unconditionally, overwriting a frontend `?:` position | a C/C++ ternary's own location | **nothing** — §21.4 item 2 is why |

M1 was run, not assumed: reverting the call fails exactly the two tests that pin
the arms and leaves `…_02` — the legacy-path control, which passes pre-patch by
construction — green.

M2 is the honest limit. The guard is kept on the semantics rather than on a
gate: `irep2_expr.h:787` says the field carries the `?` position for witness
branching, so a frontend that supplied one must win. Do not read the passing
gates as evidence the guard is exercised.

### 21.6 Phase 1 exit criterion

| suite | declines | byte-identical |
|---|---|---|
| `esbmc-cpp` | drained; residue = assert-fold | gated per patch (§11.4) |
| `esbmc` (C) | 4 / 60 tests | 138/138 (stride-12, §20.3); 1 divergence at stride-12 here, §21.4 |
| `python` | dominant site fixed by #6695 (merged) | **303/303** (stride-15) |
| `esbmc-solidity` | 0 / 26 | 502/510; 8 residuals = #6759, #6760 |
| `jimple` | 0 / 15 | 15/15 |

Python's 303/303 is a sampling result, not a proof — §21.2 is the standing
warning that a shape absent from the sample can still carry the defect, and
§21.4 item 2 bounds what the dump can observe at all. What is left is the
assert-fold in C/C++/Python, Solidity's eight residuals, and the C divergence in
§21.4 — which, unlike the Solidity eight, has not been shown to predate the
dispatcher series as a whole, only this patch.

> **Provenance.** §21.7 and §21.8 arrived from the parallel Python sweep merged
> as #6829, which sampled at stride 12 (379 tests, 24 divergences) where §21.1
> above sampled at stride 15 (303 tests, 19 divergences). Both measured the same
> defect — the location-less `if2t` of §21.2 — so their analysis carries over
> unchanged, but where they cite §21.1's or §21.6's figures they mean that
> sweep's, not this one's.

### 21.7 The obvious fix for §21.2 is refuted

Recorded in §14's spirit, because the candidate is the one anybody looking at
§21.2 will reach for first, and it is wrong in a way the A/B metric actively
hides.

`handle_floor_division` (`python_math.cpp`) builds the correction ternary with
`python_expr::build_if(cond, gen_one(div_type), gen_zero(div_type))` and sets no
location, while the float path three lines up does
`floor_call.location() = bin_expr.location()`. The obvious patch is to make the
integer path match:

```c++
if_expr.location() = bin_expr.location();
```

It works, by the metric. All 24 divergent tests plus `github_4792` go to
byte-identical — 25/25.

**It is still wrong.** Measure the *legacy* arm before and after, which the A/B
alone never does:

| | line | column |
|---|---:|---:|
| legacy, before the patch | 276 | 8 |
| legacy, after the patch | 265 | 10 |

`bin_expr.location()` is not the statement's location — 265 is up in the
function-header region, not the `y = (x + n // x) // 2` the instruction came
from. Pre-setting a location in the converter makes `restore_value_locations` a
no-op on that node, because it only fills nodes that are location-*less*
(`goto_convert_functions.cpp`). So the patch does not pull native up to legacy;
it pulls **both arms down** onto a worse location, and byte-identity improves
because fidelity got uniformly worse on both sides.

Two things follow, and the second is the more general one:

1. The real fix is on the native side, not in the converter: an IREP2-level
   equivalent of `stamp_value_locations` that fills a nil `if2t::location` from
   the enclosing statement. That touches the shared dispatcher, so it wants its
   own gates and its own patch — it is deliberately not attempted here.
2. **Byte-identity is a comparison, not a correctness measure.** It cannot tell
   "native was fixed" from "legacy was broken to match". Any patch justified by
   a divergence count must also show the legacy arm unchanged; §18.4's A/B
   protocol does not require that today, and this is the case that shows it
   should.

### 21.8 The fix landed — and the sweep understated its reach by 3 sites

§21.7's "real fix is on the native side" is #6835. Two things about it are worth
recording here, because they are properties of *this sweep*, not of that patch.

**The divergence set named one emission site; four were affected.** The nil
`if2t::location` is not specific to the assignment handler. A nested ternary can
also travel through the native `return`, the bare-call and the
expression-statement emission points, and all three dropped the location the
same way. None of them appears in §21.1's divergent set — no test in the
stride-12 sample happened to put a floor division in a return or a call
argument, so the sweep reported those paths clean.

This is §20.2's M1 result reached from the other direction. There, no verdict
test could distinguish a delegation arm that exists from one that does not, and
only the census could. Here, no A/B sweep could distinguish an emission site
that stamps from one that does not *unless the corpus happens to route a ternary
through it* — and a 379-test sample routed one through exactly one of four. The
sites were found by probing each emission point directly with a constructed
input, after reading the handlers.

**So a divergence count is a lower bound on defect reach, never a measure of
it.** §21.1's "24 divergences, one site" was accurate about what diverged and
misleading about what was broken: the correct statement is *one site observed,
reach unmeasured*. Any future sweep row should be read that way, and a fix
derived from one should enumerate the code paths that share the cause rather
than the tests that happened to catch it.

The corresponding correction to §21.6: the `isqrt` operand location is no longer
"left to a separate patch" — it is #6835, covering all four sites, with A/B test
pairs whose native arms fail without it.

## 22. §21.4's C divergence is a soundness bug — and the second half of the same premise (2026-08-08)

§21.4 filed `esbmc/cwe_uninit_array_vla` as "the first instruction-text
divergence, not yet filed." Run down, it is not a cosmetic at all: on the
default (native) path ESBMC **silently accepts a real out-of-bounds read**.

### 22.1 The reproducer

```c
int main(void)
{
  int n = 1;
  int a[n];
  a[0] = 42;
  n = 100;
  return a[5];        /* ASan: dynamic-stack-buffer-overflow */
}
```

| path | bound check emitted | verdict |
|---|---|---|
| `--no-irep2-native-body` | `5 < (signed long int)tmp$1` | **FAILED** (correct) |
| default (native) | `5 < (signed long int)n` | **SUCCESSFUL** (misses it) |

`n = 100` is what turns the stale bound into a *vacuous* one, so the missed bug
needs a reassignment; without it the two bounds are equal and the divergence is
invisible in the verdict — which is why the original test
(`cwe_uninit_array_vla`, no reassignment) passed on both paths and the defect sat
in the A/B as a text difference only.

### 22.2 Root cause: the *other* thing migrate_expr normalises

C11 6.7.6.2p5 evaluates a VLA's size expression once, at the declaration, so
`convert_decl` snapshots it into a temporary and **retypes the array symbol
mid-body** — `s->set_type(...)`, `goto_convert.cpp`. The legacy path picks that
up for free, and not by mutating its tree: `sym_name_to_symbol`
(`migrate.cpp:613`) deliberately re-reads **every level0 symbol's type from the
global symbol table** rather than trusting the expression's own, with its own
comment explaining why ("various things out there get parsed in with a partial
type"). So a statement converted *after* the retype migrates with the new type.

A native arm storing `code2` verbatim never re-migrates, so it keeps the
frontend-time `int[n]`. `goto_check`'s bounds check then reads `array_size`
straight off that stale type (`goto_check.cpp`, `ns.follow(ind.source_value->type)`).

This is the same shape of defect as §21.2 and it was hiding behind it:
**`migrate_expr` performs two normalisations that a verbatim store skips** — the
ternary location, and the symbol-table type. §21 fixed the first at four arms;
this fixes the second at the same four, behind one `normalise_native_code`
helper whose contract is now stated positively: *`code2` as `migrate_expr` would
have produced it from the legacy statement the fallback converts.*

### 22.3 Result

| sweep (divergences) | before §21 | after §21 | after §22 |
|---|---:|---:|---:|
| `esbmc` (C) + `esbmc-cpp` stride-12 | 1 / 202 | 1 / 202 | **0 / 202** |
| `python` stride-15 | 19 / 303 | 0 / 303 | 0 / 303 |

The C sample is clean for the first time. That is *not* the exit criterion met:
§22.6 is a defect this sample cannot see, found by review rather than by
measurement, and it is the second time on this patch that the sweep's silence
was mistaken for coverage.

### 22.4 Mutants

| mutant | what it breaks | caught by |
|---|---|---|
| refresh disabled, stamping kept | the VLA retype | **both** new tests — `…vla_retype_01_fail` flips to SUCCESSFUL (false negative), `…_01` to FAILED (false positive) |
| stamping disabled, refresh kept | the ternary location | the §21.5 tests only |

The two normalisations are independently pinned, which matters because they
share a helper and a call site: neither test set can pass on the other's fix.

### 22.6 Review found the same bug at the condition guards

The statement arms were fixed first, because those are what the sweep reported.
Review then reproduced the identical missed out-of-bounds read with the access
in an `if`, `while`, `do`/`while` and `for` condition — those arms fold the
condition into a GOTO guard verbatim, so they share the premise and were not
covered. The sample contains no VLA in a branch condition, so no amount of
re-running it would have surfaced this.

All six sites (the four statement arms plus the four condition guards, and
`code_assert2t`/`code_assume2t`'s guards, which are Python-reachable) now go
through the one `normalise_native_code` chokepoint.

### 22.5 What this says about the A/B as a gate

§21.4 item 2 warned the sweep's *location* coverage was partial. This is the
sharper lesson and it runs the other way: the sweep **did** report this
divergence, plainly, in instruction text — and §21 read it as "location-only
residue, filed for later" because every prior residue had been. A text
divergence is not the same class as a location one, and the C suite had never
produced one before. **Re-classify before deferring**: an unexplained A/B
divergence is a defect of unknown severity, not a cosmetic, until it has been
run down. This one was a default-on missed bug that had shipped.

## 23. The assert-fold reproduced — the last named decline residue (2026-08-08)

§12.2 named the assert-fold as the residue both C and C++ carry, and §20.4/§22
carried it forward as one of the three things left before the round-trip can go.
`generate_ifthenelse` collapses a branch that reduces to a lone `assert(false)`
into the guard; the native arm detected those shapes and **`return false`d**,
which is a *whole-function* fallback — worse than the statement-local delegation
the rest of the dispatcher uses. It now reproduces the fold.

### 23.1 The shapes, and which are corpus-reachable

| shape | native handling | reached by |
|---|---|---|
| then-branch is a lone `assert(false)`, no else (or a no-op else) | folded, guard `!c` | `…assert_fold_01` |
| else-branch is a lone `assert(false)`, then-branch a no-op | folded, guard `c` | `…assert_fold_01` |
| both branches lone `assert(false)` | both folded | `…assert_fold_01` |
| then-branch is a lone `assert(false)`, else-branch a *no-op* | folded, guard `!c` | `…assert_fold_01` |
| `(void)((cond) \|\| (assert(0),0))` — the C-library idiom | folded, guard `!c`, second instruction dropped | **`regression/esbmc/github_1565`** and 3 others; `…assert_fold_03` |

and one shape that is not a fold but a delegation:

| a fold that fires and still leaves the other branch to convert | delegated (the legacy re-entry with branches swapped is not reproduced) | `…assert_fold_02` |

**Only the `||` idiom occurs in the corpus.** A stride-6 instrumented scan of
`regression/esbmc` + `regression/esbmc-cpp/cpp` (405 tests) fires the fold on
exactly four — `github_1565`, `no_pointer_check_4`,
`interval_can_handle_global`, `github_5998-long-chain_fail` — and every one is
`idiom=1`. The other shapes were reached only by written reproducers
(`__ESBMC_assert(0, …)` in branch position), which is what the new tests pin.
Every branch this patch adds is shown live by one or the other, per the C-Live
obligation; none is dead instrumentation.

**The idiom's gate was wrong on the first cut, and review caught it.** Legacy
gates that fold on the else *program* being observationally no-op
(`is_no_op_program`); the native arm tested the *AST* (`else_case` nil), so
`if (c) { assert(0); g = 1; } else { }` folded on one path and not the other.
Not corpus-reachable, but a byte-identity break, and the third time on this
branch that a first cut was scoped by what the sweep happened to sample rather
than by what the legacy code actually says. `…assert_fold_03` pins it. The
shared predicate is now `is_no_op_program` in `remove_no_op.h` — previously a
file-static in `goto_convert.cpp` that this arm had copied, which is how the two
came to disagree.

### 23.2 Mutants — and why one of them cannot be caught

| mutant | what it breaks | caught by |
|---|---|---|
| M1 — fold arms replaced by `delegate_to_legacy()` | nothing observable | **nothing** (see below) |
| M2 — folded guard `c` instead of `!c` | the assertion's condition | `…assert_fold_01` (text) and `…_01_fail` (verdict: the assume makes the sign observable), plus the A/B on `github_1565` |
| M3 — idiom gated on an AST-empty else | the no-op-else idiom | `…assert_fold_03` |

M1 is §20.2's limit again, and it is worth restating because it is the reason
this residue survived so long: **a behaviour-preserving delegation is
indistinguishable from the arm that replaces it by any verdict or output test.**
The old code's `return false` and the new fold produce byte-identical programs.
Only a decline census can tell them apart, which is why §23.1 reports the scan
rather than resting on the green suite.

### 23.3 A/B and a harness correction

C stride-12 + C++ stride-4: **327/328**. A wider C++ stride-4 sample (189
tests) reports one divergence, `cpp_stack_top_bug`, which the §7 rule 7
self-control immediately disqualifies: it runs `--k-induction-parallel`, and the
legacy arm against **itself** produces two different hashes on consecutive runs.
The diff is interleaved whitespace from the forked workers. *Exclude
`--k-induction-parallel` tests from the A/B* — this is the third distinct
per-run artefact class the sweep has hit (§19.3 a synthetic location, §20.3 a
synthetic file name, §21.3 a temp dir encoded as character codes), and the
self-control caught all three.

### 23.4 Phase 1 exit criterion

| suite | declines | byte-identical |
|---|---|---|
| `esbmc-cpp` | assert-fold now folded; residue = `--error-label` only | 327/328 with C (the 1 is §23.3 harness noise) |
| `esbmc` (C) | assert-fold now folded; residue = `--error-label` only | as above |
| `python` | dominant site fixed by #6695 (merged) | 303/303 (stride-15) |
| `esbmc-solidity` | 0 / 26 | 502/510; 8 residuals = #6759, #6760 |
| `jimple` | 0 / 15 | 15/15 |

What is left is `--error-label` (§12.2: `convert_label` turns a matching label
into an `ASSERT(false)` carrying property metadata; invisible to any census that
does not replay `test.desc` flags) and Solidity's eight residuals. The
assert-fold row of §12.3, carried since 2026-08-04, is closed.

## 24. `--error-label` reproduced — the decline residue is now empty (2026-08-08)

§12.2 named two genuine decline sites and called both "candidates for the same
statement-local delegation." §23 closed the assert-fold; this closes the other,
and it is the last one either census found.

`convert_label` turns a label matching `--error-label` into an `ASSERT(false)`
carrying `property`/`comment`/`user_provided` metadata, and makes **that
assertion** the label's target so a `goto` lands on it. The native label arm
detected the shape and `return false`d. It now reproduces it.

### 24.1 Why it outlived the rest

§12.2 already said it: the site fires only under a flag, so it is invisible to
any census that does not replay each test's `test.desc` flags. Both the C++
census (§11) and the whole-suite sweeps developed patches against default flags,
and the arm never appeared. It is the one residue that a *better sample* would
never have found — only reading the legacy function does.

There is a second reason to be careful here, already recorded in `CLAUDE.md`:
ESBMC reports `VERIFICATION SUCCESSFUL` **silently** when the label is absent
from the GOTO program, which is indistinguishable from "label unreachable." That
shows up in the mutant table below.

### 24.2 Mutants

| mutant | what it breaks | caught by |
|---|---|---|
| M1 — arm removed (`return false`) | nothing observable | **nothing** — §23.2's limit again |
| M2 — assertion guard `true` | the error label stops failing | `regression/cbmc/01_cbmc_error-label1` (verdict) **and** the A/B on it |
| M3 — `comment("error label")` dropped | the claim's rendered text | **only** the new `…error_label_01_fail` |
| M4 — `user_provided(true)` dropped | `--no-assertions` stops skipping the claim | **only** the new `…error_label_02` |
| — `property("error label")` dropped | *nothing* | nothing, and nothing can |

The metadata splits three ways and only review caught that: `--goto-functions-only`
renders `comment` but not `property` or `user_provided`, so one test cannot pin
all three. `property` is genuinely unobservable — every reader compares it
against other literals — so the call is kept for fidelity and **is not claimed
to be tested**. `user_provided` needed a second test, because none of the 162
`--error-label`-bearing tests pairs the flag with `--no-assertions`.

M2's A/B also shows the arm is genuinely exercised, which is the reachability
evidence M1 cannot supply. Worth recording from the same run: under M2 the A/B
diverges on `01_cbmc_error-label1` but **not** on `esbmc-unix/github_2513_1`,
because that test's label is not the one it names — `CLAUDE.md`'s
silent-SUCCESSFUL trap, visible here as a test that cannot discriminate anything
about this arm.

Worth noting from the same run: under M2 the A/B diverges on
`01_cbmc_error-label1` but **not** on `esbmc-unix/github_2513_1`, because that
test's label is not the one it names — the silent-SUCCESSFUL trap above, visible
here as a test that cannot discriminate anything about this arm.

### 24.3 A/B

All 162 `--error-label`-bearing tests outside `regression/disabled`: **162/162**
byte-identical. (The first count reported here was 29 — a glob that missed the
nested suite directories, and with them the whole `esbmc-cpp/try_catch/nec_ex*`
cluster, which is the most interesting set because it combines the error label
with the `cpp_catch` legacy delegation.)

### 24.4 A test this patch invalidated

`github_4715_irep2_native_body_goto_rollback_01` existed to pin
`convert_function`'s `targets` rollback, and its stated premise was *"`--error-label`
makes the label handler decline … which is exactly the ordering that leaves the
dangling entry behind."* That premise is now false, so its comment is corrected
rather than left to rot.

Chasing it produced a finding worth keeping: **the rollback is not discriminated
by any test, and was not before this patch either.** Removing
`targets = targets_before` leaves the whole suite green, because the failure mode
is a *dangling iterator read* in `finish_gotos` — latent UB, not a crash, and
not observable without a sanitizer build. Re-pointing the test at another
declining shape would not have fixed that; four candidate decliners were tried
under the mutant and none faulted. Pinning it needs an ASan build, which this
branch does not have.

### 24.5 Phase 1 exit criterion

| suite | declines | byte-identical |
|---|---|---|
| `esbmc-cpp` | **drained** | 327/328 with C (the 1 is §23.3 harness noise) |
| `esbmc` (C) | **drained** | as above; plus 162/162 on the `--error-label` set |
| `python` | dominant site fixed by #6695 (merged) | 303/303 (stride-15) |
| `esbmc-solidity` | 0 / 26 | 502/510; 8 residuals = #6759, #6760 |
| `jimple` | 0 / 15 | 15/15 |

Every decline site either census named is now closed. What remains before the
round-trip can be deleted is **Solidity's eight residuals** (#6759, #6760 —
already established as *not* dispatcher defects) and, more honestly, the
standing caveat of §21.4/§22.6: these censuses sample, and both defects fixed on
this branch were found by reading the legacy code, not by re-running a sweep.
"0 declines" is a statement about the corpus, not a proof about the dispatcher.

## 25. The decline census, finally run properly — and it is zero (2026-08-08)

§24.5 ended on a caveat: *"'0 declines' is a statement about the corpus, not a
proof about the dispatcher"*, and every census before this one was per-suite,
sampled, and — except §12's — run without replaying `test.desc` flags. This runs
the measurement that Phase 1's exit criterion actually asks for.

### 25.1 Method

Every `return false` inside `convert_native_rec` instrumented with one
`fprintf` printing its site index and `get_expr_id(code2)` — the same technique
§18.6 describes, but over **all 18 sites at once** (17 genuine plus the
`code_block` cascade) rather than the 21 of the original C++ census, and across
four frontends in one sweep. Stride-9 over
`regression/esbmc`, `regression/esbmc-cpp/cpp`, `regression/python` and
`regression/jimple`: **778 tests**, each replayed with its own `test.desc`
flags, `KNOWNBUG`/`FUTURE`/`THOROUGH` skipped. Solidity is excluded — it does
not run on this machine (§15.1's `solc` blocker, still live).

### 25.2 Result: one site, then none

| | tests declining | sites firing |
|---|---:|---|
| before | **49 / 778** | `code_assert` — side-effecting guard (+ `code_block` cascade) |
| after the assert delegation | **1 / 778** | `code_assume` — same shape |
| after the assume delegation | **0 / 778** | — |

Both are the same one-line story: `convert_assert`/`convert_assume` hand a
side-effecting guard to `remove_sideeffects`, which owns temp-symbol machinery
this dispatcher deliberately does not reproduce. The arm `return false`d, which
is a *whole-function* fallback; it now delegates the statement, exactly as the
throw/catch/return arms do. Byte-identical by construction, and measured:
**88/88** on every test the pre-fix census flagged.

In Python this is not a corner: a call in an assert guard is ordinary code, and
`assert double(x) == 6` was taking whole functions to the round-trip.

### 25.3 What the census does and does not establish

It establishes clause 1 of §18.5's two-clause criterion — **zero declines** —
on four frontends, with flags replayed, at a sample size no previous census
reached. Combined with §21-§24's byte-identity numbers (Python 303/303, C/C++
327/328, `--error-label` 162/162), both clauses now hold everywhere they can be
measured on this machine.

It does **not** establish that the dispatcher is complete. The honest bounds,
in order of how much they cost:

- **A decline census is blind to an arm that emits the *wrong thing*.** This is
  the load-bearing one, and review demonstrated it on this very patch: the
  assert and assume arms were the only two missing the `is_if2t` disjunct that
  every sibling carries, so a *side-effect-free* top-level ternary guard sailed
  past the new delegation and emitted `ASSERT c ? a : b` where legacy lowers to
  DECL/IF/GOTO under `--validate-violation-witness`. The census counts declines;
  that arm returns `true`. Reproduced and fixed here (§25.5).
- **The A/B ran on the wrong set to catch it.** 88/88 byte-identical, but those
  88 are exactly the tests that *previously declined* — the set where both paths
  are identical by construction. The statements that were always native have
  never been swept. A full-corpus A/B, and specifically one varying
  `--validate-violation-witness`, `--no-assertions` and `--condition-coverage`
  (the three options these arms branch on, and none of which appears in any
  `regression/python/*/test.desc`), is the missing measurement.
- **Solidity is unmeasured here**, and §18.3 is the standing warning that a
  frontend can reach zero declines and still not reproduce the round-trip.
- **Stride-9 is a sample.** The site this census found fired on 49 of 778 — hard
  to miss. A site firing on one test in ten thousand would not show up.
- **A green census cannot see an arm that should exist but does not.** §23.2's
  M1 again: delegation is behaviour-preserving, so nothing distinguishes
  "delegated" from "declined" except the census itself.

### 25.5 Two defects the census could not have found

Both came out of review of this patch, and neither is a decline:

1. **The missing `is_if2t` disjunct** described above, on the assert and assume
   arms. Fixed here; `…assert_ternary` pins it.
2. **`--no-assertions` aborted ESBMC on the native path** — `assert` under that
   flag is the one native kind that emits *nothing*, and the if-arm guarded its
   then-branch against an empty program but not its else-branch, so
   `y = tmp_y.instructions.begin()` handed `end()` to `make_goto` and
   `compute_target_numbers` asserted. Reproduces on `master` with a one-line
   Python file, so it predates this branch — but delegating side-effecting
   asserts keeps more functions on the native path under that flag, which
   widens the blast radius. Fixed here rather than left, with the guard made
   symmetric.

Neither shows up as a decline; neither shows up in any suite, because **no
`regression/python` test passes `--no-assertions`** and none passes
`--validate-violation-witness`. That gap is worth closing on its own.

### 25.6 Phase 1 exit criterion

| suite | declines | byte-identical |
|---|---|---|
| `esbmc` (C) | **0** (stride-9, flags replayed) | 327/328 with C++; 162/162 on `--error-label` |
| `esbmc-cpp` | **0** (same census) | as above |
| `python` | **0** (same census) | 303/303 (stride-15) |
| `jimple` | **0** (same census) | 15/15 (§20) |
| `esbmc-solidity` | not measurable here | 502/510; 8 residuals = #6759, #6760 |

Clause 1 is met on every frontend measurable here. Clause 2 is **not** fully
measured: the byte-identity numbers cover previously-declining tests, per-patch
gates, and per-suite samples — §25.3's second bound says what is missing. The next step is not another census — it is Solidity on CI, and then
the question Phase 1 exists to answer: whether `goto_convert_rec` and the
round-trip can be deleted, which needs the fallback to be provably unreachable
rather than merely unexercised.

## 26. The option-varied A/B — §25.3's missing measurement, and what it found (2026-08-08)

§25.3 named the gap: every byte-identity sweep so far ran on **default flags**,
and the three options these arms branch on —
`--validate-violation-witness`, `--no-assertions`, `--condition-coverage` —
appear in **no** `regression/python` `test.desc` at all. This runs the A/B under
each of them.

### 26.1 Result

Stride-17 over `regression/esbmc`, `regression/esbmc-cpp/cpp` and
`regression/python` (411 dirs, 384 comparable), each replayed with its own
`test.desc` flags plus the option under test:

| option set | before | after |
|---|---:|---:|
| default | 384/384 | 384/384 |
| `--no-assertions` | 384/384 | 384/384 |
| `--condition-coverage` | 384/384 | 384/384 |
| **`--validate-violation-witness`** | **140/384 — 244 divergent** | **384/384** |

244 of 384. The default-flag sweeps that have gated every patch in this series
could not see any of it.

### 26.2 One cause, and it is the §25.5 defect again

The native call arms — the `code_assign2t` call-rhs branch and the standalone
`code_function_call2t` arm — gate their arguments on `has_sideeffect` alone,
with a comment asserting that `do_function_call`'s own `remove_sideeffects`
calls are therefore *"no-ops we can skip issuing."* Under
`--validate-violation-witness` that is false for exactly the same reason it was
false for the assert and assume arms: `remove_sideeffects` is entered for a
top-level ternary regardless of side effects, and lowers it to DECL/IF/GOTO so
the `?` column reaches the branching waypoint. The operands the arms hand it
were never stamped, so every instruction of that lowering came out **unlocated**
— 1833 unlocated instructions natively against 1641 under the round-trip on a
single test.

The fix is one disjunct, `|| is_ternary(...)` (a nil-safe `is_if2t`, §26.4), on
the callee and each argument. Review then enumerated the rest, and the tally is
the finding worth keeping: **an arm needs the disjunct exactly when its legacy
counterpart calls `remove_sideeffects` unconditionally**, and by that test seven
arms carried it and seven did not —

| carried it | did not (fixed here) | correctly exempt |
|---|---|---|
| assign lhs, assign rhs, expression, decl init, return, assert, assume | call callee, assign call args, standalone call args, `if` cond, `do`/`while` cond, `for` cond, `switch` value | `while` cond — `generate_conditional_branch` gates on `has_sideeffect` itself |

The four control-flow arms were invisible to the sweep for an incidental reason:
the C frontend wraps a control-flow condition in a `(_Bool)` typecast, so
`expr.id()` is `"typecast"`, not `"if"`. The shape only surfaces where the
ternary is already bool-typed (C++, Python) or in a `switch` value (any
language). That is a property of the frontend, not of the arms — and it is the
sharpest illustration yet of why an enumeration beats a sample: the sweep was
384/384 with four arms still wrong.

### 26.3 A fourth per-run artefact

The `--no-assertions` sweep reported one divergence,
`python/github_4792_fail`, which the §7 rule 7 self-control disqualified: two
runs of the legacy arm against itself gave two hashes. The varying token is
`unpack_<address>_0`, an operational-model temp named from a pointer — distinct
from §21.3's `ESBMC_unpack_temp_<n>`, which the normaliser already covered, and
distinct again from §19.3's synthetic location, §20.3's synthetic file name and
§21.3's character-coded temp dir. Four classes now, all found by the
self-control and none by inspection. **Run the self-control first, always.**

### 26.4 The fix segfaulted before it worked

`is_if2t(e)` is `e->expr_id == expr2t::if_id` — `operator->` on an **empty**
`expr2tc` dereferences null. The callee slot of a `sideeffect2t` function call
is nil for some shapes, so the first cut of this fix crashed ESBMC on 19 C tests
and 3 C++ ones (`github_170`, `align-deref_*`, `github_1220-*`, `github_2389_*`,
…). Every `is_*2t` predicate in the tree has this property, which is why the
codebase pairs them with `is_nil_expr` in most places; the guard is now a named
`is_ternary` helper so its six call sites cannot each forget it. (`is_symbol2t`
at the standalone-call arm's `f.function` is one place the pairing is *not*
made — reachable only for a void call with a nil callee, unobserved, and left
alone here rather than fixed blind.)

Three of those tests pin it: removing the guard segfaults them. The lesson is
narrower than "check for nil" — it is that **the suites caught this and the A/B
did not**, because a crash makes both arms fail and `ab_opt.sh` scores
`SKIP-ERR`. A sweep that skips on non-zero exit is blind to exactly the class of
bug that makes both paths exit non-zero. Run the suites, not only the sweep.

### 26.5 Phase 1 exit criterion

| suite | declines | byte-identical |
|---|---|---|
| `esbmc` (C) | 0 (§25) | 384/384 × 4 option sets, plus §21-§24's sweeps |
| `esbmc-cpp` | 0 (§25) | as above |
| `python` | 0 (§25) | as above |
| `jimple` | 0 (§25) | 15/15 (§20) |
| `esbmc-solidity` | not measurable here | 502/510; 8 residuals = #6759, #6760 |

384/384 × 4 option sets is a *sampling* result over three suites, plus
**161/161** over `regression/witnesses{,_validate}` — the suites that run
`--validate-violation-witness` natively, and the obvious place to have looked
first. It is not a proof, and §26.2 is the reason to say so plainly: the sweep
was already 384/384 while four arms were still wrong.

Outstanding, in order:

1. **Solidity**, which needs CI — §18.3's warning that zero declines does not
   imply reproduction still stands there, unmeasured.
2. **The option space is bigger than three.** Three were swept because three are
   what these arms branch on *today*. Any future arm that reads an option
   inherits the same obligation.

(Item 1 of the original list — a pre-existing `do`/`while` location bug — is
closed in §27.)

## 27. The `do`/`while` condition location — §26.5's open item, closed (2026-08-08)

§26.5 filed one defect rather than fixing it in passing: the native `do`/`while`
arm reported the *statement's* column where `convert_dowhile` reports the
condition's. It is closed here.

### 27.1 The mechanism, which is §21.2's again

`convert_dowhile` saves `code.op0().find_location()` **before** lowering, so the
loop-back branch is located at the condition. The native arm has no operand to
read — IREP2 values carry no location — so it substituted `here`, the statement
location, reasoning that `restore_value_locations` would have stamped exactly
that onto the operand.

That reasoning holds for every value kind but one. `stamp_value_locations` only
writes onto a node that *lacks* a location, and `if2t` is the single value kind
carrying its own through `migrate_expr` (irep2_expr.h:786) — the same fact §21.2
turned on. So a ternary condition arrives already located at the `?` column,
`find_location()` returns that, and the substitute was wrong:

```cpp
bool a, b, c;
int main() { do { a = true; } while (c ? a : b); return 0; }
```

| | loop-back branch |
|---|---|
| native, before | `line 16 column 12` → the `do` |
| round-trip | `line 16 column 12` → the `?` |

**On default flags** — no option needed. C hides it because the frontend wraps a
control-flow condition in a `(_Bool)` typecast, so the top node is not the
ternary; C++ and Python, whose ternaries are already bool-typed, do not.

`convert_dowhile` is the only legacy converter that calls `find_location()`, so
this arm is the only one with the substitute, and the fix is local: read the
ternary's own location when it has one, keep the existing nil-vs-empty fallback
otherwise.

### 27.2 Verification

`…dowhile_ternary_loc` pins the column and fails when the fix is reverted. The
`--validate-violation-witness` and default sweeps stay 384/384; C 1679/1682 and
C++ 752/755, pre-existing failures only.

### 27.3 What this closes, and what it says

It closes the last item this branch found and did not fix. Worth recording that
**three separate defects on this branch trace to one fact** — `if2t` is the only
value-level kind carrying a location — and each was found a different way: §21.2
by a sweep, §26.2 by an enumeration against the legacy source, §27 by review of
a patch fixing the other two. The fact is now cited at all three sites, which is
the cheapest available defence against a fourth.

## 28. Can the round-trip be deleted? Not yet — and the sample said otherwise (2026-08-08)

Phase 1 exists to answer one question: whether `goto_convert_rec` and the
whole-body round-trip can go. §25's census said **0 declines / 778 tests**, which
reads like yes. It is not, and the gap is the sampling caveat §25.3 wrote down
and this section collects on.

### 28.1 The full C/C++ corpus, not a stride

Same instrumentation, every `return false` in `convert_native_rec`, but over the
**entire** `esbmc`, `esbmc-cpp/cpp`, `esbmc-cpp11/14/17`, `cbmc`, `esbmc-unix`,
`floats`, `k-induction` and `jimple` corpus — **3 355 tests**, flags replayed.

| census | tests | declining |
|---|---:|---:|
| §25, stride-9 over four frontends | 778 | **0** |
| here, full C/C++ corpus | 3 355 | **1** |

One test: `regression/cbmc/01_cbmc_for4`. Stride-9 missed it because it is one
test in 3 355 — precisely the "one-in-ten-thousand would not show up" case
§25.3 named, arriving one section later than the warning.

### 28.2 The 15 sites, split

Reading them rather than sampling them, the sites divide cleanly:

**Cascade** (7) — fire only because a nested `convert_native_rec` returned
false, so they can never *originate* a decline: `code_block`, the `do`/`while`
body, the `for` init and iteration, the `switch` body, `switch_case`, `label`.

**Origin** (8) — a condition on the statement itself. Reachability, probed:

| site | condition | reachable? |
|---|---|---|
| `for` iteration (8) | sub-conversion left the destructor stack changed | **yes**, default flags — `for (i = 0; i < 3; acall(i++))` |
| `switch_case` (11) | sub-statement emitted nothing | **yes**, `--no-assertions` — `case 1: __ESBMC_assert(0, …);` |
| `label` (15) | sub-statement emitted nothing | **yes**, `--no-assertions` — `L: __ESBMC_assert(0, …);` |
| `code_expression` (2) | code operand that is not `cpp-throw` | not reached; try/catch/throw does not produce one |
| `code_expression` (3) | statement location nil or empty | not reached |
| `code_decl` (4) | symbol absent from the context | not reached |
| `for` condition (6) | `f.cond` nil | not reached — both C and C++ frontends synthesise a condition for `for(;;)` |
| `break` (12) / `continue` (13) | outside a loop or switch | not reached; ill-formed in C, so no frontend emits it |

"Not reached" is an honest negative from a constructed probe, not a proof: §21.2,
§26.2 and §27 were all found by reading rather than probing, and the same could
be true here. But five of the eight are defensive guards whose comments already
say so, and two (`break`/`continue` outside a loop) are ill-formed input.

### 28.3 The answer

**No.** Three origin sites are demonstrably reachable, and the fallback runs on
each. Two of the three need `--no-assertions` — the flag §25.5 recorded as
absent from every `regression/python` `test.desc`, and which has now produced a
crash (§26.4) and two live declines.

What deleting the round-trip actually requires, in order:

1. **The `for`-iteration site.** `remove_sideeffects` on an iteration statement
   containing a call with a side-effecting argument allocates a temp, whose
   `convert_decl` pushes a `code_dead`; the arm's destructor-stack invariance
   check then trips. `convert_for` handles that push; the native arm declines
   rather than assume it can. This is the only site reachable on default flags.
2. **The two "emitted nothing" sites**, which exist because `convert()` appends
   a SKIP where `convert_native_rec` may emit nothing — the same asymmetry that
   produced §26.4's crash. Fixing it at the source (make the native arms match
   `convert()`'s postamble) closes both at once and removes a whole hazard
   class rather than two symptoms.
3. **Solidity**, still unmeasured here.

Until then the fallback is load-bearing, and "0 declines" should be read as what
it is: a statement about a corpus, at a stride.

## 29. The three live sites closed — the fallback is no longer reachable from the corpus (2026-08-08)

§28 answered "can the round-trip be deleted?" with *no*, and named the three
reachable origin sites. All three are closed here, and the count that matters
moves from 1/3355 to **0/3355**.

### 29.1 The `for` iteration — the check was stricter than legacy

The arm declined when converting the iteration statement left
`targets.destructor_stack` larger than it found it. `convert_for` (goto_convert.cpp)
does no such thing: it converts the iteration, never touches the destructor
stack, and leaves any `code_dead` a declaration pushes for the **enclosing
block** to unwind — which is exactly what the arm's own comment already says
about the *init* leg three lines above. The check was symmetry with the body
leg, not a requirement.

`for (i = 0; i < 3; acall(i++))` leaks one such entry: `remove_sideeffects`
declares a temp for the side-effecting argument, and `convert_decl` pushes its
dead. Dropping the check admits it; the A/B is byte-identical on both the
reduced case and `cbmc/01_cbmc_for4` it came from.

The remaining failure legs now **delegate** rather than `return false`, matching
the body leg directly below them. That asymmetry — one leg of an arm taking a
whole-function fallback while the next takes a statement-local one — was worth
removing on its own.

### 29.2 The two "emitted nothing" sites — fixed at the source

§28.3 predicted these should be closed together, at the asymmetry rather than
the symptoms, and that is what happened. `convert()` ends with: *if the
accumulated program is still empty, add a SKIP at this statement's location*
(goto_convert.cpp). `convert_native_rec` had no counterpart, so the
`switch_case` and `label` arms — both of which need an instruction for their
target to sit on — declined when their sub-statement emitted nothing.

One helper, `ensure_nonempty`, reproduces that postamble, and both arms call it
where they previously bailed. `code_assert2t` under `--no-assertions` remains
the only native kind that can emit nothing, and a block already carries its own
SKIP, so the statement whose location is used is always one
`statement_location` knows.

### 29.3 Result

| census | tests | declining |
|---|---:|---:|
| §25, stride-9, four frontends | 778 | 0 |
| §28, full C/C++ corpus | 3 355 | 1 |
| here, full C/C++ corpus | 3 355 | **0** |

The origin-site count drops from 9 to 6 (§28.2's table put `break` and
`continue` on one row), and none of the 6 had been reached by any probe.
`return false` sites in `convert_native_rec`: **15 → 12**.

### 29.4 What this does and does not license

It does **not** license deleting the round-trip. What changed is the *evidence*:
before, one corpus input demonstrably needed the fallback; now none does. The
five remaining origin sites are unreached-by-probe, which §28.2 already flagged
as an honest negative rather than a proof — and this branch has produced three
defects found by reading rather than probing.

The gap between "no corpus input reaches it" and "no input can" is the whole of
what is left, and closing it is a different kind of work: a reachability
argument per site, of the kind `CLAUDE.md`'s Mode C prescribes, not another
sweep. Two of the five (`break`/`continue` outside a loop) are ill-formed input
and should simply be asserted rather than handled; the other three are defensive
guards whose comments already say so.

Also still open, unchanged: **Solidity**, which needs CI.

### 29.5 A note on what these tests can pin

Both new tests are verdict tests, and neither discriminates the change — a
delegation and a decline produce byte-identical programs, so §23.2's M1 limit
applies to all three sites. The census is the instrument; the tests pin the
verdict under `--no-assertions` and on a side-effecting for-iteration, which
nothing else in the C suite did, and guard the shapes against a future change
that is *not* behaviour-preserving.

## 30. The Python corpus censused in full — and the lesson repeats (2026-08-08)

§28 censused the full C/C++ corpus because §25's stride-9 had missed a live
site. Python was left at stride-9. This runs it in full, and the outcome is the
same shape of result one rung down.

### 30.1 Result

| census | tests | declining |
|---|---:|---:|
| §25, stride-9 across four frontends | 778 | 0 |
| here, full `python` + `numpy` corpus | **5 305** | **5** |
| after the fix | 5 305 | **0** |

All five fire at one origin site: `code_expression2t` whose statement carries no
location. The arm needs one for the `OTHER` it emits, so it declined — a
*whole-function* fallback.

`regression/python/print1_expr_fail` reduces it to two lines:

```python
a = nondet_int()
print((a + 1) * 2)
```

`print(a)` and `print(1)` do not reach it; a **compound** argument does.

### 30.2 Why probing missed it

§28.2 marked this site "not reached" on a constructed probe, and §29.4 warned
that such a negative is not a proof. It took nine hours of that warning to cash
out: the site is reachable from ordinary Python, in five corpus tests, and
neither the stride-9 sample nor a hand-written probe found it. The probe failed
because I guessed at C shapes — an unlocated expression statement is a *Python
frontend* artefact, and nothing about the site's guard says so.

That is now the third time on this branch that the instrument found nothing and
the defect was real (§21.4's blind spot, §26.2's four arms, this). The pattern
is consistent enough to state as a rule: **a negative from a probe is worth
less than a negative from the full corpus, and both are worth less than a
reachability argument.**

### 30.3 The fix

Delegate, not reimplement. The arm hands the statement to `convert_expression`
exactly as the shapes above it do, which is byte-identical by construction and
keeps the surrounding statements native. Working out what location legacy
actually gives that `OTHER` — the arm's comment says "at an enclosing block" —
is a question the delegation makes moot, and guessing at it would have risked
the very byte-identity the delegation guarantees.

### 30.4 Where the fallback now stands

| corpus | tests | declining |
|---|---:|---:|
| C / C++ / Jimple (§29) | 3 355 | 0 |
| Python / numpy (here) | 5 305 | 0 |
| Solidity | — | not measurable here |

`return false` sites: **12 → 11**; origin sites **6 → 5**. Every site reachable
from either corpus is closed. The five that remain — an expression statement
with a non-`cpp-throw` code operand, a `code_decl2t` whose symbol is absent from
the context, a nil `for` condition, and `break`/`continue` with no target — are
unreached by 8 660 corpus tests and by probe, which after §30.2 should be read
as *evidence*, not proof.

## 31. Reachability arguments for the five remaining sites (2026-08-08)

§29.4 said the gap between "no corpus input reaches it" and "no input can" needs
a per-site argument rather than another sweep, and §30.2 said a probe's negative
is the weakest evidence available. Here are the arguments, from reading the
producers and the legacy counterparts rather than from probing.

### 31.1 Three sites where the legacy path aborts

| site | native guard | legacy counterpart |
|---|---|---|
| `code_break2t` | `!targets.break_set` | `convert_break`: `log_error("break without target"); abort();` |
| `code_continue2t` | `!targets.continue_set` | `convert_continue`: `log_error("continue without target"); abort();` |
| `code_decl2t` | symbol absent from context | `convert_decl`: `assert(s != nullptr);` |

This is the strongest class of argument available short of a formal proof, and
it does not depend on any corpus: **the fallback at these three sites cannot
change an outcome, because the path it falls back to terminates.** A run that
reaches any of them produces no verdict either way. They are not dead code in
the compiler's sense — they are unreachable *in any run that produces a result*.

The right end-state for all three is the legacy diagnostic, not a fallback: the
native arm should abort with the same message rather than route to a converter
that will. That is a deletion, so per `CLAUDE.md` it needs its own C-Dead proof
and its own PR; recorded here rather than done in passing.

### 31.2 The nil `for` condition has no producer

Every construction path for a `code_fort` sets `cond()`:

- `clang_c_convert.cpp` initialises `exprt cond = true_exprt();` and overwrites
  it only when the AST has one, so `for(;;)` gets `true` — which is why probing
  it found nothing, and this time the probe agrees with the reading.
- `clang_cpp_convert.cpp` takes the same shape.
- The two internal builders in `builtin_functions.cpp` (array initialisation and
  `cpp_new`'s element loop) both assign `loop.cond()` explicitly.
- Python desugars `for` into `while`, so it produces no `code_for2t` at all.

Unverified: Jimple and Solidity, whose loop lowering was not read. So the claim
is "no producer in the C/C++/Python path", not "no producer".

### 31.3 The expression-statement code operand is the one genuinely open site

`code_expression2t`'s operand becomes code-typed only through the round-trip's
own lowering of a nested `side_effect_exprt("cpp-throw")` to `codet("cpp-throw")`
— which the arm handles. What it declines is *any other* code statement in that
position, and nothing was found that produces one. But unlike §31.1 this rests
on a survey of producers rather than on legacy terminating, and unlike §31.2 the
guard is open-ended (`op.statement() != "cpp-throw"`) rather than a single
field. It is the site to attack first if the round-trip is to go.

### 31.4 Summary

| # | site | argument | strength |
|---|---|---|---|
| 3 | `break`, `continue`, `decl`-symbol | legacy aborts or asserts | **strong** — corpus-independent |
| 1 | nil `for` condition | no producer in C/C++/Python | medium — Jimple/Solidity unread |
| 1 | expression code operand | no producer found | weak — open-ended guard |

Together with §29 and §30 (0 declines over 8 660 corpus tests across four
frontends), this is the state of the case for deleting `goto_convert_rec`. It is
not yet a proof, and the honest summary is that **three of the five sites can be
turned into aborts today, one is very likely dead, and one needs real work** —
plus Solidity, still unmeasured here.

## 32. Option F re-scoped: the spelling domain is 83 values, not an enum (2026-08-08)

§16.3 left Phase 0 with one open question and a revised sizing: *"add one
excluded field to two kinds, repoint one reader, run the suite."* Phase 2 is
gated on that spike, so before spending §10's estimated days on it, the premise
is worth checking. It does not hold.

### 32.1 The measurement

`#cpp_type` is written from five places. The clang C frontend's builtin-type
switch (`clang_c_convert.cpp`) alone assigns **83 distinct spellings**:

| class | count | examples |
|---|---:|---|
| ARM SVE / vector builtin names | **56** | `__clang_svint32x4_t`, `__clang_svbfloat16x2_t`, `__SVCount_t` |
| C/C++ scalar spellings | 27 | `signed_char`, `unsigned_long_long`, `char8_t`, `wchar_t`, `__int128`, `_Float16`, `bool`, `void`, `_ptrmem`, `__intcap` |

Plus the other writers: the Solidity frontend sets `bool`, `void`,
`signed_char`, `unsigned_char`; the Python frontend sets `char`, `float`,
`double`, `long_double`.

### 32.2 What that does to the design

§5.2's Option F is `enum class c_spelling` on `signedbv_type2t` /
`unsignedbv_type2t`. Two things in the measurement contradict its shape:

1. **The domain is not small or closed.** Two thirds of it is ARM SVE builtin
   names, which track a vendor extension and grow with LLVM. An enum over them
   is a maintenance liability, and they are exactly the values that carry no
   semantics for the one semantics-bearing reader.
2. **The values do not live on two kinds.** `bool`, `void`, `float`, `double`,
   `long_double`, `_ptrmem` and the 56 vector names are set on types that are
   not `signedbv`/`unsignedbv`. A field on those two kinds carries the 27-value
   scalar subset at best, and the rest still needs the `irept` key — so W3 is
   not removed, which is the entire point of B-4.

### 32.3 The re-scope this implies

The split the measurement suggests, and which §5.2 did not consider:

- **Semantics vs presentation is a real seam here, and it falls along the same
  line.** `clang_cpp_adjust_expr`'s catch-matching — §5.2's argument for why
  `#cpp_type` is semantics, not presentation — consumes scalar spellings. The
  56 vector names reach only `cpp_expr2string` and `goto2c/expr2c`, which are
  presentation. So the typed field only has to carry the scalar subset for the
  semantic reader; the rest can stay a string, or move to a presentation-only
  channel.
- **That makes Phase 0's question 3 the wrong first question.** Verdict and
  counterexample-text parity over `esbmc-cpp` matters, but only after the field
  covers a domain it can actually represent. The first question is now: *does
  catch-matching ever see a non-scalar spelling?* If no, Option F applies to a
  27-value subset on more than two kinds, and B-4 is a partial removal rather
  than a removal. If yes, Option F does not close B-4 at all.

**This does not re-open the §16 conclusions.** The two design risks §16.1/§16.2
retired — the field staying out of equality and hashing, and spellings surviving
canonicalisation — are unaffected; they were about the *mechanism*, and the
mechanism is sound. What changes is the *scope* the mechanism has to cover, and
therefore whether it closes B-4 or only shrinks it.

Recorded rather than acted on: this is a plan correction, and the plan's own
gate (§Phase 0, "a recorded answer either way") is what it feeds.

## 33. What catch-matching actually sees: four spellings (2026-08-08)

§32.3 reformulated Phase 0's first question as *"does catch-matching ever see a
non-scalar spelling?"* — because the answer decides whether Option F closes B-4
or only shrinks it. Measured here.

### 33.1 Method and result

One `fprintf` at the single `type.cpp_type()` read in
`clang_cpp_adjust_expr`'s exception-id builder — the semantics-bearing reader,
and the only one §5.2's argument rests on — run over every C++ suite:
`esbmc-cpp/cpp`, `esbmc-cpp11/14/17/20/23` and `esbmc-cpp/try_catch`, **949
test directories**, `test.desc` flags replayed.

**Four distinct spellings reach it, on 94 tests:**

```
double   float   signed_char   signed_int
```

Four of the 83 in §32.1, all scalar, and **no vector name**. `bool`, `void`,
`char8_t`, `__int128`, `_ptrmem` and the 56 SVE names never arrive.

### 33.2 What this does and does not settle

It settles the *shape* of the answer: the semantic reader consumes a tiny
scalar subset, so a typed field carrying the scalar spellings serves it. The
remaining 79 values reach only `cpp_expr2string` and `goto2c/expr2c`, both
presentation.

It does **not** settle reachability, and the argument I expected to close it is
not available. I went looking for a spec-level prohibition — sizeless SVE types
being ineligible as exception objects would make the 56 vector names
unreachable *by construction* rather than merely unobserved. The ACLE documents
sizeless-type restrictions on struct/union/class members, `sizeof`/`_Alignof`
operands and array element types, but **no restriction on throw-expressions or
catch parameters**. So the vector names are unobserved over 949 tests, which
after §30.2 is evidence and not proof.

#### 33.4 Superseded in part (2026-08-17)

§33.3's "go for the scalar subset, not a B-4 closure" reads as though the
semantics half can be taken now and the presentation half deferred. Measurement
in `scope-clang-c-irep2.md` §102 shows the two are coupled: the printers need the
spelling to tell `char` from `int8_t`, which is the same question catch-matching
asks, so a field carrying only the four catch-matching spellings does not serve
them. The split, the options and the one measurement that decides between them
are now in **`scope-c-spelling-carriage.md`**.

## 33.3 Consequence for Phase 0

The go/no-go the phase asks for, with what is now known:

- **Go, for the scalar subset.** A typed field on the kinds that carry scalar
  spellings serves the one semantic reader, and §16's mechanism conclusions
  (excluded from `fields`, no interning) hold.
- **Not a B-4 closure.** The 79 presentation-only spellings still need a
  carrier, so `#cpp_type` survives unless they move to a presentation channel
  of their own — which is a second, separable piece of work that §5.2 did not
  scope.
- **Sizing.** §16.3's "add one excluded field to two kinds, repoint one reader"
  is right *for the semantic half* and wrong for B-4 as a whole. The honest
  estimate splits: days for the semantic half, unscoped for the rest.

The remaining risk is the one §33.2 names — that a spelling outside the four
reaches catch-matching on input the corpus does not contain. Cheapest guard:
assert on an unexpected spelling in the typed-field prototype and let the suite
say so, rather than trying to enumerate the domain up front.

## 34. The break/continue equivalence §31.1 assumed (2026-08-08)

§31.1 argued three fallback sites can become aborts because their legacy
counterparts abort. That argument has a premise it did not state: **the native
arms must set `targets.break_set` / `continue_set` wherever legacy does.** If
native ever left one unset that legacy would set, the decline is a *safety net*
and replacing it with an abort would break working programs. Established here.

### 34.1 The four set points correspond, and so does their ordering

Both paths establish loop targets in exactly four places, and — the part that
matters — both do it **before** converting the body a `break`/`continue` could
appear in:

| construct | native: set | native: body | legacy: set | legacy: body |
|---|---:|---:|---:|---:|
| `while` | 1125-1126 | 1136 | 1434-1435 | 1439 |
| `do`/`while` | 1205-1206 | 1213 | 1503-1504 | 1508 |
| `for` | 1351-1352 | 1360 | 1357-1358 | 1370 |
| `switch` | 1435 (break only) | 1440 | 1599 (break only) | — |

`switch` sets `break` and deliberately leaves `continue` alone on both sides —
legacy says so in a comment (*"continue stays as is"*) — so a `continue` inside a
switch inside a loop keeps the enclosing loop's target either way. The restores
correspond too: `break_continue_targetst` / `break_switch_targetst` saved at
entry and restored at the matching point in all four arms.

No other native arm establishes or clears a loop target. `block`, `label` and
`switch_case` recurse with whatever `targets` holds, as their legacy
counterparts do.

### 34.2 What follows

The premise holds, so §31.1's argument stands: reaching those three sites means
the legacy path aborts, and the fallback cannot rescue anything.

**The change is still not made here, deliberately.** Converting the sites to a
direct abort has no functional gain — both paths abort — and a real downside if
this enumeration missed a path: today's behaviour degrades to legacy's
diagnostic, an abort degrades to a crash on a program that might have worked.
The asymmetry says wait. The change becomes forced, and safe, at the moment the
fallback is deleted, which is when the enumeration is load-bearing anyway.

Recorded so the premise does not have to be re-derived then.

## 35. The branch validated against every local suite (2026-08-09)

Every section from §21 on gated on the suites the change plausibly touches — C,
C++, Python subsets. That leaves a gap worth closing before review: this branch
edits `goto_convert_functions.cpp`, which every frontend goes through, and CI
has not run on it (the checks have been queued since the first push). So the
remaining suites were run locally.

### 35.1 What had not been run, and the result

`esbmc-cpp11/14/17/20/23`, `jimple`, `cstd`, `esbmc-unix2`, `esbmc-old`,
`goto-binary`, `goto-transcoder`, `ir-ra` — **1 022 tests, 8 failures**. Plus
the unit suite: **663/663**.

All eight fail identically on the merge-base (`9a3d7e8a6c`) with only the four
changed files reverted, so **none is a regression**:

| test | suite |
|---|---|
| `ra-fmod-inf-nan`, `ra-log-nan`, `ra-pow-nan` | `ir-ra` |
| `ra-interval-lift-mul-rdn-both-tracked`, `…-rup-both-tracked-single` | `ir-ra` |
| `builtin-template`, `builtin-template-fail` | `esbmc-cpp14/template` |
| `cbmc_fpclassify` | `goto-transcoder` |

Five of the eight are floating-point/NaN or interval-rounding cases, which is
the profile of a known macOS-local divergence rather than anything this branch
could reach.

### 35.2 The measurement that would have caught a regression, and did not

This is a negative result, and worth recording as one: running the suites a
change *does not obviously touch* found nothing, on a branch where running the
suites it does touch had already found nothing. That is the expected outcome
and it is still worth the hour — the alternative was shipping fourteen commits
to a shared converter with three of five frontends unexercised.

The branch's local validation now stands at: C 1 681/1 684, C++ 752/755, the
above 1 014/1 022, unit 663/663, Python subsets clean, and byte-identity sweeps
over four option sets — all residual failures confirmed pre-existing against the
merge base. What is still unrun anywhere is **Solidity**, and CI.

## 36. Option F's premise, and the inversion it produces (2026-08-09)

§32 and §33 measured the `#cpp_type` domain and what catch-matching consumes.
Read back against §5.2, where Option F is argued, they do something sharper than
re-scope it: **they refute one of the two measurements the option rests on, and
they invert which frontend it fits.**

### 36.1 The refuted premise

§5.2 offers two measurements. The first — Solidity's classification is already
an `enum class SolType`, stringified only to cross the `irept` boundary — holds,
and I re-read it to check.

The second does not:

> *"`#cpp_type`'s value domain is the C type-keyword set … a `c_type` variable
> drawn from the same finite vocabulary. It is an enum wearing a string."*

The `c_type` variable is assigned from LLVM's builtin-type switch, not from a
type-keyword vocabulary. §32.1 counts **83 distinct values, 56 of them ARM SVE
builtin names** that track a vendor extension and grow with the toolchain. That
is not an enum wearing a string; it is a string doing string work. §5.2 now
carries the correction inline, because a design section that states a false
measurement is worse than one that states none.

### 36.2 The inversion

§5.2 hedges against one failure mode — *"if the equality asymmetry proves
unmanageable, fall back to Option B for Solidity only"* — and prescribes a split
where **C/C++ keeps Option F and Solidity falls back.**

Both halves are wrong, in opposite directions:

| | §5.2 expected | measured |
|---|---|---|
| the risk | equality/hashing asymmetry | **retired** by §16.1/§16.2: omit from `fields`, declare `excluded_field_bytes` |
| the obstacle | — | **domain openness**, §32.1 |
| C/C++ | Option F fits | **does not** — 83 values, 56 vendor-extension names, on kinds beyond the two |
| Solidity | falls back to B | **Option F fits best** — `SolType` is already a closed enum (`solidity_grammar.h:484`) |

So the split B-4 should take is the mirror image of the one written down: apply
Option F where the domain is genuinely closed (Solidity), and do not try to
force it onto the C/C++ spelling.

### 36.3 What this changes about Phase 2

Phase 2 reads *"land `c_spelling`/`sol_class` as typed fields, repoint the four
readers, delete the `irept` accessors."* Against the measurements:

- **`sol_class` — proceed.** Closed enum, one frontend, a serialization step to
  remove rather than an escape hatch to add. This is the part that was always
  sound, and it is now the part with evidence behind it.
- **`c_spelling` — do not land as specified.** It cannot represent its domain,
  so `#cpp_type` survives and the accessors cannot be deleted. §33 leaves a
  narrower option open — a typed field for the four scalar spellings
  catch-matching actually consumes, with the rest staying a string — but that is
  a *semantics/presentation split*, not the B-4 removal Phase 2 describes, and
  it deserves its own scope document rather than inheriting this one's name.
- **Phase 0's go/no-go is answerable now, without the prototype**: no-go for
  `c_spelling` as scoped, go for `sol_class`. The prototype §16.3 sizes at days
  would confirm a conclusion the measurements already reach, and its remaining
  question (verdict/counterexample parity) only matters for a field that is
  going to land.

Recorded as a plan correction. The next executable piece of B-4 is `sol_class`
on the Solidity kinds — which, being Solidity, needs the CI leg that the rest of
this branch has been waiting on.

## 37. `sol_class` leaves the program — and B-4 has nothing executable left (2026-08-09)

§36.3 named `sol_class` the next executable piece of B-4 and said it "was
always sound". Before writing its scope document, I checked where `#sol_type`
is actually consumed. The answer removes it from the program.

### 37.1 The attribute never crosses a frontend boundary

`#sol_type` is written and read through one pair of helpers
(`solidity_convert.h:68-75`):

```cpp
static void set_sol_type(typet &t, SolidityGrammar::SolType st)
{ t.set("#sol_type", SolidityGrammar::sol_type_to_str(st)); }
static SolidityGrammar::SolType get_sol_type(const typet &t)
{ return SolidityGrammar::str_to_sol_type(t.get("#sol_type").as_string()); }
```

Every file that mentions it — 17 of them — is under `src/solidity-frontend/`.
`grep` across `src/goto-programs`, `src/goto-symex`, `src/solvers`, `src/util`
and `src/irep2` returns **nothing**.

### 37.2 Why that disqualifies it as a B-4 item

B-4 is *"no `#`-attribute legacy escape hatch **into a shared pass**"*, and
§5.2's whole argument for Option F being legitimate rather than a reinstated
escape hatch is that `#cpp_type` reaches `clang_cpp_adjust_expr`'s
catch-matching — a shared, semantics-bearing consumer.

`#sol_type` has no such consumer. It is a frontend talking to itself: it holds
`SolType` on both sides and stringifies only because `irept` cannot hold an
enum. Nothing post-migration reads it, so **a typed field on `type2t` would
carry data no consumer wants**. §5.2's phrasing was right that this "removes a
serialization step rather than adding an escape hatch" — the refinement is that
the step to remove sits *inside* the frontend, and removing it needs no IREP2
change at all.

So the work is real but it is Solidity-frontend cleanup: stop routing
frontend-internal state through `irept`. It should be filed as such, and it does
not need the CI leg this branch has been waiting on, because it does not need
`type2t` to change.

(One caveat, stated rather than hidden: a *generic* `#`-attribute walk would
still see the key. Option D seamed `#member_name` and `#cpp_type` behind typed
`irept` accessors for that reason and did not seam `#sol_type`, which is
consistent with it never having mattered outside the frontend.)

### 37.3 What is left of B-4

| item | status |
|---|---|
| `c_spelling` | **no-go as scoped** (§36) — domain is open, 83 values |
| `sol_class` | **not a migration item** (§37) — no consumer outside the frontend |

B-4 as written has **no viable executable content left.** That is not a failure
of the work; it is the measurements arriving. Two things survive it:

1. The **semantics/presentation split** §33 left open — a typed field for the
   four scalar spellings catch-matching consumes, the rest staying a string.
   Real, smaller than B-4, and needing its own scope document and name.
2. The **Solidity frontend cleanup** above, which is not this program's.

Phase 2 should be struck from the phase list rather than left as a gate on
Phases 5-9, which §6 already notes are independent of B-4. The program's
executable frontier is therefore Phase 3 (the Python flip) and Phase 4 (extract
the construction kit) — neither of which is blocked on anything measured here.

## 38. Phase 4's kit already exists (2026-08-09)

Phase 4 reads: *"Before touching a second frontend, factor what Python learned
into shared helpers: the width-reconciliation idiom
(`c_implicit_typecast_arithmetic` on `expr2tc`), the resolved-source `ns.follow`
pattern, the operand-surgery recipe. Without this, four frontends re-derive the
same lessons at four times the cost."* Checked before executing it, as §36 and
§37 were. Two of the three are already shared; the third is not code.

### 38.1 Width reconciliation — shared, with the IREP2 overload, already used

`c_implicit_typecast_arithmetic` lives in **`src/util/lang/c_typecast.h`** — a
shared location, not a frontend — and is declared **twice**: the legacy
`exprt &` form and

```cpp
bool c_implicit_typecast_arithmetic(expr2tc &expr1, expr2tc &expr2,
                                    const namespacet &ns);
```

Python already calls the `expr2tc` overload directly
(`python_adjust.cpp:403, 454, 489`). There is nothing to extract: the helper
Phase 4 names as its first deliverable is the helper the pathfinder frontend is
using.

### 38.2 The resolved-source follow — shared, and not Python-specific

`namespacet::follow` has a native IREP2 overload in
**`src/util/symtab/namespace.h:21`**, whose own comment states the point —
*"mirroring follow(typet) without the back-migrate → follow(typet) →
forward-migrate detour (hot path)"*. Its users span `goto-programs` (7 files),
`clang-cpp-frontend` (5), `pointer-analysis` (3) and `util/lang` (5), not just
Python. It is already the shared pattern.

What *is* Python-specific is `python_adjust::resolve_source` — but that is the
adjuster's member/index source resolution, a Phase 3 concern, not a
construction idiom another frontend would inherit.

### 38.3 Operand surgery is a rule, not a helper

The third item cannot be extracted because it is not code: *mutate an operand
in place through `Foreach_operand`; never round-trip a resolved subtree through
`migrate_expr_back` → `migrate_expr`, which reverts resolved `member2t`/`index2t`
sources to by-name `symbol_type2t`.* That belongs in prose, and this section is
where the next frontend will look for it.

### 38.4 What Phase 4 actually needs

Not a refactor — a pointer. For whoever opens `scope-jimple-irep2.md`:

| lesson | where it already lives |
|---|---|
| width reconciliation over `expr2tc` | `src/util/lang/c_typecast.h` — use the `expr2tc` overload, not `gen_typecast_arithmetic` on legacy `exprt` |
| symbol-type resolution over `type2tc` | `src/util/symtab/namespace.h:21`, `ns.follow(const type2tc &)` |
| operand surgery | §38.3 — in-place via `Foreach_operand`, never a round-trip |
| the min-promotion trap | `c_implicit_typecast_arithmetic` promotes sub-`int` widths to `int`; sub-`int` numpy dtypes must be narrowed back afterwards |
| `if2t` carries a location | the only value-level kind that does; §21.2, §26.2, §27 are three defects from forgetting it |

**Phase 4 is closed as already-done.** Three phases in a row (2, 4, and B-4's
content) have now turned out to be satisfied or misframed on inspection — the
later phases were written before the work that made them moot, which is the
ordinary fate of a plan that survives contact with its own execution. The
remaining executable phase is **3** (finish the Python flip), and then **5-9**,
whose gate on Phase 2 (Phase 8's text) is void now that B-4 has no content.

## 39. Phase 5 (jimple) is complete — and what Phase 6 inherits (2026-08-09)

`scope-jimple-irep2.md` closed at §31. Every expression kind
`jimple_expr::get_expression` can construct and every statement kind the body
dispatcher can construct is now built natively or left on the migrating default
with a stated reason; twenty-one PRs, all byte-identical, all mutant-checked.
Phase 5 named jimple "the pathfinder for the kit", and the kit it produced is
not the one Phase 4 predicted.

### 39.1 The transferable artefact is a diagnostic table, not code

§38.4 lists five construction lessons and their locations. Jimple added no
sixth: the shared helpers were already adequate, and every slice was mechanical
once the target was understood. What jimple actually produced is a way to tell
whether a migration has been *verified* or merely *not contradicted*.

The gate used throughout was A/B byte-identity of `--goto-functions-only`
(G3) plus a mutant that must change it. The mutant is the load-bearing half,
and it fails silently in five distinct ways:

| An unmoved mutant means | Response | jimple §|
|---|---|---|
| the corpus is thin | write a test | §20, §21.1 |
| the code is unreachable by construction | do not mirror the branch | §22.3, §23.1 |
| a caller downstream re-does the work | test in a position where it does not | §23.2, §30.1 |
| the printer normalises the field away | argue from source; do not claim it measured | §21.2, §24.1 |
| the mutation makes the operation invalid, and the error path is also a no-op | mutate to a valid alternative | §30.2 |

Only the second is a fact about the code. The other four are facts about the
harness, and three of them read exactly like the second if not chased.

Two procedural rules go with it:

- **Census before writing.** Five of jimple's expression kinds occur zero times
  in its corpus. One of them (`nondet`) was migrated before this was known, and
  its byte-identity claim held for nine PRs because nothing executed the
  override (§28). A single census, run once, prices every construct at the
  start.
- **Corrupt the arm, do not delete it** (§27.1). Deletion asks whether an arm is
  *necessary*; a correct migration slice never is, because producing identical
  output is the premise of the gate. Only corruption asks whether it *runs*.
- **Anchor mutants to the native function.** The parallel-method technique
  leaves a near-twin of every override a few hundred lines away, and a
  text-targeted mutant that hits the legacy copy returns a false zero (§26.3).

### 39.2 A program-level defect jimple surfaced

`scope-jimple-irep2.md` §16 blocked its largest slice on whether
`c_typecastt::implicit_typecast`'s two copies agree. They did not:
`do_typecast`'s irept copy folds a cast of a constant and its `expr2tc` copy did
not, so a literal assigned to a differently-typed lvalue folded on one path and
not the other. Fixed in **#6873** with a differential harness in
`unit/util/c_typecast.test.cpp`.

That is not a jimple defect. Every frontend in Phases 6-9 implicit-casts at
assignment, and each would have inherited it. It is the second divergence found
between these independently-written copies, after the `floatbv` omission the
same test file documents — which is itself a standing reason to run the
differential harness whenever either copy is touched.

`scope-coupled-arith-assign-conversion.md` §20 records the seven *structural*
gaps that remain between the two `implicit_typecast_followed` copies. Four are
C++-shaped (references, pointer-to-member, derived-to-base, string-to-array) and
are dormant for jimple and Python but **live for Phase 7 (clang-cpp)**, which
should treat §20.1 as its own pre-flight list.

### 39.3 Next

Phase 6 is **clang-c** (971 mentions, 49 already IREP2). Its first action is the
census §39.1 asks for, not a slice. Phase 3 (the Python flip) remains open and
independent.
