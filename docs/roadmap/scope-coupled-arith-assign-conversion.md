# Scope — the coupled arithmetic + assignment conversion (the `python_adjust` flip blocker)

> **Status: Phases 0-2 discharged against the gates this scope owns
> (2026-07-31, §11-§13); Phase 3 blocked on two foreign mechanisms, not on
> this scope's implementation.**
> **Status: Phases 0 and 1 discharged (2026-07-30); Phases 2-3 not started.**
> This document exists because `docs/roadmap/scope-v1k-adjuster.md` §"Flip gate
> (2026-07-29)" closes that scope with exactly one remaining prerequisite and
> hands it off: *"Next owner: take the coupled conversion effort as its own
> scope, then re-run this whole-corpus census as the flip gate."* This is that
> scope.
>
> §9 records the Phase 0 census and the prototype measurements it enabled; its
> instrumentation and prototype arms were reverted, as Phase 0 requires. §10
> records what Phase 1 shipped, which is **smaller than §3.4 sized it**: two of
> the three pieces §3.3/§9.1 asked for are measured dead on this path and were
> not built.

## 1. What this unblocks, and what it does not

`python_adjust` is the IREP2-native replacement for the legacy
`clang_cpp_adjust` pass on the Python path. It is complete enough to run as the
sole adjuster behind `--python-irep2-adjust-only` (`src/esbmc/options.cpp:214`,
experimental, default off), and every *structural* gap the adjuster scope
enumerated is closed or refuted. One it did not enumerate has since surfaced:
`python_adjust` has no `equality2t`/`notequal2t` arm at all (§9.3). That gap is
not the defect class below and does not block the flip on its own.

The flip to default-on was thought to be blocked on one defect class. Phase 0
found that four of the six flip-gate regressions are a *second*, unrelated
mechanism (§9.4), so clearing this one is necessary but not sufficient. Clearing
it:

- **unblocks** the `python_adjust` flip (`--python-irep2-adjust-only` becomes
  the default; the legacy `clang_cpp_adjust` hop on the Python path goes away);
- **does not** advance §V.1 bars #1, #2 or #4 — those are V.2/W3 and the
  symbol-write boundary, tracked elsewhere;
- **does not** touch C or C++, which keep using `clang_c_adjust` /
  `clang_cpp_adjust` unchanged. The blast radius is the Python suite only.

## 2. The defect, restated precisely

From the flip-gate census (1 108 tests, every 4th directory): 6 genuine
regressions, ~~all one mechanism~~ — **Phase 0 refutes the "all one mechanism"
reading: only two of the six respond to the conversions this scope adds, and
four are a separate defect (§9.4).** The mechanism below is real and is the one
this scope clears. The witness is `builtin_all_nonliteral`, whose normalized
`--goto-functions-only` dumps differ in exactly one line:

```
legacy:   5: ASSIGN element=(_Bool)tmp$5;
hop-off:  5: ASSIGN element=tmp$5;
```

A `_Bool`-typed target receives an unconverted integer. The ill-typed
assignment survives goto-conversion and symex and reaches the solver, where the
destination AST is not the sort the source expects — SIGSEGV in
`smt_solver_baset::convert_assign` (`smt_solver.cpp:366`), or a Bitwuzla
"terms with mismatching sort" abort. This is a **crash class, not a
false-alarm class**; that severity finding is what moved the flag from
"tolerable for an experimental flag" to "does not ship".

**Why the obvious fix is unsafe.** `clang_c_adjust::adjust_assign`
(`clang_c_adjust_code.cpp:175-181`) is two statements:

```cpp
adjust_operands(code);
gen_typecast(ns, code.op1(), code.op0().type());
```

Mirroring only the second line in `python_adjust` fixes `precedence2` **and
makes `neural-net_fail` (`--fixedbv`) report SUCCESSFUL where legacy correctly
reports FAILED** — it masks a real bug. The reason is the first line:
`adjust_operands` has already recursively applied the usual arithmetic
conversions to the right-hand side. Converting only at the assignment seam,
over operands that were never reconciled, changes the stored value. The two
halves are only sound **together**.

## 3. Sizing correction — the conversion engine is already IREP2-native

The adjuster scope sized the operand half as "mirroring
`clang_c_adjust::adjust_expr_binary_arithmetic` (~114 lines) … a multi-PR
effort in its own right". That estimate treats the whole legacy function as
work to be re-implemented. A re-audit of the tree (2026-07-30) finds two
reasons it is an over-estimate, and one reason it is an under-estimate.

### 3.1 `c_typecastt` already has full `expr2tc` overloads

The usual-arithmetic-conversion engine does **not** need building. Every
routine the legacy path uses already has a native IREP2 sibling, and they are
exported as free functions:

| legacy entry point | IREP2 sibling | location |
|---|---|---|
| `c_implicit_typecast(exprt&, typet, ns)` | `c_implicit_typecast(expr2tc&, const type2tc&, ns)` | `c_typecast.h:33` |
| `c_implicit_typecast_arithmetic(exprt&, exprt&, ns)` | `c_implicit_typecast_arithmetic(expr2tc&, expr2tc&, ns)` | `c_typecast.h:43` |
| `implicit_typecast_followed(exprt&, …)` | `implicit_typecast_followed(expr2tc&, …)` | `c_typecast.cpp:784` |
| `get_c_type(const typet&)` | `get_c_type(const type2tc&)` | `c_typecast.cpp:364` |
| `do_typecast(exprt&, const typet&)` | `do_typecast(expr2tc&, const type2tc&)` | `c_typecast.cpp:947` |

These are not stubs — `implicit_typecast_arithmetic(expr2tc&, c_typet)`
(`c_typecast.cpp:490-565`) implements the full promotion ladder including the
array→pointer decay case.

**They are already in use on the Python path.** `python_adjust.cpp:428` calls
the `expr2tc` arithmetic overload today, and so do `python_math.cpp:47`,
`list_comprehension.cpp:299`, `list_mutation.cpp:981`, `python_set.cpp:180,252`
and `builtins.cpp:1490`. So the engine is native, exercised, and no
`migrate_expr` round-trip is involved.

What is narrow is the **guard**, not the engine. `python_adjust.cpp:421-429`
fires only when the node has exactly 2 operands, *both* are `bv`, and their
signedness differs — the round-13 relational fix (#6462). Arithmetic binops and
assignments have **no arm at all**: `adjust_expr`'s dispatch
(`python_adjust.cpp:247-767`) handles `member2t`, `index2t`, `dereference2t`,
`if2t`, `not2t`, `constant_struct2t`, `code_function_call2t` and
`code_cpp_throw2t`, and nothing else.

### 3.2 The complex branch is very likely unreachable on the Python path

Of the 114 lines in `adjust_expr_binary_arithmetic`
(`clang_c_adjust_expr.cpp:428-541`), **lines 435-522 (~88) are the complex
branch** — promotion to `{val, 0}`, component-wise lowering, the `ieee_*`
remap, and `bind_sideeffect_operands`. The scalar path is lines 524-540: follow
both operand types, one `gen_typecast_arithmetic` call, adopt the result type
if both operands agree and are numeric, then `adjust_float_arith`.

The Python converter **already lowers complex arithmetic itself**, before the
adjuster ever runs: `math/complex_handler.cpp:98-110` builds `ieee_mul2tc` /
`ieee_add2tc` over `.real` / `.imag` `member2tc` accesses directly, carrying
`symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode")` as the rounding mode
(`complex_handler.cpp:92`). If no `complex_type2t`-typed binary operation
survives into `python_adjust`, ~88 of the 114 lines are not this scope's work.

**This was a hypothesis with a named discharge, not a finding.** It is now
**confirmed** corpus-wide — see §9.1.

### 3.3 The under-estimate — `adjust_float_arith` is not an id rewrite in IREP2

`clang_c_adjust::adjust_float_arith` works by **mutating the node's id in
place** (`expr.id("ieee_add")`) and then setting a `rounding_mode` sub-irep.
Neither operation exists in IREP2: `add2t` and `ieee_add2t` are distinct
classes (`src/irep2/expr_kinds.inc:58`), nodes are immutable, and the rounding
mode is a constructor **operand**, not an attribute. The IREP2 arm must
therefore *rebuild* the node — `ieee_add2tc(type, lhs, rhs, rm)` — rather than
retype it, and must source `rm` the same way the converter already does
(`c:@__ESBMC_rounding_mode`), or the goto output will not be byte-identical.

The legacy function also carries a `// BUG: setting rounding_mode breaks
migration` comment and an early return for vector types. Do not port the bug;
do check whether the vector arm is reachable from Python at all (it likely is
not — the same census as G0 answers it).

### 3.4 Revised sizing

| half | estimate | basis |
|---|---|---|
| operand-level reconciliation | **1 PR**, ~40-60 lines | scalar path only (§3.2), engine already native (§3.1), node rebuild instead of id rewrite (§3.3) |
| assignment conversion | **1 PR**, ~15-25 lines | `c_implicit_typecast(expr2tc&, type2tc, ns)` at a new `code_assign2t` arm |
| flip + census | **1 PR** | re-run the §6 gate |

So **3 PRs, not "multi-PR effort" in the open-ended sense** — *conditional on
G0*. If G0 shows complex or vector binops do reach `python_adjust`, the operand
half reverts to roughly the original estimate and gains a fourth PR.

**G0 came back clean, so this sizing stands** (§9.1). Two adjustments the census
forced, neither of which adds a PR: the operand half must exclude `pointer` and
`code` operands (§9.2), and it must perform legacy's node-type adoption, which in
IREP2 is a `with_type` rebuild rather than an assignment to `expr.type()`.

## 4. Phased decomposition

Strictly ordered. The ordering is the soundness argument from §2, not a
preference.

### Phase 0 — the reachability census (no code change)

Discharge G0. Instrument `python_adjust::adjust_expr` to log the `type_id` and
`expr_id` of every binary arithmetic node it sees, run the whole `python`
suite, and tabulate. Deliverable: a table of reachable operand type kinds.
Revert the instrumentation.

### Phase 1 — operand-level arithmetic reconciliation — **LANDED, §10**

Add the binary-arithmetic arm to `adjust_expr`, covering the kinds Phase 0
found reachable. Widen the `python_adjust.cpp:421` guard from
"both-bv-different-signedness" to the usual arithmetic conversions, rebuilding
`ieee_*` nodes per §3.3.

**Ships alone.** It must, because it has its own parity gate and because
shipping it *with* Phase 2 would leave no way to attribute a regression to one
half. Landing it alone is safe in the direction that matters: it adds
conversions the legacy path also performs, so it moves the hop-off *toward*
legacy, and the assignment seam stays as under-converted as it is today.

### Phase 2 — the assignment conversion

Add the assignment-conversion arm calling
`c_implicit_typecast(source, target->type, ns)`. Only after Phase 1 has landed
and passed its gate.

**Confirm the node kind first.** ~~A `code_assign2t`-only arm is partly dead on
this path … so the arm must cover both kinds or it will silently skip most
Python source assignments.~~ **Measured and refuted — see §11.** Every
Python-source assignment, plain or augmented, arrives as `code_assign2t`; the
expression-form traffic is `sideeffect_assign2t` (not `sideeffect2t{assign}`,
which is not a class) and is **entirely C operational-model bodies**. A
`code_assign2t`-only arm is therefore the correct scope, and it does see
`precedence2`.

**The `neural-net_fail` check is the acceptance test for this phase**, not a
regression to notice later. It must report FAILED. §9.3 shows the prototype
coupling achieves this.

### Phase 3 — the flip

Make `python_adjust` the sole adjuster; `--python-irep2-adjust-only` becomes
the default with an opt-out, mirroring how the W1-loc keystone shipped
(`--irep2-native-body` → deprecated no-op, `--no-irep2-native-body` the escape
hatch, `src/esbmc/options.cpp:964-975`).

## 5. Gates

| # | Gate | Discharged by |
|---|---|---|
| **G0** | ~~The reachable operand-kind census exists, and the complex/vector claim in §3.2 is confirmed or refuted~~ **DISCHARGED, §9.1** | Phase 0 |
| G1 | ~~`builtin_all_nonliteral` and `chained-comparison2_fail`~~ **`builtin_all_nonliteral` only** produces legacy-identical verdicts under the hop-off; `chained-comparison2_fail` belongs to the second mechanism (§13.1). **Met** | Phase 2 |
| G2 | **`neural-net_fail` (`--fixedbv`) reports FAILED** — the anti-masking gate. **Met** | Phase 2 |
| G3 | ~~3 regressions~~ **`github_4344` only** — the `github_5571` pair is the array-typecast mechanism, not this scope (§13). `lambda15`, `precedence2`, `sum_tuple` stay out of scope per §9.4. **Met** | Phase 2 |
| G4 | Whole-corpus census re-run, 0 attributable divergences | Phase 3 |
| G5 | Dual-solver agreement (Bitwuzla + Z3) on the corpus | Phase 3 |

**Census methodology — inherited, non-optional.** Both prior censuses on this
track were first invalidated by harness artifacts. Reuse the recorded rules:

1. **Skip tests whose `test.desc` already passes the flag** — adding it twice
   makes boost throw `multiple_occurrences` (9 false divergences in the
   flip-gate run).
2. **Count both-paths-no-verdict separately** — differing only in `rc=134` vs
   `rc=139` is pre-existing, not attributable.
3. **Exclude or serialize `--k-induction-parallel` tests** — forked children
   share stderr and the capture garbles; it is UNSTABLE against itself.
4. **Minimum-size guard on captured output** (`< 200 bytes → SKIP`) — both
   sides collapsing to one error line otherwise counts as a *match*.
5. Sample **unbiased and dense**: stride-20 missed a 0.5 % defect rate
   entirely. Directory-order prefixes are biased.

## 6. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Phase 1 lands and Phase 2 never does, leaving the hop-off half-converted indefinitely | Phase 1 is verdict-neutral-or-better by construction; the flag stays default-off until G2-G4 |
| R2 | The conversions change goto bytes for tests that currently pass, on the *default* path | They cannot — `python_adjust` runs only under `--python-irep2-adjust-only` until Phase 3 |
| R3 | G0 refutes §3.2 and the effort triples | Phase 0 is deliberately first and cheap; re-size before committing to Phase 1 |
| R4 | Another masking case exists that `neural-net_fail` does not represent | G3 + G5; and prefer `_fail` tests when sampling — a masked bug only shows on a test that should FAIL |

## 7. Non-goals

- Touching `clang_c_adjust` / `clang_cpp_adjust`, or the C and C++ paths.
- The four flip-gate regressions Phase 0 found to be a *separate* mechanism
  (`chained-comparison2_fail`, `lambda15`, `precedence2`, `sum_tuple` — §9.4).
  They need their own diagnosis and, on the evidence, their own scope; they are
  not per-case stragglers of this one. This supersedes the earlier "do not
  re-triage per case" instruction, which rested on the refuted single-mechanism
  reading.
- V.1a, V.2/W3, V.5, V.6 — different scopes, see
  `docs/roadmap/irep2-migration.md` §V.7 and
  `docs/roadmap/scope-v2-w3-attribute-carriage.md`.
- §V.1 bars #1/#2/#4. This scope moves none of them.

## 8. One-line summary

The engine is already IREP2-native and already called from `python_adjust`;
what is missing is a binary-arithmetic arm and an assignment arm, which are
unsound apart and must land in that order — sized at 3 PRs conditional on a
cheap reachability census that could shrink the first one by ~88 lines.

Phase 0 discharged that census: the ~88 lines are indeed out of scope, the
coupling is confirmed by direct measurement, and the arithmetic arm turns out to
matter for only two tests in the whole suite — but half the flip-gate
regressions are a second mechanism this scope does not clear, so the flip needs
one more scope than it looked like it did.

## 9. Phase 0 — results (2026-07-30)

### 9.1 The census, and G0

`python_adjust::adjust_expr` was instrumented to log `expr_id`, node type and
operand types for the IREP2 counterparts of the eight ids legacy routes to
`adjust_expr_binary_arithmetic` — `+ - * / mod bitand bitxor bitor`
(`clang_c_adjust_expr.cpp:124-130`) — plus the four `ieee_*` variants those
lower to on a floating-point path, then the
whole `python` suite was run under `--python-irep2-adjust-only`, replaying each
`test.desc`'s own flags. 4 396 tests ran; 4 301 produced at least one node;
~5.25 M nodes were logged. The instrumentation has been reverted.

Two methodology notes worth reusing:

- **Probe placement is load-bearing.** `adjust_expr` completes the node's own
  type and then recurses operands *bottom-up*, both **after** function entry. A
  probe at entry measures pre-completion, pre-recursion state, so a
  `symbol_type2t` resolving to `complex` would be invisible — an entry-only
  census does not discharge G0. Both points were logged; they never disagreed,
  but that has to be measured rather than assumed.
- **`--goto-functions-only` is sound here and ~5× faster.** The adjuster runs
  before symex, so skipping the solve yields byte-identical shape sets while
  taking each test off its solve timeout.

Every type kind observed, across all 5.25 M nodes:

| kind | appears as |
|---|---|
| `signedbv`, `unsignedbv` | node type and operands |
| `floatbv`, `fixedbv` | node type and operands |
| `pointer` | operands of `add`/`sub` (and `sub`'s node type, for differences) |
| `code` | one operand of `ieee_div` only — §9.2 |

**`complex_type2t` and `vector_type2t` do not occur at all.** §3.2 is confirmed:
~88 of the legacy 114 lines are not this scope's work, and the `adjust_float_arith`
vector arm is unreachable from Python. `ieee_*` nodes arrive already reconciled —
both sides `floatbv`, operands ordered `rounding_mode, side_1, side_2`
(`irep2_expr.cpp:762`) with the rounding mode an `int32` symbol, as §3.3
predicted — so the arm must not re-lower an already-`ieee_*` node.

**Heterogeneous nodes are 165 of ~5.25 M (0.003 %), in four shapes, confined to
two tests:**

| shape | count | test | note |
|---|---:|---|---|
| `ieee_div floatbv \| signedbv code floatbv` | 160 | 32 `cmath_*` / `complex_*` | §9.2 |
| `mul signedbv \| signedbv fixedbv` | 3 | `neural-net_fail` (`--fixedbv`) | the masking witness |
| `add fixedbv \| fixedbv signedbv` | 1 | `neural-net_fail` (`--fixedbv`) | same |
| `sub unsignedbv \| signedbv signedbv` | 1 | `constants` | node-type only |

So **Phase 1's arithmetic arm has a corpus-wide blast radius of two tests.**
Under the default (floatbv) configuration the converter already emits
homogeneous operands; heterogeneity appears only under `--fixedbv`. A census
sample containing no `--fixedbv` test will report zero heterogeneity and is
worthless for this question.

`constants` is worth noting separately: `E = uint64(2**4 - 1)` yields a
`unsignedbv`-typed `sub` over two `signedbv` operands. Legacy's
`if (type0 == type1 && is_number(type0)) expr.type() = type0;`
(`clang_c_adjust_expr.cpp:532-533`) retypes the node; the hop-off does not. The
work here is a **node-type adoption**, not an operand conversion — and in IREP2
that is a `with_type` rebuild, since nodes are immutable.

### 9.2 Two operand kinds the arm must decline

- **`pointer` is the most common operand kind in the corpus.**
  `add pointer pointer signedbv` alone is ~31 % of all nodes, plus
  `sub pointer pointer signedbv` and `sub signedbv pointer pointer` (pointer
  difference). This traffic is the C operational-model bodies, not Python source.
  Legacy's `is_number` guard already declines to adopt for it; the IREP2 arm must
  reproduce that or it will corrupt pointer arithmetic.
- **`code`-typed operands occur.** `cmath.pi / 2.0` lowers to
  `IEEE_DIV(pi, 2.000000)` where `pi` is a bare `code`-typed symbol — 160 nodes
  across 32 `cmath_*`/`complex_*` tests, reproducible in six lines, and present
  **on the default path too**, so it is not hop-off-specific. It is not currently
  unsound: bounding probes on `cmath.pi / 2.0` (`> 1.5` SUCCESSFUL, `> 1.6`
  FAILED, `< 0.0` FAILED) bracket the value correctly at π/2. But a new arm must
  not hand a `code`-typed operand to `c_implicit_typecast_arithmetic`. Worth
  filing separately as a node type-hygiene defect; it may be related to the
  known `cmath_*` SMT-sort abort.

### 9.3 §2's coupling argument, measured

Prototype arms for Phase 1 and Phase 2 were built behind environment gates and
measured on `neural-net_fail --fixedbv`, where G2 requires FAILED. Both
prototypes have been reverted; this is evidence for phasing, not an
implementation.

| config | verdict |
|---|---|
| legacy | FAILED |
| hop-off, bare | FAILED |
| hop-off **+ assignment arm only** | **SUCCESSFUL — the §2 masking, reproduced** |
| hop-off + assignment arm + equality arm | SUCCESSFUL |
| hop-off + arithmetic arm only | FAILED |
| hop-off **+ arithmetic arm + assignment arm** | **FAILED — G2 satisfied** |

**§2 is confirmed and the §4 ordering is correct**: the assignment conversion
masks a real bug when it lands alone, and the arithmetic arm is what prevents
that. Phase 1 must precede Phase 2.

The third row rules out a tempting alternative. `python_adjust` has **no arm for
`equality2t`/`notequal2t` at all**, while legacy routes `=` and `notequal`
through `adjust_expr_rel` alongside the four ordering relations `python_adjust`
does handle (`clang_c_adjust_expr.cpp:109-115` vs `python_adjust.cpp:393-396`);
and the `neural-net_fail` GOTO diff is an equality —
`x == (double)((signed int)x)` (legacy) vs `x == (signed int)x` (hop-off). That
makes a missing-equality-arm explanation look compelling, and it is wrong:
adding an equality arm does not prevent the masking. The gap is real and is
tracked in `scope-v1k-adjuster.md`, but it is not this mechanism.

### 9.4 Gate status under the coupled prototype

Four of the eight G1/G3 tests reach legacy parity, including all three crashes
and the abort:

| test | legacy | coupled prototype | |
|---|---|---|---|
| `builtin_all_nonliteral` | SUCCESSFUL | SUCCESSFUL | was SIGSEGV |
| `github_4344` | SUCCESSFUL | SUCCESSFUL | was SIGSEGV |
| `github_5571_fail` | FAILED | FAILED | was abort |
| `github_5571_tuple_str_annotation` | SUCCESSFUL | SUCCESSFUL | was abort |
| `chained-comparison2_fail` | FAILED | *no verdict* | unchanged |
| `lambda15` | SUCCESSFUL | *no verdict* | unchanged |
| `precedence2` | SUCCESSFUL | FAILED | unchanged |
| `sum_tuple` | SUCCESSFUL | FAILED | unchanged |

The remaining four are **insensitive to all three prototype arms** — identical in
the bare hop-off and in every configuration — so they are a separate mechanism
and G3 does not close on Phases 1-2 as scoped. `precedence2` localises it: `x` is
a `double`, and `x |= 7` gives legacy
`ASSIGN x=(double)((signed long int)((signed int)x) | 7);` against hop-off
`ASSIGN x=(signed long int)((signed int)x) | 7;`. That is an assignment
conversion the `code_assign2t` arm does not see, which is the node-kind
correction recorded in §4 Phase 2.

One claim in §2 did not reproduce: mirroring the assignment conversion alone
does **not** fix `precedence2` (it stays FAILED in that configuration). The
`neural-net_fail` half of that sentence reproduced exactly.

## 10. Phase 1 — what landed (2026-07-30)

An `else if` arm in `python_adjust::adjust_expr` over the eight IREP2
counterparts of the ids legacy routes to `adjust_expr_binary_arithmetic`
(`add sub mul div modulus bitand bitxor bitor`), calling
`c_implicit_typecast_arithmetic(expr2tc&, expr2tc&, ns)` on the two operands
when both are `bv`/`floatbv`/`fixedbv`, then adopting the reconciled type for
the node. Well under §3.4's 40-60 estimate, because one of the three pieces
that estimate included is dead on this path.

### 10.1 One sub-arm §3.3 asked for, measured dead

It was not shipped; keeping it would be dead instrumentation. The node-type
adoption was *also* judged dead on a first pass — that judgement was wrong, and
CI caught it; see §10.4.

- **The `ieee_*` rebuild (§3.3).** `map_operator` already returns
  `ieee_add`/`ieee_sub`/`ieee_mul`/`ieee_div` whenever the result type is
  floatbv (`converter/converter_internal.h:78-89`), so a plain floatbv-typed
  arithmetic node never reaches the adjuster in the first place. Confirmed
  empirically: an instrumented build logging every floatbv-typed `add`/`sub`/
  `mul`/`div` reaching the arm fired **0 times over 597 tests**. §3.3's concern
  — that IREP2 cannot retype a node in place — is real but does not arise here,
  because nothing needs retyping.

### 10.2 Gates

| gate | result |
|---|---|
| **G2** (`neural-net_fail --fixedbv` FAILED) | **holds** — legacy FAILED, bare hop-off FAILED, arm FAILED. Matches §9.3 row 5 |
| G1/G3 tests | all eight verdict-identical to the bare hop-off; unchanged, as expected — they need Phase 2 |
| default path | unaffected by construction: `python_adjust` runs only under `--python-irep2-adjust-only` / `--python-irep2-adjust`, both default-off (`python_language.cpp:299,331`). 400-test default-path slice green |
| hop-off suite | 146/150; the 4 failures are `esbmc-solidity`, blocked on macOS (no `solc`, empty `sol64` models) |

### 10.3 The verdict witness, and a residual gap

Phase 1 is not merely verdict-neutral — it clears a false alarm the bare hop-off
reports, which is what the new `python_irep2_adjust_only_binary_arith{,_fail}`
pair pins:

```python
def mix(n: int, x: float) -> float:
    return n * x + n
assert mix(2, 1.5) == 5.0     # --fixedbv: legacy SUCCESSFUL,
                              # bare hop-off FAILED, arm SUCCESSFUL
```

A neighbouring shape does **not** clear, and is worth recording for Phase 2:
`return 3 * x - 1` under `--fixedbv` stays FAILED under the arm where legacy
reports SUCCESSFUL. The operands reconcile; what is missing is the conversion at
the **return seam**, which is the same node-kind question §4 Phase 2 already
flags for assignments.

### 10.4 The node-type adoption is required — a correction

The first version of this arm converted the operands and left the node type
alone, on the finding that legacy's adoption never fires here. **That finding
was an artefact of the probe, and the resulting arm was ill-formed.** CI caught
it: both new tests aborted on macOS with

```
Assertion failed: (p1 || (is_bv_type(t) == is_bv_type(v2->type) &&
  t->get_width() == v2->type->get_width())), function
  assert_arith_2ops_consistency, file irep2_expr.cpp, line 678.
```

`arith_2ops` requires the node type to agree with **both** operands in bv-ness
and width. Promoting an operand without adopting the type violates that
directly.

**Why the probe said "0 firings".** It logged only the case
`lhs->type == rhs->type && reconciled != expr->type`. But
`c_implicit_typecast_arithmetic` frequently promotes *one* operand and leaves
the other — a `signedbv` literal against a `fixedbv` variable is the common
shape — so the equality precondition was false exactly when the node became
inconsistent. The probe measured a strictly narrower condition than the
invariant, and reported the difference as absence. **Probe the invariant you
depend on, not a proxy for it.**

**Why local testing did not catch it.** `assert_arith_2ops_consistency` is
`#ifndef NDEBUG`, so a `RelWithDebInfo` build compiles it out; the corpus census,
the flip-gate tests and both new regression tests all passed locally with an
ill-formed IR. Two dead ends worth not repeating: a bare `#undef NDEBUG` in
`irep2_expr.cpp` makes `c2goto` abort (exit 138) on a *pre-existing* violation
in the C operational-model build, so it cannot be used to isolate Python-path
regressions; and a full `DebugOpt` build is the documented route but costs a
complete rebuild.

**What replaced it.** The arm now reconciles on copies and commits only when the
promotion lands both operands on one type, which it then adopts via
`with_type`. Verified by a postcondition probe that checks the invariant itself
on every 2-op arith node the arm touches — **validated by first confirming it
reports 2 violations on the pre-fix code**, then 0 after the fix, then 0 across
the corpus. A probe that has not been shown to fire proves nothing.

## 11. Phase 2 reconnaissance — the assignment node kind, measured (2026-07-31)

§4 Phase 2 opens with "confirm the node kind first". Done, and it changes the
phase's scope: **two claims in this document were wrong, and both made Phase 2
look harder than it is.**

### 11.1 What the two node kinds actually are

There is no `sideeffect2t{assign}`. `sideeffect2t`'s `allockind` enum
(`irep2_expr.h:71-93`) has no `assign` member. Legacy assignment side-effects
migrate to a **distinct class**, `sideeffect_assign2t`
(`irep2_expr.h:2382`, built at `migrate.cpp:1905`), carrying an `op` string —
`assign`, `assign+`, `assign_bitor`, `assign_lshr`, … The statement form is
`code_assign2t` as expected.

### 11.2 The measurement

`python_adjust::adjust_expr` was instrumented to log each assignment node's
kind, operand type ids and whether they agree. Controlled minimal pairs isolate
what Python *source* contributes from the operational-model background:

| program | delta vs its control |
|---|---|
| `x = 1.0` → `x = 1.0; x \|= 7` | **+1 shape: `code_assign2t floatbv signedbv DIFF`**; no new `sideeffect_assign2t` shape |
| `assert True` → `y = n; y += 1; return y` | **+2 `code_assign2t signedbv signedbv`**; `sideeffect_assign2t assign+ signedbv` stays at **9, unchanged** |

The `sideeffect_assign2t` population is *invariant under the Python source* —
identical counts across all four programs. It is the C operational-model bodies,
which `clang_c_adjust` already reconciled before `c2goto` froze them.

### 11.3 Consequences for Phase 2

- **A `code_assign2t`-only arm is the correct scope**, not a partial one. §4
  Phase 2's warning is withdrawn.
- **§9.4's diagnosis of `precedence2` is withdrawn too.** It reads the missing
  cast as "an assignment conversion the `code_assign2t` arm does not see"; the
  measurement shows `x |= 7` *is* a `code_assign2t`, with target `floatbv` and
  source `signedbv` (`x` is monomorphically float because of the earlier
  `x /= 3`). Whatever made the §9.3 prototype miss it, it was not the node kind.
  `precedence2` should be re-tried against a plain `code_assign2t` arm before
  being re-homed to the "second mechanism" scope.
- **Do not extend the arm to `sideeffect_assign2t`.** It would touch only OM
  bodies, for no Python-source benefit, and the traffic includes
  `assign+ pointer signedbv` (pointer arithmetic) — the same operand kind §9.2
  requires the arithmetic arm to decline.

### 11.4 Legacy semantics the arm must mirror

For the record, `clang_c_adjust::adjust_side_effect_assign`
(`clang_c_adjust_expr.cpp:888-923`) is three cases, only the first of which the
`code_assign2t` path needs (`clang_c_adjust_code.cpp:30`):

| op | legacy action |
|---|---|
| `assign` | node type := lhs type; `gen_typecast(rhs, lhs_type)` |
| `assign_shl` / `assign_shr` | promote **rhs alone**; `shr` becomes `lshr`/`ashr` by lhs signedness — an *op rewrite*, so IREP2 must rebuild the node |
| all other compound ops | `gen_typecast_arithmetic(lhs, rhs)` — reconcile **both** |

## 12. Phase 2 — what landed, and what it does not clear (2026-07-31)

A `code_assign2t` arm calling `c_implicit_typecast(source, target->type, ns)`,
placed after the existing narrow assign arms so they keep priority. Scope is
§11's measured one: `code_assign2t` only.

### 12.1 The guard, and the §2 witness

Numeric-to-numeric, **plus pointer-source-into-Boolean**. That second clause is
not incidental — it *is* the §2 witness. `builtin_all_nonliteral`'s
`ASSIGN element=(_Bool)tmp$5` looked like an integer-to-Boolean narrowing, but
in `all()`'s model `tmp$5` is `void *`:

```
ASSIGN tmp$5 = *(void * *)return_value$___ESBMC_list_at$4->value;
5: ASSIGN element = (_Bool)tmp$5;
```

A numeric-only guard declines it and the arm is inert on the very test §2 names.
Legacy's `gen_typecast` to bool is a null test, so the pointer source is correct
here; a pointer source is still declined for every other target (§9.2), and a
pointer *target* is owned by the decay arms above.

### 12.2 Gate status — G2 holds, G1/G3 do not fully close

| test | legacy | P1 alone | P1+P2 | §9.4 prototype |
|---|---|---|---|---|
| **`neural-net_fail` (G2)** | FAILED | FAILED | **FAILED** | FAILED |
| `builtin_all_nonliteral` (G1) | SUCCESSFUL | *no verdict* | **SUCCESSFUL** | SUCCESSFUL |
| `github_4344` (G3) | SUCCESSFUL | *no verdict* | **SUCCESSFUL** | SUCCESSFUL |
| `github_5571_fail` (G3) | FAILED | *no verdict* | *no verdict* | FAILED |
| `github_5571_tuple_str_annotation` (G3) | SUCCESSFUL | *no verdict* | *no verdict* | SUCCESSFUL |
| `chained-comparison2_fail` (G1) | FAILED | *no verdict* | *no verdict* | *no verdict* |
| `lambda15` | SUCCESSFUL | *no verdict* | *no verdict* | *no verdict* |
| `precedence2` | SUCCESSFUL | FAILED | FAILED | FAILED |
| `sum_tuple` | SUCCESSFUL | FAILED | FAILED | FAILED |

**G2 — the anti-masking gate — holds**, which is the property that makes the
coupling sound and the reason Phase 1 had to precede this. Two of the four
crash/abort cases clear. **No test regresses against Phase 1 alone.**

Three things this does *not* discharge, stated rather than glossed:

1. **The `github_5571` pair still produces no verdict**, where §9.4's prototype
   recorded both clearing ("was abort"). Either that prototype's arm was wider
   than §11's measured scope, or its numbers were taken under a different tree.
   This is the first thing to chase.
2. **G1 as written cannot close.** It names `chained-comparison2_fail`, but §9.4
   and §7 both put that test in the *second mechanism* that this scope
   explicitly does not own. G1's test list contradicts §7; it should be
   rewritten to name only `builtin_all_nonliteral`.
3. **`precedence2` is still FAILED**, so §11.3's suggestion — that a plain
   `code_assign2t` arm would clear it once the node-kind confusion was removed —
   is **refuted**. The node kind was never the obstacle; `precedence2` belongs
   with the second mechanism after all, as §9.4 originally placed it.

### 12.3 Not yet run

The whole-corpus census (G4) and dual-solver agreement (G5) are Phase 3 gates
and have not been run for this arm. The hop-off regression subset is green
(44/44) and the flag remains default-off, so nothing here reaches the default
path.

## 13. The `github_5571` pair is not this scope's work (2026-07-31)

§12.2 flagged the pair as the first thing to chase. Chased, and the answer is
that **§9.4's table over-credits the §9.3 prototype**: both tests abort
identically in the bare hop-off, under Phase 1 alone, and under Phase 1+2 —

```
ERROR: Typecast for unexpected type
typecast
* from : constant_array   (char[1], i.e. the "" literal + NUL)
* type : array            (char[16])
```

— an **array-to-array typecast between different sizes**, produced by `s = ""`
where `s` is a `char[16]`. Neither the arithmetic arm nor the assignment arm
constructs or consumes such a node, and the three binaries' error output is
byte-identical. §9.4 records both as "was abort → cleared by the coupled
prototype"; that is not reproducible on this tree, and the mechanism is not one
the coupled conversion can reach.

This is a third mechanism, distinct from both this scope and the "second
mechanism" of §9.4. It is in the same family as the existing array→pointer decay
arm (`python_adjust.cpp`, the `is_typecast2t && is_pointer_type(target)` case)
but with an **array** target, which that arm's guard declines. It wants its own
diagnosis.

### 13.1 G1 and G3 must be rewritten

Both gates name tests that other mechanisms own, so as written neither can ever
close, no matter how correct Phases 1-2 are:

| gate | as written | owns | should name |
|---|---|---|---|
| G1 | `builtin_all_nonliteral` + `chained-comparison2_fail` | §7 puts `chained-comparison2_fail` in the second mechanism | `builtin_all_nonliteral` only |
| G3 | `github_4344`, `github_5571_fail`, `github_5571_tuple_str_annotation` | the 5571 pair is the array-typecast mechanism above | `github_4344` only |

**Against the corrected lists, Phases 1-2 discharge every gate this scope owns:**
G1 (`builtin_all_nonliteral` SUCCESSFUL), G2 (`neural-net_fail` FAILED),
G3 (`github_4344` SUCCESSFUL). What remains before the Phase 3 flip is not this
scope's implementation but the two foreign mechanisms — the §9.4 second
mechanism (`chained-comparison2_fail`, `lambda15`, `precedence2`, `sum_tuple`)
and the array-typecast class (the `github_5571` pair) — plus G4/G5.

## 14. §13 refined — the `github_5571` abort *is* an assignment conversion

§13 concluded the pair was "a third mechanism this scope cannot reach". A GOTO
diff of `github_5571_tuple_str_annotation` shows that is half right: the mechanism
is distinct, but it is reachable by an assignment arm — an **array-aware** one,
which §12's numeric-plus-pointer-into-Boolean guard declines.

The whole difference in `f` is one line:

```
legacy:   DECL signed char [0] s;  ASSIGN s = (signed char [16])(&{ 0 }[0]);
hop-off:  DECL signed char [0] s;  ASSIGN s = { 0 };
```

Legacy does **two** things at the assignment seam — decays the `char[1]` string
literal to `&{0}[0]`, then casts to `char[16]`. The hop-off does neither and
assigns the bare `constant_array`. Crucially, `typecast → char[16]` appears
**twice in the legacy dump and only once in the hop-off dump**: the cast the SMT
layer chokes on is not in the hop-off GOTO at all. It is synthesised later,
during symex, when the `char[1]` value meets its differently-sized destination —
and `convert_typecast` has no array arm, hence
`ERROR: Typecast for unexpected type`.

So the defect class is the one §2 describes — an unconverted assignment reaching
the solver — with an array type instead of a scalar one. The reason it survived
Phase 2 is the guard, not the node kind: §11's measurement stands, the
assignment *is* a `code_assign2t`.

**What an array arm has to reproduce**, and why it was not attempted blind: the
legacy shape `(signed char [16])(&{ 0 }[0])` is a cast of a *pointer* to an
*array* type, assigned to a variable declared `char[0]`. All three widths differ
and the existing `is_pointer_type(target)` decay arm cannot emit it. Getting that
wrong risks the array/pointer mismatch at symex rename that the decay arms were
added to fix, so it wants its own measured pass rather than an extension bolted
onto §12's guard.

This supersedes §13's "cannot reach" framing. G3's re-homing still stands — the
pair is not cleared by Phases 1-2 as built — but the follow-up belongs to this
scope's family, not to an unrelated one.

## 15. Both flip blockers now have owner documents (2026-08-03)

§13.1 discharges every gate this scope owns and leaves Phase 3 blocked on two
mechanisms it disowns. Neither had an owner; both now do.

| blocker | owner | finding |
|---|---|---|
| the `github_5571` array-typecast pair (§14) | `scope-array-assignment-conversion.md` | every existing arm declines it because they all guard on a **pointer** target, while this shape casts to an **array** type |
| the §9.4 second mechanism | `scope-relational-float-reconciliation.md` | all four tests share an integer meeting a floating-point operand at a comparison or bitwise assignment; the relational arm admits only `bv`/`bv` pairs (`python_adjust.cpp:406-409`) while the arithmetic arm admits floatbv too (`:449`) |

Both are recorded as hypotheses gated on a Phase 0 measurement, not as
conclusions — this document's own §13→§14 reversal is the reason.
