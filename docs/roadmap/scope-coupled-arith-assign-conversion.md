# Scope — the coupled arithmetic + assignment conversion (the `python_adjust` flip blocker)

> **Status: not started.** This document exists because
> `docs/roadmap/scope-v1k-adjuster.md` §"Flip gate (2026-07-29)" closes that
> scope with exactly one remaining prerequisite and hands it off: *"Next owner:
> take the coupled conversion effort as its own scope, then re-run this
> whole-corpus census as the flip gate."* This is that scope.
>
> It is a **forward plan**, not a record of work. Nothing below has landed.

## 1. What this unblocks, and what it does not

`python_adjust` is the IREP2-native replacement for the legacy
`clang_cpp_adjust` pass on the Python path. It is complete enough to run as the
sole adjuster behind `--python-irep2-adjust-only` (`src/esbmc/options.cpp:214`,
experimental, default off), and every *structural* gap the adjuster scope
enumerated is closed or refuted.

The flip to default-on is blocked on one defect class. Clearing it:

- **unblocks** the `python_adjust` flip (`--python-irep2-adjust-only` becomes
  the default; the legacy `clang_cpp_adjust` hop on the Python path goes away);
- **does not** advance §V.1 bars #1, #2 or #4 — those are V.2/W3 and the
  symbol-write boundary, tracked elsewhere;
- **does not** touch C or C++, which keep using `clang_c_adjust` /
  `clang_cpp_adjust` unchanged. The blast radius is the Python suite only.

## 2. The defect, restated precisely

From the flip-gate census (1 108 tests, every 4th directory): 6 genuine
regressions, all one mechanism. The witness is `builtin_all_nonliteral`, whose
normalized `--goto-functions-only` dumps differ in exactly one line:

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

**This is a hypothesis with a named discharge, not a finding.** It must be
verified before it is used to size anything — see gate G0 in §5.

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

## 4. Phased decomposition

Strictly ordered. The ordering is the soundness argument from §2, not a
preference.

### Phase 0 — the reachability census (no code change)

Discharge G0. Instrument `python_adjust::adjust_expr` to log the `type_id` and
`expr_id` of every binary arithmetic node it sees, run the whole `python`
suite, and tabulate. Deliverable: a table of reachable operand type kinds.
Revert the instrumentation.

### Phase 1 — operand-level arithmetic reconciliation

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

Add the `code_assign2t` arm calling
`c_implicit_typecast(source, target->type, ns)`. Only after Phase 1 has landed
and passed its gate.

**The `neural-net_fail` check is the acceptance test for this phase**, not a
regression to notice later. It must report FAILED.

### Phase 3 — the flip

Make `python_adjust` the sole adjuster; `--python-irep2-adjust-only` becomes
the default with an opt-out, mirroring how the W1-loc keystone shipped
(`--irep2-native-body` → deprecated no-op, `--no-irep2-native-body` the escape
hatch, `src/esbmc/options.cpp:964-975`).

## 5. Gates

| # | Gate | Discharged by |
|---|---|---|
| **G0** | The reachable operand-kind census exists, and the complex/vector claim in §3.2 is confirmed or refuted | Phase 0 |
| G1 | `builtin_all_nonliteral` and `chained-comparison2_fail` produce legacy-identical verdicts under the hop-off | Phase 2 |
| G2 | **`neural-net_fail` (`--fixedbv`) reports FAILED** | Phase 2 — the anti-masking gate |
| G3 | All 6 flip-gate regressions clear: `github_4344`, `github_5571_fail`, `github_5571_tuple_str_annotation`, `lambda15`, `precedence2`, `sum_tuple` | Phase 2 |
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
- The remaining `python_adjust` per-case divergences — the flip-gate census
  established there are none left outside this mechanism. **Do not re-triage
  per case**; the census names the single mechanism.
- V.1a, V.2/W3, V.5, V.6 — different scopes, see
  `docs/roadmap/irep2-migration.md` §V.7 and
  `docs/roadmap/scope-v2-w3-attribute-carriage.md`.
- §V.1 bars #1/#2/#4. This scope moves none of them.

## 8. One-line summary

The engine is already IREP2-native and already called from `python_adjust`;
what is missing is a binary-arithmetic arm and an assignment arm, which are
unsound apart and must land in that order — sized at 3 PRs conditional on a
cheap reachability census that could shrink the first one by ~88 lines.
