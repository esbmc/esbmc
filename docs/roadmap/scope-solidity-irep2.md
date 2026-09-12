# Scope: solidity frontend to IREP2 (Phase 8)

Opened per `frontends-to-irep2.md` §6, which requires each of Phases 5-9 to
start with its own scope doc: census, phased decomposition, gates, risks. Phase
5 (jimple) closed at `scope-jimple-irep2.md` §31; Phase 6 (clang-c) is
`scope-clang-c-irep2.md`; Phase 7 (clang-cpp) is `scope-clang-cpp-irep2.md` and
is **not** closed, which matters here for the reason §2 gives.

## 1. Census

### 1.1 The blocker the parent records is stale

`frontends-to-irep2.md` §15.1 records Solidity as *blocked, not zero*: a decline
census got `ERROR: `' is not a goto-binary`, nothing ran, and the note concludes
the suite needs Linux CI. Re-tested 2026-09-11 and it is measurable here.

```
$ grep ENABLE_SOLIDITY build/CMakeCache.txt
ENABLE_SOLIDITY_FRONTEND:BOOL=On
$ ctest -j24 -L esbmc-solidity
99% tests passed, 2 tests failed out of 525
```

The two are not failures: both print `passed but is marked as KNOWNBUG.
Consider reclassifying it as CORE` (`testing_tool.py` exit 77) —
`delegate_shadow_3` and `nested_array_deep_1`, each expecting `^VERIFICATION
SUCCESSFUL$` and getting it, in half a second, so not the timeout that
`KNOWNBUG` silently accepts. They are master drift, and flipping them wants its
own change; they are recorded here because a census that reports "2 failures"
without saying which kind is the §15.1 mistake again.

§15.1's own rule is what makes this census admissible: it must show the thing
under test executed before a zero means anything.

### 1.2 Surface

`src/solidity-frontend`, 23 599 LOC over 25 files:

| File | LOC | Legacy construction sites |
|---|---:|---:|
| `solidity_convert_call.cpp` | 3 682 | 404 |
| `solidity_convert_expr.cpp` | 3 409 | 250 |
| `solidity_convert.h` | 1 129 | 221 |
| `solidity_convert_ref.cpp` | 937 | 89 |
| `solidity_convert_type.cpp` | 1 588 | 86 |
| `solidity_convert_constructor.cpp` | 1 055 | 78 |
| `solidity_convert_stmt.cpp` | 978 | 75 |
| `solidity_convert_decl.cpp` | 1 526 | 73 |
| `solidity_convert_mapping.cpp` | 708 | 70 |
| `solidity_convert_contract.cpp` | 850 | 67 |
| `solidity_convert_tuple.cpp` | 653 | 59 |
| `solidity_convert_builtin.cpp` | 524 | 54 |
| `solidity_convert_modifier.cpp` | 752 | 53 |
| `solidity_convert.cpp` | 1 266 | 51 |
| `solidity_convert_util.cpp` | 1 046 | 36 |
| `solidity_convert_literals.cpp` | 159 | 15 |
| `solidity_grammar.cpp` | 1 804 | **0** |
| `pattern_check.cpp` | 153 | **0** |
| remainder (`typecast`, `language`, headers) | — | 4 |
| **total** | **23 599** | **1 685** |

"Construction sites" counts lines mentioning `exprt`, `typet`, a `code_*t`,
`symbol_expr`, `gen_zero` or `from_integer` — the same rough proxy the parent
used for its 971 and 1 420 figures, not a precise obligation count. The
parent's 1 420 was measured at `f14cd73ff8`; the surface has grown since.

**IREP2 use today: zero.** `grep -c 'expr2tc\|type2tc' *.cpp *.h` over the
whole frontend returns 0, so unlike clang-c (49 sites already native) this
phase has no head start. `solidity_grammar.cpp` is the AST-kind classifier and
constructs no IR at all — the analogue of `clang_c_lexer.cpp` in Phase 6's
§1.1, and it needs no migration.

### 1.3 Corpus

| Suite | Tests | CORE | THOROUGH | KNOWNBUG |
|---|---:|---:|---:|---:|
| `esbmc-solidity` | 525 | 324 | 193 | 8 |

Tests ship a pre-generated `contract.solast` beside `contract.sol`, and the
flags line names both (`--sol contract.sol --contract <Name>`), so `solc` is
not needed to run them. That is why the suite is measurable without the
toolchain §15.1 assumed was required.

## 2. The finding that sets this phase's shape: there is no Solidity adjust pass

`solidity_languaget::typecheck` (`src/solidity-frontend/solidity_language.cpp`)
has four phases — intrinsics, the `sol64` operational models, the converter,
then the adjuster — and the adjuster it runs is **`clang_cpp_adjust`**, Phase
7's subject, not one of its own:

```cpp
  clang_cpp_adjust adjuster(new_context);
  if (adjuster.adjust())
    return true;
```

Two consequences, and they pull in opposite directions.

**The adjust half of this phase is inherited, not owed.** Every arm Phase 7
ports serves Solidity at no extra cost, and none of Solidity's 1 685 sites is
in an adjust pass. Phase 8's own work is entirely in the *converter*.

**Solidity is therefore exposed to every Phase 7 divergence, and is not
currently measuring any of them.** `clang_cpp_language.cpp` gates its adjuster
on `clang-cpp-irep2-adjust-only`; the line above does not, so the flag does not
reach this path — a Solidity test run with it is byte-identical to one run
without (checked on `abi_decode_1`; the claim rests on the source, since one
identical verdict would not distinguish "flag ignored" from "flag made no
difference here").

That makes the first action of this phase cheap and valuable: wire the existing
flag through, and the 525-test Solidity corpus becomes a **second, independent
corpus for Phase 7's pass** — one whose input the C++ frontend never produces,
over a converter with different habits. Phase 7's divergence census has been run
on `esbmc-cpp` only.

### 2.1 The save/restore dance is a seam hazard

Phase 4 of that function saves every library symbol's value as an `exprt`
before adjusting and restores it afterwards, because `clang_cpp_adjust` would
corrupt bodies that `c2goto`'s `clang_c_adjust` already adjusted.
`clang_c_adjust_irep2` writes back only values it changed, and writes them back
through `migrate_expr_back`. So the restore interacts with a different
write-back discipline than it was written against, and any seam loss on a
library body would be masked by the restore rather than observed. This wants a
measurement before the flag is wired, not after.

## 3. Proposed decomposition (not yet executed)

1. **S.1** Wire `clang-cpp-irep2-adjust-only` into
   `solidity_languaget::typecheck` and report the divergence count over the
   corpus. Measured baseline, no porting. **Done — §7.**
2. **S.2** A read-only `migrate` census for the Solidity converter's output, as
   `--clang-cpp-irep2-migrate-census` is for C++: run every value through
   `get_value2()` and report what `migrate_expr` cannot represent. This is the
   only way to price the converter before touching it, and it is the step
   §15.1's void figure was trying to be.
3. **S.3** The converter's construction sites, in the order the census ranks
   them. `solidity_convert_call.cpp` and `solidity_convert_expr.cpp` are 39 %
   of the surface between them.
4. **S.4** Remove the legacy path once S.1's divergence count is zero.

Steps S.1 and S.2 are both measurement, and both are cheap. Neither is
committed to a porting order, deliberately: Phase 6 §60 found the adjuster's
arms to be one strongly-coupled component that could not move singly, and Phase
7 §3 found its pass was not extensible in the way Phase 6 assumed. A census
first is the lesson those two paid for.

## 4. Gates

The parent's §7 gates apply unchanged. Two are worth restating for this phase:

- **A census must show the thing under test executed.** §15.1's "14 tests, 0
  declines" is void because nothing ran. Any figure this doc reports names the
  command that produced it.
- **`KNOWNBUG` and `FUTURE` rows cannot be read as green.** They accept a
  timeout as satisfying the expectation, so a Solidity divergence count must
  separate them out; `grep` for `accepted under KNOWNBUG` before reading a run
  as clean.

## 5. Risks

| # | Risk |
|---|---|
| R1 | This phase's adjust half depends on Phase 7 closing. Wiring the flag before Phase 7's divergences are down exposes Solidity to all of them at once, which is why S.1 is a measurement and not a flip. |
| R2 | The §2.1 restore can mask a seam loss on a library body. A divergence count taken without checking it may be optimistic. |
| R3 | The `sol64` operational models are a second `c2goto`-compiled artefact, adjusted by `clang_c_adjust` at build time. The `building-c-library` exemptions that protect the C models have no Solidity analogue recorded, and Phase 6's name-matched-builtin section found one such exemption already unreachable from the IREP2 pass. |
| R4 | 1 685 sites and zero head start make this the second-largest phase after Python. Its ordering before Python is the parent's §215 judgement and this doc does not revisit it. |
| R5 | The two XPASS rows in §1.1 mean the suite's expectations are drifting from master. A divergence count is only meaningful against a suite whose baseline is green. |

## 6. Next

S.3, the converter's own 1 685 sites. §7.22's padding row is the one open
defect in the adjust and seam half; §7.23 has the corpus figures.

## 7. S.1 executed: the baseline, and it is one cause (2026-09-11)

The wiring mirrors `clang_cpp_language.cpp` — there is no C++ shadow mode, only
`clang-cpp-irep2-adjust-only`, so the gate is a single `if`/`else`. Default path
unchanged.

Before the wiring the flag was inert here: `abi_decode_1` run with it was
byte-identical to the same run without. After it, that test SIGSEGVs, which is
how the wiring is observed at all — see §7.3.

### 7.1 The measurement

Stride-8 sample of the corpus, each row run twice on one binary, flag off
against flag on:

| | rows |
|---|---:|
| measured | 65 |
| verdicts agree | **5** |
| verdicts diverge | 9 |
| crash under the flag | **51** |

A sample, not the whole corpus, and labelled as one: the machine this ran on
was under memory pressure heavy enough to have an earlier build OOM-killed, and
a 517-row sweep is 1 034 runs. The conclusion does not turn on the precision —
78 % of a stride sample crashing is not a figure a fuller run reverses into
health.

This is R1 arriving as a number rather than a prediction. Wiring the flag
exposes Solidity to every open Phase 7 divergence at once, which is exactly why
S.1 was specified as a measurement and not a flip.

### 7.2 The crashes are one site

Seven crashing rows sampled across the corpus — `abi_decode_1`, `bitwise_ops_2`,
`array_2`, `clearing_mapping_1`, `error_3`, `mapping_12`, `super_3` — symbolised
with `--segfault-handler` and `addr2line`. All seven share one top frame:

```
is_constant_bool2t(irep_container<expr2t> const&)   src/irep2/expr_kinds.inc:23
goto_convertt::optimize_guarded_gotos(goto_programt&)
                                     src/goto-programs/goto_convert.cpp:102
```

`optimize_guarded_gotos` tests `is_true(it_goto_y->guard)`, which is inlined,
and `is_constant_bool2t` dereferences the container. A GOTO instruction is
therefore reaching that pass with a **nil guard** where the legacy path leaves
a `true` one — so the defect is upstream of `goto_convert`, in what the IREP2
pass fails to fill in, not in the optimisation.

Naming the site is not naming the cause. Which expression is left nil, and by
which missing arm, is the next investigation; Phase 7's §3.3 found the same
shape (23 crashes, one site) and the cause was an unpopulated list the
converter leaves empty. That is a hypothesis here, not a finding.

### 7.3 What pins this, and what cannot

Nothing end-to-end. A flag-pinned Solidity test over one of the 5 agreeing rows
passes with the wiring in or out, so it pins nothing; a test pinning a diverging
or crashing row would be pinning the defect. The evidence for S.1 is the
measurement above, reproducible with the two commands in §1.1 plus the flag.

The instrument becomes available the moment §7.2's cause is fixed: a Solidity
row that today crashes, pinned for *producing a verdict at all*, is a real gate
— `^VERIFICATION SUCCESSFUL$` cannot match a SIGSEGV. That test belongs to the
change that fixes the crash, not to this one. #7717 shipped the C++ pass the
same way, with its census as the evidence.

### 7.4 The crash localised: the generated harness, and it is not a missing arm

`gdb -batch -ex run -ex bt` on the reduced input gives the frame `addr2line`
could not:

```
#0  make_not (expr=...)                        src/irep2/irep2_utils.cpp:8
#1  goto_convertt::optimize_guarded_gotos      goto_convert.cpp:101
#2  goto_convert_functionst::try_convert_body_native
#3  goto_convert_functionst::convert_function
```

So it is `make_not(it->guard)` at line 101, not the `is_true` on the line before
it, and `it` is a **conditional GOTO whose guard is nil**.

**It reduces to four lines.** `solc` is available at `/tmp/extest/solc`, so new
contracts can be generated rather than picked from the corpus:

```solidity
pragma solidity >=0.8.0;
contract C {
    uint x;
}
```

`--goto-functions-only` plus the flag crashes on that; without the flag it dumps
the program. No function, no statement, no expression of the user's is needed —
which places the guard in code the converter *generates*, and the legacy dump
names it: `_ESBMC_Main_C (sol:@C@C@F@_ESBMC_Main_C#)`, the per-contract harness,
whose first branch is `IF !return_value$_nondet_bool$1 THEN GOTO 2`. That is
exactly the `if(x) goto z; goto y; z:` shape `optimize_guarded_gotos` rewrites.

Two things this rules out.

**It is not the native body converter.** Frame #2 is `try_convert_body_native`,
but `--no-irep2-native-body` does not avoid the crash: the legacy
`goto_convert_rec` path runs `optimize_guarded_gotos` over the same sequence
and dies identically. So the malformed guard is in the adjusted body, not in
either converter.

**It is probably not a missing arm.** An A/B of `--symbol-table-only` over
`string_concat_1`, flag off against flag on, differs in ten lines, of which
eight are a temp-directory path and blank lines in one printed body. The one
structural difference is a dropped `#location` that was **present and empty**
on a constructor call — the irept tri-state where `find()` cannot distinguish
absent from nil, and a mutable `location()` creates a third state. An arm that
failed to run would not produce a symbol table this close.

That pointed at the seam rather than the arm table. §7.5 tests it, and the lead
does not survive.

Stated as a hypothesis, not a finding: the symbol-table closeness makes a seam
loss the better explanation, but nothing here has yet shown that *this* loss is
what empties the guard.

### 7.5 The location lead is weaker than §7.4 read it, and gdb cannot close it

Run on the four-line contract instead of `string_concat_1`, the
`--symbol-table-only` A/B is tighter still: **one** structural line, a dropped
`* #location:` that was present and empty. But it sits on the *callee symbol of
a constructor call inside* `sol:@_ESBMC_Object_C#` — not on any function that
holds a conditional goto. Four generated functions do hold one:

| generated function | conditional gotos |
|---|---:|
| `sol:@C@C@F@$transfer#0` | 2 |
| `sol:@C@C@F@$send#0` | 2 |
| `_sol_init_` | 1 |
| `sol:@C@C@F@_ESBMC_Main_C#` | 1 |

So §7.4's "try the location restore first" was too strong a reading. The loss is
real and worth fixing on its own account, but it is in a different symbol from
the crash and nothing connects the two.

The harness shape is not the cause either. `_ESBMC_Main_C` is
`while (nondet_bool()) { _ESBMC_Nondet_Extcall_C(); }` after a `__ESBMC_HIDE:`
label, and the direct C analogue —

```c
_Bool nondet_bool(); void body();
int main(void) { HIDE:; while (nondet_bool()) { body(); } return 0; }
```

— converts cleanly under `--clang-c-irep2-adjust-only`, under
`--clang-cpp-irep2-adjust-only`, and on the default path. A side-effect loop
condition reached through a `__ESBMC_HIDE` label is handled.

**Why this stops here.** Naming the function needs the symbol at frame 3, and
`gdb` reports `symbol = <optimized out>`; `dest.instructions` cannot be walked
either, because every accessor is inlined (`Cannot evaluate function -- may be
inlined`). A `-O2 -DNDEBUG` build will not give up that name. The next step is
the technique the earlier phases used for exactly this: a temporary `fprintf`
in `convert_function` printing `symbol.id` before `optimize_guarded_gotos`,
which needs a rebuild. Four candidates and a two-second reproducer make that a
short run once a build is available.

What is settled: the guard is nil in a converter-generated body, in one of four
named functions, and neither body converter, nor the loop shape, nor — on
present evidence — the location seam explains it. §7.6 closes it, and overturns
the "probably not a missing arm" reading above.

### 7.6 The cause: an unported arm, and the A/B that hid it

A temporary `fprintf` in `convert_function` names the function in one run:
`sol:@C@C@F@_ESBMC_Main_C#`, the 207th of 207 conversions, with the SIGSEGV
immediately after it. A second `fprintf` dumping `symbol.get_value().pretty()`
for that symbol gives the two trees the printed C form could not. The `while`
condition is a `sideeffect` function call, and legacy against flag-on reads:

| | legacy | `--clang-cpp-irep2-adjust-only` |
|---|---|---|
| the call's type | `bool` (`#cpp_type: bool`) | **empty**, keeping `#sol_type: BOOL` |
| the callee symbol's type | `code`, with `arguments` and `return_type: bool` | **empty** |

An expression with no type is what `goto_convert` turns into a nil guard, and
`make_not` then dereferences it.

Both losses have one cause, in
`clang_c_adjust::adjust_side_effect_function_call` (`clang_c_adjust_expr.cpp`):
when the callee resolves to a context symbol it replaces `f_op` with
`symbol_expr(symbol)` — restoring the callee's `code` type from the symbol
table — and then calls `align_se_function_call_return_type(f_op, expr)`, which
sets the call's type to the callee's `return_type`. `clang_cpp_adjust`
overrides that helper to skip constructors. Neither the callee replacement nor
the alignment is ported: `grep -n 'align_se\|return_type'` over both IREP2
adjust passes returns nothing.

So the Solidity converter emits the call with an incomplete type carrying only
`#sol_type: BOOL`, legacy repairs it, and the IREP2 pass leaves it as it found
it. It is a missing arm after all, and a **C** one — so porting it serves the
C++ frontend too.

**The instrument was the mistake, not the reasoning.** §7.5 concluded "probably
not a missing arm" from a `--symbol-table-only` A/B that differed in one line.
That dump renders values as C source, and `expr2c` prints a call with an empty
type exactly as it prints a typed one: `nondet_bool()`. The tree differed all
along; the printer flattened it. Any future A/B over adjusted bodies wants
`pretty()`, not the C rendering — the same shape of error as measuring a decline
census from a goto dump instead of a verdict.

Ported in §7.7, which measures what it moves.

### 7.7 One arm, and it moves 50 of the 51 crashes (2026-09-12)

`adjust_call_signature` rebuilds the callee from the symbol table when the
converter left its type incomplete, and then calls a new
`align_call_return_type` hook — empty in `clang_c_adjust_irep2`, as
`clang_c_adjust::align_se_function_call_return_type` is empty for C, and
overridden in `clang_cpp_adjust_irep2` to take the callee's `return_type` and
skip constructors. The row goes before `adjust_call_arguments`, whose parameter
types come from the callee type this repairs.

Same stride-8 sample, same binary discipline, flag off against flag on:

| | before | after |
|---|---:|---:|
| verdicts agree | 5 | **56** |
| crash | 50 | **0** |
| neither | 9 | 9 |

Those are the corrected figures; §7.9 says what was wrong with the first set.

**The row has to be in two tables.** `clang_cpp_adjust_irep2` substitutes its
own arm table rather than adding to the C one (§3.1 of the clang-cpp scope
doc). With the row in the C table alone the arm compiles, links, and never
dispatches: all four reduced contracts still crashed. That A/B is the row's
mutation evidence, and it is worth stating because a null result there invites
discarding a correct diagnosis.

**The residue is one cause, not nine.** Every non-agreeing row now ends in
`ERROR: migrate expr failed` — `constructor_4`, `enum_2`,
`function_overload_2_fail`, `import_2`, `inheritance_1`, `inheritance_8`,
`return_6`, `send_ether_via_creation_1`, `try_catch_1`. They looked like
timeouts in the sweep (`on=[]`), and they are not: one re-run with a 300 s
budget exits in 0.6 s with that error. The row first reported as a remaining
crash, `github_6759_02`, was never one — §7.9.

`regression/esbmc-solidity/irep2_only_call_return_type{,_fail}` pin it,
generated with `solc --ast-compact-json` so the checked-in `.solast` matches
the corpus convention, and bounded with `--unwind 1 --no-unwinding-assertions`
because the harness runs past 200 s unbounded. Both halves flip to a SIGSEGV
when the arm is disabled, so neither verdict regex is satisfiable without it.
Default path unchanged: `esbmc-solidity` is 525 of 527 with the same two
`KNOWNBUG` rows that already passed, and `irep2_only` is 99 of 99.

What this does not claim: any gain for the C frontend. The callee refresh is in
the C pass because that is where legacy has it, but clang's converter does not
leave an incomplete callee type, and the alignment hook is empty for C by
design. The 97 pre-existing C rows passing is consistent with the arm being
inert there, not evidence of a C-side improvement.

### 7.8 The residue is Phase 7's cpp_new size defect, already fixed in #7726

The nine `migrate expr failed` rows are **not** this branch's doing: the
diverging set is identical before and after §7.7's arm (`comm -13` over the two
sweeps is empty). They are a separate defect that closing the crashes merely
uncovered.

`gdb -batch -ex 'catch throw'` against the current build — line numbers shift
between builds, which is why a probe placed from an older backtrace never fired
— gives the chain:

```
code_block operand loop          src/util/irep/migrate.cpp:2568
  -> sideeffect_assign, lhs      src/util/irep/migrate.cpp:2093
    -> cpp_new size operand      src/util/irep/migrate.cpp:2125   throws
```

So it is a `sideeffect_assign` whose lhs is a `cpp_new` whose size is an **empty
irept**. The site reads

```cpp
const exprt &sz = expr.cmt_size().is_not_nil() ? … cmt_size() : … size_irep();
migrate_expr(sz, thesize);
```

and `is_not_nil()` is true for a *present-but-empty* irept, so the empty `#size`
is selected and handed to a `migrate_expr` that has no handler for it.

That is the defect PR **#7726** already fixes, in commit `d28be5a7e2`, with a
tri-state-aware test:

```cpp
const auto carries_size = [](const irept &i) {
  return !i.id().empty() && !i.is_nil();
};
```

All four stacked Phase 7 branches carry `cpp_new_size`; master does not. So
Phase 8's residue and Phase 7's `cpp_new` size fix are one defect, and this
corpus is a second, independent corpus for it — worth recording on that PR
rather than fixing twice here.

Two corrections this section makes to §7.5-§7.7's reading. The empty operand is
the `cpp_new` **size**, not a sideeffect's `op0`; and the five-kind exclusion
list at migrate.cpp:2116 is a red herring, since `cpp_new` is *in* it. The
one-line `&& expr.op0().is_not_nil()` guard that suggested itself would have
papered over a defect that already has a correct fix in review.

What is left after this: confirming the link above by running this corpus on a
build of #7726, which needs that branch built rather than argued from the diff.

### 7.9 The crash count was wrong, and the instrument was mine (2026-09-12)

`github_6759_02` never crashed. Its `test.desc` runs `--goto-functions-only`,
and the GOTO dump contains the literal string `uncaught exception` from ESBMC's
own exception machinery; the sweep classified a row as a crash with an
unanchored `grep 'SIGSEGV\|uncaught exception'` over the whole output, so the
dump matched itself. Re-running that row under the flag, with and without
`--no-irep2-native-body`, produces no crash at all.

So the figures are 50 real crashes before the arm and **0** after, not 51 and
1, and the same row inflated both ends. The classifier now anchors on the
diagnostic lines:

```sh
grep -qE '^ESBMC caught SIGSEGV|^ERROR: uncaught exception'
```

This is the second time in this section a clean-looking measurement was the
instrument's fault rather than the subject's — §7.6 was a `--symbol-table-only`
A/B that renders a typeless call identically to a typed one. Both belong in §4's
gates: a census must state what it greps for, and a sweep over program *output*
must anchor its patterns, because a dump quotes the verifier's own diagnostics
back at it.

### 7.10 The whole corpus, and the residue is three named buckets (2026-09-12)

The stride sample was a stand-in; this is the corpus. 509 rows measured — 525
directories less 8 `KNOWNBUG` and 8 with no source file for the flags line to
name — each run twice on one binary:

| bucket | rows |
|---|---:|
| verdicts agree | **441** |
| `migrate expr failed` | 63 |
| SIGSEGV | 3 |
| `cannot remove side effect (assign…)` | 2 |

**No row produces a wrong verdict.** Every residual row fails to produce one,
which is the failure mode to want: the hop-off declines loudly rather than
answering differently. That is worth stating plainly, because it is the
property the migration's gates exist to protect, and a 67-row divergence list
reads much worse than it is until the buckets are named.

Two figures moved while writing this up, both my instrument's fault rather than
the subject's. 441, not 439: two of the "divergences" were the pair this branch
itself adds, whose `test.desc` already pins the flag, so the sweep supplied it
a second time and ESBMC rejected the repeated option. The sweep now skips a row
that pins it, the way the Phase 7 reach probe already did. And the 63 is the
§7.8 bucket, still one cause, still #7726's.

**The 3 SIGSEGVs are new work**, and the stride sample missed all three:
`interface_7`, `struct_1`, `struct_2`. So is the 2-row `cannot remove side
effect (assign…)` bucket. Neither has been reduced yet; both are named here so
the next pass starts from a list rather than a sweep.

### 7.11 The 3 SIGSEGVs are one site, and it is in the solver (2026-09-12)

They do not reproduce from a reduced struct: a contract with a one-field
struct, its literal constructor, and a member read all convert and verify under
the flag. What the three rows share is not the construct but the *strategy* —
`--k-induction` on `interface_7` and `struct_1`, `--incremental-bmc` on
`struct_2` — and they crash only when symex actually runs:
`--goto-functions-only` on `struct_1` converts cleanly, and legacy with
`--k-induction` is fine.

`gdb -batch -ex run -ex bt` on two of the three gives the same top frame:

```
#0 smt_solver_baset::convert_assign   src/solvers/smt/smt_solver.cpp:366
#1 smt_convt::convert_assign          src/solvers/smt/smt_conv.cpp:80
#2 symex_target_equationt             src/goto-symex/equation/symex_target_equation.cpp:139
```

Line 366 is `side2->assign(this, side1)`, so an SSA assignment reaches the
encoder with sides it cannot assign across — the shape a sort or structure
mismatch takes, and the same shape as the mixed-width `ieee_fma` the C work
declined in §138.2 of the clang-c doc rather than hand to the solver.

So this bucket is **not** a missing adjust arm producing a nil: the body
converts. It is a type the pass leaves inconsistent, surfacing only once an
equation is built. Naming the mismatched assignment needs the two sides printed
at that frame, which `-O2` will not give up — the same wall as §7.5 — so it
wants the `fprintf` treatment next, not another A/B.

§7.12 answers which assignment, and refutes the sort-mismatch reading.

### 7.12 Not a sort mismatch: a nested member read the solver cannot project

The mismatch reading was wrong, and the probe that refuted it was written to be
able to. A `fprintf` at `smt_solver.cpp`'s assign site, firing only when
`eq.side_1->type != eq.side_2->type`, prints **nothing** before the crash: the
two sides' types are equal, so the assignment is not ill-sorted at the top
level.

What `gdb` does give up, once the fields are read directly rather than through
accessors it refuses to call:

```
info locals            side1 = 0xa54c2f0        side2 = 0x51
p eq.side_1.ptr_->expr_id                       expr2t::symbol_id
p eq.side_2.ptr_->expr_id                       expr2t::member_id
p eq.side_2.ptr_->type.ptr_->type_id            type2t::unsignedbv_id
p ((member2t*)eq.side_2.ptr_)->source_value.ptr_->expr_id
                                                expr2t::member_id
p …->source_value.ptr_->type.ptr_->type_id      type2t::struct_id
```

`side2` printed as `0x51`, and the crash is the virtual call on it rather than
the assignment itself. Treat the *value* with suspicion: these are `-O2`
locals, where `info locals` can show a stale register. What the frame supports
is that the RHS AST is unusable at the call, not that it is specifically `0x51`
— §7.14.

The RHS is a **nested member read**: `member(member(…, struct), unsignedbv)`,
which in `struct_1` is `this->book.book_id`. The outer component's name is an
`irep_idt` the debugger can only show as a pool index, so it is not resolved
here.

That makes the hypothesis worth testing next: the struct type reached through
the inner member differs between the two paths, so projecting the outer
component by name finds nothing and the flattener returns a bad AST. The pass
has two places that could do it — `adjust_struct` pads struct *literals*,
`pad_type_symbol` pads type *symbols* — and a type padded on one path and not
the other would behave exactly like this. Comparing the two struct types at
that site is the next instrument; it is not established yet, and the earlier
`--symbol-table-only` A/B cannot settle it, for the reason §7.6 records.
### 7.14 Both probes refute their hypothesis, and one earlier reading was over-read

The `fprintf` at `get_member_name_field`'s fall-off — printing the wanted name
and the names present whenever the scan runs off the end — **never fires** on
any of the three rows. So the name is always found, `idx` is in range, and
§7.13's out-of-range projection, though real as a mechanism, is not what these
rows hit.

That is two probes in a row that refuted the hypothesis they were built for,
which is the intended use: each was placed so that silence was an answer rather
than an absence of one. Where it leaves the bucket:

| claim | status |
|---|---|
| three rows, one frame — `side2->assign(this, side1)` | measured |
| crash needs symex — `--goto-functions-only` is clean, legacy is clean | measured |
| the assignment's two sides have equal types | measured (silent probe) |
| RHS is `member(member(…, struct), unsignedbv)` | measured |
| an ill-sorted assignment | **refuted** |
| a member name missing from its struct type | **refuted** |
| `convert_ast` returned the value `0x51` | **over-read** — an `-O2` local |

The retraction matters for the next step: with the lookup exonerated, the RHS
AST may be null rather than a stray non-pointer, and those two suggest
different culprits. So the next instrument prints `src` and its AST kind
*inside* `convert_member`, which separates "`project` misbehaves on a valid AST
and a valid index" from "the inner member conversion already failed and the
outer call inherited it".
### 7.15 Root cause: a padded struct type against an unpadded tuple (2026-09-12)

Printing `project`'s *result* as well as its input settles it. Both member
projections on the crashing path, with the source AST pointer, its sort kind,
the index, and what came back:

```
XPROJ src=0x2dad4d50 sort=6 idx=1 srckind=5    -> res=0x2dc28e10
XPROJ src=0x2dc28e10 sort=6 idx=3 srckind=58   -> res=0x51
```

The inner projection is fine. The outer one takes that AST and asks for field
**index 3**, and gets `0x51` back.

Read against §7.14, which established that `get_member_name_field` *finds* the
name: `idx = 3` is a valid position in the member-name list of
`member.source_value->type`, so that type has **at least four** members. `Book`
in `struct_1` declares three — `title`, `author`, `book_id` — so the type
carries a synthetic pad. And `project(3)` returning garbage means the AST's
tuple has **at most three** fields.

So the expression's struct type and the AST's tuple sort disagree on member
count: **the type is padded, the AST was built unpadded.** That is §7.11's
padding hypothesis with both halves measured instead of assumed, and it
explains why the two probes before it were silent — neither the assignment's
types nor the name lookup is wrong. The disagreement is between a type and an
AST built from a different version of the same type.

It also reverses §7.14's retraction of `0x51`. That value is real: it is
`project`'s return, printed by the probe, not an `-O2` local. The caution was
right in kind and unnecessary in fact, and what resolved it was printing the
*result* beside the input — a probe that reports only its inputs cannot tell
"went in bad" from "came out bad".

Two candidates for which side is stale, and this does not yet choose between
them: `pad_type_symbol`, which pads type symbols under `sole_adjuster`, and
whatever built the tuple sort — a sort cached from the symbol's type before
padding would behave exactly like this. Choosing needs the member counts of
`member.source_value->type` and of `src->sort` printed side by side, which is
one more line in the same probe.

Also worth separating out, as §7.13 noted for a different reason: `project`
taking an index it cannot bounds-check, from a lookup whose only guard is an
assert compiled out of release builds, turns any such disagreement into
undefined behaviour rather than a diagnosable failure.
### 7.16 Both counts measured: one struct type, padded in one place and not another

The inference in §7.15 was indirect — a found index on one side, a bad pointer
on the other — so the probe was extended to print both counts outright, firing
only when the index is out of range for the AST:

```
XCNT OOR idx=3 type_members=4 ast_members=3
```

The expression's struct type has **four** members: `Book`'s three declared
(`title`, `author`, `book_id`) plus a synthetic pad. The AST's tuple sort has
**three**. So the projection is out of range, and the type is padded while the
AST is not — as inferred, now read directly.

The locating detail is *which* AST. The inner projection returned a valid
struct AST with three fields, and that AST is the outer member's source. Its
sort was not built from the outer member's `source_value->type`, which has
four; a projected field's sort comes from its **parent tuple's** declared field
sorts. So the enclosing struct declares its `Book` field with an *unpadded*
`Book`, while the member expression reading that field carries the *padded*
one. Two `Book` types coexist, and they disagree.

That is the defect, stated as narrowly as the evidence allows: the pass pads
some occurrences of a struct type and not others, so a field's declared type
inside an enclosing struct disagrees with the type on expressions that read it.
`pad_type_symbol` pads *type symbols* under `sole_adjuster` and `adjust_struct`
pads struct *literals*; a `Book` inlined into another struct's member list is
reached by neither. Which of those two should also cover it is a
padding-ownership question larger than this branch, and it is not answered
here.

Three probes were needed and two of them refuted their own hypothesis, which is
the shape to want: each printed something whose *absence* was also informative.
The one that finally landed differs from its predecessors only in printing both
sides of the comparison rather than one.
### 7.17 The last unexplained bucket, closed by two docstrings (2026-09-12)

Both `cannot remove side effect` rows abort on `(assign_shr)`, and the cause is
written down in the tree twice over. `clang_c_adjust_expr.cpp`:

> The C converter now picks the kind (§76); Solidity still emits the untyped
> `assign_shr`, so the rewrite below stays for it.

so legacy resolves `>>=` to `assign_lshr` or `assign_ashr` by the target's
signedness, and `goto_sideeffects.cpp` handles only those two. The IREP2 arm
returns early on every shift spelling, and its helper says why that looked safe:

> The shift spellings clang_c_adjust returns early on: it promotes only the right
> operand there, which **the corpus shows** is already the migrated shape.

That was measured on the **C** corpus, where the converter resolves the kind.
Solidity emits a shape the C corpus does not contain, so the early return was a
sound conclusion from an incomplete population — which is §2's argument for
wiring this frontend in as a second corpus, paying for itself.

`adjust_compound_assignment` now performs the rewrite before that early return,
guarded as legacy guards it: `assign_shr`, a numeric right operand, and a
signed or unsigned target.

| row | before | after |
|---|---|---|
| `compound_assign_1` | `cannot remove side effect` | **agrees** |
| `op_binary_3` | `cannot remove side effect` | `migrate expr failed` |

So the bucket is empty and the **total residue is unchanged**: one row closes,
one advances past its first blocker onto §7.8's, which already has an owner in
#7726. Worth stating that way round — "two rows fixed" would be wrong.

`irep2_only_shift_assign_kind{,_fail}` pin it, and both halves flip to the
abort when the rewrite is disabled: an abort prints no verdict line, so neither
`^VERIFICATION SUCCESSFUL$` nor `^VERIFICATION FAILED$` is satisfiable without
it. `irep2_only` is 101 of 101, and the default path is unchanged at 527 of 529
with the same two `KNOWNBUG` rows that already passed.

Phase 8's residue is now 64 rows owned by #7726 and 3 by §7.16's padding
disagreement, with nothing unexplained, against 442 of 509 agreeing and no row
anywhere answering differently.
### 7.18 The padding fix that did not work, and what it rules out (2026-09-12)

§7.16 offered two candidates for the stale side. The first was tried and is
wrong.

`pad_type_symbol` pads only the top-level type of each type symbol, while
`clang_c_adjust::adjust_type` recurses — it walks each component before padding
the enclosing type, so a struct inlined into another's member list is padded
too. That looked like the whole story, so the IREP2 side was made to recurse
the same way (array subtypes, then components, then the enclosing type, leaning
on `add_padding`'s idempotence, which `adjust_type` asserts). All three rows
still SIGSEGV, so the change was reverted rather than parked: a change that
fixes nothing measurable should not ship.

What that rules out is useful. The short type is **not** a nested type symbol
the pass failed to reach, and the tree says where it does come from —
`adjust_struct`'s own comment:

> The literal's own type is an inline copy the converter recorded before
> `add_padding` ran, so `ns.follow` leaves it short the synthetic members.

So the 3-member `Book` is an inline copy carried **on an expression**, while
the 4-member one is the same struct resolved through the padded symbol table.
Two versions of one type in one tree, which is what §7.16 measured, but the
stale copy is on an expression and no amount of padding type *symbols* reaches
it.

That reframes the row: it is not a padding-ownership question but the seam
question §2.1 raised from the other end — which types on expressions are inline
snapshots and which resolve through the table. `adjust_struct` repairs that for
struct *literals* by padding their operands; nothing repairs it for a struct
type reached through a member read. The candidate fix is therefore to resolve
such a type through the table at the point the member is adjusted, in the way
`adjust_call_signature` already does for a callee's `code` type (§7.7) — the
precedent is on this branch, and it is the same class of repair.
### 7.19 S.2 done: the census names the symbol, and the bucket is not uniform

`migrate_census` was a `static` in `clang_cpp_language.cpp`. It walks a
`contextt` and is frontend-agnostic, so it moves into
`src/util/irep/migrate.{h,cpp}` beside the migration it measures, the C++
frontend's copy is deleted, and both frontends call one definition. Two copies
that can drift is the arm-table hazard (§7.7) applied to a helper.

**It answers a different question depending on the other flag, and that is easy
to misreport.** On `enum_2`, census alone:

```
IREP2 migrate census: 807 symbols, 357 values migrated, 8 type kinds, 0 failures
```

and census with `--clang-cpp-irep2-adjust-only`:

```
ERROR: IREP2 migrate census: migrate expr failed:  on symbol
       sol:@C@FreshJuiceSize@F@FreshJuiceSize#
IREP2 migrate census: 807 symbols, 356 values migrated, 8 type kinds, 1 failures
```

The first prices the **converter's** output, which is clean. The second prices
the **round trip** through the IREP2 pass. Quoting the first as "migration is
clean" would be wrong in the way §7.9 and §7.10 were wrong: a measurement that
runs, produces a plausible number, and answers a different question than the
one asked.

**What it says about the residue.** Over the 65 diverging rows it can run (two
pin the flag themselves):

| failures reported | rows |
|---|---:|
| 1 | 58 |
| 2 | 3 |
| 3 | 1 |
| 0 | 1 (`compound_assign_1`, closed by §7.17) |
| no census line | 2 (`bitwise_ops_1`, `op_binary_1`) |

and the failing symbols are **29 constructors and 33 methods** — so the bucket
is not constructor-shaped, as the one reduced row suggested, and some rows
carry more than one failure.

What this does *not* establish: that all 62 share §7.8's `cpp_new` size cause.
The error text is identical everywhere — `migrate expr failed:` with an empty
id, which is what an empty irept prints — and the one row traced by backtrace
was that cause, but identical text is not identical cause. #7726 merging is the
cheap test: re-run this census on that branch and the bucket either empties or
splits.

**Two flaws in the census harness, found before its numbers were used.** The
first run reported "46 rows: 1 failure, 15 rows: 0" and both figures were junk:
the symbol grep assumed no spaces between "census:" and "on symbol", where the
text is `migrate expr failed: `, so every symbol read as absent; and the script
passed each row's source file but dropped its **flags line**, losing
`--contract` and converting a different contract than the row verifies. That is
the fourth harness error in this section, against a comparable number of real
defects. The standing correction: do not quote a sweep figure without
re-reading what the sweep passed and what it grepped.
### 7.20 The #7726 link, tested rather than argued (2026-09-12)

§7.19 refused to claim that all 63 `migrate expr failed` rows shared §7.8's
cause: the error text is identical everywhere because an empty irept prints as
nothing, and identical text is not identical cause. So it was tested.

Cherry-picking `d28be5a7e2` conflicts — S.2 touches the same file — and that
commit also moves the typecast arms and the pre-dispatch forms, none of which
this question needs. So only the tri-state size selection was hand-applied, on
a scratch branch, marked in-source as an experiment:

```cpp
const auto carries_size = [](const irept &i) {
  return !i.id().empty() && !i.is_nil();
};
```

Census over the same 65 rows, before and after:

| failures reported | before | with the guard |
|---|---:|---:|
| 1 | 58 | 0 |
| 2 | 3 | 0 |
| 3 | 1 | 0 |
| 0 | 1 | **63** |
| no census line | 2 | 2 |

Every row with a migrate failure clears, across both symbol shapes — 29
constructors and 33 methods — so the bucket really is one cause, and #7726
closes all of it. The generalisation held; the point is that it was checked
instead of assumed.

A verdict sweep with the guard applied reports **64 agree, 0 diverge, 1 crash**
over its sample, the crash being `struct_2` from §7.16. Read that against the
56/9/0 of §7.9 with care: the two stride samples do not have identical
membership, because the skip lists differ between the two versions of the sweep
(one skipped no `KNOWNBUG` row, the other one). The robust statement is that
divergences in the sample fall to zero and the only residual is the padding
crash.

This is #7726's fix, not work owed here. The guard stays on the scratch branch;
what is owed is a note on that PR that this corpus exercises it across 63 rows.

One measurement detail worth stating, since it makes two numbers non-comparable
if left out: `enum_2` cannot answer at all without `--goto-functions-only` —
the full run exceeds 200 s — so the census and the verdict sweep measure that
row at different depths. Fine for counting migrate failures, which precede
symex; not fine to quote side by side unremarked.
### 7.21 A fourth bucket, and why this one had to be fixed at the seam

The two rows S.2 could not census at all (§7.19) name themselves, unlike the
empty-irept bucket:

```
ERROR: migrate expr failed: shr
```

An id, not a blank. `migrate_expr` has no handler for a plain `shr`, and cannot
have one usefully: **IREP2 has no kind-less shift node** — `is_shift` covers
`shl2t`, `ashr2t`, `lshr2t`. `clang_c_adjust` resolves it before anything
migrates, by the left operand's signedness (`unsignedbv` -> `lshr`, `signedbv`
-> `ashr`).

This is the *expression* twin of §7.17's `assign_shr`: the Solidity converter
emits both `>>` and `>>=` without picking the kind, the C converter picks both,
and the
IREP2 side assumed resolution because the C corpus always had it. Three of Phase
8's four fixes are now that same defect class.

**The asymmetry is where each has to be repaired.** `assign_shr` survives
migration, so an adjust arm can rewrite it. `shr` cannot be migrated at all, so
no arm ever sees it and the seam is the only code on the flag path that runs
early enough. That is a more invasive place for a semantic decision, so the
helper says why rather than leaving a bare special case.

| row | before | after |
|---|---|---|
| `bitwise_ops_1` | `migrate expr failed: shr` | **agrees** |
| `op_binary_1` | `migrate expr failed: shr` | `migrate expr failed:` (§7.8's bucket) |

One row closes; one advances past `shr` onto the `cpp_new` size blocker that
§7.20 showed #7726 closes.

**The complexity gate rejected the first attempt, correctly.** Three branches
added inline took `migrate_expr` from 291 to 294, and the gate blocks any
increase on a function already over threshold. #7726's own commit message
describes the remedy — move arms out into named helpers — so the existing
`lshr` and `ashr` arms were extracted alongside the new `shr`: two branches
removed, one added, net decrease, gate clear. Pleading the case was not an
option, and should not have been the first instinct.

Because the extraction moved arms every frontend uses on the **default** path,
the regression net is wider than for a flag-gated arm: 864 of 864 unit tests,
103 of 103 `irep2_only`, both shift pairs re-run against the refactored build
rather than assumed, and the Solidity default path unchanged.

One thing to hand forward: `adjust_compound_assignment` now sits at CCN **15**,
exactly the `core` gate line, so the next branch added there fails the gate.
Extract before adding.
### 7.22 Why the padding row is not a one-arm port (2026-09-12)

§7.18 ruled out nested type symbols. The reading is sharper than that, and it
explains why the obvious repair cannot work. Both numbers come from the **same**
expression:

- `get_member_name_field(member.source_value->type, …)` finds **4** names, so that
  expression's type is the padded `Book`;
- `convert_ast(member.source_value)` yields a tuple of **3**, so the AST was built
  from a different version of that type.

The AST cannot come from the expression's own `->type`. The inner member's AST
is `src->project(idx)`, and a projected field's sort comes from the **parent
tuple's** declared field sorts — so the grandparent's struct type carries an
*unpadded* `Book` component while the child member's own type is the *padded*
one. The disagreement is within a single expression tree, between a
grandparent's component type and a child's type.

That kills the repair §7.18 proposed. Retyping the member, or its source, does
not touch the AST: the sort is already fixed by the grandparent. Nor does
padding type symbols harder, which §7.18 measured. The fix has to retype the
**base** of the member chain so the whole tuple chain is built from padded
types — a recursive type-normalisation over expressions, not an arm.

That is a design question about which types on expressions are inline snapshots
and which resolve through the table — §2.1's seam question, reached from the
other end — and it is not worth a speculative broad change for 3 rows of 509.
It is recorded here as the remaining Phase 8 defect, with the two dead ends
marked so the next attempt does not repeat them:

| attempt | outcome |
|---|---|
| pad nested type symbols recursively (`pad_type_tree`) | no change; the stale type is on an expression |
| resolve the member's own source type through the table | cannot work; the AST's sort predates it |
| retype the base of the member chain | untried, and the only one that reaches the sort |
### 7.23 The corpus after four fixes, and what is left (2026-09-12)

Full sweep, 531 directories, 507 measured — 8 `KNOWNBUG`, 6 that pin the flag
themselves (this branch's own tests, skipped by the rule §7.10 added), and the
rest lacking a source for their flags line to name:

| | earlier (§7.10) | now |
|---|---:|---:|
| verdicts agree | 441 | **441** |
| diverge | 67 | **63** |
| crash | 3 | 3 |
| measured | 509 | 507 |

Divergences fall by four: `bitwise_ops_1` and `compound_assign_1` agree, and
the two artefacts §7.10 found are now excluded rather than miscounted. The
agreement count is flat because the rows the other two fixes touched moved
*within* the residue rather than out of it — `op_binary_1` and `op_binary_3`
each advanced past their first blocker onto §7.8's.

**Nothing answers differently.** Of the 63 divergences, rows where both paths
reach a verdict and disagree: **zero**. Every residual row declines — loudly,
with an error — rather than returning a different answer. That is the property
the migration's gates exist to protect, and it has held through every sweep in
this section.

The residue is fully owned:

| bucket | rows | owner |
|---|---:|---|
| `migrate expr failed` | 63 | #7726, measured closing all 63 (§7.20) |
| padding disagreement | 3 | §7.22, open — `interface_7`, `struct_1`, `struct_2` |

So with #7726 merged this corpus reaches **504 of 507**, and the only open
defect is the type normalisation §7.22 describes. Phase 8 opened on a roadmap
entry that recorded the suite as unmeasurable (§1.1); it now has a measured
census, the hop-off wired, four mutation-checked fixes, and a residue with
named owners.

What is *not* done, and should not be read as done: S.3, the converter's 1 685
construction sites (§1.2), which is the phase's actual bulk. Everything above
is the adjust and seam half — the half Phase 7 shares — and it is what made the
converter's half measurable, not a substitute for it.
### 7.24 S.3 opened: the converter's output has no representational wall

Before ranking 1 685 sites, the prior question: does what `solidity_convertert`
builds fit in IREP2 at all? An IR that cannot hold a construct would gate the
whole phase behind a representation change, as W3 did for earlier ones.

The migrate census answers it when run **without** the hop-off flag — that
configuration prices the converter's own output rather than the round trip,
which is the distinction §7.19 had to learn the hard way. Stride-8 sample:

```
--- converter census, stride 8: 65 rows, 65 clean, 0 with failures ---
```

Every row migrates: `s.get_type2()` and `s.get_value2()` succeed on every
symbol the corpus produces. So **S.3 has no W3-style wall.** Its risk is volume
and behaviour preservation, not representation, and the porting order can be
chosen on cost rather than dictated by a blocker.

Two limits on that claim, stated rather than left implied. It is a stride-8
sample, not the whole corpus; and it measures only what the corpus exercises —
a converter path no test reaches is unmeasured either way, which is §4's own
rule about a census showing the thing under test executed.

**Pathfinder.** `solidity_convert_literals.cpp` is 159 lines with 15 sites
(§1.2): the smallest self-contained target, and the right place to establish
the pattern before `solidity_convert_call.cpp` (404) and
`solidity_convert_expr.cpp` (250), which are 39 % of the surface between them.
Phase 5 used jimple the same way — smallest surface first, as the kit's
pathfinder.

What S.3 inherits from §7's half: a corpus that agrees on 441 of 507 rows with
nothing answering differently, so a converter change that breaks something will
show as a *new* divergence against a known baseline rather than disappearing
into noise. That baseline is the deliverable of this section, more than any
single fix in it.
### 7.25 S.3's first blocker, before a line is ported: `#cformat`

§7.24 said the converter's output faces no representational wall. That is true
of *forward* migration, which is what the census measures, and it is not the
whole question. Any boundary that stays legacy needs the **reverse**, and the
reverse loses something.

The pathfinder's first function, `convert_integer_literal`
(`solidity_convert_literals.cpp`), builds

```cpp
the_val = constant_exprt(
  integer2binary(z_ext_value, bv_width(type)),
  integer2string(z_ext_value),
  type);
```

and that three-argument constructor records the decimal spelling as `#cformat`
(`util/irep/std_expr.h`):

```cpp
constant_exprt(const irep_idt &_value, const irep_idt &_cformat, const typet &_type)
{
  set("#cformat", _cformat);
  set_value(_value);
}
```

A native port would build `constant_int2tc(type, value)` instead, and
`migrate_expr_back` reconstructs a `constant_exprt` from **`value` alone**
(`migrate.cpp`, `constant_int_id`): it sets the binary string and nothing else.
`#cformat` does not survive the round trip.

So porting this one function, with its callers left legacy, either carries
`#cformat` across the seam or changes what ESBMC *prints* for every Solidity
integer literal — counterexamples and goto dumps include it. That is a
default-path output change of the kind the clang-c doc's §137.5 treats as
SV-COMP-relevant, not a silent internal refactor.

Three ways out, none free, and the choice belongs to S.3's design rather than
to this section:

| option | cost |
|---|---|
| carry `#cformat` through `migrate_expr_back` | a seam change affecting every frontend's integer literals |
| port the callers too, so no round trip happens | pushes the boundary into `solidity_convert_expr.cpp` (250 sites) |
| accept the printed-output change | needs an SV-COMP run and a sweep of tests that pin literal spellings |

This is the same shape as the spelling carriage `scope-c-spelling-carriage.md`
records, arrived at from a different direction. Worth knowing before 1 685
sites are ranked: the smallest file in the phase is blocked on a seam question,
so "mechanical" in §7.24 means *representable*, not *free*.
### 7.26 504 of 507, measured on the merged tree (2026-09-12)

#7717 and #7742 landed on master, and #7742's merge carried the stacked #7726
content with it — `cpp_new_size` is on master although #7726 is still open as a
PR. So the fix §7.20 measured on a scratch branch is now the shipped one.

That had to be re-measured rather than carried across: what landed is #7726's
fuller restructuring of the cast arms and pre-dispatch forms *around* the
guard, not the hand-applied `carries_size` lambda alone. Full sweep against the
merged tree:

```
--- 507 measured (stride 1), skipped 8 KNOWNBUG/FUTURE: agree=504 diverge=0 crash=3 ---
```

| | §7.10 | §7.23 | now |
|---|---:|---:|---:|
| verdicts agree | 441 | 441 | **504** |
| diverge | 67 | 63 | **0** |
| crash | 3 | 3 | 3 |

Zero divergences of any kind, and zero rows where both paths reach a verdict
and disagree — the property that has held through every sweep in this section.
The whole residue is §7.22's padding row: `interface_7`, `struct_1`,
`struct_2`.

So the adjust-and-seam half of Phase 8 is closed but for one defect, at 3 rows
in 507. The phase opened on a roadmap entry recording the suite as unmeasurable
(§1.1), and the first sweep after wiring the flag had 50 of 65 sampled rows
crashing (§7.1, §7.9).

What of that is this branch's: four mutation-checked fixes —
`adjust_call_signature` (§7.7), the `assign_shr` rewrite (§7.17), the shared
migrate census (§7.19), and the kind-less `shr` at the seam (§7.21) — plus the
flag wiring and the baseline itself. The 63-row bucket is #7726's, and this
corpus's contribution there was to measure that it closes all of them.

What is *not* done: S.3, the converter's 1 685 sites, now gated on §7.25's
`#cformat` question rather than on anything in this section.