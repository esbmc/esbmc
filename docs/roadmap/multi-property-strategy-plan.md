# Plan — `--multi-property` under the k-step strategies

**Status:** In progress. W1, W2, W2b, W2c, W3a (#7923), W3b and W3c landed.
**Origin:** Discussion
[#7900](https://github.com/esbmc/esbmc/discussions/7900), *"Current state of
--multi-property support"*: is `--multi-property` orthogonal to the analysis
mode, and which modes can its results be trusted under? Related open issues:
[#1361](https://github.com/esbmc/esbmc/issues/1361),
[#1902](https://github.com/esbmc/esbmc/issues/1902),
[#1599](https://github.com/esbmc/esbmc/issues/1599),
[#2075](https://github.com/esbmc/esbmc/issues/2075),
[#7503](https://github.com/esbmc/esbmc/issues/7503).
**Last updated:** 2026-09-23.

**Measurement environment.** aarch64 macOS, ESBMC 8.5.0 built from master
`25b71af213`, default solver (Bitwuzla 0.9.1). Every result below is a verdict
or a property-table row, not a timing, so it does not depend on the host.

---

## 1. The answer today

`--multi-property` is not orthogonal to the analysis mode. The per-claim loop in
`bmct::multi_property_check` is sound for a single BMC run. The strategies that
run BMC more than once — one run per k step, or one per process — each carry
their own handling of it, and every one of them differs:

| Strategy with `--multi-property` | Behaviour | Trust |
|---|---|---|
| `--unwind N` (± `--interval-analysis`, `--parallel-solving`, `--smt-during-symex`) | Every violated claim reported, one table, verdict matches — except that claims sharing a comment and a location share a row (D5) | Yes, per row |
| `--k-induction`, `--incremental-bmc` (± `--interval-analysis`) | Every violated claim reported, one table at the end with stable ids, verdict matches — except D5 | Yes, per row (W1, W2, W3b) |
| `--falsification` | Reports the violations it reaches; a claim it never settles is UNKNOWN, not PASSED (W3b). It leaves the k loop at the first violation, so a claim violated at a larger k stays UNKNOWN and its table prints after that phase's verdict (D4) | Failed rows only |
| `--k-induction-parallel` | Stops at the first violation and prints PASSED for claims violated at a larger k (D2, D4, D5) | Failed rows only |
| `--falsify-context-bound` | Rejects the combination with an error (`bmc_strategy.cpp:608-622`) | n/a |

`--loop-invariant` runs the same k-step loop (`driver.cpp:297-300`), so W1, W2
and W3b reach it too. `--loop-invariant-check` is not routed there (measured: no
`Checking base case` line), so it runs once and only D5 applies.

D1 was the one wrong *program* verdict — an `--incremental-bmc` run that had
printed violations could still end `VERIFICATION UNKNOWN`, exit 0 — and W1
closed it. Everywhere else the verdict is right and it was the per-property
rows that were wrong, which is the whole point of the flag. D5 is what W3a
leaves; the rest of this table is history the defect sections still record.

§8 is the reply to the discussion once the fixes land; until then this table is
the reply.

---

## 2. Reproducers

Each goes into the PR that fixes the defect it pins. `i1361.c`, `vac2.c` and
`vac3.c` have two violated claims each, the second reachable only after a loop
has run; `vac2_safe.c` and the `assert(i < 5)` variant of `i1361.c` are the
passing halves of the test pairs.

**`i1361.c`** — the program from #1361.

```c
#include <assert.h>
int main()
{
   for(int i = 0; i< 2; i++)
   {
      assert(2==3);
   }
   assert(1==2);
   return 0;
}
```

`assert(2==3)` fails at k = 1, `assert(1==2)` at k = 3.

**`vac2.c`**

```c
#include <assert.h>
int main()
{
  unsigned n = nondet_uint();
  unsigned x = 0;
  assert(n != 7);                /* violated at k = 1 */
  for (unsigned i = 0; i < n; ++i)
    ++x;
  assert(x != 3);                /* violated at k = 4 */
}
```

**`vac2_safe.c`** — the safe twin of `vac2.c`, for the passing half of W1's
and W4's test pairs. The forward condition closes at k = 5, so it already ends
`VERIFICATION SUCCESSFUL`, exit 0, under `--incremental-bmc`, `--k-induction`
and `--k-induction-parallel`; it pins that neither workstream turns a proof into
FAILED. Under `--falsification` it ends `VERIFICATION UNKNOWN`, exit 0: that
branch runs base cases only (`bmc_strategy.cpp:461-475`) and has no forward
condition, so it can report no proof at all.

```c
#include <assert.h>
int main()
{
  unsigned n = nondet_uint();
  unsigned x = 0;
  __ESBMC_assume(n < 5);
  for (unsigned i = 0; i < n; ++i)
    ++x;
  assert(x == n);
  assert(x < 5);
}
```

**`vac3.c`** — a control for the case D3 is not: a claim violated early must
not be assumed while proving a later one.

```c
#include <assert.h>
int main()
{
  int x = 0;
  while (1)
  {
    assert(x < 3);    /* violated when x == 3 */
    assert(x != 10);  /* violated when x == 10 */
    ++x;
  }
}
```

`vac3.c --k-induction --multi-property --max-k-step 15` reports `x != 10`
failed at k = 11. Sequential k-induction does **not** keep a violated claim as
an assumption, so that is not the cause of D3.

**`d5.c`** — two claims with one signature.

```c
int a[4];
int main()
{
  unsigned i = nondet_uint(), j = nondet_uint();
  __ESBMC_assume(i < 4);
  int s = a[i] + a[j];   /* a[i] safe, a[j] not */
  return s;
}
```

---

## 3. Defects

### D1 — `--incremental-bmc` ends UNKNOWN after finding violations

```
$ esbmc vac2.c --incremental-bmc --multi-property --max-k-step 10
  FAILED       [main.assertion.1]  line 6  assertion n != 7
  ...
  FAILED       [main.assertion.1]  line 9  assertion x != 3
Bug found (k = 4)
...
Unable to prove or falsify the program, giving up.
VERIFICATION UNKNOWN          # exit 0
```

The loop in `bmc_strategy.cpp` records `any_violation_found`, and the
`conclude()` lambda (`:253`) honours it. The exhausted-k exit at the end of the
same function (`:501-503`, `log_fail("VERIFICATION UNKNOWN"); return 0;`) does
not. `conclude()` is reached only from a phase that settles the program: the
forward condition (`:458` for `--incremental-bmc`) or, under `--k-induction`,
the inductive step. On `i1361.c` the forward condition does close, and
`--incremental-bmc --multi-property` ends `VERIFICATION FAILED`, exit 1. The
UNKNOWN exit is what a program whose loop the strategy cannot exhaust gets —
`vac2.c`, whose trip count is nondeterministic. Every violation has been printed
by then; only the verdict and the exit code contradict them.

The run also does not stop once every claim has a verdict: it keeps unwinding
to `--max-k-step`. #1902 reports the same symptom (k keeps growing) on a
loop-free program. A two-assert loop-free program does *not* reproduce it:
both strategies stop at k = 2. #1902's own program (`--overflow
--memory-leak-check`) is re-triaged in W5.

### D2 — per-step tables, renumbered ids, interim SUCCESSFUL

Each k step runs `multi_property_check` on that step's equation and prints its
own property table. Three things follow:

1. **Interim success lines.** The k = 1 base case of `vac3.c` prints
   `** 0 of 2 properties failed, 2 passed` and `VERIFICATION SUCCESSFUL`, for a
   program where both claims fail.
   PASSED there means "not violated within k", a bounded fact, printed in the
   form of a proof.
2. **Ids are per equation, not per program.** In `vac2.c` the k = 1 table
   prints `x != 3` as `main.assertion.2`. Once `n != 7` has failed and is
   skipped, later tables print the same claim as `main.assertion.1`. A
   consumer that joins tables by id joins the wrong rows.
3. **No final table.** The last line is a verdict with no summary of which
   claims failed, passed, or were left undecided.

### D3 — a claim missed under `--k-induction --interval-analysis`

```
$ esbmc i1361.c --k-induction --interval-analysis
  FAILED       [main.assertion.1]  line 6  assertion 2==3
  FAILED       [main.assertion.2]  line 8  assertion 1==2
$ esbmc i1361.c --k-induction --interval-analysis --multi-property
  FAILED       [main.assertion.1]  line 6  assertion 2==3
Bug found (k = 1)
...
Solution found by the inductive step (k = 2)
VERIFICATION FAILED
```

`assert(1==2)` is never reported, and the inductive step at k = 2 accepts a
program in which it is violated at k = 3. The verdict is still FAILED because of
the first claim, so this is not a wrong program verdict. It is a wrong
per-property result, and that is the whole point of `--multi-property`.
Rewriting the loop body to `assert(i < 5)` (it holds) makes the same flags find
`1==2` at k = 3, so the earlier violated claim is involved.

What is measured: `--interval-analysis` folds both claims to `ASSERT 0` and
deletes the rest of the loop body after the first one
(`--interval-analysis --goto-functions-only`). What is **not** established is
why the inductive step then discharges the second claim. That is the first
task of W2.

### D4 — `--falsification` and `--k-induction-parallel` stop with a bad table

Both stop at the first violation (`i1361.c`, `vac2.c`, `vac3.c`). On `i1361.c`
the second claim is simply absent from the table. On `vac2.c` and `vac3.c` it is
present and wrong:

```
$ esbmc vac2.c --falsification --multi-property --max-k-step 10
  FAILED       [main.assertion.1]  line 6  assertion n != 7
  PASSED       [main.assertion.2]  line 9  assertion x != 3
** 1 of 2 properties failed, 1 passed
$ esbmc vac2.c --unwind 10 --multi-property
  FAILED       [main.assertion.3]  line 9  assertion x != 3    # id 2: unwinding
```

`--k-induction-parallel` prints the same rows on `vac2.c`, and both print
`PASSED ... assertion x != 10` on `vac3.c`, where it is violated at k = 11.

Each PASSED there is the base-case result at the k the run stopped at — "not
violated within k" — recorded as a proof by the UNSAT arm of
`multi_property_check` (`bmc.cpp:3154`), which does not consult the `bs` in
scope at `:2892`. Under `--k-induction` and `--incremental-bmc` the same row
appears, but the run continues and a later table corrects it (D2); here the run
stops, so it is the answer.

The wrong row reproduces without `--multi-property` (`vac2.c --falsification
--max-k-step 10` alone prints it), so the fix belongs at the recording site, not
in the strategy loop — it is the same fix §4 requires. What is specific to
this pair is the early return: `--falsification` leaves the k loop
unconditionally on a violation (`bmc_strategy.cpp:467`,
`if (violated && !is_coverage) return 1;`), so no later k can correct the row.
`--k-induction-parallel` forks base case, forward condition and inductive step
into three processes (`k_induction.cpp:163-183`) that report back over a pipe
as `resultt` values; the parent stops at the first result that settles the
program, and no per-claim state crosses the boundary.

### D5 — claims are matched by comment and location

`claim_slicer` builds two signatures for a claim (`slice.cpp:390-397`):
`claim_msg`, the claim's guard expression, and `claim_cstr`, its
`comment + " at " + location`. Outside `--assertion-coverage` the skip of
already-verified claims keys on `claim_cstr` (`bmc.cpp:3013`), and so does the
verdict store the table is printed from (`bmc.cpp:3155`). The code's own comment
(`:3001`) says the signature is unsound. `goto-check` produces colliding pairs
routinely, because a comment names the check, not the expression checked:

```
$ esbmc d5.c --show-claims
  Claim 1: line 6 column 3, array bounds violated: array `a' upper bound
           (signed long int)i < 4
  Claim 2: line 6 column 3, array bounds violated: array `a' upper bound
           (signed long int)j < 4
$ esbmc d5.c --claim 1 → VERIFICATION SUCCESSFUL
$ esbmc d5.c --claim 2 → VERIFICATION FAILED
$ esbmc d5.c --multi-property
  FAILED       [main.array-bounds-violated.1]  line 6  array bounds violated: ...
** 1 of 1 properties failed
```

Two claims, one row. The table cannot say that the `a[i]` check holds, and under
a k-step strategy the surviving signature makes the second claim skipped at
every later k. `--keep-verified-claims` does not recover the row: it suppresses
the skip, not the collision in the verdict store.

Note that `claim_msg` does **not** collide here — the guards differ. A fix
that only rekeys the `claim_msg + "\t" + claim_loc` signature used by the
`--assertion-coverage` path (`bmc.cpp:3003-3008`) leaves D5 untouched. Unlike
D1–D4, this one affects plain BMC too.

### D6 — `havoc_slot` abort (not a multi-property defect)

```
$ esbmc vac3.c --k-induction --interval-analysis
Assertion failed: (loop_head->is_goto()), function havoc_slot,
file goto_k_induction.cpp, line 181.
```

Reproduces without `--multi-property`, so it is out of this plan's scope. It is
listed so that a matrix run in W6 does not attribute it to the flag. The
comment at `driver.cpp:196` records the same assert for
`--loop-invariant-check` + `--termination`. With interval analysis, a
`while (1)` loop whose head is an `ASSERT` hits it by another route.

### D7 — the `(c) || assert(0)` fold drops code (plain BMC)

```
$ esbmc pw.c --multi-property      # if (nondet) { __ESBMC_assert(0,"v"); x = 1; } assert(x == 0);
  FAILED       [main.assertion.1]  line  8  v
  PASSED       [main.assertion.2]  line 11  assertion x == 0
```

`x == 0` is violated on the path through the failed assertion.
`--goto-functions-only` shows the block folded to `ASSERT !nondet` with
`x = 1` gone. This is a wrong row under plain BMC, so §1's first row holds
only for programs without that idiom. W2b.

### D8 — k-induction havoc misses an entry past `assert(0)`

See W2c. Wrong row under `--k-induction --multi-property`, no interval
analysis needed.

---

## 4. Target semantics

Under a k-step strategy each claim moves through one lifecycle, keyed by its
**program** claim id (its position in the GOTO program's claim order, as in
`--show-claims`), not the per-equation index:

```
NotChecked ──base case SAT at k──▶ Failed(k)          final
NotChecked ──forward condition UNSAT at k──▶ Passed   final (bounded exhaustively)
NotChecked ──inductive step UNSAT at k──▶ Passed      final (proved)
NotChecked ──max-k reached──▶ printed as UNKNOWN      reported as such
```

- A base-case UNSAT records nothing, never `Passed` — and not `Unknown`
  either: the store's dominance order (below) ranks `Unknown` above `Passed`,
  so an `Unknown` recorded at k could never be overwritten by the proof a
  later forward condition or inductive step finds. The claim stays
  `NotChecked` until a final outcome, and a row still `NotChecked` when the
  run ends is printed as UNKNOWN. It means "not violated
  within k", and `multi_property_check` records `Passed` for it today
  (`bmc.cpp:3154`, which does not read the `bs` already in scope at `:2892`).
  That line is what puts the PASSED rows in D4's final table and in D2's
  interim ones, and a table with no failed row is what the interim
  `VERIFICATION SUCCESSFUL` follows from; without this rule the stop condition
  below turns bounded results into proofs. `all_properties_proved` (`bmc.cpp:3594`) already
  states the rule for the single-run path and exempts `--multi-property` from
  it; the exemption goes.
- A table is printed once, at the end, from
  `goto_functionst::property_verdicts`, the store the single-run summary
  (#7064, `report_property_verdicts`) already reads.
- The interim per-step output keeps its progress lines (`Bug found (k = 4)`)
  but prints no PASSED rows and no `VERIFICATION SUCCESSFUL`.
- The run stops as soon as no claim is `Unknown`.
- The exit code follows the table: any `Failed` → 1; otherwise all
  `Passed` → 0; otherwise `VERIFICATION UNKNOWN`.

`property_verdictt` (`property_verdict.h`) already has all four states,
ordered `NotChecked < Passed < Unknown < Failed`, so a later, weaker outcome for
a claim cannot overwrite a stronger one. That order is why a bounded base-case
result must leave the row `NotChecked` rather than record `Unknown` (above).
Most of W3 is routing the k-step strategies into that store rather than
adding a new mechanism.

---

## 5. Workstreams

One PR each, in this order. W1 and W2 are independent; W3 builds on W1, and
W4 on §4's base-case rule from W3.

### W0 — reply to #7900

Post §1's table as the answer, and link this plan. No code change.

### W1 — final verdict (D1) — done

- Route the exhausted-k exit through `conclude()`, so a recorded violation
  ends `VERIFICATION FAILED`, exit 1.
- **Tests.** `vac2.c --incremental-bmc --multi-property --max-k-step 10` and
  `vac3.c --k-induction --multi-property --max-k-step 6` →
  `^VERIFICATION FAILED$` with no `Unable to prove or falsify` line;
  `vac2_safe.c` under both strategies → `^VERIFICATION SUCCESSFUL$`, so W1
  does not turn a proof into FAILED; a `while (1)` program whose only claim
  is not violated within the bound → `^VERIFICATION UNKNOWN$`, so the new
  exit stays scoped to a recorded violation. `vac2.c` under `--k-induction`
  is **not** a W1 test: master already ends FAILED there through the
  inductive step, so it pins nothing.
- Mutation check: restore the old exit and confirm both FAILED tests change
  verdict.
- Labels: `needs-svcomp-run` (changes a strategy's exit code).

**Dropped: the early stop.** The second bullet of this workstream — after
each k step, stop when every claim has a final verdict — was implemented and
then removed as unsound. "No claim left" is only observable through the GOTO
program, because `clear_verified_claims_in_goto` is what records it, and a
claim generated during symex has no GOTO `ASSERT` behind it. Measured: in

```c
int main() {
  unsigned n = nondet_uint();
  assert(n != 1);                 /* the program's only GOTO ASSERT */
  int *p = 0;
  for (unsigned i = 0; i < n; ++i)
    if (i == 4) *p = 1;           /* NULL deref, first claimed at k = 5 */
}
```

the whole GOTO program holds one `ASSERT`. Violating it at k = 1 empties the
program, the stop fires, and `main.null-pointer-dereference.1` — which the
same program without the first assert reports at k = 5 — is never checked,
under a line claiming every property has a final verdict. The defect it would
report is silently lost; `--keep-verified-claims` recovers it, the default
path does not.

The generalisation is that a claim can first appear at a larger k, so no claim
set observed at k bounds the claim set at k+1. The only sound stopping signal
is proof that the program is fully unwound — the forward condition or the
inductive step — and both already return through `conclude()`. #1902's own
reproducer confirms there is nothing left to win here: it stops at k = 2
through the forward condition on master and on the W1 build alike. Anything
further belongs to W3, which gives claims program-wide ids and a verdict
store, and even there the bound-vs-proof distinction above still applies.

### W2 — the missed claim under interval analysis (D3) — done

**Cause.** `goto_programt::get_successors` gives an `ASSERT` whose guard is
the literal `0` no successor, as though a failed assertion ended the run.
Symex does not stop there: `int x = 0; assert(0); x = 1; assert(x == 0);`
under `--multi-property` reports both claims FAILED. By the time the
post-k-induction interval pass (`instrument_loop_bounds_after_kind`) runs,
`assert(2==3)` has been folded to `ASSERT 0`, so the fixpoint never follows
the loop's increment and back edge. It computes `i == 0` at the loop head and
inserts `ASSUME 0 == i` there for the inductive step. Once
`multi_property_check` clears the violated `ASSERT 0`, the path continues,
`i` becomes 1, the assumption kills it, and `assert(1==2)` is unreachable:
the inductive step generates no VCC and reports a proof.

A guard must be literally false to trigger it: `assert(2==3)` does once
folded, `assert(i < 0)` never does. `__ESBMC_assert(0, ...)` is literal from
the start, so the first interval pass (plain BMC included) was exposed too.
Without `--multi-property` the pruning costs nothing: it drops only states
after a reachable violation, and the run is already FAILED.

**Fix.** `ai_baset::continue_past_failed_assertions` restores the fall-through
edge that `get_successors` drops, and stops `--interval-analysis-assume-asserts`
from narrowing the state at an assertion, which had the same effect. Both
interval passes set it when claims after a violation are still checked:
`multi-property` (which `--parallel-solving`, `--all-witnesses` and
`--synthesise-loop-invariants` imply) or a coverage run. `remove_unreachable`
was skipped for a narrower, command-line-only version of that predicate, so it
still deleted the code after a failed assertion under `--all-witnesses`; both
now use one predicate. `get_successors` itself is unchanged.
`goto_contractor.cpp` builds its own interval analyses without the flag; it is
not compiled in the default build, so this was not measured.

- **Tests.** #1361's program (`--k-induction --interval-analysis
  --multi-property`) reports `1 == 2` FAILED and no inductive-step proof; its
  `assert(i < 5)` twin keeps the PASSED row. A `github_1092_2_true` variant is
  still proved only with the interval bound. `--interval-analysis-assume-asserts`,
  the first pass on `__ESBMC_assert(0, ...)`, and `--all-witnesses` each have a
  FAILED test that fails with its own fix site reverted, and a twin.
- Labels: `needs-svcomp-run`.

### W2b — the `(c) || assert(0)` fold drops code (D7) — done (#7917)

`goto_convert` folds `if (c) { assert(0); x = 1; }` to `ASSERT !c`, erasing
the statement after the assertion (`goto_convert.cpp`, `is_or_idiom` in
`goto_convert_functions.cpp`). Under plain `--multi-property`, with no other
flag, `assert(x == 0)` after that block is reported PASSED although it is
violated. Fix at the fold: apply it only when nothing follows the assertion,
or only when the run stops at the first violation. Test pair on that program.

### W2c — k-induction havoc skips an entry jump past `assert(0)` (D8) — done (#7918)

`reaches_back_edge` in `goto_k_induction.cpp` walks `get_successors`, so a
literal-false assertion between a jump into the loop and its back edge hides
that entry, and no havoc is inserted for it. `goto L;` into a loop whose body
starts `L: __ESBMC_assert(0, "v");` then gets its later `assert(i < 500)`
"proved" by the inductive step under `--k-induction --multi-property`. Fix in
the walk, with the same predicate as W2. Test pair on that program.

### W3a — claims that share a position (D5) — done

- `property_key` is the one key the verdict store and the skip set use. When a
  claim is its instruction's own assertion (the description is the
  instruction's comment), the key also names that instruction, so the two
  bound checks of `a[i] + a[j]` are two rows and
  one's violation no longer skips the other. Rows that still read the same
  print their condition: `[(signed long int)i < 4]`.
- The key does not change when the `ASSERT` becomes a `SKIP`.
  `clear_verified_claims_in_goto` does that in place once a claim is violated,
  clearing the guard, and a later instance of the claim -- another unrolled
  copy, or another interleaving under `--smt-during-symex` -- must key the same
  way. The instruction is named by its node, not its `location_number`:
  `--bidirectional` inserts an `ASSERT` mid-run and renumbers the program. The
  condition shown in a row is captured while the `ASSERT` is intact and
  rendered only for rows that collide and differ, so the default mode pays no
  `from_expr` per assertion.
- Residuals: `__ESBMC_assert(c, "")` gets the description `assertion <c>`,
  which is not its instruction's (empty) comment, so two such assertions at
  one position still share a row. Under `--parallel-solving`, reading a
  claim's guard races with another thread's `make_skip`; `claim_slicer`
  already did so before W3a.
- A dereference check raised while evaluating an assertion keeps description
  and position: the instruction is not that claim's assertion.
- Coverage goals keep the old key, which their reports print verbatim.
- **Tests.** `d5.c` under plain BMC, single-property BMC and `--k-induction`
  gives `i < 4` PASSED (or NOT CHECKED) and `j < 4` FAILED, pinned with the
  summary line so a duplicate row fails the test; its twin with both indices
  bounded is SUCCESSFUL with two PASSED rows. `condition_coverage_goal_key`
  pins a coverage goal line; `multi_property_same_position_asserts_loop` keys
  instances off a `SKIP`, and `same_position_assert_bidirectional` survives a
  renumbering. The three `synth_loop_invariant_calleeinv*` tests
  now show each synthesised clause as its own row.

### D9 — two claims symex raises at one instruction share a row

`return *p + *r;` with only `r` possibly NULL prints one
`null-pointer-dereference` row, FAILED: the `*p` check, which holds, has no row
of its own, and the same goes for the out-of-bounds and alignment pairs. Such
claims are raised during symbolic execution at an instruction that is not their
own assertion, so neither the instruction nor its guard tells them apart, and
the SSA step carries no un-renamed condition that would. Fixing it needs an
identity recorded where symex raises the claim. Plain BMC is affected. Open.

### W3b — one table across k steps (D2, #1361) — done

- The strategy owns the verdict store. `do_bmc_strategy` clears it once and
  sets `k-step-property-table`; every phase records into it; the phase that
  concludes the run prints the table, and where the k steps run out the
  driver prints it. A claim keeps one row and one id across every k, so
  `main.assertion.2` stays `main.assertion.2`.
- §4's lifecycle at the recording sites. A base case records a violation and
  nothing else: its UNSAT, its proof-cache hit and the simplifier's discharge
  in `goto_symext::claim` all mean "not violated within k", so the claim
  stays `NotChecked` and the forward condition or the inductive step settles
  it through `promote_unchecked_to_passed`. `all_properties_proved` no longer
  exempts `--multi-property` from the base-case rule, and `report_result`
  prints `No bug has been found in the base case` where it printed an interim
  `VERIFICATION SUCCESSFUL`.
- A row still `NotChecked` when a k-step run ends prints UNKNOWN: every base
  case checked it and none settled it, which is not the same as the
  single-run "this mode never separated it out".
- Two guards the cross-phase promotion needs. `kind-violation-found` stops
  disqualifying it, or a violation at k = 1 would cost every *other* claim
  the proof the forward condition finds; and a phase that stopped short
  (`--multi-fail-fast`, the interleaving budget) calls
  `property_verdict_tablet::note_incomplete`, which disarms it for the run.
- The forward condition runs with `--no-assertions`, so its only claims are
  the unwinding assertions that ask whether the loop is exhausted. They are
  the strategy's device, not the program's properties, and no longer seed the
  table.
- **Interface change.** Checked against `parse_result()` in
  `scripts/competitions/svcomp/esbmc-wrapper.py`, which reads `VERIFICATION
  FAILED` before `VERIFICATION SUCCESSFUL` and never sets `--multi-property`;
  `python3 scripts/competitions/svcomp/test_esbmc_wrapper.py` passes.
- **Tests.** `github_1361` and its safe twin under `--k-induction`;
  `multi_property_incremental_one_table` and its twin under
  `--incremental-bmc`; `multi_property_bounded_base_case_row` pins the
  bounded row as UNKNOWN under `--falsification`, and its twin pins that
  plain `--unwind` still reads a per-claim UNSAT as a proof.
- Closes #1361.
- Labels: `needs-svcomp-run`.
- Residuals, none of them unsound, each conservative in the direction that
  under-reports rather than over-reports:
  - `--falsification` still leaves the k loop at the first violation, so its
    table prints after that phase's `VERIFICATION FAILED` rather than before.
    W4 routes it through `conclude()` and the order follows.
  - `note_incomplete` latched for the whole run, but the invariant it guards
    is per k. Closed by W3c.
  - The exhausted-k table has no `Solver: … • Decision procedure total time`
    footer: `solver_stats` is per `bmct` and the driver has no run-wide total.
  - Nothing enforces "printed once". It holds because every phase for which
    `reports_final_verdict` is true also ends the strategy, and the two
    driver-side calls are on mutually exclusive returns, but W4 moves one of
    those returns.
  - `goto_symext::claim` applied the bounded-round rule through its own
    reading of the driver options. Closed by W3c's `withholds_proofs`.

### W3c — make the completeness signal per round — done

- A proof at k needs only that k's base case to have solved every claim: a
  base case at k covers every shorter path too. So the completeness flag is
  per round: `property_verdict_tablet::begin_round` marks the round
  incomplete at the start of each k-step base case, and `complete_round`
  clears it only when that base case reaches its report without leaving a
  claim undecided. A base case that throws -- an `--smtlib` solver process
  that dies, say -- never completes its round. One flag suffices; W3b's
  residual asked for a second.
- `withholds_proofs` (`property_verdict.h`) is the one rule both recording
  sites read -- the solver's UNSAT in `multi_property_check` and the
  simplifier's discharge in `goto_symext::claim`: under a k-step table, a base
  case withholds every `Passed`, and any other phase withholds it while its
  round is incomplete. That is what makes the diagnostic pass at max-k decline
  a proof when the base case at max-k stopped short; before, it recorded
  `Passed` regardless. It also closes W3b's last residual: both sites now
  demote or skip under the same predicate.
- A withheld proof leaves the claim in the program. The forward condition and
  inductive step turned every claim they discharged into a SKIP, withheld or
  not, so the next base case never saw it, completed its round, and the
  promotion proved a claim no base case had solved.
- `clear_verified_claims_in_goto` skips only the assertion that is the claim
  itself, not every assertion with its position and guard. `--loop-invariant`
  copies a loop body, so one copy's violation skipped the other; and a check
  raised inside an assertion -- the NULL-pointer check in `assert(*p == 2)` --
  skipped the assertion it was sliced from. Either way the next base case
  never solved the skipped claim, completed its round, and the promotion
  proved it. `goto_symext::assertion_message` names an assertion's own claim
  for both symex and this match.
- **Tests.** `multi_property_kinduction_fail_fast` now pins `PASSED`: the
  claim `--multi-fail-fast` skipped at k = 1 is solved at k = 2, and the
  forward condition proves it there. `multi_property_kinduction_diagnose_fail_fast`
  pins the diagnostic pass declining both proofs, the solver's and the
  simplifier's (`UNKNOWN`, partial-report note); `multi_property_kinduction_diagnose`
  pins that without the skip the same pass still proves them.
  `multi_property_loop_invariant_copy_fail` pins both copies of a violated
  claim `FAILED` under `--loop-invariant --multi-fail-fast`; matching by
  position and guard reports the second `PASSED`. Its twin
  `multi_property_loop_invariant_copy` pins both copies of a safe claim
  `PASSED`. `multi_property_kinduction_check_in_assert_fail` pins
  `assertion *p == 2` `FAILED` after its NULL-pointer check fails at k = 1;
  skipping the assertion with the check reports it `PASSED`. Its twin
  `multi_property_kinduction_check_in_assert` pins the safe program proved.
- Not pinned by a test: a withheld proof keeping its claim, and a throwing
  base case leaving its round incomplete. Both need a phase to end in
  `P_ERROR` or an exception, which no regression input forces.
- The partial-report note now describes the closing round only: a skip at an
  earlier k whose claims a later base case solved no longer prints it.
- Residuals:
  - The signal is per round, not per claim: one skipped claim withholds the
    proofs of claims the same base case did solve, and a forward condition or
    inductive step that ends in `P_ERROR` withholds the diagnostic pass's
    proofs too. A per-claim skip set recorded by the base case would be exact.
  - The per-claim inductive step under `--loop-invariant` is gated by the same
    predicate but has no test of its own.
  - `multi_property_kinduction_diagnose_fail_fast` depends on claims being
    solved in SSA order, which is what puts the skipped instances after the
    violated one.

### W4 — falsification and parallel k-induction (D4)

- `--falsification`: when `--multi-property` is set, continue escalating past
  a violation, as the incremental-bmc branch does, and end through
  `conclude()`. Small change. Pair of tests on `vac2.c`: `--falsification
  --multi-property --max-k-step 10` must list `assertion x != 3` FAILED, and
  `vac2_safe.c` under the same flags must stay `^VERIFICATION UNKNOWN$`, exit 0
  — `--falsification` has no forward condition and so proves nothing; the twin
  pins that escalating past a violation does not set `any_violation_found` on a
  clean run. §4's base-case rule landed in W3b, so the escalating run no
  longer prints the k = 1 PASSED row.
- `--k-induction-parallel`: merging per-claim verdicts across forked processes
  is a separate design. For now, reject the combination with a
  `log_error`, as `--falsify-context-bound` does (`bmc_strategy.cpp:608-622`),
  and open a follow-up issue. An error is better than a run that reports half
  the violations and presents that as the full answer.
- **Tests (pair) for the rejection**, modelled on
  `regression/esbmc-unix/github_6831_falsify_prepass_reject`: one pinning
  `^ERROR: --k-induction-parallel cannot be combined with --multi-property$` on
  `vac2.c`, and one pinning that `vac2.c --k-induction-parallel --max-k-step 10`
  without the flag still ends `^VERIFICATION FAILED$`, so the rejection is
  scoped to the combination.
- Labels: `needs-svcomp-run`. `esbmc-wrapper.py:359` selects `--falsification`
  for the `falsi` strategy, so this changes a strategy SV-COMP runs.

### W5 — re-triage

Re-run the reproducers of #1599, #2075 and #7503 on the W1–W4 build. Close
each one that passes with a `github_<N>` test. Check the KNOWNBUG suite for
multi-property entries in the same pass. File D6 separately, after checking
it against existing issues and KNOWNBUG tests.

### W6 — documentation

Add a `--multi-property` section to the website documentation with the §1
table as it stands after W4. Also add a matrix test that runs §2's reproducers
under every strategy in §1 and pins each verdict, so that the next strategy
added cannot silently diverge.

---

## 6. Regression surface

Two surfaces, and the workstreams do not share one.

**W1, W2 and W4 change `do_bmc_strategy`.** 20 tests combine `--multi-property`
with a strategy routed there (`driver.cpp:297-300`): `--k-induction` 8,
`--incremental-bmc` 5, `--loop-invariant` 7; `--falsification` and
`--termination` have none, so W4's `--falsification` change is pinned only by
the tests it adds. Find them with:

```sh
grep -rl --include=test.desc -e '--multi-property' regression \
  | xargs grep -lE -e '--(k-induction|incremental-bmc|falsification|termination|loop-invariant)([[:space:]]|$)'
```

`--include` goes before `--`, or `--` makes it a file operand and the search
covers all of `regression` unfiltered. The trailing `([[:space:]]|$)` is not
decoration: `\b` matches inside `--loop-invariant-check`, which is the
standalone havoc schema and is *not* k-stepped, and pulls in 45 further tests
(mostly `regression/loop-invariants`) that these workstreams cannot reach.

**W3 changes what every `--multi-property` run prints.** Its surface is all 254
test.desc that set the flag:

```sh
grep -rl --include=test.desc -e '--multi-property' regression
```

Run the matching subset first on every workstream, then `ctest -L regression`
under the 10-minute cap. Any test that matches a table row by regex is expected
to need updating under W3. Each such update is reviewed as part of the PR, not
waved through.

---

## 7. Open questions

1. **D3's cause** (W2). Resolved: see W2.
2. **Inductive-step granularity.** The inductive step proves all remaining
   claims together. A claim that is inductive on its own can fail to be proved
   because a non-inductive sibling shares the step. Per-claim inductive steps
   (as `diagnose_unknown_properties` already runs at the last k) would give
   more `Passed` rows but multiply solver calls. This is not a correctness
   defect, so it is out of scope unless measurement shows a large difference.
3. **Parallel k-induction.** Whether to implement per-claim merging after W4's
   rejection depends on whether anyone asks for it.

---

## 8. Reply to #7900 after W4

> `--multi-property` composes with plain BMC (`--unwind`) and with
> `--k-induction`, `--incremental-bmc` and `--falsification`. Under the k-step
> strategies a claim is reported FAILED at the k where the base case finds it,
> PASSED when the forward condition or inductive step proves it, and UNKNOWN if
> neither happens by `--max-k-step`, in one table at the end.
> `--k-induction-parallel` and `--falsify-context-bound` reject it.
