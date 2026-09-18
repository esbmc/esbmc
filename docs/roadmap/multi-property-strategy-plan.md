# Plan — `--multi-property` under the k-step strategies

**Status:** Proposed. Nothing implemented.
**Origin:** Discussion
[#7900](https://github.com/esbmc/esbmc/discussions/7900), *"Current state of
--multi-property support"*: is `--multi-property` orthogonal to the analysis
mode, and which modes can its results be trusted under? Related open issues:
[#1361](https://github.com/esbmc/esbmc/issues/1361),
[#1902](https://github.com/esbmc/esbmc/issues/1902),
[#1599](https://github.com/esbmc/esbmc/issues/1599),
[#2075](https://github.com/esbmc/esbmc/issues/2075),
[#7503](https://github.com/esbmc/esbmc/issues/7503).
**Last updated:** 2026-09-18.

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
| `--unwind N` (± `--interval-analysis`, `--parallel-solving`, `--smt-during-symex`) | Every violated claim reported, one table, verdict matches — except that same-text claims on one line share a row (D5) | Yes, per row |
| `--k-induction`, `--incremental-bmc` | Every violated claim found, but one table per k step, with claim ids renumbered and interim `VERIFICATION SUCCESSFUL` lines (D2) | Final verdict only |
| `--incremental-bmc` | Ends `VERIFICATION UNKNOWN`, exit 0, after reporting violations (D1) | No |
| `--k-induction --interval-analysis` | Misses a violated claim that the same flags without `--multi-property` report (D3) | No |
| `--k-induction-parallel`, `--falsification` | Stop at the first violation without saying so (D4) | First violation only |
| `--falsify-context-bound` | Rejects the combination with an error | n/a |

§8 is the reply to the discussion once the fixes land; until then this table is
the reply.

---

## 2. Reproducers

Each goes into the PR that fixes the defect it pins. The first three have two
violated claims each, the second reachable only after a loop has run.

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
same function (`log_fail("VERIFICATION UNKNOWN"); return 0;`) does not call
`conclude()`. Under `--k-induction` the same programs end `FAILED` only because
an inductive step succeeds once every claim has failed, and goes through
`conclude()`. The `--incremental-bmc` loop has no inductive step to reach it.

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

### D4 — `--falsification` and `--k-induction-parallel` ignore the flag

Both stop at the first violation (`i1361.c`, `vac2.c`, `vac3.c`), and neither
warns about it. `--falsification` returns from the k loop unconditionally on a
violation (`bmc_strategy.cpp:467`, `if (violated && !is_coverage) return 1;`).
`--k-induction-parallel` forks base case, forward condition and inductive step
into separate processes. The parent stops at the first result that settles the
program, and no per-claim state crosses the process boundary.

### D5 — claims are matched by message and location

The skip of already-violated claims in `multi_property_check` keys a claim on
`claim_msg + "\t" + claim_loc` (`bmc.cpp:~3000`); the code's own comment says
this is unsound. Two distinct claims with the same text at the same location
share one signature, and `goto-check` produces such pairs routinely:

```
$ esbmc d5.c --show-claims          # Claim 1 and Claim 2, both
  line 6 column 3 function main
  array bounds violated: array `a' upper bound
$ esbmc d5.c --multi-property
  FAILED       [main.array-bounds-violated.1]  line 6  array bounds violated: ...
** 1 of 1 properties failed
```

Two claims, one row. The table cannot say that the `a[i]` check holds, and
under a k-step strategy the surviving signature makes the second claim skipped
at every later k. Unlike D1–D4, this one affects plain BMC too.

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

---

## 4. Target semantics

Under a k-step strategy each claim moves through one lifecycle, keyed by its
**program** claim id (its position in the GOTO program's claim order, as in
`--show-claims`), not the per-equation index:

```
Unknown ──base case SAT at k──▶ Failed(k)          final
Unknown ──forward condition UNSAT at k──▶ Passed   final (bounded exhaustively)
Unknown ──inductive step UNSAT at k──▶ Passed      final (proved)
Unknown ──max-k reached──▶ Unknown                 reported as such
```

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
a claim cannot overwrite a stronger one. Most of W3 is routing the k-step
strategies into that store rather than adding a new mechanism.

---

## 5. Workstreams

One PR each, in this order. W1 and W2 are independent; W3 builds on W1.

### W0 — reply to #7900

Post §1's table as the answer, and link this plan. No code change.

### W1 — final verdict and early stop (D1, #1902)

- Route the exhausted-k exit through `conclude()`, so a recorded violation
  ends `VERIFICATION FAILED`, exit 1.
- After each k step, stop when every claim in the program has a final
  verdict.
- **Tests (pair).** `vac2.c --incremental-bmc --multi-property --max-k-step 10`
  → `^VERIFICATION FAILED$`. The safe twin bounds the loop and asserts what
  holds — `__ESBMC_assume(n < 5)`, then `assert(x == n)` and `assert(x < 5)`
  — and already ends `^VERIFICATION SUCCESSFUL$` (forward condition, k = 5)
  under both `--incremental-bmc` and `--k-induction`; it pins that W1 does not
  turn a proof into FAILED.
- Mutation check: restore the old exit and confirm the FAILED test changes
  verdict.
- Labels: `needs-svcomp-run` (changes a strategy's exit code).

### W2 — the missed claim under interval analysis (D3)

- Localise with `esbmc-rca`, starting from `--show-vcc` of the k = 2 inductive
  step of `i1361.c`, with and without `--multi-property`. The question to
  answer is which constraint makes `assert(1==2)` unreachable in the inductive
  step when the base case has already recorded `assert(2==3)` as failed.
- Fix at the cause. If the cause is the interval pass's rewrite composing
  badly with the per-claim skip, the fix belongs in whichever of the two is
  wrong, not in the strategy loop.
- **Tests (pair).** `i1361.c --k-induction --interval-analysis
  --multi-property` must list `assertion 1==2` as FAILED; the `assert(i < 5)`
  variant must report only `1==2` failed and nothing proved.
- Labels: `needs-svcomp-run`.

### W3 — one table with stable ids (D2, D5, #1361)

- Key the skip set and the property table on the program claim id instead of
  message + location. This removes D5 and the `//! This algo is unsound`
  comment together. A k-step run's labels then match the first table's
  (`main.assertion.2` stays `main.assertion.2`).
- Record each k step's per-claim outcomes into `property_verdicts` under that
  id. Print the table once from `conclude()` / the exhausted-k exit.
- Suppress interim PASSED rows and interim `VERIFICATION SUCCESSFUL` under a
  k-step strategy.
- **Interface change.** This changes what ESBMC prints. Check it against
  `parse_result()` in `scripts/competitions/svcomp/esbmc-wrapper.py` and run
  `python3 scripts/competitions/svcomp/test_esbmc_wrapper.py`; #7250 is what
  happens otherwise.
- **Tests (pair).** `i1361.c --k-induction --multi-property`: exactly one
  table, `main.assertion.1 ... line 6` and `main.assertion.2 ... line 8` both
  FAILED, no `VERIFICATION SUCCESSFUL` anywhere. A safe program: one table, all
  PASSED, `VERIFICATION SUCCESSFUL` once. `d5.c --multi-property`: two rows,
  `a[i]` PASSED and `a[j]` FAILED, under plain BMC and `--k-induction`.
- Close #1361 with its program as the `github_1361` CORE test.
- Labels: `needs-svcomp-run`.

### W4 — falsification and parallel k-induction (D4)

- `--falsification`: when `--multi-property` is set, continue escalating past
  a violation, as the incremental-bmc branch does, and end through
  `conclude()`. Small change. Pair of tests on `vac2.c`.
- `--k-induction-parallel`: merging per-claim verdicts across forked processes
  is a separate design. For now, reject the combination with a
  `log_error`, as `--falsify-context-bound` does, and open a follow-up issue.
  An error is better than a run that reports half the violations and
  presents that as the full answer.

### W5 — re-triage

Re-run the reproducers of #1599, #2075 and #7503 on the W1–W4 build. Close
each one that passes with a `github_<N>` test. Check the KNOWNBUG suite for
multi-property entries in the same pass. File D6 separately, after checking
it against existing issues and KNOWNBUG tests.

### W6 — documentation

Add a `--multi-property` section to the website documentation with the §1
table as it stands after W4. Also add a matrix test that runs the four
reproducers under every strategy in §1 and pins each verdict, so that the next
strategy added cannot silently diverge.

---

## 6. Regression surface

About 13 existing tests combine `--multi-property` with a k-step strategy
(`github_1890_1`, `github_1890_2`, `github_6070`, `github_6070_pass`,
`proof_cache_kinduction`, `proof_cache_incremental`, `github_6387_6`,
`github_6387_15`, `multi_property_unknown_diagnosis`, and Python, C++ and
Solidity ones). Find them with:

```sh
grep -rl -- '--multi-property' regression --include=test.desc \
  | xargs grep -lE 'k-induction|incremental-bmc|falsification'
```

Run that subset first on every workstream, then `ctest -L regression` under
the 10-minute cap. W3 changes output text, so any of these that match a table
row by regex are expected to need updating. Each such update is reviewed as
part of the PR, not waved through.

---

## 7. Open questions

1. **D3's cause** (W2). Until it is known, W2 has no fix sketch.
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
