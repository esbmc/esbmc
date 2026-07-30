---
title: Linear Temporal Logic
---

ESBMC can check **linear temporal logic** properties — including liveness
properties such as "whenever the button is pressed, the charge eventually
exceeds the minimum" — over unmodified C programs. The approach is described in
[1] and [4]: the negated formula is translated to a Büchi automaton, the
automaton is emitted as a C *monitor thread*, and ESBMC checks the monitor
interleaved with the program under analysis. Because bounded model checking
explores only finite prefixes, the verdict is drawn from a four-valued lattice
rather than true/false.

This page documents what the `--ltl` path does in ESBMC 8.4.0, including where
the implementation diverges from [1].

> **Warning**: LTL is a legacy, lightly maintained path, and it currently
> **misses violations**. The monitor is not synchronised to updates of the
> variables its propositions read, so a definitively violated property can be
> reported as *presumably true*
> ([#6546](https://github.com/esbmc/esbmc/issues/6546)). Separately,
> `VERIFICATION SUCCESSFUL` and the exit code do not reflect the LTL verdict at
> all ([#6548](https://github.com/esbmc/esbmc/issues/6548)) — you must read the
> `Final lowest outcome:` line — and several ordinary invocations report a false
> ⊤ ([#6547](https://github.com/esbmc/esbmc/issues/6547)). Read
> [Limitations](#limitations) before relying on any of it.

## At a glance

| | |
| --- | --- |
| Flag | `--ltl` |
| Languages | C only — no C++, Python, Solidity or Jimple support |
| Temporal operators | `G` `F` `X` `U` `V` — see [Operators](#operators) |
| Soundness | **Misses violations** ([#6546](https://github.com/esbmc/esbmc/issues/6546)) |
| Formula translator | [libltl2ba](https://github.com/esbmc/libltl2ba), external; **not bundled and not fetched by `DOWNLOAD_DEPENDENCIES`** |
| Verdict reported as | the `Final lowest outcome:` log line |
| Exit code reflects verdict | **No** — always `0` / `VERIFICATION SUCCESSFUL` |
| Counterexample | **No** — no `[Counterexample]` is emitted for an LTL outcome |
| Compatible strategies | plain BMC only; `--unwind N` works if `N` is large enough |
| Incompatible strategies | `--k-induction`, `--incremental-bmc`, `--termination`, `--falsification` |
| Monitor prefix bound | `-DLTL_PREFIX_BOUND=N`, **required in practice** (default is 2³¹) |
| Other properties in the same run | **None** — safety and unwinding assertions are masked |
| Regression coverage | 3 `CORE` tests in `regression/ltl/`, run on Linux, macOS and Windows |
| Translator last updated | libltl2ba 2.1 (April 2024), C output targeting ESBMC 7.6 |

## The four-valued verdict

Standard LTL is interpreted over infinite traces, but a bounded model checker
explores finite ones. ESBMC therefore evaluates the formula under the *bounded
trace semantics* of [1, Def. 2], built on the four-valued deMorgan lattice
`{⊥, ⊥ᵖ, ⊤ᵖ, ⊤}` of [2]. A finite prefix is stutter-extended — its final state
repeated forever — and the verdict records whether *every*, *some*, or *no*
infinite continuation satisfies the formula φ.

| Reported outcome | Lattice value | Meaning |
| --- | --- | --- |
| `LTL_BAD` | ⊥ | A **bad prefix** was found: no continuation of this trace can satisfy φ. The property is definitively violated. |
| `LTL_FAILING` | ⊥ᵖ | *Presumably false*: the trace ends in a state that violates φ when stutter-extended, but some other continuation would satisfy it. |
| `LTL_SUCCEEDING` | ⊤ᵖ | *Presumably true*: the program halts in a state that satisfies φ when stutter-extended, but some other continuation would violate it. |
| `LTL_GOOD` | ⊤ | A **good prefix**: every continuation satisfies φ. The property definitively holds. |

The lattice order is `⊥ ⊑ ⊥ᵖ ⊑ ⊤ᵖ ⊑ ⊤`. ESBMC checks each interleaving
separately and reports the **least** value seen across all of them
(`src/esbmc/bmc.cpp:1750`), which is why the log line reads
`Final lowest outcome:`.

For a liveness formula, ⊤ is generally unreachable: no finite prefix of
`G(p -> F q)` can rule out a later violation, so the best attainable verdict is
⊤ᵖ [1, §7.1]. A `LTL_SUCCEEDING` result on a liveness property is therefore the
expected *success* outcome, not a near-miss.

## Checking an LTL property

### 1. Translate the negated formula

Build [libltl2ba](https://github.com/esbmc/libltl2ba) — ESBMC's fork of
`ltl2ba` [3], extended with a C output format — and emit the monitor for the
**negation** of the property you want to hold:

```bash
ltl2ba -O c -f '!(G({pressed} -> F {charge > min}))' > notphi.c
```

Note the negation: the automaton is a Spin-style *never claim*, so what you hand
to `ltl2ba` is the complement of the property, and the reported verdict is about
the property. Feeding `!(φ)` for a φ that holds yields ⊤ᵖ; feeding `!(φ)` for a
φ that fails yields ⊥ᵖ.

C expressions over the program's global variables are written inside curly
brackets and act as the atomic propositions; they must be side-effect free and
evaluate to something usable as a truth value.

#### Operators

All five LTL temporal operators are accepted, in both their letter and
ASCII-art spellings:

| Operator | Syntax | Notes |
| --- | --- | --- |
| always | `G`, `[]` | |
| eventually | `F`, `<>` | |
| until | `U` | |
| release | `V` | written `R` in the mathematical notation of [1] |
| next | `X` | translates and runs, but see the caveat below |

Propositional syntax is `true`, `false`, `{C expression}`, a lowercase
identifier, `!`, `&&` (or `/\`), `||` (or `\/`), `->` and `<->`. Precedence,
highest first, is `U`/`V` (right-associative), `&&`, `||`, `<->`, `->`; unary
operators bind tighter than binary ones, and libltl2ba wants spaces between
symbols. See its
[README](https://github.com/esbmc/libltl2ba/blob/master/README) for the full
grammar.

`X` is the one to be careful with. Its intended reading here is "φ holds after
the next update of a global variable used in the propositions" [1, §3.1], which
relies on the monitor being stepped at each such update — exactly the directed
scheduling that is currently disabled
([#6546](https://github.com/esbmc/esbmc/issues/6546)). A formula containing `X`
translates and produces a verdict, but that verdict is not backed by the
intended semantics. [1, §3.2] recommends X-free, stutter-invariant formulas in
any case, since `X` is awkward to interpret over finite traces at all.

The generated file contains one pure C accessor per proposition, plus a
`char __ESBMC_property_*[]` marker that tells ESBMC which functions to treat as
propositions:

```c
char __ESBMC_property__ltl2ba_cexpr_0[] = "pressed";
int _ltl2ba_cexpr_0_status(void) { return pressed; }
char __ESBMC_property__ltl2ba_cexpr_1[] = "charge > min";
int _ltl2ba_cexpr_1_status(void) { return charge > min; }
```

### 2. Declare the variables the propositions read

libltl2ba does not know the types of the program's globals, so the generated
file needs declarations for them. Put them in a header:

```c
extern int pressed;
extern int charge, min;
```

and have libltl2ba `#include` it directly with `-H`:

```bash
ltl2ba -O c -H '"tau.h"' -f '!(G({pressed} -> F {charge > min}))' > notphi.c
```

Alternatively, add the `extern` declarations to `notphi.c` by hand, or leave the
header out of the monitor and pass it to ESBMC with `--include-file`:

```bash
esbmc program.c --ltl notphi.c --include-file tau.h -DLTL_PREFIX_BOUND=10
```

### 3. Run ESBMC

```bash
esbmc program.c --ltl notphi.c -DLTL_PREFIX_BOUND=10
```

`LTL_PREFIX_BOUND` bounds the monitor's own transition loop. Its default of
`2147483648` is not tractable — a run left at the default does not finish — so
pass an explicit bound. libltl2ba emits an
`assert(num_iters == iters, "Unwind bound on ltl2ba_fsm insufficient")` to catch
a bound that is too small, but `--ltl`
[masks that assertion](#other-properties-are-masked) along with every other
non-LTL property, so an inadequate bound is not reported.

Taking `regression/ltl/basic` as the worked example — a program that presses the
button twice but only tops up the charge when pressed:

```c
int pressed, charge, min;

int main()
{
	charge = nondet_int();
	min = nondet_int() % 1024;
	for (int i = 0; i < 2; i++) {
		pressed = nondet_int();
		if (pressed)
			charge = min + 1;
	}
}
```

ESBMC reports:

```
Checking for LTL_BAD
WARNING: Couldn't find LTL_BAD assertion
Checking for LTL_FAILING
Found trace satisfying LTL_FAILING
...
Final lowest outcome: LTL_FAILING

VERIFICATION SUCCESSFUL
```

⊥ᵖ is the correct answer: the loop can exit with `pressed` true and `charge` not
yet topped up, and stuttering that final state forever violates
`F {charge > min}` — but a longer trace could still satisfy it. Note that
`VERIFICATION SUCCESSFUL` on the last line says nothing about the LTL result.

## How it works

Given the program and the generated monitor, ESBMC:

1. **Finds the propositions.** `add_property_monitors`
   (`src/esbmc/parseoptions/property_monitors.cpp:20`) scans the symbol table
   for `__ESBMC_property_<name>` markers and, for each, extracts the returned
   expression from the matching `<name>_status` function.
2. **Makes proposition updates atomic.** Every assignment whose target is one of
   the globals a proposition reads is wrapped in `ATOMIC_BEGIN` / `ATOMIC_END`
   (`property_monitors.cpp:162`), so the monitor cannot observe a half-updated
   state.
3. **Starts and stops the monitor.** Calls to `ltl2ba_start_monitor` and
   `ltl2ba_finish_monitor` are injected at the top of the entry function and
   before each of its `return` instructions (`property_monitors.cpp:97`).
   `ltl2ba_start_monitor` spawns the automaton as a pthread and registers it via
   the `__ESBMC_register_monitor` intrinsic (`src/goto-symex/symex_main.cpp:609`).
4. **Explores the automaton symbolically.** The monitor keeps the current
   automaton state in a single nondeterministic-but-constrained integer, with
   each transition guarded by `__ESBMC_assume`. The automaton is never
   determinised; the solver explores the alternatives [1, §6.2.1].
5. **Evaluates the prefix at the end.** `ltl2ba_finish_monitor` kills the
   monitor and asserts three properties against the precomputed reachability
   tables `_ltl2ba_bad_prefix_states`, `_ltl2ba_stutter_accept_table` and
   `_ltl2ba_good_prefix_excluded_states`, labelled `LTL_BAD`, `LTL_FAILING` and
   `LTL_SUCCEEDING`.
6. **Solves once per lattice level.** `bmct::ltl_run_thread`
   (`src/esbmc/bmc.cpp:2083`) masks all but one of the three assertions and runs
   a separate solver call for each, from ⊥ upwards, returning the first level for
   which a trace exists. If none is satisfiable it returns `LTL_GOOD`.

Two smaller accommodations: the context-switch threshold is raised from 2 to 3
under `--ltl` (`src/goto-symex/execution_state.cpp:106`), and the assertion
cache is disabled (`src/esbmc/bmc.cpp:118`) because the LTL assertions are
re-checked with different maskings.

## Limitations

> **Note**: Everything below was reproduced against ESBMC 8.4.0. The `--ltl`
> path has no dedicated issue label; the open reports are
> [#6546](https://github.com/esbmc/esbmc/issues/6546),
> [#6547](https://github.com/esbmc/esbmc/issues/6547) and
> [#6548](https://github.com/esbmc/esbmc/issues/6548).

### Violations are missed

The monitor does not observe updates to the globals its propositions read, so a
property that is *definitively* violated can be reported as ⊤ᵖ, "presumably
true" ([#6546](https://github.com/esbmc/esbmc/issues/6546)). For a program that
assigns `s = 1`:

```c
int s;

int main()
{
	s = 0;
	s = 1;
	s = 0;
	return 0;
}
```

the property `G {s == 0}` is definitively false — once `s` is 1 no continuation
repairs a `G` — so the verdict should be ⊥. Instead:

```bash
ltl2ba -O c -H '"tau.h"' -f 'F {s != 0}' > safety.c
esbmc prog.c --ltl safety.c -DLTL_PREFIX_BOUND=6
```

```
Final lowest outcome: LTL_SUCCEEDING
```

The generated automaton does have a reachable bad-prefix state, so this is not a
translation artefact — but running the same files without `--ltl` shows the
monitor never enters it in any explored interleaving:

```
✓ PASSED: 'LTL_BAD at file safety.c line 204 column 2 function ltl2ba_finish_monitor'
```

The cause is the disabled directed scheduling described under
[Divergences from the published algorithm](#divergences-from-the-published-algorithm):
with the monitor free-running as an ordinary thread, no interleaving is
guaranteed to sample the state at the moment a proposition changes. Raising
`LTL_PREFIX_BOUND`, `--context-bound` or `--unwind` does not help.

### The verdict is not in the exit code

`ltl_run_thread`'s result is folded into a log message and nothing else — the
LTL path always returns "unsatisfiable" to its caller
([#6548](https://github.com/esbmc/esbmc/issues/6548), `src/esbmc/bmc.cpp:1995`).
A definitive violation still prints
`VERIFICATION SUCCESSFUL` and exits `0`:

```
Final lowest outcome: LTL_BAD

VERIFICATION SUCCESSFUL
```

No `[Counterexample]` is emitted either, for any outcome, so a failing run gives
no trace to inspect. Parse the `Final lowest outcome:` line, or use the
[workaround below](#getting-a-usable-exit-code).

### A missing assertion is read as ⊤

`ltl_run_thread` reports `LTL_GOOD` both when it has *proved* every prefix
assertion unsatisfiable and when it could not find the assertions at all
([#6547](https://github.com/esbmc/esbmc/issues/6547), `src/esbmc/bmc.cpp:2114`
and `2147`). "Not instrumented" and "definitively
correct" are indistinguishable in the output beyond a
`WARNING: Couldn't find LTL_* assertion` line. Three situations trigger it:

- **`--ltl` with no monitor file.** Warns `No LTL traces seen, apparently` and
  reports `VERIFICATION SUCCESSFUL`.
- **A program that leaves `main` other than by returning.** `exit()` and
  `abort()` bypass the injected `ltl2ba_finish_monitor` call, because it is only
  placed before `return` instructions. Every VCC is then simplified away and
  ESBMC returns before the LTL check runs, printing no verdict line at all —
  only `WARNING: No LTL traces seen, apparently` and `VERIFICATION SUCCESSFUL`.
- **An incompatible strategy** — see below.

### Do not combine `--ltl` with another strategy

`--ltl` is only meaningful for a plain single-shot BMC run. On
`regression/ltl/basic`, whose correct verdict is ⊥ᵖ:

| Invocation | Reported outcome |
| --- | --- |
| `--ltl` | `LTL_FAILING` ✓ |
| `--ltl --unwind 3` (and above) | `LTL_FAILING` ✓ |
| `--ltl --unwind 1`, `--ltl --unwind 2` | `LTL_GOOD` ✗ |
| `--ltl --k-induction` | `LTL_GOOD` ✗ |
| `--ltl --incremental-bmc` | `LTL_GOOD` ✗ |
| `--ltl --termination` | `LTL_GOOD` ✗ |
| `--ltl --falsification` | `LTL_FAILING`, but `VERIFICATION UNKNOWN` and one verdict line per iteration |

The k-induction and incremental drivers restructure the program per iteration,
so the prefix assertions are no longer present when `ltl_run_thread` looks for
them; the fall-through to `LTL_GOOD` then reports the *top* of the lattice for a
program that plainly fails. k-induction additionally logs
`k-induction does not support concurrency yet. Disabling inductive step`,
because the monitor is a thread.

An `--unwind` bound that is too small fails the same way, and just as silently:
the bound applies to the monitor's transition loop as well as the program's, so
too few iterations leave the automaton short of the state that would witness the
violation.

### Other properties are masked

To isolate one lattice level at a time, `ltl_run_thread` turns **every**
assertion that is not the one it is currently seeking into a `SKIP`
(`src/esbmc/bmc.cpp:2103`). That includes all the ordinary ones: unwinding
assertions, bounds and NULL checks, overflow checks, and user `assert`s. An
`--ltl` run therefore says nothing about safety, and it also disables the two
checks that would otherwise catch a bad bound — the unwinding assertions, and
the `"Unwind bound on ltl2ba_fsm insufficient"` assertion libltl2ba builds into
the monitor for exactly this purpose.

This is why the `--unwind 1` row above reports `LTL_GOOD` rather than an error.
Re-running the same command without `--ltl` shows what was masked:

```
✗ FAILED: 'unwinding assertion loop 3 at file notphi.c line 131 column 2 function ltl2ba_fsm'
✗ FAILED: 'unwinding assertion loop 4 at file program.c line 7 column 2 function main'
```

Check safety properties and bound adequacy in a separate run without `--ltl`.

### Divergences from the published algorithm

- **The dedicated monitor scheduler is disabled.** [1, §6.2.3] replaces
  general-purpose scheduling of the monitor with a directed context switch to it
  after each global-variable update, reported there as the change that made the
  analysis practical. The code that inserts those `__ESBMC_switch_to_monitor`
  calls is `#if 0`'d out (`property_monitors.cpp:205`, disabled in commit
  `4146a8e387` as broken), and libltl2ba emits the corresponding calls commented
  out. The monitor therefore runs as an ordinary schedulable thread — the
  earlier, slower behaviour that [1] set out to replace. The
  `switch_to_monitor` / `switch_away_from_monitor` machinery still exists in the
  symbolic execution engine (`src/goto-symex/execution_state.cpp:1269`) but is
  unreachable from the `--ltl` path.
- **Propositions are re-evaluated, not cached.** [1, §6.2.2] describes inserting
  an update to a Boolean variable per proposition after each assignment. The
  current code only makes the assignment atomic; the automaton calls
  `_ltl2ba_cexpr_N_status()` directly each time it needs a proposition. This
  replaced an earlier scheme that stored the propositions as strings for ESBMC
  to parse, which the frontends no longer support.
- **Only direct assignments to a named global are monitored.**
  `add_monitor_exprs` returns immediately unless the assignment target is a
  plain symbol (`property_monitors.cpp:175`), so a write through a pointer, to
  an array element, or to a struct member does not count as a proposition
  update. This limitation is acknowledged in [1, §6.2.2].

### Other restrictions

- **C only.** The property-monitor machinery keys on C symbol names
  (`c:@F@<name>_status`) and the monitor is a C file; no other frontend is
  wired up.
- **`ltl2ba` is not bundled.** libltl2ba must be built and installed separately.
  Its C output was last adapted for ESBMC 7.6, several releases behind the
  current one.
- **Test coverage is thin.** `regression/ltl/` holds three tests, and the two
  automata they ship exercise only ⊥ᵖ and ⊤ᵖ — neither `LTL_BAD` nor `LTL_GOOD`
  is covered, because both formulas have an empty bad-prefix state set and mark
  every state as excluded from being a good prefix.
- Multi-threaded liveness checking remains practical only for small programs
  [1, §7.2]: liveness needs enough interleavings for every thread to complete
  whole loop iterations, whereas safety violations are typically shallow.

## Getting a usable exit code

The instrumentation is driven by the `__ESBMC_property_*` markers in the monitor
file, not by `--ltl` (`src/esbmc/parseoptions/process_goto_program.cpp:458`).
Passing the monitor **without** `--ltl` therefore checks the three prefix
assertions as ordinary assertions, which restores a non-zero exit code,
per-claim results and counterexamples:

```bash
esbmc program.c notphi.c -DLTL_PREFIX_BOUND=10 --multi-property
```

```
✓ PASSED: 'LTL_BAD at file notphi.c line 239 ...'
✗ FAILED: 'LTL_FAILING at file notphi.c line 241 ...'
✗ FAILED: 'LTL_SUCCEEDING at file notphi.c line 243 ...'

VERIFICATION FAILED
```

The **lowest failing claim** in the order `LTL_BAD` → `LTL_FAILING` →
`LTL_SUCCEEDING` is the lattice verdict, and all three passing means ⊤. This
reproduces the `--ltl` outcome on each of the shipped tests — `LTL_FAILING` for
`basic` and `basic-func`, `LTL_SUCCEEDING` for `basic-success`. Without
`--multi-property` ESBMC stops at the first violated assertion, which need not
be the lowest one, so the exit code is usable but the reported claim is not the
verdict.

## Regression tests

All three tests check the property `G({pressed} -> F {charge > min})`, or its
negation, against the same small program.

| Test | Monitor | Expected outcome |
| --- | --- | --- |
| `regression/ltl/basic` | automaton for the formula, propositions inlined as C expressions | `LTL_FAILING` |
| `regression/ltl/basic-func` | same automaton, propositions via `_ltl2ba_cexpr_N_status()` accessors | `LTL_FAILING` |
| `regression/ltl/basic-success` | automaton for the negated formula | `LTL_SUCCEEDING` |

All three are `CORE` and pass:

```bash
ctest -R "regression/ltl/" --output-on-failure
```

Each checks both `^VERIFICATION SUCCESSFUL$` and the `Final lowest outcome:`
line — the latter is what actually pins the LTL behaviour, since the former
holds regardless.

## References

[1] Jeremy Morse, Lucas C. Cordeiro, Denis A. Nicole, Bernd Fischer: *Model
checking LTL properties over ANSI-C programs with bounded traces.* Software and
Systems Modeling 14(1):65–81, 2015.
[doi:10.1007/s10270-013-0366-0](https://doi.org/10.1007/s10270-013-0366-0)

[2] Andreas Bauer, Martin Leucker, Christian Schallhart: *Comparing LTL
Semantics for Runtime Verification.* Journal of Logic and Computation
20(3):651–674, 2010.
[doi:10.1093/logcom/exn075](https://doi.org/10.1093/logcom/exn075)

[3] Paul Gastin, Denis Oddoux: *Fast LTL to Büchi Automata Translation.* CAV
2001, LNCS 2102: 53–65.
[doi:10.1007/3-540-44585-4_6](https://doi.org/10.1007/3-540-44585-4_6)

[4] Jeremy Morse: *Expressive and efficient bounded model checking of concurrent
software.* PhD thesis, University of Southampton, 2015.
[PDF](https://ssvlab.github.io/esbmc/papers/phd_thesis_morse.pdf)
