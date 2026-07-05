---
title: Termination and Non-termination
weight: 22
---

Most ESBMC modes prove *safety* — that nothing bad happens within a bound. The
`--termination` mode proves a *liveness* property instead:

> for every input, every execution of the program eventually halts.

```sh
esbmc file.c --termination
```

A `VERIFICATION SUCCESSFUL` here means every loop in the program is guaranteed to
finish; a `VERIFICATION FAILED` means ESBMC found an input under which some loop
runs forever, and reports the non-terminating execution as a counterexample.
Termination is undecidable in general [1], so — like every other ESBMC mode —
the analysis is sound but incomplete: it answers `TRUE`, `FALSE`, or `UNKNOWN`.

## Reduction to a safety property {#reduction}

ESBMC does not reason about termination directly. It rewrites the question into
a reachability problem it already knows how to solve. For each natural loop, a
GOTO-to-GOTO pass inserts an `assert(false)` *marker* on every edge that leaves
the loop body — the top-test fall-through, every `break`, and every jump to a
label outside the loop. The marker carries exactly one bit of information:
*control left the loop.* The same pass applies the
[k-induction](/docs/theory/verification-algorithms#k-induction) havoc [4] to the
loop head, so the marker is evaluated from an arbitrary in-loop state rather than
only the concrete one.

The two k-induction obligations then read directly as termination verdicts:

- **Forward condition unsatisfiable at *k*** — from the concrete initial state,
  no execution reaches a loop exit within *k* unwindings *because the loop has
  fully unwound*. Every iteration is dominated by the guard, so the loop always
  exits. **The program terminates.**
- **Inductive step unsatisfiable at *k*** — from a havoc'd in-loop state, no
  execution can leave the loop within *k* iterations. An adversary controlling
  the inputs can stay inside forever. **A non-terminating execution exists.**

Anything still open at the maximum *k* is reported as `UNKNOWN`.

## The three-pass pipeline {#pipeline}

The marker reduction above is the fallback. Two cheaper structural analyses run
first, because each settles a large class of loops without paying for full
symbolic execution. The passes run in order, and ESBMC stops as soon as one
returns a definite verdict:

1. **Recurrent-set non-termination.** Looks for a cheap witness that a loop can
   stay live forever: a reachable set of states the loop can re-enter
   indefinitely under some input. It targets `while (1)` controller loops that
   read an input, validate it, and dispatch to a state-machine update — if the
   solver finds an input under which no branch fires, the loop sits at a fixed
   point and never exits. This pass only ever reports `FALSE` (non-terminating)
   or abstains.

2. **Ranking-function synthesis.** Tries to *prove* termination by exhibiting a
   **ranking function** — an integer measure of the loop state that is bounded
   below and strictly decreases on every iteration [2]. A measure that cannot
   decrease forever forces the loop to halt. ESBMC derives candidate measures
   from the loop guard, synthesises the supporting invariants the measure needs
   to hold [3], and discharges two SMT obligations per candidate: *bounded below*
   (the measure cannot drop under its floor while the guard holds) and *strict
   decrease* (every path through the body lowers it). When no single measure
   works, a lexicographic pair of measures is tried. This pass only ever reports
   `TRUE` (terminating) or abstains.

3. **k-induction on the markers.** If neither structural pass settles the loop,
   ESBMC falls back to the marker-and-havoc reduction of the previous section and
   lets the incremental [k-induction](/docs/theory/verification-algorithms#k-induction)
   loop discharge it as an ordinary safety property.

The ordering is deliberate: the structural detectors are cheap and each covers a
distinct shape (live controller loops; loops with an obvious decreasing measure),
so the expensive symbolic pipeline only runs on what they leave behind.

## Soundness {#soundness}

The non-negotiable constraint is **no wrong `TRUE`**: ESBMC must never claim a
non-terminating program terminates. Two places need care.

- **Ranking arguments rest on their supporting invariants.** A measure derived
  from the guard only proves termination relative to facts the synthesiser can
  establish about the reachable loop-head states; ranking and reachability are
  proved together, not separately [3]. If the supporting invariant cannot be
  established, the candidate is rejected rather than trusted.
- **The havoc must cover the variables termination depends on.** The inductive
  step is only a valid non-termination witness if the loop-head havoc actually
  freed every variable the loop's exit condition reads. A loop that exits on the
  *contents* of a heap buffer, while only the pointer is in the modified set,
  would otherwise report non-termination spuriously. ESBMC detects these shapes
  and treats the inductive-step verdict as inconclusive for them, falling back to
  the forward condition or the recurrent-set witness.

## Command-line flags {#flags}

```sh
esbmc file.c --termination --max-inductive-step 3 --interval-analysis
```

- `--termination` — enable the reduction and the three-pass pipeline above. It
  is mutually exclusive with `--k-induction` (k-induction takes precedence).
- `--max-inductive-step N` — cap how far the inductive step unwinds. The decisive
  non-termination proofs typically fire at small *k* (2 or 3), so a small cap is
  usually enough.
- `--interval-analysis` — supply the ranking and recurrent-set passes with
  per-loop numeric bounds (see [Interval Analysis](/docs/theory/interval-analysis));
  tighter ranges help the supporting-invariant synthesis converge.

## References

[1] Byron Cook, Andreas Podelski, Andrey Rybalchenko: *Proving Program
Termination.* Communications of the ACM 54(5):88–98, 2011.
[doi:10.1145/1941487.1941509](https://doi.org/10.1145/1941487.1941509)

[2] Andreas Podelski, Andrey Rybalchenko: *A Complete Method for the Synthesis of
Linear Ranking Functions.* VMCAI 2004, LNCS 2937: 239–251.
[doi:10.1007/978-3-540-24622-0_20](https://doi.org/10.1007/978-3-540-24622-0_20)

[3] Aaron R. Bradley, Zohar Manna, Henny B. Sipma: *Linear Ranking with
Reachability.* CAV 2005, LNCS 3576: 491–504.
[doi:10.1007/11513988_48](https://doi.org/10.1007/11513988_48)

[4] Mikhail Y. R. Gadelha, Hussama Ibrahim Ismail, Lucas C. Cordeiro: *Handling
loops in bounded model checking of C programs via k-induction.* Int. J. Softw.
Tools Technol. Transf. 19(1):97–114, 2017.
[doi:10.1007/s10009-015-0407-9](https://doi.org/10.1007/s10009-015-0407-9)
