---
title: Limitations
weight: 4
---

The Ladder Diagram frontend is under active development and is gated behind the
`ENABLE_LD_FRONTEND` build option. This page records what is currently
supported and the known restrictions.

## Supported constructs

- **Contacts and coils** — normally-open and normally-closed contacts; output,
  Set, and Reset coils. Contacts carrying `edge="rising"` / `edge="falling"`
  (and the vendor spellings `positive`/`negative`, `R`/`P`, `F`/`N`) are sensed
  against a previous-scan shadow rather than treated as level contacts.
- **Rung topology** — parallel paths that reach the same coil are OR-ed, not
  overwritten by the last branch. Rungs are evaluated sequentially, as in the
  code Beremiz generates for MATIEC: each coil re-reads its contacts, so a rung
  sees an earlier rung's write in the same scan, and a block steps once per
  scan. Graphical coils run in Beremiz's layout order (coils less than 10 units
  apart vertically share a row and are ordered left to right; rows top to
  bottom), whether or not the right power rail lists them. When one variable
  has both set and reset coils, the last powered coil wins and ESBMC warns. A rung path that passes through a function
  block resolves the block into synthesised pins instead of being dropped, and
  no path is silently dropped for being unmodellable. Power flow is solved per
  node — `pf(n) = (OR over predecessors) AND cond(n)` — rather than by
  enumerating rail-to-sink paths, so a network with re-convergent branches
  lowers in time linear in its size instead of exponentially.
- **Declared initial values** — `<initialValue>` on a variable declaration is
  parsed, so declared presets no longer read as zero.
- **Timers** — `TON` (on-delay) and `TOF` (off-delay), with their retained
  `ET`/`Q` state evaluated per scan. `ET` stops at `PT` as IEC 61131-3
  §2.5.2.3.2 requires, so a timer held on indefinitely cannot overflow `ET` and
  flip `Q` back. `TP` (pulse) blocks are accepted but currently simplified to
  `TON` semantics — see Restrictions below.
- **Counters** — `CTU` (count-up) and `CTD` (count-down), edge-triggered on the
  count input. As in MATIEC, `CTU` counts only while `CV < PV` and `CTD` only
  while `CV > 0`. `CTU`'s `R` and `CTD`'s `LD` (or CODESYS's `LOAD`) take the
  power flow of the contact or block `Q` wired to them; `LD` reloads
  `CV := PV` before the count-down.
- **Non-numeric presets** — a `PT`/`PV` given as a variable rather than a
  literal is resolved rather than aborting the run.
- **Arithmetic function blocks** — `ADD`, `SUB`, `MUL`, `DIV`, and `MOVE`.
- **User-defined function blocks** — function blocks with a Structured Text (ST)
  body are translated and inlined into the scan, so custom logic (assignments,
  `IF`/`WHILE`, arithmetic, comparisons, and `AND`/`&`, `OR`, `XOR`, `NOT` on
  `BOOL` operands) participates in the proof. An instance's inputs take the
  value of their wiring — a program variable, another block's output pin, or a
  `BOOL` or integer literal; an unwired input, or one fed by something not
  modelled, is nondeterministic with a warning. Constructs the translator
  cannot lower over-approximate the block's outputs as nondeterministic; a body
  that cannot be translated at all makes every output and in/out pin
  nondeterministic each scan (see [Usage](/docs/ld/usage)).
- **Variables written outside the program** — `VAR_IN_OUT`, `VAR_EXTERNAL` and
  `%M` marker variables take any value at the start of each scan, since a
  caller, an HMI or a fieldbus may write them; `--ld-closed-world` assumes only
  the program writes them.
- **Variable types** — `BOOL`, the integer types `INT`/`DINT`/`TIME`
  (modelled as 32-bit integers), and `REAL` analog values (modelled as
  floating-point).
- **Properties** — the five kinds described in
  [Property Format](/docs/ld/property-format).

## Restrictions

- **Input format.** Programs must be supplied as PLCopen XML. Other LD
  serialisations are not parsed.
- **POU body notations.** Only the `<LD>` / `<ladderDiagram>` body of a program
  POU, and the `<ST>` body of a function block, are translated. A POU whose body is `<ST>`, `<FBD>`,
  `<SFC>` or `<IL>` is rejected with
  `UnsupportedConstruct(<notation> body of POU '<name>', tier=2)`. Ladder nested
  in `<SFC>` step actions is rejected with the rest of the chart: the step and
  transition sequencing that gates those actions is not modelled, so running
  them is a different program. Rejecting rather than skipping is deliberate —
  a body that is skipped leaves the scan cycle empty, and every property then
  holds vacuously.
- **One program, one task.** A document must contain one program POU, run by
  one task as one instance. More than one program POU, task or program instance
  is rejected with `UnsupportedConstruct`: a single scan loop cannot model a
  faster task preempting the program mid-scan. A function block with an LD
  body is rejected too; only ST bodies of function blocks are translated.
- **Data-path wiring.** Graphical rungs translate power flow and user
  function-block outputs. An operator block or a timer/counter output wired to
  an `outVariable`, a block wired to a `PT`/`PV` pin, a `CTU` `R` or `CTD` `LD`
  pin fed by anything other than power flow or a block's `Q`, and an
  unparsable literal or initial value (`"1.5"` for an integer) are rejected
  with `UnsupportedConstruct` rather than read as 0.
- **`TP` pulse timers.** `TP` blocks are modelled with `TON` (on-delay)
  semantics: `Q` rises after `IN` has been held for `PT` ticks, rather than
  emitting a fixed-width pulse on a rising edge of `IN`. Properties that depend
  on accurate pulse-timer behaviour are not faithfully checked yet.
- **Property expression syntax.** Expressions in `invariant` and `absence`
  properties are Boolean-only: variable names combined with `!`, `&&`, `||`, and
  parentheses. Arithmetic relations (for example `Counter >= 5`) are not yet
  accepted in property expressions.
- **Bounded results.** `response` properties are complete only up to their
  justified `max_scans`. Under BMC, `reachability` and all other properties are
  checked only up to the unwind bound; use `--k-induction` for an unbounded
  safety proof.
- **Integer width.** Integer variables are fixed at 32 bits; configurable widths
  are not modelled.

## Reporting issues

The frontend is evolving. Please report bugs or missing constructs on the
[GitHub issue tracker](https://github.com/esbmc/esbmc/issues), ideally with the
PLCopen XML and the property file that reproduce the problem.
