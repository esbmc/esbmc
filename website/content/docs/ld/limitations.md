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
  overwritten by the last branch, and network feedback is snapshotted per scan
  as required by IEC 61131-3 §4.1.3. A rung path that passes through a function
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
  count input, with reset handling. `CV` saturates at the integer bounds instead
  of wrapping.
- **Non-numeric presets** — a `PT`/`PV` given as a variable or expression rather
  than a literal is resolved rather than aborting the run.
- **Arithmetic function blocks** — `ADD`, `SUB`, `MUL`, `DIV`, and `MOVE`.
- **User-defined function blocks** — function blocks with a Structured Text (ST)
  body are translated rung-by-rung and inlined into the scan, so custom logic
  (assignments, `IF`/`WHILE`, arithmetic and comparisons) participates in the
  proof. Constructs the translator cannot lower over-approximate the block's
  outputs as nondeterministic by default; `--ld-sound-mode` makes them fall back
  to a no-op instead (see [Usage](/docs/ld/usage)).
- **Variable types** — `BOOL`, the integer types `INT`/`DINT`/`TIME`
  (modelled as 32-bit integers), and `REAL` analog values (modelled as
  floating-point).
- **Properties** — the five kinds described in
  [Property Format](/docs/ld/property-format).

## Restrictions

- **Input format.** Programs must be supplied as PLCopen XML. Other LD
  serialisations are not parsed.
- **POU body notations.** Only `<LD>` / `<ladderDiagram>` bodies, and the `<ST>`
  body of a function block, are translated. A POU whose body is `<ST>`, `<FBD>`,
  `<SFC>` or `<IL>` is rejected with
  `UnsupportedConstruct(<notation> body of POU '<name>', tier=2)`. Ladder nested
  in `<SFC>` step actions is rejected with the rest of the chart: the step and
  transition sequencing that gates those actions is not modelled, so running
  them is a different program. Rejecting rather than skipping is deliberate —
  a body that is skipped leaves the scan cycle empty, and every property then
  holds vacuously.
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
