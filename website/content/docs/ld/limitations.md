---
title: Limitations
weight: 4
---

The Ladder Diagram frontend is under active development and is gated behind the
`ENABLE_LD_FRONTEND` build option. This page records what is currently
supported and the known restrictions.

## Supported constructs

- **Contacts and coils** — normally-open and normally-closed contacts; output,
  Set, and Reset coils.
- **Timers** — `TON` (on-delay) and `TOF` (off-delay), with their retained
  `ET`/`Q` state evaluated per scan. `TP` (pulse) blocks are accepted but
  currently simplified to `TON` semantics — see Restrictions below.
- **Counters** — `CTU` (count-up) and `CTD` (count-down), edge-triggered on the
  count input, with reset handling.
- **Arithmetic function blocks** — `ADD`, `SUB`, `MUL`, `DIV`, and `MOVE`.
- **Variable types** — `BOOL`, and the integer types `INT`/`DINT`/`TIME`
  (modelled as 32-bit integers).
- **Properties** — the five kinds described in
  [Property Format](/docs/ld/property-format).

## Restrictions

- **Input format.** Programs must be supplied as PLCopen XML. Other LD
  serialisations are not parsed.
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
