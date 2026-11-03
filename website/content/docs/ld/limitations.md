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
  against the operand's value at the contact's previous evaluation, one sample
  per coil the contact feeds, rather than treated as level contacts.
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
- **Timers** — `TON` (on-delay), `TOF` (off-delay) and `TP` (pulse), with their
  retained `ET`/`Q` state evaluated per scan. `ET` stops at `PT` as IEC 61131-3
  §2.5.2.3.2 requires, so a timer held on indefinitely cannot overflow `ET` and
  flip `Q` back. As in MATIEC, no time elapses on the scan in which `IN`
  changes: a `TON` does not advance `ET` or set `Q` on the scan `IN` rises, and
  a `TOF` holds `Q` through the scan `IN` falls. A rising `IN` starts a `TP`
  pulse of `PT` scans that ignores `IN` until it expires.
- **Direction inference** — a variable with no address and no interface
  section is an input when something reads it and nothing writes it. The output
  pins of timers, counters, arithmetic blocks and user function blocks count as
  writes, so a contact that reads a timer's `Q` does not turn it into a free
  input ([#8445](https://github.com/esbmc/esbmc/pull/8445)).
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
- **Variable types** — `BOOL`; `INT` as a 16-bit signed integer and `DINT` and
  `TIME` as 32-bit signed integers, as in IEC 61131-3 and MATIEC; `REAL` as IEEE
  binary32 and `LREAL` as binary64. Timers, counters and arithmetic compute in
  the common type of their operands, so an `INT` meeting a `DINT` or `TIME` is
  computed at 32 bits, and an unsuffixed real literal that meets a `REAL` is
  rounded once to binary32 from its source text
  ([#8198](https://github.com/esbmc/esbmc/pull/8198),
  [#8910](https://github.com/esbmc/esbmc/pull/8910)). `--overflow-check`
  reports an `INT` sum, difference or product outside 16 bits; without it the
  result wraps, as MATIEC's C does.
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
- **Declarations and block pins.** The parser refuses, rather than reads as
  something else: a name declared twice in one interface, a name that starts
  with `__edge_prev_`, `__edge_q_`, `__timer_prev_` or `__ctr_prev_`, a variable
  of a derived type used as a contact, coil or block pin, a coil `kind` or
  `storage` other than `set`, `reset` or `none`, two timers or two counters that
  resolve to one instance name (an unnamed block takes the name of its `Q`
  variable), an `EN` of a standard or user function block wired to anything but
  the left power rail, and an `ENO` that feeds anything but the right power
  rail ([#8859](https://github.com/esbmc/esbmc/pull/8859)). A Boolean
  variable that only a block pin reads (`IN`, `CU`, `CD`, `R`, `LD`) and nothing
  writes is a free input, as one that only a contact reads is.
- **Types and values.** An elementary type with no model, such as `BYTE`,
  `WORD`, `UINT`, `SINT` or `LINT`, is refused rather than read as another
  type. So are an `INT` initial value or literal outside 16 bits,
  a `DINT` or `TIME` initial value outside 32 bits, and a timer whose `ET` is
  `INT` with a wider `PT`. A coil on an input variable is refused, a timer's
  `IN` and `Q` and its `PT` and `ET`, and a counter's `PV` and `CV`, must be
  distinct variables, and a counter with an `INT` `CV` needs an `INT` preset.
  An unparsable task interval is refused (`PT0S` is accepted).
- **Coils, wires and blocks the translation cannot honour.** Refused with
  `UnsupportedConstruct` rather than translated as something else: a negated or
  edge-triggered coil; a negation or edge attribute spelled in a way the parser
  does not recognise; a continuation wire; a block wire whose `formalParameter`
  is not a pin of the block's type (such as `PTT` for a `TON`'s `PT`); a
  graphical block other than a timer or counter acting as a power-flow source;
  a textual `<rung>` that mixes a timer, counter or arithmetic block with
  contacts or coils; a block or output variable with a positive
  `executionOrderId`; and a graphical `jump`, `label` or `return`, which would
  otherwise be dropped so that every rung ran
  ([#8216](https://github.com/esbmc/esbmc/pull/8216),
  [#8217](https://github.com/esbmc/esbmc/pull/8217),
  [#8186](https://github.com/esbmc/esbmc/pull/8186),
  [#8435](https://github.com/esbmc/esbmc/pull/8435),
  [#8858](https://github.com/esbmc/esbmc/pull/8858)).
- **Property expression syntax.** Expressions in `invariant` and `absence`
  properties are Boolean-only: variable names combined with `!`, `&&`, `||`, and
  parentheses. Arithmetic relations (for example `Counter >= 5`) are not yet
  accepted in property expressions, and a variable that is not `BOOL` is
  rejected with a diagnostic ([#8846](https://github.com/esbmc/esbmc/pull/8846)).
- **Bounded results.** `response` properties are complete only up to their
  justified `max_scans`. Under BMC, `reachability` and all other properties are
  checked only up to the unwind bound; use `--k-induction` for an unbounded
  safety proof.
- **Integer width.** Each integer type has its IEC width (`INT` 16 bits, `DINT`
  and `TIME` 32 bits); the types without a model are refused, and configurable
  widths are not modelled. ST intermediates between two `INT` operands are
  16-bit, as in IEC; MATIEC's C promotes them to `int`, so the two differ only
  when an unstored intermediate overflows 16 bits.

## Reporting issues

The frontend is evolving. Please report bugs or missing constructs on the
[GitHub issue tracker](https://github.com/esbmc/esbmc/issues), ideally with the
PLCopen XML and the property file that reproduce the problem.
