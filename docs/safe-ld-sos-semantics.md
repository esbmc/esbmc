# ESBMC-PLC Structural Operational Semantics for IEC 61131-3 Ladder Diagram

**Status:** DRAFT (WP1 / T1.2)
**Version:** 0.2
**Date:** 2026-10-05

This document gives a Structural Operational Semantics (SOS) for the Tier-1
subset of IEC 61131-3 Ladder Diagram that ESBMC-PLC verifies. It is the
semantic ground truth referenced by the M1 gate in
`docs/roadmap/safe-ld-implementation-plan.md` §5, and it is the left-hand side of the
semantic-preservation theorem in §3.7 of that document: `ld_converter` is
correct exactly when the GOTO program it emits refines the transition relation
defined here.

The rules below describe what the front-end **actually implements**. Where the
implementation deliberately restricts or approximates IEC 61131-3, §8 says so.
Each rule carries the tag used in `src/ld-frontend/semantics/sos_semantics.h`,
so an `LdIRNode` can be traced back to the rule that produced it.

---

## 1. Notation and state space

### 1.1 Values and stores

Tier-1 variables take values in

> V = B ∪ Z ∪ R,  where B = {tt, ff}

A **variable store** σ ∈ Σ is a total map from variable names to values,
respecting the declared type of each variable (§2). We write σ[v ↦ x] for the
store that agrees with σ everywhere except at v, where it takes the value x.

One derived store is carried alongside σ:

- **π ∈ Π** — the *edge store*, mapping each operand sensed by a
  transition-sensing contact to its value at the previous scan boundary.

A full configuration is the pair ⟨σ, π⟩. Where π is not mentioned in a rule it
is threaded through unchanged.

### 1.2 Boolean projection

Contacts and coils may name non-Boolean operands. Define the projection

> ⌊x⌋ = ff if x = 0 or x = ff, and tt otherwise

and its inverse on assignment to a numeric coil, ⌈tt⌉ = 1, ⌈ff⌉ = 0. All
Boolean rules below are stated over ⌊σ(v)⌋; for a `BOOL` variable this is the
identity.

### 1.3 Judgements

Three judgement forms are used.

| Form | Reads |
|---|---|
| ⟨e, σ, π⟩ ⇓ b | element e evaluated in the given state yields power flow b ∈ B |
| ⟨e, σ, π⟩ → σ' | element e transforms the store to σ' |
| ⟨P, σ, π⟩ ⟹ ⟨σ', π'⟩ | one full scan cycle of program P |

Power flow is threaded left to right along a rung: an element's *input* power
flow is written `p` and its output `p'`.

### 1.4 Grammar of the supported subset

A program P is a sequence of networks (§6). A **textual** network is already
a sequence of rungs, each a sequence of elements:

> rung ::= element*
> element ::= contact | coil | fb_step

A **graphical** network is a connection graph G = (N, E): N is the set of
contacts, coils, blocks and the left power rail; E is the set of power-flow
wires (§6.2). G is not required to be series-parallel. A rung-equivalent
sequence is derived from G by the power-flow computation of §6.2, which
accepts any directed acyclic G with every sink reachable from the rail,
including **bridge networks** (two parallel branches tied together by a
cross-connection, so the graph is not reducible to nested series/parallel
contact groups). The grammar therefore does not restrict G to series-parallel
form; §6.4 states the well-formedness conditions G must satisfy instead, and
gives the argument that checking them terminates.

A graph that is *not* acyclic (a wire cycle, where power flow into a node
depends, through some path, on that node's own output) is outside the
grammar: it is rejected at parse time (§6.4) rather than given a semantics.
Feedback expressed through a **variable** (a coil in one network and a
contact on the same variable in a later network, or a later rung of the same
network) is in the grammar: §6.3's sequential evaluation already handles it,
since a contact reads whatever the store holds when its rung runs.

contact ::= `--[ ]--` v | `--[/]--` v | `--[P]--` v | `--[N]--` v
coil    ::= `--( )--` v | `--(S)--` v | `--(R)--` v
fb_step ::= TON(IN, PT, Q, ET) | TOF(IN, PT, Q, ET) | TP(IN, PT, Q, ET)
          | CTU(CU, R, PV, Q, CV) | CTD(CD, LD, PV, Q, CV)
          | ARITH(op, IN1, IN2, OUT)

v ranges over declared program variables (§1.1, §2). An element outside this
grammar (an unrecognised block `typeName`, a data pin wired to something
other than a variable or a literal, or a body language other than LD) is
diagnosed with `UnsupportedConstructError` and excluded from the semantics
(§8).

---

## 2. Type rules

A program is well-typed when every rule below is derivable. The type checker
(`semantics/type_checker.cpp`) rejects programs that are not.

| Construct | Obligation |
|---|---|
| contact on v | v declared; ⌊σ(v)⌋ defined, i.e. v : BOOL, INT, DINT or REAL |
| coil on v | v declared and not an input (§3.1) |
| TON/TOF/TP | IN : BOOL; PT, ET : INT or TIME; Q : BOOL |
| CTU/CTD | CU/CD/R : BOOL; PV, CV : INT or DINT; Q : BOOL |
| ADD/SUB/MUL/DIV | IN1, IN2, OUT numeric and of one type |

`TIME` is represented as a tick count (§5.1), so `TIME` and `INT` share a
representation and are interchangeable at FB pins.

---

## 3. The cyclic scan

### 3.1 Scan rule

A program P is a sequence of networks, each a sequence of rungs
R₁ … R_n (§6 explains how a graphical network is put into that form).
One scan cycle is:

```
                    σ₁ = read_inputs(σ)
      ⟨R₁, σ₁, π⟩ → σ₂    …    ⟨R_n, σ_n, π⟩ → σ_{n+1}
                    π' = latch(σ_{n+1})
      ───────────────────────────────────────────────────────  [SCAN]
                    ⟨P, σ, π⟩ ⟹ ⟨σ_{n+1}, π'⟩
```

where

- `read_inputs(σ)` = σ with every variable declared as a physical input
  reassigned an arbitrary value of its type. Inputs are *free*: the semantics
  admits every input sequence, which is what makes a proof over this relation
  a proof over all environments.
- `latch(σ)` binds each edge-sensed operand v to ⌊σ(v)⌋, at the end of the
  scan and after every rung, so that all contacts sensing v within one scan
  compare against the same previous-scan sample regardless of rung order.

The scan relation is total and deterministic in σ given the input choice: for
each scan there is exactly one σ_{n+1}. Non-determinism enters only through
`read_inputs`.

The execution model is a **single periodic task**. Programs declaring
interrupt tasks or multiple tasks are rejected with
`UnsupportedConstruct(InterruptTask, tier=2)` and are outside this semantics.

The read-inputs/execute/write-outputs structure of one scan cycle is
[IEC 61131-3 §TBD: confirm the clause stating the PLC cyclic execution
model]. The implementation collapses the write-outputs phase into the coil
rules directly, which §8 already records as a deliberate restriction.

### 3.2 Rung rule

A rung is a sequence of elements e₁ … e_m evaluated left to right, starting
from the left power rail, which always supplies power:

```
   p₀ = tt      ⟨e_i, σ_i, π⟩ ⇓ p_i    ⟨e_i, σ_i, π⟩ → σ_{i+1}
   ─────────────────────────────────────────────────────────────────  [RUNG]
                     ⟨e₁ … e_m, σ₁, π⟩ → σ_{m+1}
```

Contacts contribute to p and leave σ unchanged; coils and FB steps consume p
and update σ.

Left-to-right, rail-first evaluation of a rung's elements is
[IEC 61131-3 §TBD: confirm the clause giving LD's execution order within a
rung].

---

## 4. Contacts and coils

Let `val(v) = ⌊σ(v)⌋`, the value of v in the current store (§6.3), and let `p`
be the input power flow.

### 4.1 Static contacts

```
      val(v) = tt                        val(v) = ff
  ───────────────────────  [NO-TRUE]   ───────────────────────  [NO-FALSE]
   ⟨--[ ]-- v, …⟩ ⇓ p              ⟨--[ ]-- v, …⟩ ⇓ ff

      val(v) = ff                        val(v) = tt
  ───────────────────────  [NC-TRUE]   ───────────────────────  [NC-FALSE]
   ⟨--[/]-- v, …⟩ ⇓ p              ⟨--[/]-- v, …⟩ ⇓ ff
```

Normally-open and normally-closed contact semantics are
[IEC 61131-3 §TBD: confirm the clause defining `--[ ]--` / `--[/]--`].

### 4.2 Transition-sensing contacts

A transition is sensed on the *operand*, against the edge store; the
contact's own polarity is applied to the result, so `--[/P]--` conducts on
every scan on which `--[P]--` does not.

```
   val(v) = tt      π(v) = ff              val(v) = ff     π(v) = tt
  ─────────────────────────────  [P-EDGE]   ─────────────────────────────  [N-EDGE]
     ⟨--[P]-- v, …⟩ ⇓ p                        ⟨--[N]-- v, …⟩ ⇓ p
```

and ⇓ ff otherwise. Because π is updated only by `latch` at the end of the
scan, an edge contact conducts for exactly one scan per transition, and two
contacts sensing the same operand agree unless a coil writes the operand
between them. Beremiz instead gives each edge contact its own `R_TRIG`/`F_TRIG`
instance; the two coincide for operands no coil writes, such as inputs.
Applying the contact's own negation after the edge test (so `--[/P]--` is the
Boolean negation of `--[P]--`, not an edge test on `¬v`) is
[IEC 61131-3 §TBD: confirm against §2.5.1.1, open item 1 of §10].

### 4.3 Coils

```
  ─────────────────────────────────────  [COIL]
   ⟨--( )-- v, σ, …⟩ → σ[v ↦ ⌈p⌉]

       p = tt                                p = ff
  ─────────────────────────  [SET]      ─────────────────────  [SET-SKIP]
   ⟨--(S)-- v, σ⟩ → σ[v ↦ tt]            ⟨--(S)-- v, σ⟩ → σ

       p = tt                                p = ff
  ─────────────────────────  [RESET]    ─────────────────────  [RESET-SKIP]
   ⟨--(R)-- v, σ⟩ → σ[v ↦ ff]            ⟨--(R)-- v, σ⟩ → σ
```

A coil writes σ directly, so a later contact on the same variable, in the same
rung or a later one, reads the value just written (§6.3).

Output-coil, set-coil and reset-coil semantics are
[IEC 61131-3 §TBD: confirm the clause defining `--( )--` / `--(S)--` /
`--(R)--`]. Where one scan drives both a set coil and a reset coil on the
same variable with both conditions true, this semantics gives the result
whichever coil's rung the sequential order (§6.3) evaluates last: [SET] and
[RESET] are stated independently with no priority between them, so the
outcome is decided entirely by rung order, not by a reset-dominant or
set-dominant rule. Whether IEC 61131-3 states a dominance rule for this case
, as it does for the CTU/CTD reset input (§5.5), is
[IEC 61131-3 §TBD: confirm]. `warn_set_and_reset` (#8080) flags exactly this
configuration (unconditionally, no flag needed) without changing the
verdict, which is the right response if no dominance rule applies and the
wrong one if IEC 61131-3 states that one does.

---

## 5. Function blocks

### 5.1 The fixed-tick time model

Time is not tracked in wall-clock units. Every scan cycle advances time by
exactly **one tick**, and every preset is a tick count. A duration literal is
converted at parse time:

> ticks(d) = ⌈d / τ⌉

where τ is the period of the declared cyclic task and d the literal's value in
milliseconds. When the program declares no task, τ = 1 ms. Rounding is upward
so that a preset shorter than one scan still takes one scan to expire.

This makes time progression concrete and deterministic: a TON with preset N
fires N scans after the scan its enable rises (one scan after when N = 0), so timer-dependent properties have a known
induction depth and no `__ESBMC_assume` over Δt is needed. What it does not
model is scan-period jitter — see §8.

Throughout this section, `IN` denotes the Boolean projection of the block's
enable pin and `PT`, `ET`, `Q` its preset, elapsed count and output.

### 5.2 TON — on-delay

```
 IN = tt, π(IN) = tt, σ(ET) < σ(PT)    IN = tt, π(IN) = tt, σ(ET) ≥ σ(PT)
 ──────────────────────────────────    ──────────────────────────────────  [TON]
   σ' = σ[ET ↦ σ(ET)+1]                  σ' = σ

            IN = ff  or  π(IN) = ff
      ────────────────────────────────  [TON-RESET]
       σ' = σ[ET ↦ 0], σ'' = σ'[Q ↦ ff]

              σ'' = σ'[Q ↦ (σ'(ET) ≥ σ(PT))]   (IN = tt, π(IN) = tt)
```

Equivalently `Q := IN ∧ π(IN) ∧ ET ≥ PT`. The scan on which IN rises starts
the interval and counts as no elapsed time, as in MATIEC's TON
(`lib/timer.txt`), so Q stays ff on that scan even at `PT = 0`; with
`PT = 0`, Q rises on the next scan.

ET is bounded above by PT, so the count stops once the interval is up
[IEC 61131-3 §TBD: confirm, open item 7 of §10; an earlier draft of this
document cited §2.5.2.3.2 for the 0..PT range without that citation having
been checked against the standard's text]. An unbounded ET would rise on
every scan IN holds and eventually overflow its machine width, which is
undefined behaviour and wraps ET negative so that Q drops back to ff.

### 5.3 TOF — off-delay

```
        IN = tt                     IN = ff, π(IN) = ff, σ(Q) = tt
  ────────────────────────────    ──────────────────────────────────  [TOF]
   σ' = σ[ET ↦ 0][Q ↦ tt]          σ' = σ[ET ↦ σ(ET)+1]
                                   σ'' = σ'[Q ↦ (σ'(ET) < σ(PT))]

          IN = ff, π(IN) = tt          or          IN = ff, σ(Q) = ff
                  ────────────────────────────  [TOF-IDLE]
                            σ' = σ
```

Q rises with IN. The scan on which IN falls starts the interval and counts as
no elapsed time, as in MATIEC's TOF, so Q holds for max(PT, 1) scans counting
that scan: through the falling scan alone when `PT ≤ 1`. The idle rule is what
keeps a TOF from reporting an expired interval at power-up: with Q initialised
to ff and ET to 0, a timer that has never been enabled stays off, rather than
reading ET = 0 as "just dropped".

### 5.4 TP — pulse

```
       σ(Q) = tt                     σ(Q) = ff, IN = tt, π(IN) = ff
  ──────────────────────────    ─────────────────────────────────────  [TP]
   σ' = σ[ET ↦ σ(ET)+1]              σ' = σ[ET ↦ 0][Q ↦ tt]
   σ'' = σ'[Q ↦ σ'(ET) < σ(PT)]
```

and σ' = σ otherwise. A pulse runs for PT scans from a rising IN and ignores
IN until it expires; like TON and TOF, the block keeps its own previous-IN
entry in π, so a TP is retriggerable only after its pulse has completed.

### 5.5 CTU / CTD — counters

```
   σ(CU) = tt, π(CU) = ff, σ(CV) < σ(PV)         σ(R) = tt
  ─────────────────────────────────────  [CTU]  ───────────────────  [CTU-RESET]
   σ' = σ[CV ↦ σ(CV)+1]                          σ' = σ[CV ↦ 0]

                    σ'' = σ'[Q ↦ (σ'(CV) ≥ σ(PV))]

   σ(CD) = tt, π(CD) = ff, σ(CV) > 0
  ─────────────────────────────────────  [CTD]  σ'' = σ'[Q ↦ (σ'(CV) ≤ 0)]
   σ' = σ[CV ↦ σ(CV)−1]
```

Counters are edge-triggered on their count pin, using a per-instance entry in
the edge store. The reset arm applies after the count arm, so a scan in which
both fire leaves CV at 0; whether IEC 61131-3 instead specifies reset as
dominant over counting in the same scan is
[IEC 61131-3 §TBD: confirm, open item 2 of §10]. CTU stops at the preset and
CTD at 0, as the bodies of MATIEC's `CTU` and `CTD` (`lib/counter.txt`) do;
whether IEC 61131-3 instead bounds CV by the type's range rather than by PV/0
is [IEC 61131-3 §TBD: confirm, open item 4 of §10]. An unwired PV reads 0,
the INT default, so a CTU without one never counts.

### 5.6 Arithmetic blocks

```
  ────────────────────────────────────────────  [ARITH]
   ⟨op, σ⟩ → σ[OUT ↦ σ(IN1) op σ(IN2)]
```

for op ∈ {+, −, ×, ÷}; MOVE is the unary case OUT := IN1. Division by zero is
a checked property, not a semantic side condition: the store is undefined
there and ESBMC reports the violation.

---

## 6. Networks

### 6.1 Textual bodies

A `<rung>` body is already a sequence of elements and maps onto [RUNG]
directly.

### 6.2 Graphical bodies

A graphical PLCopen body is a connection graph, not a sequence. Let G be that
graph, with an edge x → y whenever y lists x as a `refLocalId` on a
power-flow pin. The power flow out of a node n is

```
            pf(n) = (⋁ over predecessors p of n in G of pf(p)) ∧ ⟨n⟩ ⇓
  ────────────────────────────────────────────────────────────────────  [NET]
                  a sink s receives the power flow of its predecessors
```

which is the disjunction over rail-to-sink paths of the conjunction of their
contacts, computed per node rather than per path. Sequential evaluation (§6.3)
recomputes it for each sink, so a body with C sinks costs O(C·(V+E)).

Only power-flow pins (`IN`, `CU`, `CD`) induce edges; data pins (`PT`, `PV`)
carry values, and a literal wired to one is read as a constant via ticks(·).

A path running through a function block is cut at the block: the segment
before it is what drives the block's enable, and the segment after it resumes
from the block's output pin. A block is stepped once per scan, immediately
before the first sink that consumes it; a block no sink consumes is stepped
after every sink.

Sinks run in the order the right power rail lists them, the order the vendor
tool draws them. Coils the rail does not list follow in Beremiz's order
(`PLCGenerator.SortInstances`): coils less than 10 apart vertically share a row
and are ordered by x, other rows by y, sorted stably from document order.

### 6.3 Sequential evaluation

Each sink is a statement evaluated in turn: its power flow is recomputed from
the current σ when the sink runs, so it reads every write made by an earlier
sink, in the same rung or in an earlier one. The exception is a function block:
it steps once, and a later sink downstream of it reads the output of that step. This is the ST that Beremiz
generates for MATIEC and OpenPLC. For the toggle of
`regression/ld/stairs_light_safe/stairs_light.ld` it emits

```
IF NOT(lights_buttons_state) AND (R_TRIG1.Q OR R_TRIG2.Q) THEN
  lights_buttons_state := TRUE;
END_IF;
IF lights_buttons_state AND (R_TRIG3.Q OR R_TRIG4.Q) THEN
  lights_buttons_state := FALSE;
END_IF;
```

so a button press sets the variable and the reset clears it in the same scan.
The semantics reproduces that behaviour rather than reading a variable at its
value on entry to the network. Whether IEC 61131-3 instead specifies an
entry-value rule for a variable both read and written within one network is
[IEC 61131-3 §TBD: confirm against §4.1.3 / Ed. 3 §8.1.5, open item 3 of
§10; this reference toolchain's behaviour is reproduced here regardless, and
any departure from the normative text is recorded, not corrected, since the
semantics' purpose is to match what ESBMC-PLC actually verifies against].

### 6.4 Graphical well-formedness

A graphical network G (§6.2) must satisfy three conditions before §6.2's
power-flow computation is defined on it. All three are checked by
`PlcopenXmlParser::parse_network` (`ensure_pf`, `emit_sink`); each violation
is diagnosed and the program is rejected rather than given a semantics.

1. **Acyclic power flow.** The directed graph of power-flow edges (§6.2) must
   contain no cycle: no node's power flow may depend, through any chain of
   predecessors, on its own output. This excludes a wire cycle (a block or
   contact wired, directly or transitively, from its own power-flow output)
   while leaving **feedback through a variable** unrestricted: a coil in one
   network read by a contact in a later network, or a later rung of the same
   network, is not a wire cycle, since the read goes through σ, not through a
   graph edge, and §6.3's sequential evaluation already gives it a semantics.
   This is the acyclic/legal-feedback distinction M31 asks the paper to state.

2. **Every sink is driven.** Each coil, block enable pin, and FB-consumed
   output must have at least one live predecessor (`live_preds`, filtered by
   §6.2's `rail_reaches`). A sink with none would hold its initial value for
   every scan, so any property over it would be vacuously true; this is
   excluded by construction rather than left as a latent vacuity risk.

3. **Recognised element and pin names.** A node tag outside the grammar's
   `contact | coil | block` (§1.4), or a block `typeName` the converter does
   not implement, is diagnosed with `UnsupportedConstructError` before power
   flow is computed from it.

   The plan (`REVISION_PLAN.md` item 7) also names a fourth condition, "known
   `formalParameter`s": that a block's wired pin names (e.g. a `TON`'s `IN`,
   `PT`, `Q`, `ET`) are checked against the pins that block type declares, and
   an unrecognised pin name is rejected. **This is not implemented.** The
   converter looks up each expected pin by name (`get_var("IN")`,
   `resolve_data_pin(block_id, "PT", ...)`, `ir/ld_ir_builder.cpp:47-50`); a
   wire whose `formalParameter` attribute does not match any pin the lookup
   asks for is silently never read, rather than causing a diagnostic. A
   program with a misspelled or vendor-specific pin name on an otherwise
   recognised block type is accepted and verified over a model that ignores
   that wire. Closing this needs a small change (a known-pin-name set per
   block type, checked when the block's `<variable>` children are visited)
   and a regression test (an unrecognised pin name, expected to be rejected);
   until it lands, condition 3 should be read as covering only the element
   and block-type level, not individual pins.

**Termination.** `ensure_pf` is depth-first search over G with an
"in-progress" set keyed by node id: a node already in that set when recursion
reaches it again is reported as a cycle (condition 1) rather than recursed
into, and a node already fully emitted (`pf_emitted`) is not recursed into
again. Every node is therefore visited at most once by a completed call and
at most once more while in progress, so the recursion depth is bounded by
|N| and the total work is O(|N| + |E|): on an acyclic G the search always
terminates, and on a cyclic G it terminates by raising the diagnostic at the
first back-edge found, rather than by stack exhaustion.

---

## 7. Correspondence with the GOTO IR

The translation `ld_converter` performs is a rule-by-rule refinement of the
above. With R ⊆ Σ × S the relation of §3.7 — (σ, s) ∈ R iff σ(v) = s(`ld::v`)
for every LD variable v, extended to π through the shadow symbols
`ld::__edge_prev_v`, `ld::__timer_prev_<instance>` and
`ld::__ctr_prev_<instance>`, each rule maps to:

| Rule | GOTO IR |
|---|---|
| [SCAN] | `code_whilet(true, scan_body)` in `ld::scan_loop` |
| read_inputs | `code_assignt(v, side_effect_expr_nondett)` per input, at scan top |
| latch | `code_assignt(prev_v, ⌊v⌋)` per sensed operand, at scan bottom |
| [NO-*] / [NC-*] | `and_exprt(pf, v)` / `and_exprt(pf, not_exprt(v))` |
| [P-EDGE] / [N-EDGE] | `and_exprt(v, not_exprt(prev_v))` / `and_exprt(not_exprt(v), prev_v)` |
| [COIL] | `code_assignt(v, pf)` |
| [SET] / [RESET] | `code_ifthenelset(pf, code_assignt(v, true/false))` |
| [TON] / [TOF] / [TP] | the `code_ifthenelset` chains of `translate_timer` |
| [CTU] / [CTD] | the `code_ifthenelset` chains of `translate_counter` |
| [ARITH] | `code_assignt(OUT, exprt(op, T))` with an explicit result type |
| [NET] | one rung per path; disjunction via a scratch accumulator (§6.2) |

The disjunction in [NET] is realised without an OR node in the IR: the
accumulator a is cleared unconditionally, each path's chain drives `--(S)-- a`,
and the sink is driven from `--[ ]-- a`. Since the clear precedes every set
and the sink read follows them all, a holds ⋁ of the path conjunctions when
the sink is evaluated.

The proof obligation for each row is that, assuming (σ, s) ∈ R and the rule's
premises, the emitted instructions produce s' with (σ', s') ∈ R. For contacts
and coils this is immediate from the table. For the FB rules it is the
non-trivial obligation named in §3.7 as the primary proof obligation of WP2;
the regression suite under `regression/ld/` discharges it by testing rather
than by proof, and fault injection (`--ld-fault-injection`) checks that
perturbing the rules is detected.

---

## 8. Scope, and what this semantics does not model

The following are deliberate restrictions. Each is either rejected at parse
time or documented as an approximation.

- **Multi-task and interrupt-driven execution.** Rejected
  (`UnsupportedConstruct(InterruptTask, tier=2)`). [SCAN] is single-task.
- **Wall-clock timing and jitter.** §5.1 counts scans, not seconds. A property
  proved here is a property about scan counts; mapping to real time additionally
  requires the scan period to be bounded, which is not modelled.
- **WRITE_OUTPUTS as a distinct phase.** Output coils write σ directly; there
  is no separate output-image latch. This is unobservable to properties that
  are checked at the scan boundary, which is where the encoder places them.
- **Counter reset from a contact chain.** [CTU-RESET] takes R from a variable.
  A reset pin driven by a contact chain in a graphical body is diagnosed and
  left unconnected rather than silently approximated.
- **Integer width.** CV and ET are machine integers of the configured width.
  Neither can wrap: CV stays between 0 and PV (§5.5) and ET saturates at PT.
  See the open item in §10 on which counter bound IEC intends.
- **Non-timer, non-counter blocks on a rung path.** A path through an
  arithmetic or unknown block is diagnosed and dropped rather than modelled,
  so a program using one verifies over strictly less behaviour. User-defined
  function blocks are executed from their Structured Text body instead
  (`ir_gen/st_fb_translator.cpp`), which is outside this semantics.
- **The semantics itself is not proved against the normative text.** It is
  validated by review and by fault injection. It is the assumed ground truth
  of §3.7's theorem, not a consequence of it.

---

## 9. Relation to the property format

`docs/safe-ld-property-format.md` defines the YAML property language. A
property is evaluated in σ at the scan boundary, i.e. after σ_{n+1} in [SCAN]
and before the next `read_inputs`. Properties may name:

- any declared program variable;
- `<instance>__<pin>` for a function-block pin synthesised by the graphical
  resolver (§6.2), e.g. `TOF0__Q`.

---

## 10. Open items for M1

The M1 gate requires two independent reviewers to validate this specification
against IEC 61131-3 §2. That review has not yet been carried out. Known gaps
to raise in it:

1. §4.2 applies contact polarity after the edge test. IEC's operator ordering
   for a negated edge contact should be confirmed against §2.5.1.1.
2. §5.5 orders the counter's reset arm after its count arm. Whether
   IEC 61131-3 specifies reset as dominant over counting within the same
   scan, and if so against which clause, is open and unconfirmed; no clause
   number is given here since none has been checked against the standard's
   text.
3. §6.3 follows the sequential evaluation of Beremiz/MATIEC. IEC 61131-3
   §4.1.3 (Ed. 3 §8.1.5) states an entry-value rule for feedback paths within a
   network; confirm whether it applies to LD coils and contacts on the same
   variable, and record where the reference toolchain departs from it.
4. §5.5 stops CTU at the preset and CTD at 0, as MATIEC does. Secondary
   sources render the normative bodies with `CV < PVmax` and `CV > PVmin`
   (the type bounds); confirm which IEC 61131-3 §2.5.2.3.3 specifies. The two
   agree on Q in every reachable state and differ only in CV beyond the preset
   or below 0, so the choice changes only verdicts that read CV.
5. Every `[IEC 61131-3 §TBD: ...]` marker inline in §§3.1, 3.2, 4.1, 4.2, 4.3,
   5.2, 5.5, 6.3 (added alongside items 1 to 4 above, WS1, October 2026)
   needs a clause number from the actual standard text. This session had no
   verified copy
   of IEC 61131-3 and could not source normative clause numbers from web
   search, which returns only secondary descriptions of PLC behaviour, not
   the standard's text; filling these in needs a copy of the standard.
6. §6.4 condition 3 ("known `formalParameter`s") is not implemented: an
   unrecognised pin name on an otherwise recognised block type is silently
   ignored rather than rejected (`get_var`, `ir/ld_ir_builder.cpp:47-50`).
   This is a code gap, not a semantics gap; §6.4 records it so the paper
   does not claim a check that does not exist. WS0-style fix: a known-pin
   set per block type and a regression test, filed as a candidate fix
   before WS1's M1 review rather than during it.
7. §5.2 bounds a TON's ET above by PT. An earlier draft of this document
   cited IEC 61131-3 §2.5.2.3.2 for the 0..PT range; that citation predates
   this session's inline-marker pass and was not independently checked
   against the standard, so it is now an open item rather than a settled
   fact, alongside items 2 and 4's citations in the same §2.5.2.3 family.
