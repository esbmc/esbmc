# ESBMC-PLC Structural Operational Semantics for IEC 61131-3 Ladder Diagram

**Status:** DRAFT (WP1 / T1.2)
**Version:** 0.1
**Date:** 2026-07-24

This document gives a Structural Operational Semantics (SOS) for the Tier-1
subset of IEC 61131-3 Ladder Diagram that ESBMC-PLC verifies. It is the
semantic ground truth referenced by the M1 gate in
`docs/safe-ld-implementation-plan.md` §5, and it is the left-hand side of the
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

Two derived stores are carried alongside σ:

- **π ∈ Π** — the *edge store*, mapping each operand sensed by a
  transition-sensing contact to its value at the previous scan boundary.
- **φ ∈ Φ** — the *feedback store*, mapping each network-feedback variable
  (§6.3) to its value on entry to the current network.

A full configuration is the triple ⟨σ, π, φ⟩. Where π and φ are not mentioned
in a rule they are threaded through unchanged.

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
| ⟨e, σ, π, φ⟩ ⇓ b | element e evaluated in the given state yields power flow b ∈ B |
| ⟨e, σ, π, φ⟩ → σ' | element e transforms the store to σ' |
| ⟨P, σ, π⟩ ⟹ ⟨σ', π'⟩ | one full scan cycle of program P |

Power flow is threaded left to right along a rung: an element's *input* power
flow is written `p` and its output `p'`.

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
                    φ  = snapshot(σ₁)                         [FEEDBACK]
      ⟨R₁, σ₁, π, φ⟩ → σ₂    …    ⟨R_n, σ_n, π, φ⟩ → σ_{n+1}
                    π' = latch(σ_{n+1})
      ───────────────────────────────────────────────────────  [SCAN]
                    ⟨P, σ, π⟩ ⟹ ⟨σ_{n+1}, π'⟩
```

where

- `read_inputs(σ)` = σ with every variable declared as a physical input
  reassigned an arbitrary value of its type. Inputs are *free*: the semantics
  admits every input sequence, which is what makes a proof over this relation
  a proof over all environments.
- `snapshot(σ)` binds each feedback variable to its value here, before any
  rung runs (§6.3).
- `latch(σ)` binds each edge-sensed operand v to ⌊σ(v)⌋, at the end of the
  scan and after every rung, so that all contacts sensing v within one scan
  compare against the same previous-scan sample regardless of rung order.

The scan relation is total and deterministic in σ given the input choice: for
each scan there is exactly one σ_{n+1}. Non-determinism enters only through
`read_inputs`.

The execution model is a **single periodic task**. Programs declaring
interrupt tasks or multiple tasks are rejected with
`UnsupportedConstruct(InterruptTask, tier=2)` and are outside this semantics.

### 3.2 Rung rule

A rung is a sequence of elements e₁ … e_m evaluated left to right, starting
from the left power rail, which always supplies power:

```
   p₀ = tt      ⟨e_i, σ_i, π, φ⟩ ⇓ p_i    ⟨e_i, σ_i, π, φ⟩ → σ_{i+1}
   ─────────────────────────────────────────────────────────────────  [RUNG]
                     ⟨e₁ … e_m, σ₁, π, φ⟩ → σ_{m+1}
```

Contacts contribute to p and leave σ unchanged; coils and FB steps consume p
and update σ.

---

## 4. Contacts and coils

Let `val(v, φ) = ⌊φ(v)⌋` if v is a feedback variable and `⌊σ(v)⌋` otherwise
(§6.3), and let `p` be the input power flow.

### 4.1 Static contacts

```
      val(v, φ) = tt                        val(v, φ) = ff
  ───────────────────────  [NO-TRUE]   ───────────────────────  [NO-FALSE]
   ⟨--[ ]-- v, …⟩ ⇓ p              ⟨--[ ]-- v, …⟩ ⇓ ff

      val(v, φ) = ff                        val(v, φ) = tt
  ───────────────────────  [NC-TRUE]   ───────────────────────  [NC-FALSE]
   ⟨--[/]-- v, …⟩ ⇓ p              ⟨--[/]-- v, …⟩ ⇓ ff
```

### 4.2 Transition-sensing contacts

A transition is sensed on the *operand*, against the edge store; the
contact's own polarity is applied to the result, so `--[/P]--` conducts on
every scan on which `--[P]--` does not.

```
   val(v, φ) = tt      π(v) = ff              val(v, φ) = ff     π(v) = tt
  ─────────────────────────────  [P-EDGE]   ─────────────────────────────  [N-EDGE]
     ⟨--[P]-- v, …⟩ ⇓ p                        ⟨--[N]-- v, …⟩ ⇓ p
```

and ⇓ ff otherwise. Because π is updated only by `latch` at the end of the
scan, an edge contact conducts for exactly one scan per transition, and two
contacts sensing the same operand always agree.

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

A coil writes σ directly, never φ: within a network, later contacts on a
feedback variable still read the entry snapshot (§6.3).

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
fires after exactly N scans, so timer-dependent properties have a known
induction depth and no `__ESBMC_assume` over Δt is needed. What it does not
model is scan-period jitter — see §8.

Throughout this section, `IN` denotes the Boolean projection of the block's
enable pin and `PT`, `ET`, `Q` its preset, elapsed count and output.

### 5.2 TON — on-delay

```
    IN = tt, σ(ET) < σ(PT)              IN = tt, σ(ET) ≥ σ(PT)
  ─────────────────────────────    ─────────────────────────────  [TON]
   σ' = σ[ET ↦ σ(ET)+1]             σ' = σ

                    IN = ff
              ────────────────────────────  [TON-RESET]
               σ' = σ[ET ↦ 0], σ'' = σ'[Q ↦ ff]

              σ'' = σ'[Q ↦ (σ'(ET) ≥ σ(PT))]   (IN = tt)
```

Equivalently `Q := IN ∧ ET ≥ PT`. Conjoining `IN` matters at `PT = 0`, where
a TON must follow its enable directly rather than latch on.

ET is bounded above by PT (IEC 61131-3 §2.5.2.3.2 gives ET the range 0..PT), so
the count stops once the interval is up. An unbounded ET would rise on every
scan IN holds and eventually overflow its machine width, which is undefined
behaviour and wraps ET negative so that Q drops back to ff.

### 5.3 TOF — off-delay

```
        IN = tt                          IN = ff, σ(Q) = tt
  ────────────────────────────    ──────────────────────────────────  [TOF]
   σ' = σ[ET ↦ 0][Q ↦ tt]          σ' = σ[ET ↦ σ(ET)+1]
                                   σ'' = σ'[Q ↦ (σ'(ET) < σ(PT))]

                        IN = ff, σ(Q) = ff
                  ────────────────────────────  [TOF-IDLE]
                            σ' = σ
```

Q rises with IN and holds for PT scans after IN drops. The idle rule is what
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
IN until it expires; the block keeps its own previous-IN entry in π, so a TP
is retriggerable only after its pulse has completed.

### 5.5 CTU / CTD — counters

```
   σ(CU) = tt, π(CU) = ff, σ(CV) < INTmax        σ(R) = tt
  ─────────────────────────────────────  [CTU]  ───────────────────  [CTU-RESET]
   σ' = σ[CV ↦ σ(CV)+1]                          σ' = σ[CV ↦ 0]

                    σ'' = σ'[Q ↦ (σ'(CV) ≥ σ(PV))]

   σ(CD) = tt, π(CD) = ff, σ(CV) > INTmin
  ─────────────────────────────────────  [CTD]  σ'' = σ'[Q ↦ (σ'(CV) ≤ 0)]
   σ' = σ[CV ↦ σ(CV)−1]
```

Counters are edge-triggered on their count pin, using a per-instance entry in
the edge store. The reset arm applies after the count arm, so a scan in which
both fire leaves CV at 0.

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
power-flow pin. Define, for a sink s (a coil, or a function block's enable
pin), the set

> paths(s) = { simple paths from a leftPowerRail to s in G }

Each path is a series chain, so it contributes the conjunction of its
contacts; paths to the same sink are alternatives, so the sink's power flow is
their disjunction:

```
                   pf(s) = ⋁ over q ∈ paths(s) of ⋀ over c ∈ q of ⟨c⟩ ⇓
  ────────────────────────────────────────────────────────────────────  [NET]
                       sink s receives power flow pf(s)
```

Only power-flow pins (`IN`, `CU`, `CD`) induce edges; data pins (`PT`, `PV`)
carry values, and a literal wired to one is read as a constant via ticks(·).

A path running through a function block is cut at the block: the segment
before it is what drives the block's enable, and the segment after it resumes
from the block's output pin. Sinks are ordered by the order in which the
right power rail lists them — the order the vendor tool draws them — and a
block is stepped immediately before the first sink that consumes it.

**Complexity.** |paths(s)| is exponential in the number of parallel branches
reaching s. This is tractable for the programs evaluated here; a path-count
guard is future work.

### 6.3 Feedback variables

A variable both written by a coil and sensed by a contact of the same network
closes a feedback loop, and [NET] alone does not determine its meaning — the
answer would depend on the order the resolver happened to emit sinks in. IEC
61131-3 §4.1.3 fixes this by requiring the loop variable to be read at its
value on entry to the network. That is the role of φ:

```
      F = { v : v written by a coil of N and sensed by a contact of N }
  ─────────────────────────────────────────────────────────────────────  [FEEDBACK]
                  φ = { v ↦ σ(v) : v ∈ F }, before any rung of N
```

Contacts on v ∈ F read φ(v); coils on v write σ(v). A latch therefore takes
effect on the scan *after* the one that set it, which is the standard
behaviour of a feedback coil in a single network.

---

## 7. Correspondence with the GOTO IR

The translation `ld_converter` performs is a rule-by-rule refinement of the
above. With R ⊆ Σ × S the relation of §3.7 — (σ, s) ∈ R iff σ(v) = s(`ld::v`)
for every LD variable v, extended to π and φ through the shadow symbols
`ld::__edge_prev_v` and `ld::v__prev` — each rule maps to:

| Rule | GOTO IR |
|---|---|
| [SCAN] | `code_whilet(true, scan_body)` in `ld::scan_loop` |
| read_inputs | `code_assignt(v, side_effect_expr_nondett)` per input, at scan top |
| latch | `code_assignt(prev_v, ⌊v⌋)` per sensed operand, at scan bottom |
| [FEEDBACK] | a rung `--[ ]-- v --( )-- v__prev` emitted before all others |
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
  Both saturate rather than wrap: CV at INTmax/INTmin, ET at PT. Saturating at
  the type bound rather than at PV over-approximates CV above the preset, which
  can raise a false alarm but cannot hide a violation; see the open item in §10
  on which bound IEC intends.
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
  resolver (§6.2), e.g. `TOF0__Q`;
- `<var>__prev` for the entry snapshot of a feedback variable (§6.3).

---

## 10. Open items for M1

The M1 gate requires two independent reviewers to validate this specification
against IEC 61131-3 §2. That review has not yet been carried out. Known gaps
to raise in it:

1. §4.2 applies contact polarity after the edge test. IEC's operator ordering
   for a negated edge contact should be confirmed against §2.5.1.1.
2. §5.5 orders the counter's reset arm after its count arm. IEC 61131-3
   defines CTU with reset dominant; confirm the intended order when both fire
   in one scan.
3. §6.3 applies the entry-snapshot rule to all feedback variables of a
   network. IEC 61131-3 §4.1.3 states it for feedback paths specifically;
   confirm the two coincide for LD bodies, or narrow the rule.
4. §5.5 saturates CV at the integer type's bound. Secondary sources render the
   normative CTU body with both `CV < PVmax` (the type bound, as here) and
   `CV < PV` (the preset); confirm which IEC 61131-3 §2.5.2.3.3 specifies. The
   two agree on Q for every reachable state and differ only in CV's value above
   the preset, so this changes no verdict that does not read CV directly.
