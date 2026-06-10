# SAFE-LD Property Specification Format

**Status:** DRAFT (WP1 / T1.3)  
**Version:** 0.1  
**Date:** 2026-06-09

This document specifies the YAML format used to express safety properties for
IEC 61131-3 Ladder Diagram programs verified with SAFE-LD.

---

## Implementation Status (#5289)

A property file is consumed either by the `esbmc` driver directly —
`esbmc program.ld --ld-props props.yaml --k-induction` — or via `ld-verify`
(`ld-verify program.{ld,xml} --props props.yaml`). The property encoder turns
each entry into a `code_assertt` node appended to the scan-loop body.

| Kind | Status |
|---|---|
| `mutual_exclusion` | **Implemented & exercised** (regression test `missing_interlock_fail`) |
| `invariant` | **Implemented**, encodes and verifies |
| `absence` | **Implemented & exercised** (regression test `motor_interlock`) |
| `reachability` | **Implemented**, encodes and verifies |
| `response` | **Not yet working** — the auxiliary scan counter trips an irep2 bit-width assertion during GOTO generation (shared root cause with the timer/counter encoding; see the implementation plan §10) |

The `description` field is propagated into the `code_assertt` comment, so a
violated property is named in the ESBMC counterexample and in the `ld-verify`
JSON report's `description` field.

---

## File Structure

```yaml
properties:
  - id: <string>
    kind: <property-kind>
    description: <string>           # optional; used in counterexample reports
    justification: <string>         # required for 'response' and 'reachability'
    # kind-specific fields (see below)
```

Each property has a unique `id` (used to name the `code_assertt` node and appear
in the verification result JSON).

---

## Property Kinds

### `mutual_exclusion`

Asserts that two or more BOOL variables are never simultaneously `TRUE`.

```yaml
- id: P1
  kind: mutual_exclusion
  variables: [Motor_Forward, Motor_Reverse]
  description: "Forward and Reverse coils must never be energised simultaneously"
```

**GOTO IR emitted:** `code_assertt(not_exprt(and_exprt(A, B, ...)))`

**Guarantees:** Sound and complete. Checked at every scan iteration.

---

### `invariant`

Asserts that a Boolean expression over LD variables holds at all times.

```yaml
- id: P2
  kind: invariant
  expression: "ESD_Valve_Closed || !High_Pressure_Alarm"
  description: "ESD valve must be closed when high-pressure alarm is active"
```

**Expression syntax (Tier 1):**
- Variable names as declared in the PLCopen XML
- `!var` — logical negation
- `A && B` — conjunction
- `A || B` — disjunction
- `(expr)` — parenthesised sub-expression

**GOTO IR emitted:** `code_assertt(expr)`

**Guarantees:** Sound and complete.

---

### `absence`

Asserts that a Boolean expression never holds (dual of `invariant`).

```yaml
- id: P3
  kind: absence
  expression: "OL_Trip && Motor_Forward"
  description: "Motor must not run forward when overload trip is active"
```

**GOTO IR emitted:** `code_assertt(not_exprt(expr))`

**Guarantees:** Sound and complete.

---

### `response`

Asserts that whenever a `trigger` variable is `TRUE`, a `response` variable
becomes `TRUE` within `max_scans` scan iterations.

```yaml
- id: P4
  kind: response
  trigger: Start_Button
  response: Conveyor_Running
  max_scans: 5
  justification: "TON1_PT is 2 ticks; worst-case 3 scans + 2-scan margin"
```

**GOTO IR emitted:** auxiliary scan counter + `code_assertt(ctr <= max_scans)`

**Guarantees:** Sound but **bounded**. The bound `max_scans` must be justified by
timing analysis or IEC 61508 §7 requirements and documented in the `justification`
field. `ld-verify` rejects `response` properties without a `justification`.

---

### `reachability`

Asserts that a target state is reachable (used for liveness checks and
fault-injection validation). Under BMC, a `VIOLATION` result means the target
state was reached; `SAFE` under k-induction means it was proved unreachable.

```yaml
- id: P5
  kind: reachability
  expression: "Belt_1 && Belt_2 && Conveyor_Running"
  description: "Full-speed operation state must be reachable"
  justification: "Liveness check: all three outputs true simultaneously"
```

**GOTO IR emitted:** `code_ifthenelset(guard, code_assertt(false))`

**Guarantees:** Under k-induction, unreachability is sound and complete for the
cyclic-scan model. Under BMC, only unreachability up to the unwind bound is
established.

---

## Soundness and Completeness Table

| Kind               | Sound? | Complete?       | Notes                                               |
|--------------------|--------|-----------------|-----------------------------------------------------|
| `mutual_exclusion` | Yes    | Yes             | Checked every scan; k-ind and BMC give exact result |
| `invariant`        | Yes    | Yes             | Same as above                                       |
| `absence`          | Yes    | Yes             | Same as above                                       |
| `response`         | Yes    | **Bounded**     | Requires justified `max_scans`                      |
| `reachability`     | Yes    | No (BMC-bounded) | k-ind unreachability proof is sound and complete    |

---

## Naming Conventions

- Property `id` values must be unique within a file.
- Variable names in expressions must match the identifiers declared in the
  PLCopen XML `<variable name="...">` elements exactly (case-sensitive).
- The `description` field is copied verbatim into the `code_assertt` comment
  field and appears in the ESBMC counterexample trace and the `ld-verify` JSON
  report.

---

## Example: Motor Interlock

```yaml
properties:
  - id: P1
    kind: mutual_exclusion
    variables: [Motor_Forward, Motor_Reverse]
    description: "Forward and Reverse coils must never be simultaneously energised"

  - id: P2
    kind: response
    trigger: Start_Button
    response: Motor_Forward
    max_scans: 3
    justification: "Direct coil; no timer; response in same scan cycle + 2 margin"

  - id: P3
    kind: invariant
    expression: "Motor_Forward || !Start_Fwd || Motor_Reverse"
    description: "Forward start only activates forward coil (not reverse)"
```
