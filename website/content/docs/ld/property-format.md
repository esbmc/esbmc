---
title: Property Format
weight: 3
---

Safety properties are written in a YAML file passed to ESBMC with `--ld-props`.
Each property is encoded as an assertion checked on every scan cycle. This page
specifies the file structure and the five supported property kinds.

## File structure

```yaml
properties:
  - id: <string>                  # unique within the file
    kind: <property-kind>
    description: <string>         # optional; shown in counterexamples
    justification: <string>       # required for 'response' and 'reachability'
    # kind-specific fields (see below)
```

The `id` names the property. The optional `description` is copied verbatim into
the assertion comment, so it appears in the ESBMC counterexample and identifies
which property failed. Variable names in expressions must match the identifiers
declared in the PLCopen XML `<variable name="...">` elements exactly
(case-sensitive).

## Property kinds

### `mutual_exclusion`

Two or more BOOL variables are never simultaneously `TRUE`.

```yaml
- id: P1
  kind: mutual_exclusion
  variables: [Motor_Forward, Motor_Reverse]
  description: "Forward and Reverse coils must never be energised simultaneously"
```

Sound and complete.

### `invariant`

A Boolean expression over LD variables holds at all times.

```yaml
- id: P2
  kind: invariant
  expression: "ESD_Valve_Closed || !High_Pressure_Alarm"
  description: "ESD valve must be closed when the high-pressure alarm is active"
```

Expression syntax: declared variable names, `!` (negation), `&&` (conjunction),
`||` (disjunction), and `(...)` for grouping. Sound and complete.

### `absence`

A Boolean expression never holds (the dual of `invariant`).

```yaml
- id: P3
  kind: absence
  expression: "OL_Trip && Motor_Forward"
  description: "Motor must not run forward when the overload trip is active"
```

Sound and complete.

### `response`

Whenever a `trigger` variable becomes `TRUE`, a `response` variable becomes
`TRUE` within `max_scans` scan iterations.

```yaml
- id: P4
  kind: response
  trigger: Start_Button
  response: Conveyor_Running
  max_scans: 5
  justification: "TON1_PT is 2 ticks; worst-case 3 scans + 2-scan margin"
```

Sound, but **bounded** by `max_scans`. The bound must be justified by timing
analysis (recorded in the required `justification` field).

### `reachability`

A target state is reachable — useful for liveness checks. Under BMC, a
`VIOLATION` means the target state was reached; `SAFE` under k-induction means it
was proved unreachable.

```yaml
- id: P5
  kind: reachability
  expression: "Belt_1 && Belt_2 && Conveyor_Running"
  description: "Full-speed operation state must be reachable"
  justification: "Liveness check: all three outputs true simultaneously"
```

Under k-induction the unreachability proof is sound and complete; under BMC only
unreachability up to the unwind bound is established.

## Soundness and completeness

| Kind               | Sound? | Complete?        | Notes                                            |
|--------------------|--------|------------------|--------------------------------------------------|
| `mutual_exclusion` | Yes    | Yes              | Checked every scan; k-induction and BMC agree    |
| `invariant`        | Yes    | Yes              | As above                                         |
| `absence`          | Yes    | Yes              | As above                                         |
| `response`         | Yes    | Bounded          | Requires a justified `max_scans`                 |
| `reachability`     | Yes    | No (BMC-bounded) | k-induction unreachability proof is complete     |

## Example: motor interlock

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
    description: "Forward start only activates the forward coil (not reverse)"
```
