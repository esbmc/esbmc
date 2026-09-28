# Multi-Conveyor Sequential Startup (CS2)

**Source:** synthetic, written for WP3 case study CS2
**Constructs:** contacts, coils, TON timer with a declared preset

## Description

A start request, gated on the emergency stop being clear, enables a TON
timer. Once the timer confirms the request for `TON1_PT` scans the conveyor
starts, and both belts follow it. `TON1_PT` is declared as 2 ticks; at the
20 ms task period modelled in §3.3 that is a 2-scan confirmation delay.

## Safety Properties

| ID | Kind | Description |
|---|---|---|
| P4 | invariant | Belt_1 runs whenever the conveyor runs |
| P5 | invariant | Belt_2 runs whenever the conveyor runs |
| P6 | absence | Conveyor must not run while Emergency_Stop is active |
| P7 | invariant | Conveyor runs only on a confirmed timer output |
| P8 | invariant | Timer output is held only while its enable is held |

## Expected Result

VERIFICATION SUCCESSFUL — k-induction, k=2. Wired as
`regression/ld/conveyor_sequencing_safe`.

## Validation notes

The original property set carried a `response` property requiring
`Conveyor_Running` within 5 scans of `Start_Button`. That property is not
sound for this program: `Stop_Button` is a free input, and holding it blocks
the start indefinitely, so no finite bound exists. Its `justification` field
accounted for the timer delay but not for the stop input. It has been dropped
rather than given a larger bound — see §3.5 on bounded `response` properties.

The preset was also never declared: `TON1_PT` had no `<initialValue>`, so it
read as zero and the timer confirmed instantly, which is what made P6 fail
before this benchmark was validated.
