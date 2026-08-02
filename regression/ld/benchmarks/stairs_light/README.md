# Staircase Light Control

**Source:** CONTROLLINO-PLC/OpenPLC_examples (MIT License)
**Hardware:** CONTROLLINO MAXI Automation PLC
**Created:** 2024-11-18 | **Modified:** 2024-12-05

## Description

Real-world building automation program for staircase lighting.
Two push buttons toggle the light. A PIR motion sensor activates
the light automatically for 20 seconds via a TOF timer.

Graphical PLCopen (tc6_0201): the logic is a connection graph, and it
exercises every part of the graphical resolver — rising-edge contacts, a
function block on a rail-to-coil path, parallel branches OR-ing into one
coil, and a feedback variable read and written in the same network. The
declared 20 ms task period turns the timer's `T#20s` preset into 1000 ticks.

## Safety Properties

| ID | Kind | Description |
|---|---|---|
| P1 | invariant | Timer output energises the light |
| P2 | invariant | Latched button toggle energises the light |
| P3 | invariant | Light is only ever on when the timer or the toggle drives it |

## Expected Result

VERIFICATION SUCCESSFUL — k-induction, k=2. Wired as
`regression/ld/stairs_light_safe`.

## Validation notes

The original property set required the light to be off whenever the PIR was
clear and no button was latched. That contradicts the program: the TOF holds
the light on for its full 20 s preset after the PIR clears, which is the
purpose of a stairwell light, and the source comment in the export says so.
The properties passed only because the resolver was dropping the timer from
the rung path. `regression/ld/stairs_light_hold_fail` keeps the old property
as a negative test so a resolver that drops the timer again is caught.

Properties may name function-block pins using the `<instance>__<pin>`
symbols the resolver synthesises for graphical blocks — `TOF0__Q` here — and
`<var>__prev` for the entry-value snapshot of a feedback variable.
