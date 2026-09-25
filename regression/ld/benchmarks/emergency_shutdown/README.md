# Emergency Shutdown System (CS3)

**Source:** synthetic, written for WP3 case study CS3
**Constructs:** contacts, set/reset coils, latch

## Description

A high-pressure alarm, a high-temperature alarm, or a manual trip each set the
ESD latch. A reset command clears the latch, gated on the alarms having
cleared. The latch drives valve closure and process shutdown.

## Safety Properties

| ID | Kind | Description |
|---|---|---|
| P8 | invariant | Valve closed while the high-pressure alarm is active (SIL-2 immediacy) |
| P9 | invariant | Valve closed while the high-temperature alarm is active |
| P10 | invariant | Valve closed while the manual ESD is triggered |
| P11 | invariant | Process shut down whenever the latch is set (persistence) |
| P12 | absence | High-pressure alarm never occurs without valve closure |

## Expected Result

VERIFICATION FAILED on P10 — k-induction, k=1. Wired as
`regression/ld/esd_manual_reset_fail`.

## Validation notes

The violation is a true positive, not a modelling artefact. The reset rung
requires the two alarms to have cleared but does not require the manual trip
to have been released, so a reset command issued while `Manual_ESD` is still
asserted clears the latch and reopens the valve with the trip still active.
For a SIL-2 emergency shutdown that is a real hazard: the manual trip is the
operator's last line of defence and must not be overridable by a reset.

`regression/ld/emergency_shutdown_safe` holds the corrected program, with
`Manual_ESD` added to the reset rung's guard; it discharges all five
properties under k-induction at k=2.
