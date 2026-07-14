# Staircase Light Control

**Source:** CONTROLLINO-PLC/OpenPLC_examples (MIT License)
**Hardware:** CONTROLLINO MAXI Automation PLC
**Created:** 2024-11-18 | **Modified:** 2024-12-05

## Description

Real-world building automation program for staircase lighting.
Two push buttons toggle the light. A PIR motion sensor activates
the light automatically for 20 seconds via a TOF timer.

## Safety Properties

| ID | Kind | Description |
|---|---|---|
| P1 | invariant | Light must not be on without PIR or button activation |
| P2 | invariant | PIR detection must activate the light |
| P3 | invariant | Simultaneous button presses must not cause undefined state |

## Expected Result

VERIFICATION SUCCESSFUL — k-induction k=2 in ~0.03s
