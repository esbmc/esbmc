# Water Reserve Control

**Source:** CONTROLLINO-PLC/OpenPLC_examples (MIT License)
**Hardware:** CONTROLLINO MAXI Automation PLC
**Created:** 2024-11-13 | **Modified:** 2024-12-05

## Description

Real-world industrial program for automatic water pump control.
A pump fills an elevated tank from a cistern, controlled by
level sensors and manual Start/Stop buttons.

## Safety Properties

| ID | Kind | Description |
|---|---|---|
| P1 | invariant | Pump must not run when pool is empty (cavitation hazard) |
| P2 | invariant | Pump must not run when tank is full (overflow hazard) |
| P3 | absence | Stop button must immediately halt the pump |

## Expected Result

VERIFICATION SUCCESSFUL — k-induction k=2 in ~0.03s
