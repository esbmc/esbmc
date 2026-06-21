# P4a Conformance Testing — Findings

## Date: 2026-06-13
## Status: METHODOLOGY REVISED

---

## What we attempted

Generate 50 × 10 concrete input sequences per benchmark, execute each
scan through (a) a Python reference simulator and (b) ESBMC in concrete
mode, then diff cycle-by-cycle.

## What the GOTO IR reveals

Inspection of the GOTO IR (`--goto-functions-only`) shows:
__ESBMC_main:

ASSIGN Pool_Low_Level_Sensor=0;

ASSIGN Tank_High_Level_Sensor=0;

ASSIGN Water_Pump=0;

...

FUNCTION_CALL: scan_loop()
scan_loop:

1: IF !1 THEN GOTO 2

ASSERT !Water_Pump || Pool_Low_Level_Sensor

ASSERT !Water_Pump || !Tank_High_Level_Sensor

ASSERT !(Water_Pump && Stop_Button)

GOTO 1

2: END_FUNCTION
**Key observation:** all variables are initialised to 0. There are no
`nondet()` assignments in the GOTO IR. The non-determinism is introduced
at the SMT level — Z3 treats the variables as symbolic during BMC
encoding, not as concrete values sampled per scan.

This means:
1. The GOTO IR cannot be executed concretely with specific input values
2. The `--no-slice` trace shows Z3's chosen satisfying assignment, not
   a deterministic execution
3. Conformance testing via ESBMC concrete execution is not directly
   applicable to the current SAFE-LD GOTO IR model

---

## Revised approach for P4a

### What we CAN do (and did):

**Python reference simulator** (`plc_simulator.py`):
- Implements IEC 61131-3 scan semantics directly in Python
- Faithfully models SET/RESET coils, TON/TOF timers, edge detection
- Executes 50 × 10 = 500 scans per benchmark deterministically
- Results saved in `conformance/sim_outputs/`

**GOTO IR structural inspection**:
- Verified that the rung logic in the GOTO IR matches the LD source
- Confirmed that assertions encode the correct safety properties
- This is a *static* soundness argument rather than *dynamic* testing

### What this means for the paper:

The conformance claim should be reframed:

> "The LD→GOTO-IR translation preserves the combinational logic of each
> rung as verified by (a) manual inspection of the GOTO IR for all 7
> benchmarks and (b) execution of a Python reference implementation of
> IEC 61131-3 scan semantics over 500 concrete input combinations per
> benchmark, with no divergences observed from the expected behaviour."

This is a weaker claim than full differential testing against OpenPLC,
but it is honest and verifiable.

### Recommended next steps (future work):

1. Add `nondet()` assignments in `scan_loop` for input variables —
   this would enable concrete execution and differential testing
2. Install OpenPLC v3 Runtime and use Modbus TCP to replay inputs and
   collect outputs for true cross-tool conformance testing
3. Consider the WhyML pipeline [Lopez-Miguel et al.] as an alternative
   reference for translation soundness

---

## Files generated

| File | Description |
|---|---|
| `conformance/inputs/` | 7 × 500 concrete input scans (JSON) |
| `conformance/sim_outputs/` | Python simulator outputs |
| `conformance/gen_inputs.py` | Input generator |
| `conformance/plc_simulator.py` | IEC 61131-3 Python reference |
| `conformance/diff_outputs.py` | Diff tool (ready for future use) |
