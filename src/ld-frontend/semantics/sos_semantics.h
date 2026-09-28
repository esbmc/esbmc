#pragma once

// SOS semantic rule tags — used to annotate LdIR nodes with the rule that
// generated them, enabling structured proof obligations in T1.2.
// The rules are specified in docs/safe-ld-sos-semantics.md; the summaries
// below are the rule conclusions, not the full premises.
enum class SosRule
{
  // Contact rules
  NO_Contact_True,  // [NO-TRUE]  IN=T, var=T  => pf_out = T
  NO_Contact_False, // [NO-FALSE] IN=T, var=F  => pf_out = F
  NC_Contact_True,  // [NC-TRUE]  IN=T, var=F  => pf_out = T
  NC_Contact_False, // [NC-FALSE] IN=T, var=T  => pf_out = F

  // Transition-sensing contacts. The operand is compared against its value at
  // the previous scan boundary, which the scan epilogue latches.
  Rising_Contact,  // [P-EDGE]  pf_out = IN and var and not prev(var)
  Falling_Contact, // [N-EDGE]  pf_out = IN and not var and prev(var)

  // Coil rules
  Output_Coil, // [COIL]   var := pf
  Set_Coil,    // [SET]    if pf then var := true
  Reset_Coil,  // [RESET]  if pf then var := false

  // Timer rules (fixed-tick model). Every timer starts with Q false: at
  // power-up it has not run, so ET must not read as an expired interval.
  TON_Step, // [TON] if IN then ET++ else ET:=0; Q := IN and ET >= PT
  TOF_Step, // [TOF] if IN then {ET:=0; Q:=T} elif Q then {ET++; Q := ET < PT}
  TP_Step,  // [TP]  if Q then {ET++; Q := ET < PT}
            //       elif rising IN then {ET:=0; Q:=T}

  // Counter rules
  CTU_Step, // [CTU]    rising CU => CV++; Q := (CV >= PV); R => CV:=0
  CTD_Step, // [CTD]    rising CD => CV--; Q := (CV <= 0); LD => CV:=PV

  // Arithmetic rules
  Arith_Step, // [ARITH]  OUT := IN1 op IN2

  // Network rules
  Feedback_Snapshot, // [FEEDBACK] prev(var) := var, before any rung runs
};
