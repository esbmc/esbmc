#pragma once

// SOS semantic rule tags — used to annotate LdIR nodes with the rule that
// generated them, enabling structured proof obligations in T1.2.
enum class SosRule
{
  // Contact rules
  NO_Contact_True,  // [NO-TRUE]  IN=T, var=T  => pf_out = T
  NO_Contact_False, // [NO-FALSE] IN=T, var=F  => pf_out = F
  NC_Contact_True,  // [NC-TRUE]  IN=T, var=F  => pf_out = T
  NC_Contact_False, // [NC-FALSE] IN=T, var=T  => pf_out = F

  // Coil rules
  Output_Coil, // [COIL]   var := pf
  Set_Coil,    // [SET]    if pf then var := true
  Reset_Coil,  // [RESET]  if pf then var := false

  // Timer rules (fixed-tick model)
  TON_Step, // [TON]    if IN then ET++ else ET:=0; Q := (ET >= PT)
  TOF_Step, // [TOF]    if !IN then ET++ else ET:=0; Q := (ET < PT)
  TP_Step,  // [TP]     if IN and Q then ET++; Q := (ET < PT)

  // Counter rules
  CTU_Step, // [CTU]    rising CU => CV++; Q := (CV >= PV); R => CV:=0
  CTD_Step, // [CTD]    rising CD => CV--; Q := (CV <= 0); LD => CV:=PV

  // Arithmetic rules
  Arith_Step, // [ARITH]  OUT := IN1 op IN2
};
