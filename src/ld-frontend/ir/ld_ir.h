#pragma once

#include <ld-frontend/parser/ld_ast.h>
#include <ld-frontend/semantics/sos_semantics.h>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// LdIR — cyclic control-flow graph over SOS state-transition blocks.
//
// Structure:
//   INIT_BLOCK
//   └── SCAN_LOOP (while true)
//       ├── READ_INPUTS
//       ├── RUNG_1 ... RUNG_n
//       └── WRITE_OUTPUTS
// -----------------------------------------------------------------------

enum class LdIRNodeKind
{
  ContactEval, // evaluate a contact → produces power-flow value
  CoilAssign,  // assign coil output
  TimerStep,   // TON/TOF/TP fixed-tick step
  CounterStep, // CTU/CTD per-scan step
  ArithStep,   // arithmetic FB step
};

// One IR instruction corresponding to a single SOS rule application.
struct LdIRNode
{
  LdIRNodeKind kind;
  SosRule rule; // which SOS rule generated this node

  // Fields used by ContactEval / CoilAssign
  std::string variable;
  ContactKind contact_kind = ContactKind::NormallyOpen;
  CoilKind coil_kind = CoilKind::Output;

  // Fields used by TimerStep
  std::string timer_IN;
  std::string timer_ET;
  std::string timer_PT;
  std::string timer_Q;
  FBKind timer_kind = FBKind::TON;

  // Fields used by CounterStep
  std::string ctr_instance; // FB instance name (for shadow edge variable)
  std::string ctr_CU;
  std::string ctr_CD;
  std::string ctr_R;
  std::string ctr_CV;
  std::string ctr_PV;
  std::string ctr_Q;
  FBKind ctr_kind = FBKind::CTU;

  // Fields used by ArithStep
  std::string arith_IN1;
  std::string arith_IN2;
  std::string arith_OUT;
  FBKind arith_kind = FBKind::ADD;

  LdLocation loc;
};

// A rung block is a sequence of IR nodes evaluated in order within one scan.
struct LdIRRung
{
  std::string id;
  std::vector<LdIRNode> nodes;
  LdLocation loc;
};

// A user-defined FB instance to execute once per scan cycle, with its ST body.
struct UserFBExec
{
  std::string type_name;
  std::string instance_name;
  std::vector<std::string> input_vars; // formal inputs (IN1, IN2, ...)
  std::vector<std::string> local_vars; // FB locals (e.g. "i")
  std::string output_var;
  bool output_is_bool = true;
  std::string in1_var; // program var feeding IN1 ("" => nondet)
  std::vector<FBOutWire> out_wires; // FB pin -> program variable assignments
  std::string st_body; // raw Structured Text
};

// Top-level IR: variable declarations + the ordered list of rungs.
struct LdIR
{
  std::string source_file;
  std::vector<VarDecl> variables;     // copied from LdAst
  std::vector<LdIRRung> rungs;        // ordered scan body
  std::vector<UserFBExec> user_fbs;   // user-defined FB bodies to execute
};
