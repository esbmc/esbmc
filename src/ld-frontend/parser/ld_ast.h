#pragma once

#include <string>
#include <vector>

// Source location carried by every AST node so counterexample traces
// reference original PLCopen XML positions.
struct LdLocation
{
  std::string file;
  unsigned line = 0;
  unsigned col = 0;
};

// -----------------------------------------------------------------------
// Contact nodes
// -----------------------------------------------------------------------

enum class ContactKind
{
  NormallyOpen,  // --[ ]--
  NormallyClosed // --[/]--
};

struct ContactNode
{
  ContactKind kind = ContactKind::NormallyOpen;
  std::string variable; // PLCopen XML identifier
  LdLocation loc;
};

// -----------------------------------------------------------------------
// Coil nodes
// -----------------------------------------------------------------------

enum class CoilKind
{
  Output, // --( )--
  Set,    // --( S )--
  Reset   // --( R )--
};

struct CoilNode
{
  CoilKind kind = CoilKind::Output;
  std::string variable;
  LdLocation loc;
};

// -----------------------------------------------------------------------
// Function-block nodes (timers, counters, arithmetic FBs)
// -----------------------------------------------------------------------

enum class FBKind
{
  TON,
  TOF,
  TP,
  CTU,
  CTD,
  ADD,
  SUB,
  MUL,
  DIV,
  MOVE
};

struct FBInputPort
{
  std::string port_name; // e.g. "IN", "PT", "CU"
  std::string variable;  // connected variable or literal
};

struct FBOutputPort
{
  std::string port_name; // e.g. "Q", "ET", "CV"
  std::string variable;
};

struct TimerFBNode
{
  FBKind kind = FBKind::TON;
  std::string instance_name;
  std::string IN_var; // enable input variable
  std::string PT_var; // preset time variable (tick count)
  std::string Q_var;  // output variable
  std::string ET_var; // elapsed time variable
  LdLocation loc;
};

struct CounterFBNode
{
  FBKind kind = FBKind::CTU;
  std::string instance_name;
  std::string CU_var; // count-up input variable
  std::string CD_var; // count-down input variable (CTD only)
  std::string R_var;  // reset variable
  std::string PV_var; // preset value variable
  std::string Q_var;  // output variable
  std::string CV_var; // counter value variable
  LdLocation loc;
};

struct ArithFBNode
{
  FBKind kind = FBKind::ADD;
  std::string instance_name;
  std::string IN1_var;
  std::string IN2_var;
  std::string OUT_var;
  LdLocation loc;
};

// -----------------------------------------------------------------------
// Rung and network nodes
// -----------------------------------------------------------------------

// A rung element is one of the Tier 1 constructs
enum class RungElementKind
{
  Contact,
  Coil,
  TimerFB,
  CounterFB,
  ArithFB
};

struct RungElement
{
  RungElementKind kind;
  ContactNode contact;
  CoilNode coil;
  TimerFBNode timer_fb;
  CounterFBNode counter_fb;
  ArithFBNode arith_fb;
  LdLocation loc;
};

// A rung is a list of elements evaluated left-to-right within a scan cycle.
struct RungNode
{
  std::string id; // rung number / label from XML
  std::vector<RungElement> elements;
  LdLocation loc;
};

// A network groups rungs and the variable declarations for one POU.
struct NetworkNode
{
  std::string name;
  std::vector<RungNode> rungs;
  LdLocation loc;
};

// -----------------------------------------------------------------------
// Variable declarations
// -----------------------------------------------------------------------

enum class VarKind
{
  BOOL,
  INT,
  DINT,
  TIME, // used for timer PT/ET fields
  REAL, // IEEE floating point (analog process variables)
};

struct VarDecl
{
  std::string name;
  VarKind kind = VarKind::BOOL;
  bool is_input = false;
  bool is_output = false;
  LdLocation loc;
};

// -----------------------------------------------------------------------
// User-defined function blocks (e.g. EQ_0/GT_0/SUB_0 wrappers that may carry
// Ladder Logic Bombs).  Their Structured Text bodies are translated and
// executed in the scan cycle so the embedded logic reaches the GOTO IR.
// -----------------------------------------------------------------------

// A function-block formal/local variable with its declared IEC 61131-3 type.
// Carrying the type (rather than just the name) keeps REAL/analog FB variables
// from being silently demoted to int when the body is translated.
struct FBVarDecl
{
  std::string name;
  VarKind kind = VarKind::INT; // numeric default for an untyped FB variable
};

struct UserFBDef
{
  std::string type_name;              // e.g. "EQ_0"
  std::vector<FBVarDecl> input_vars;  // formal inputs (IN1, IN2, ...) + types
  std::vector<FBVarDecl> local_vars;  // FB-local variables (e.g. "i") + types
  std::string output_var;             // formal output name (OUT)
  VarKind output_kind = VarKind::BOOL; // OUT declared type
  std::string st_body;                // raw Structured Text body
};

// Wiring of an FB output pin to a program variable: "prog_var := <inst>__pin".
struct FBOutWire
{
  std::string prog_var; // program variable driven by the pin (e.g. MV1)
  std::string pin;       // FB formal output name (e.g. OUT_MV1)
};

struct UserFBInstance
{
  std::string type_name;     // references a UserFBDef
  std::string instance_name; // e.g. "EQ_00"
  std::string block_id;      // graphical localId of the block
  std::string in1_var; // program variable feeding IN1 ("" => nondet sample)
  std::vector<FBOutWire> out_wires; // pins consumed by program outVariables
  LdLocation loc;
};

// -----------------------------------------------------------------------
// Top-level AST
// -----------------------------------------------------------------------

struct LdAst
{
  std::string source_file;           // path of the PLCopen XML file
  std::vector<VarDecl> variables;    // all declared variables
  std::vector<NetworkNode> networks; // one per POU (Tier 1: single POU)
  std::vector<UserFBDef> user_fb_defs;
  std::vector<UserFBInstance> user_fb_instances;
  bool has_interrupt_tasks = false; // triggers Tier-2 rejection
};
