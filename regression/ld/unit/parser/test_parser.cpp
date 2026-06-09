// Unit tests for PlcopenXmlParser and TypeChecker.
// Tests are standalone (no external framework); return 0 on pass.

#include <ld-frontend/parser/plcopen_xml_parser.h>
#include <ld-frontend/semantics/type_checker.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>

// Path to fixture files relative to the build directory (CMake sets this).
#ifndef FIXTURE_DIR
#  define FIXTURE_DIR "."
#endif

static std::string fixture(const char *name)
{
  return std::string(FIXTURE_DIR) + "/" + name;
}

// -----------------------------------------------------------------------
// Parser tests
// -----------------------------------------------------------------------

// Fix 1: coil kind encoded as a `kind` attribute must be parsed correctly.
static void test_coil_kind_from_attribute()
{
  PlcopenXmlParser parser;
  LdAst ast = parser.parse(fixture("coil_kind_attr.ld"));

  // Three rungs: set, reset, output
  assert(ast.networks.size() == 1);
  auto &rungs = ast.networks[0].rungs;
  assert(rungs.size() == 3);

  auto coil_in_rung = [&](size_t rung_idx) -> const CoilNode & {
    auto &elems = rungs[rung_idx].elements;
    for (auto &e : elems)
      if (e.kind == RungElementKind::Coil)
        return e.coil;
    throw std::logic_error("no coil in rung");
  };

  assert(coil_in_rung(0).kind == CoilKind::Set);
  assert(coil_in_rung(1).kind == CoilKind::Reset);
  assert(coil_in_rung(2).kind == CoilKind::Output);
  std::cout << "PASS: test_coil_kind_from_attribute\n";
}

// Fix 2: XML comment nodes inside a rung must be silently skipped.
static void test_non_element_children_skipped()
{
  PlcopenXmlParser parser;
  LdAst ast = parser.parse(fixture("rung_with_comment.ld"));

  assert(ast.networks.size() == 1);
  auto &rungs = ast.networks[0].rungs;
  assert(rungs.size() == 1);
  // Only the real elements (contact + coil) should appear; comments are dropped.
  assert(rungs[0].elements.size() == 2);
  std::cout << "PASS: test_non_element_children_skipped\n";
}

// Fix 3: unknown rung elements must throw UnsupportedConstructError.
static void test_unknown_element_throws()
{
  PlcopenXmlParser parser;
  bool threw = false;
  try
  {
    parser.parse(fixture("unknown_element.ld"));
  }
  catch (const UnsupportedConstructError &)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "PASS: test_unknown_element_throws\n";
}

// -----------------------------------------------------------------------
// TypeChecker tests (build LdAst programmatically)
// -----------------------------------------------------------------------

static LdAst make_single_contact_ast(const std::string &var, bool empty_var = false)
{
  LdAst ast;
  ast.source_file = "<test>";
  if (!empty_var)
  {
    VarDecl v;
    v.name = var;
    v.kind = VarKind::BOOL;
    ast.variables.push_back(v);
  }

  ContactNode c;
  c.variable = empty_var ? "" : var;
  c.kind = ContactKind::NormallyOpen;

  RungElement elem;
  elem.kind = RungElementKind::Contact;
  elem.contact = c;

  RungNode rung;
  rung.id = "1";
  rung.elements.push_back(elem);

  NetworkNode net;
  net.name = "main";
  net.rungs.push_back(rung);

  ast.networks.push_back(net);
  return ast;
}

static LdAst make_single_coil_ast(const std::string &var, bool empty_var = false)
{
  LdAst ast;
  ast.source_file = "<test>";
  if (!empty_var)
  {
    VarDecl v;
    v.name = var;
    v.kind = VarKind::BOOL;
    ast.variables.push_back(v);
  }

  CoilNode c;
  c.variable = empty_var ? "" : var;
  c.kind = CoilKind::Output;

  RungElement elem;
  elem.kind = RungElementKind::Coil;
  elem.coil = c;

  RungNode rung;
  rung.id = "1";
  rung.elements.push_back(elem);

  NetworkNode net;
  net.name = "main";
  net.rungs.push_back(rung);

  ast.networks.push_back(net);
  return ast;
}

// Fix 4: empty contact variable must be a TypeCheckError.
static void test_empty_contact_variable_rejected()
{
  LdAst ast = make_single_contact_ast("", true);
  TypeChecker tc;
  bool threw = false;
  try
  {
    tc.check(ast);
  }
  catch (const TypeCheckError &)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "PASS: test_empty_contact_variable_rejected\n";
}

// Fix 4: empty coil variable must be a TypeCheckError.
static void test_empty_coil_variable_rejected()
{
  LdAst ast = make_single_coil_ast("", true);
  TypeChecker tc;
  bool threw = false;
  try
  {
    tc.check(ast);
  }
  catch (const TypeCheckError &)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "PASS: test_empty_coil_variable_rejected\n";
}

static LdAst make_timer_ast(bool omit_et = false)
{
  LdAst ast;
  ast.source_file = "<test>";

  auto add_var = [&](const std::string &name, VarKind k) {
    VarDecl v;
    v.name = name;
    v.kind = k;
    ast.variables.push_back(v);
  };
  add_var("In1", VarKind::BOOL);
  add_var("Q1", VarKind::BOOL);
  add_var("PT1", VarKind::INT);
  if (!omit_et)
    add_var("ET1", VarKind::INT);

  TimerFBNode t;
  t.kind = FBKind::TON;
  t.instance_name = "TON1";
  t.IN_var = "In1";
  t.PT_var = "PT1";
  t.Q_var = "Q1";
  t.ET_var = omit_et ? "" : "ET1";

  RungElement elem;
  elem.kind = RungElementKind::TimerFB;
  elem.timer_fb = t;

  RungNode rung;
  rung.id = "1";
  rung.elements.push_back(elem);

  NetworkNode net;
  net.name = "main";
  net.rungs.push_back(rung);
  ast.networks.push_back(net);
  return ast;
}

// Fix 5: timer with missing ET port must be rejected.
static void test_timer_missing_et_rejected()
{
  LdAst ast = make_timer_ast(/*omit_et=*/true);
  TypeChecker tc;
  bool threw = false;
  try
  {
    tc.check(ast);
  }
  catch (const TypeCheckError &)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "PASS: test_timer_missing_et_rejected\n";
}

// Fix 5: fully connected timer must pass type check.
static void test_timer_fully_connected_passes()
{
  LdAst ast = make_timer_ast(/*omit_et=*/false);
  TypeChecker tc;
  bool threw = false;
  try
  {
    tc.check(ast);
  }
  catch (const TypeCheckError &)
  {
    threw = true;
  }
  assert(!threw);
  std::cout << "PASS: test_timer_fully_connected_passes\n";
}

static LdAst make_counter_ast(bool omit_cv = false)
{
  LdAst ast;
  ast.source_file = "<test>";

  auto add_var = [&](const std::string &name, VarKind k) {
    VarDecl v;
    v.name = name;
    v.kind = k;
    ast.variables.push_back(v);
  };
  add_var("CU1", VarKind::BOOL);
  add_var("Q1", VarKind::BOOL);
  if (!omit_cv)
    add_var("CV1", VarKind::DINT);

  CounterFBNode c;
  c.kind = FBKind::CTU;
  c.instance_name = "CTU1";
  c.CU_var = "CU1";
  c.Q_var = "Q1";
  c.CV_var = omit_cv ? "" : "CV1";

  RungElement elem;
  elem.kind = RungElementKind::CounterFB;
  elem.counter_fb = c;

  RungNode rung;
  rung.id = "1";
  rung.elements.push_back(elem);

  NetworkNode net;
  net.name = "main";
  net.rungs.push_back(rung);
  ast.networks.push_back(net);
  return ast;
}

// Fix 6: counter with missing CV port must be rejected.
static void test_counter_missing_cv_rejected()
{
  LdAst ast = make_counter_ast(/*omit_cv=*/true);
  TypeChecker tc;
  bool threw = false;
  try
  {
    tc.check(ast);
  }
  catch (const TypeCheckError &)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "PASS: test_counter_missing_cv_rejected\n";
}

static LdAst make_arith_ast(bool omit_in2 = false)
{
  LdAst ast;
  ast.source_file = "<test>";

  auto add_var = [&](const std::string &name, VarKind k) {
    VarDecl v;
    v.name = name;
    v.kind = k;
    ast.variables.push_back(v);
  };
  add_var("X", VarKind::INT);
  add_var("Y", VarKind::INT);
  add_var("Z", VarKind::INT);

  ArithFBNode a;
  a.kind = FBKind::ADD;
  a.instance_name = "ADD1";
  a.IN1_var = "X";
  a.IN2_var = omit_in2 ? "" : "Y";
  a.OUT_var = "Z";

  RungElement elem;
  elem.kind = RungElementKind::ArithFB;
  elem.arith_fb = a;

  RungNode rung;
  rung.id = "1";
  rung.elements.push_back(elem);

  NetworkNode net;
  net.name = "main";
  net.rungs.push_back(rung);
  ast.networks.push_back(net);
  return ast;
}

// Fix 7: arith FB with missing IN2 (for non-MOVE) must be rejected.
static void test_arith_missing_in2_rejected()
{
  LdAst ast = make_arith_ast(/*omit_in2=*/true);
  TypeChecker tc;
  bool threw = false;
  try
  {
    tc.check(ast);
  }
  catch (const TypeCheckError &)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "PASS: test_arith_missing_in2_rejected\n";
}

// Fix 7: MOVE FB with empty IN2 must pass (MOVE doesn't use IN2).
static void test_move_fb_no_in2_passes()
{
  LdAst ast;
  ast.source_file = "<test>";

  auto add_var = [&](const std::string &name, VarKind k) {
    VarDecl v;
    v.name = name;
    v.kind = k;
    ast.variables.push_back(v);
  };
  add_var("Src", VarKind::INT);
  add_var("Dst", VarKind::INT);

  ArithFBNode a;
  a.kind = FBKind::MOVE;
  a.instance_name = "MOVE1";
  a.IN1_var = "Src";
  a.IN2_var = ""; // empty — valid for MOVE
  a.OUT_var = "Dst";

  RungElement elem;
  elem.kind = RungElementKind::ArithFB;
  elem.arith_fb = a;

  RungNode rung;
  rung.id = "1";
  rung.elements.push_back(elem);

  NetworkNode net;
  net.name = "main";
  net.rungs.push_back(rung);
  ast.networks.push_back(net);

  TypeChecker tc;
  bool threw = false;
  try
  {
    tc.check(ast);
  }
  catch (const TypeCheckError &)
  {
    threw = true;
  }
  assert(!threw);
  std::cout << "PASS: test_move_fb_no_in2_passes\n";
}

// derived name="INT" must map to VarKind::INT; unknown derived name falls back to BOOL.
static void test_derived_type_parsing()
{
  PlcopenXmlParser parser;
  LdAst ast = parser.parse(fixture("derived_type.ld"));

  assert(ast.variables.size() == 2);
  // First variable: <derived name="INT"/> → INT
  assert(ast.variables[0].name == "Count");
  assert(ast.variables[0].kind == VarKind::INT);
  // Second variable: <derived name="UserFBType"/> → BOOL (unknown, default)
  assert(ast.variables[1].name == "Flag");
  assert(ast.variables[1].kind == VarKind::BOOL);
  std::cout << "PASS: test_derived_type_parsing\n";
}

int main()
{
  test_coil_kind_from_attribute();
  test_derived_type_parsing();
  test_non_element_children_skipped();
  test_unknown_element_throws();
  test_empty_contact_variable_rejected();
  test_empty_coil_variable_rejected();
  test_timer_missing_et_rejected();
  test_timer_fully_connected_passes();
  test_counter_missing_cv_rejected();
  test_arith_missing_in2_rejected();
  test_move_fb_no_in2_passes();
  std::cout << "All parser/type-checker tests passed.\n";
  return 0;
}
