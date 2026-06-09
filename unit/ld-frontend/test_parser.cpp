#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <ld-frontend/parser/plcopen_xml_parser.h>
#include <ld-frontend/semantics/type_checker.h>

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

TEST_CASE("coil kind read from 'kind' attribute", "[parser]")
{
  PlcopenXmlParser parser;
  LdAst ast = parser.parse(fixture("coil_kind_attr.ld"));

  REQUIRE(ast.networks.size() == 1);
  auto &rungs = ast.networks[0].rungs;
  REQUIRE(rungs.size() == 3);

  auto coil_in_rung = [&](size_t idx) -> const CoilNode & {
    for (auto &e : rungs[idx].elements)
      if (e.kind == RungElementKind::Coil)
        return e.coil;
    FAIL("no coil in rung");
    throw std::logic_error("unreachable");
  };

  REQUIRE(coil_in_rung(0).kind == CoilKind::Set);
  REQUIRE(coil_in_rung(1).kind == CoilKind::Reset);
  REQUIRE(coil_in_rung(2).kind == CoilKind::Output);
}

TEST_CASE("derived type name resolved from attribute", "[parser]")
{
  PlcopenXmlParser parser;
  LdAst ast = parser.parse(fixture("derived_type.ld"));

  REQUIRE(ast.variables.size() == 2);
  REQUIRE(ast.variables[0].name == "Count");
  REQUIRE(ast.variables[0].kind == VarKind::INT);
  REQUIRE(ast.variables[1].name == "Flag");
  REQUIRE(ast.variables[1].kind == VarKind::BOOL); // unknown derived → BOOL default
}

TEST_CASE("XML comment children inside a rung are skipped", "[parser]")
{
  PlcopenXmlParser parser;
  LdAst ast = parser.parse(fixture("rung_with_comment.ld"));

  REQUIRE(ast.networks.size() == 1);
  REQUIRE(ast.networks[0].rungs.size() == 1);
  REQUIRE(ast.networks[0].rungs[0].elements.size() == 2);
}

TEST_CASE("unknown rung element throws UnsupportedConstructError", "[parser]")
{
  PlcopenXmlParser parser;
  REQUIRE_THROWS_AS(
    parser.parse(fixture("unknown_element.ld")), UnsupportedConstructError);
}

// -----------------------------------------------------------------------
// TypeChecker tests
// -----------------------------------------------------------------------

static LdAst make_contact_ast(const std::string &var, bool empty_var = false)
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

static LdAst make_coil_ast(const std::string &var, bool empty_var = false)
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

TEST_CASE("empty contact variable rejected by type checker", "[type_checker]")
{
  TypeChecker tc;
  REQUIRE_THROWS_AS(tc.check(make_contact_ast("", true)), TypeCheckError);
}

TEST_CASE("empty coil variable rejected by type checker", "[type_checker]")
{
  TypeChecker tc;
  REQUIRE_THROWS_AS(tc.check(make_coil_ast("", true)), TypeCheckError);
}

static LdAst make_timer_ast(bool omit_et)
{
  LdAst ast;
  ast.source_file = "<test>";
  auto add = [&](const std::string &n, VarKind k) {
    VarDecl v;
    v.name = n;
    v.kind = k;
    ast.variables.push_back(v);
  };
  add("In1", VarKind::BOOL);
  add("Q1", VarKind::BOOL);
  add("PT1", VarKind::INT);
  if (!omit_et)
    add("ET1", VarKind::INT);
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

TEST_CASE("timer with missing ET port rejected", "[type_checker]")
{
  TypeChecker tc;
  REQUIRE_THROWS_AS(tc.check(make_timer_ast(true)), TypeCheckError);
}

TEST_CASE("fully connected timer passes type check", "[type_checker]")
{
  TypeChecker tc;
  REQUIRE_NOTHROW(tc.check(make_timer_ast(false)));
}

static LdAst make_counter_ast(bool omit_cv)
{
  LdAst ast;
  ast.source_file = "<test>";
  auto add = [&](const std::string &n, VarKind k) {
    VarDecl v;
    v.name = n;
    v.kind = k;
    ast.variables.push_back(v);
  };
  add("CU1", VarKind::BOOL);
  add("Q1", VarKind::BOOL);
  if (!omit_cv)
    add("CV1", VarKind::DINT);
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

TEST_CASE("counter with missing CV port rejected", "[type_checker]")
{
  TypeChecker tc;
  REQUIRE_THROWS_AS(tc.check(make_counter_ast(true)), TypeCheckError);
}

static LdAst make_arith_ast(bool omit_in2, FBKind kind = FBKind::ADD)
{
  LdAst ast;
  ast.source_file = "<test>";
  auto add = [&](const std::string &n, VarKind k) {
    VarDecl v;
    v.name = n;
    v.kind = k;
    ast.variables.push_back(v);
  };
  add("X", VarKind::INT);
  add("Y", VarKind::INT);
  add("Z", VarKind::INT);
  ArithFBNode a;
  a.kind = kind;
  a.instance_name = "FB1";
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

TEST_CASE("arith FB missing IN2 rejected for non-MOVE", "[type_checker]")
{
  TypeChecker tc;
  REQUIRE_THROWS_AS(tc.check(make_arith_ast(true)), TypeCheckError);
}

TEST_CASE("MOVE FB with empty IN2 passes type check", "[type_checker]")
{
  TypeChecker tc;
  REQUIRE_NOTHROW(tc.check(make_arith_ast(true, FBKind::MOVE)));
}
