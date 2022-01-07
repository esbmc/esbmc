/*******************************************************************
 Module: Jimple AST generatation unit test

 Author: Rafael Sá Menezes

 Date: October 2021

 Test Plan:
   - Initialize ast from json string
 \*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <catch2/catch.hpp>
#include <jimple-frontend/AST/jimple_ast.h>
#include <jimple-frontend/AST/jimple_file.h>
#include <jimple-frontend/AST/jimple_type.h>
#include <jimple-frontend/AST/jimple_expr.h>
#include <jimple-frontend/AST/jimple_statement.h>
#include <jimple-frontend/AST/jimple_declaration.h>
#include <nlohmann/json.hpp>

// ** Try to initialize an structure with a JSON string
SCENARIO("AST initialization from JSON (basic constructs)", "[jimple-frontend]")
{
  GIVEN("A Main class")
  {
    std::istringstream file(R"json({
    "object": "Class",
    "modifiers": [
        "public"
    ],
    "name": "Main",
    "extends": "java.lang.Object",
    "content": []
})json");
    nlohmann::json j;
    file >> j;

    jimple_file f;
    j.get_to(f);

    REQUIRE(f.getClassName() == "Main");
    REQUIRE_FALSE(f.getClassName() == "FalseClassName");
    REQUIRE(!f.is_interface());
    REQUIRE_FALSE(f.is_interface());
    REQUIRE(f.getExtends() == "java.lang.Object");
    REQUIRE(f.getImplements() == "(No implements)");
    REQUIRE(f.getM().at(0) == jimple_modifiers::modifier::Public);
    REQUIRE(f.getBody().size() == 0);
  }
}

SCENARIO("AST initialization from JSON (methods)", "[jimple-frontend]")
{
  GIVEN("A Class method")
  {
    std::istringstream file(R"json({
    "object": "Method",
    "modifiers": [
        "public"
    ],
    "type": {"identifier": "int",
             "dimensions": 0,
             "mode": "basic"
            },
    "body":[],
    "name": "method",
    "throws": "(No throw)",
    "parameters": [],
    "content": []
})json");
    nlohmann::json j;
    file >> j;

    jimple_class_method f;
    j.get_to(f);

    REQUIRE(f.getName() == "method");
    REQUIRE_FALSE(f.getName() == "WrongMethodName");
    REQUIRE(f.getThrows() == "(No throw)");
    REQUIRE_FALSE(f.getThrows() == "exception");
    REQUIRE(f.getT().is_array() == false);
    REQUIRE_FALSE(f.getT().is_array());
    REQUIRE(f.getParameters().size() == 0);
    REQUIRE(f.getM().at(0) == jimple_modifiers::modifier::Public);
    REQUIRE_FALSE(f.getM().at(0) == jimple_modifiers::modifier::Private);
  }
}

SCENARIO("AST initialization from JSON (declarations)", "[jimple-frontend]")
{
  GIVEN("A variable declaration")
  {
    std::istringstream file(R"json({
    "object": "Variable",
    "name": "a",
    "type": {"identifier": "int",
             "dimensions": 0,
             "mode": "basic"
            }
})json");
    nlohmann::json j;
    file >> j;

    jimple_declaration f;
    j.get_to(f);

    REQUIRE(f.getName() == "a");
    REQUIRE_FALSE(f.getName() == "b");
    REQUIRE(f.getT().getTName() == "int");
    REQUIRE_FALSE(f.getT().getTName() == "void");
    REQUIRE(f.getT().getTDim() == 0);
    REQUIRE_FALSE(f.getT().getTDim() == 1);
  }
}

SCENARIO("AST initialization from JSON (statements)", "[jimple-frontend]")
{
  GIVEN("A goto statement")
  {
    std::istringstream file(R"json({
    "object": "goto",
    "goto": "label1"
})json");
    nlohmann::json j;
    file >> j;

    jimple_goto f;
    j.get_to(f);

    REQUIRE(f.getLabel() == "label1");
    REQUIRE_FALSE(f.getLabel() == "label");
  }

  GIVEN("A label statement")
  {
    std::istringstream file(R"json({
    "object": "label",
    "label_id": "label2",
    "content": []
})json");
    nlohmann::json j;
    file >> j;

    jimple_label f;
    j.get_to(f);

    REQUIRE(f.getLabel() == "label2");
    REQUIRE_FALSE(f.getLabel() == "label");
  }

  GIVEN("An assertion statement")
  {
    std::istringstream file(R"json({
    "object": "Assert",
    "equals": {"value" : "42",
               "symbol": "x"
              }
})json");
    nlohmann::json j;
    file >> j;

    jimple_assertion f;
    j.get_to(f);

    REQUIRE(f.getValue() == "42");
    REQUIRE_FALSE(f.getValue() == "15");
    REQUIRE(f.getVariable() == "x");
    REQUIRE_FALSE(f.getVariable() == "y");
  }

  GIVEN("An invoke statement")
  {
    std::istringstream file(R"json({
    "object": "StaticInvoke",
    "base_class": "URI1495",
    "method": "foo",
    "parameters": []
})json");
    nlohmann::json j;
    file >> j;

    jimple_invoke f;
    j.get_to(f);

    REQUIRE(f.getBaseClass() == "URI1495");
    REQUIRE_FALSE(f.getBaseClass() == "BaseClass");
    REQUIRE(f.getMethod() == "foo");
    REQUIRE_FALSE(f.getMethod() == "bar");
    REQUIRE(f.getParameters().size() == 0);
    REQUIRE_FALSE(f.getParameters().size() == 1);
  }

  GIVEN("An if statement")
  {
    std::istringstream file(R"json({
    "object": "if",
    "goto": "label2",
    "expression": {
            "expr_type": "binop",
            "operator": "<=",
            "lhs": {
                    "expr_type": "symbol",
                    "value": "y"
                    },
            "rhs": {
                    "expr_type": "constant",
                    "value": "10"
                    }
            }
})json");
    nlohmann::json j;
    file >> j;

    jimple_if f;
    j.get_to(f);

    REQUIRE(f.getLabel() == "label2");
    REQUIRE_FALSE(f.getLabel() == "label");
  }

  GIVEN("An assignment statement")
  {
    std::istringstream file(R"json({
    "object": "SetVariable",
    "name": "x",
    "value": {"expr_type": "constant",
              "value": "10"}
})json");
    nlohmann::json j;
    file >> j;

    jimple_assignment f;
    j.get_to(f);

    REQUIRE(f.getVariable() == "x");
    REQUIRE_FALSE(f.getVariable() == "y");
  }

  GIVEN("An identity statement")
  {
    std::istringstream file(R"json({
    "object": "identity",
    "identifier": "this",
    "name": "r0",
    "type": {"identifier": "int",
             "dimensions": 0,
             "mode": "basic"}
})json");
    nlohmann::json j;
    file >> j;

    jimple_identity f;
    j.get_to(f);

    REQUIRE(f.getLocalName() == "r0");
    REQUIRE_FALSE(f.getLocalName() == "r1");
    REQUIRE(f.getAtIdentifier() == "this");
    REQUIRE_FALSE(f.getAtIdentifier() == "");
    REQUIRE(f.getT().getTName() == "int");
    REQUIRE_FALSE(f.getT().getTName() == "void");
    REQUIRE(f.getT().getTDim() == 0);
    REQUIRE_FALSE(f.getT().getTDim() == 1);
  }
}

SCENARIO("AST initialization from JSON (expressions)", "[jimple-frontend]")
{
  GIVEN("An expression constant")
  {
    std::istringstream file(R"json({
    "object": "expression constant",
    "value": "1"
})json");
    nlohmann::json j;
    file >> j;

    jimple_constant f;
    j.get_to(f);

    REQUIRE(f.getValue() == "1");
    REQUIRE_FALSE(f.getValue() == "15");
  }

  GIVEN("An expression symbol")
  {
    std::istringstream file(R"json({
    "object": "expression symbol",
    "value": "x"
})json");
    nlohmann::json j;
    file >> j;

    jimple_symbol f;
    j.get_to(f);

    REQUIRE(f.getVarName() == "x");
    REQUIRE_FALSE(f.getVarName() == "y");
  }

  GIVEN("An expression binary operator")
  {
    std::istringstream file(R"json({
    "object": "expression binop",
    "operator": "-",
    "lhs": {
            "expr_type": "symbol",
            "value": "x"
            },
    "rhs": {
            "expr_type": "constant",
            "value": "1"
            }
})json");
    nlohmann::json j;
    file >> j;

    jimple_binop f;
    j.get_to(f);

    REQUIRE(f.getBinop() == "-");
    REQUIRE_FALSE(f.getBinop() == "+");
  }
}
