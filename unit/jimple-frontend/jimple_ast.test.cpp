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
    "name": "MainKt",
    "extends": "java.lang.Object",
    "content": []
})json");
    nlohmann::json j;
    file >> j;

    jimple_file f;
    j.get_to(f);

    REQUIRE(f.class_name == "MainKt");
    REQUIRE_FALSE(f.class_name == "FalseClassName");
    REQUIRE(!f.is_interface());
    REQUIRE_FALSE(f.is_interface());
    REQUIRE(f.extends == "java.lang.Object");

    REQUIRE(f.implements == "(No implements)");
    REQUIRE(f.modifiers.is_public());
    REQUIRE(f.body.size() == 0);
  }
}

SCENARIO("AST initialization from JSON (methods)", "[jimple-frontend]")
{
  GIVEN("A Class method")
  {
    std::istringstream file(R"json({
    "object": "Method",
    "modifiers": [
        "static", "public"
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

    jimple_method f;
    j.get_to(f);

    REQUIRE(f.name == "method_0");
    REQUIRE_FALSE(f.name == "WrongMethodName");
    REQUIRE(f.throws == "(No throw)");
    REQUIRE_FALSE(f.throws == "exception");
    REQUIRE(f.type.is_array() == false);
    REQUIRE_FALSE(f.type.is_array());
    REQUIRE(f.parameters.size() == 0);
    REQUIRE(f.modifiers.is_public());
    REQUIRE_FALSE(f.modifiers.is_private());
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

    REQUIRE(f.name == "a");
    REQUIRE_FALSE(f.name == "b");
    REQUIRE(f.type.name == "int");
    REQUIRE_FALSE(f.type.name == "void");
    REQUIRE(f.type.dimensions == 0);
    REQUIRE_FALSE(f.type.dimensions == 1);
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

    REQUIRE(f.label == "label1");
    REQUIRE_FALSE(f.label == "label");
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

    REQUIRE(f.label == "label2");
    REQUIRE_FALSE(f.label == "label");
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

    REQUIRE(f.value == "42");
    REQUIRE_FALSE(f.value == "15");
    REQUIRE(f.variable == "x");
    REQUIRE_FALSE(f.variable == "y");
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

    REQUIRE(f.base_class == "URI1495");
    REQUIRE_FALSE(f.base_class == "BaseClass");
    REQUIRE(f.method == "foo_0");
    REQUIRE_FALSE(f.method == "bar");
    REQUIRE(f.parameters.size() == 0);
    REQUIRE_FALSE(f.parameters.size() == 1);
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

    REQUIRE(f.label == "label2");
    REQUIRE_FALSE(f.label == "label");
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

    REQUIRE(f.value == "1");
    REQUIRE_FALSE(f.value == "15");
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

    REQUIRE(f.binop == "-");
    REQUIRE_FALSE(f.binop == "+");
  }
}
