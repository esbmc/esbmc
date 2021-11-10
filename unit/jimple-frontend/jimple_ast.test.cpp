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
    REQUIRE(!f.is_interface());
    REQUIRE(f.getExtends() == "java.lang.Object");
    REQUIRE(f.getImplements() == "(No implements)");
    REQUIRE(f.getM().at(0) == jimple_modifiers::modifier::Public);
    REQUIRE(f.getBody().size() == 0);
  }
}
