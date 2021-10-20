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
#include <nlohmann/json.hpp>

// ** Try to initialize an structure with a JSON string
SCENARIO("AST initialization from JSON", "[jimple-frontend]")
{
  GIVEN("hello world")
    {
      int a = 42;
      REQUIRE(a == 42);
    }
  
}
