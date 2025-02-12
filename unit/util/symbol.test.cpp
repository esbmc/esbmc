/*******************************************************************\

Module: Unit tests of symbol_tablet

Author: Diffblue Ltd.

\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/symbol.h>

SCENARIO(
  "Constructed symbol validity checks",
  "[core][utils][symbol__validity_checks]")
{
  GIVEN("A valid symbol")
  {
    symbolt symbol;
    irep_idt symbol_name = "Test_TestBase";
    symbol.name = symbol_name;
    // symbol.base_name = "TestBase";
    symbol.module = "TestModule";
    symbol.mode = "C";

    THEN("Symbol should be well formed")
    {
      //REQUIRE(symbol.is_well_formed());
    }
  }
  /* TODO: Add check for well formed symbols
  GIVEN("An improperly initialized symbol")
  {
    symbolt symbol;

    WHEN("The symbol doesn't have a valid mode")
    {
      symbol.mode = "";

      THEN("Symbol well-formedness check should fail")
      {
        REQUIRE_FALSE(symbol.is_well_formed());
      }
    }

    WHEN("The symbol doesn't have base name as a suffix to name")
    {
      symbol.name = "TEST";
      symbol.base_name = "TestBase";
      symbol.mode = "C";

      THEN("Symbol well-formedness check should fail")
      {
        REQUIRE_FALSE(symbol.is_well_formed());
      }
    }
  }
  */
}
