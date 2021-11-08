/*******************************************************************\
Module: Unit tests for binary2integer
Author: Franz Brauße

\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/mp_arith.h>

TEST_CASE("signed binary2integer leading one matters", "[core][util][bin2int]")
{
  auto b2i = binary2integer;
  auto i2b = integer2binary;
  REQUIRE(b2i(/**/ "01111111111111111111111111101010", true) == 2147483626);
  REQUIRE(b2i(/***/ "1111111111111111111111111101010", true) == -22);
  REQUIRE(b2i(/**/ "10000000000000000000000000000100", true) == -2147483644);
  REQUIRE(i2b(-2147483644, 32) == "10000000000000000000000000000100");
}

TEST_CASE("signed binary2integer leading zero is zero", "[core][util][bin2int]")
{
  REQUIRE(binary2integer("00", true) == 0);
}
