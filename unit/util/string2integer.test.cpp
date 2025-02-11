/*******************************************************************\
Module: Unit tests for string2int.h
Author: Diffblue Ltd.

Notes:
    I've replaced the original tests to use the string2integer and
    the BigInt API, which doesn't throw errors
\*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <util/mp_arith.h>

TEST_CASE(
  "converting optionally to a valid integer should succeed",
  "[core][util][string2int]")
{
  REQUIRE(string2integer("13") == 13);
  REQUIRE(string2integer("-5") == -5);
  REQUIRE(string2integer("c0fefe", 16) == 0xc0fefe);
}

TEST_CASE(
  "optionally converting invalid string to integer should return 0",
  "[core][util][string2int]")
{
  REQUIRE(string2integer("thirteen") == 0);
  REQUIRE(string2integer("c0fefe") == 0);
}

/* TODO: Is this the behavior for string2integer?
TEST_CASE(
  "optionally converting string out of range to integer should return 0",
  "[core][util][string2int]")
{
  REQUIRE(
    string2integer("0xfffffffffffffffffffffffffffffffffffffffffff", 16) ==
    0);
}
*/

TEST_CASE(
  "converting optionally to a valid unsigned should succeed",
  "[core][util][string2int]")
{
  BigInt v = string2integer("13");
  REQUIRE_FALSE(v.is_negative());
  REQUIRE(v == 13u);

  BigInt q = string2integer("c0fefe", 16);
  REQUIRE_FALSE(q.is_negative());
  REQUIRE(q == 0xc0fefeu);
}

TEST_CASE(
  "converting optionally to a valid size_t should succeed",
  "[core][util][string2int]")
{
  REQUIRE(string2integer("13") == std::size_t{13});
  REQUIRE(string2integer("c0fefe", 16) == std::size_t{0xc0fefe});
}