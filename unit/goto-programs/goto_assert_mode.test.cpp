/*******************************************************************
 Module: Goto Programs asserts unit test

 Author: Rafael Sá Menezes

 Date: June 2022

 Test Plan:
   - Checks if is_mode_enabled is working properly.
 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <goto-programs/goto_assert_mode.h>

SCENARIO("is_mode_enabled can confirm values properly", "[goto-assert]")
{
  GIVEN("An goto_assertions value START")
  {
    goto_assertions::goto_assertion_mode START;

    WHEN("START contains all modes")
    {
      START = goto_assertions::ALL_MODES;
      REQUIRE(goto_assertions::is_mode_enabled(START, goto_assertions::USER));
      REQUIRE(goto_assertions::is_mode_enabled(
        START, goto_assertions::POINTER_SAFETY));
      REQUIRE(
        goto_assertions::is_mode_enabled(START, goto_assertions::ARRAY_SAFETY));
      REQUIRE(goto_assertions::is_mode_enabled(
        START, goto_assertions::ARITHMETIC_SAFETY));
      REQUIRE(goto_assertions::is_mode_enabled(START, goto_assertions::OTHER));
    }

    WHEN("START contains one mode")
    {
      START = goto_assertions::POINTER_SAFETY;
      REQUIRE(goto_assertions::is_mode_enabled(
        START, goto_assertions::POINTER_SAFETY));
      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::USER));

      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::ARRAY_SAFETY));
      REQUIRE_FALSE(goto_assertions::is_mode_enabled(
        START, goto_assertions::ARITHMETIC_SAFETY));
      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::OTHER));
    }

    WHEN("START contains multiple modes")
    {
      START = (goto_assertions::goto_assertion_mode)(
        goto_assertions::POINTER_SAFETY | goto_assertions::ARITHMETIC_SAFETY);
      REQUIRE(goto_assertions::is_mode_enabled(
        START, goto_assertions::POINTER_SAFETY));
      REQUIRE(goto_assertions::is_mode_enabled(
        START, goto_assertions::ARITHMETIC_SAFETY));

      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::USER));
      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::ARRAY_SAFETY));
      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::OTHER));
    }

    WHEN("START does not contain any modes")
    {
      START = (goto_assertions::goto_assertion_mode)0;
      REQUIRE_FALSE(goto_assertions::is_mode_enabled(
        START, goto_assertions::POINTER_SAFETY));
      REQUIRE_FALSE(goto_assertions::is_mode_enabled(
        START, goto_assertions::ARITHMETIC_SAFETY));
      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::USER));
      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::ARRAY_SAFETY));
      REQUIRE_FALSE(
        goto_assertions::is_mode_enabled(START, goto_assertions::OTHER));
    }
  }
}
