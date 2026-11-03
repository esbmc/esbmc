/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * host_rounding_mode: a scope runs under the requested FPU rounding mode and
 * leaves the mode it found behind, however it is left.
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cfenv>
#include <stdexcept>
#include <util/base/host_rounding_mode.h>

TEST_CASE("host_rounding_mode sets and restores the mode", "[core][util]")
{
  std::fesetround(FE_TONEAREST);
  {
    const host_rounding_mode upward(FE_UPWARD);
    REQUIRE(std::fegetround() == FE_UPWARD);
  }
  REQUIRE(std::fegetround() == FE_TONEAREST);
}

TEST_CASE("host_rounding_mode pins nearest and restores upward", "[core][util]")
{
  std::fesetround(FE_UPWARD);
  {
    const host_rounding_mode nearest(FE_TONEAREST);
    REQUIRE(std::fegetround() == FE_TONEAREST);
    {
      const host_rounding_mode already_nearest(FE_TONEAREST);
      REQUIRE(std::fegetround() == FE_TONEAREST);
    }
    REQUIRE(std::fegetround() == FE_TONEAREST);
  }
  REQUIRE(std::fegetround() == FE_UPWARD);
  std::fesetround(FE_TONEAREST);
}

TEST_CASE("host_rounding_mode guards nest", "[core][util]")
{
  std::fesetround(FE_TONEAREST);
  {
    const host_rounding_mode upward(FE_UPWARD);
    {
      const host_rounding_mode downward(FE_DOWNWARD);
      REQUIRE(std::fegetround() == FE_DOWNWARD);
    }
    REQUIRE(std::fegetround() == FE_UPWARD);
  }
  REQUIRE(std::fegetround() == FE_TONEAREST);
}

TEST_CASE("host_rounding_mode restores on an exception", "[core][util]")
{
  std::fesetround(FE_TONEAREST);
  try
  {
    const host_rounding_mode upward(FE_UPWARD);
    throw std::runtime_error("unwind");
  }
  catch (const std::runtime_error &)
  {
  }
  REQUIRE(std::fegetround() == FE_TONEAREST);
}
