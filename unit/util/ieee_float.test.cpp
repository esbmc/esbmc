#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cmath>
#include <util/ieee_float.h>

TEST_CASE("ieee float can handle 1", "[core][util][ieee_floatt]")
{
  int one = 1;
  double d_one = 1;
  ieee_floatt ieee_one(ieee_float_spect(52, 11));

  SECTION("Basic context")
  {
    REQUIRE(std::isnormal(d_one)); // Holds
  }

  SECTION("From double")
  {
    ieee_one.from_double(d_one);
    CAPTURE(
      ieee_one.get_exponent(), ieee_one.get_fraction(), ieee_one.get_sign());
    REQUIRE(std::isnormal(ieee_one.to_double())); // Holds
  }

  SECTION("From integer")
  {
    ieee_one.from_integer(one);
    CAPTURE(
      ieee_one.get_exponent(), ieee_one.get_fraction(), ieee_one.get_sign());
    REQUIRE(std::isnormal(ieee_one.to_double())); // Holds
  }
}
