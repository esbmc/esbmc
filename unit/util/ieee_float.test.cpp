#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cmath>
#include <limits>
#include <util/arith/ieee_float.h>

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

TEST_CASE("ieee float converts zero to double", "[core][util][ieee_floatt]")
{
  // Zero is not "normal" (isnormal(0.0) is false) but it must still convert
  // back to exactly 0.0 -- see #1037, where the interval domain mistook this
  // for a conversion failure.
  ieee_floatt ieee_zero(ieee_float_spect(52, 11));
  ieee_zero.from_integer(0);

  CAPTURE(
    ieee_zero.get_exponent(), ieee_zero.get_fraction(), ieee_zero.get_sign());

  REQUIRE(ieee_zero.is_zero());
  REQUIRE_FALSE(std::isnormal(ieee_zero.to_double()));
  REQUIRE(ieee_zero.to_double() == 0.0);
}

TEST_CASE("ieee float classifies infinities", "[core][util][ieee_floatt]")
{
  // #7320: the division-by-zero and overflow sections are the ones that bite;
  // the others pin the rest of the classification.
  const ieee_float_spect spec(52, 11);

  ieee_floatt one(spec);
  one.from_integer(1);
  REQUIRE(one.is_normal());

  SECTION("division by zero")
  {
    ieee_floatt zero(spec);
    zero.from_integer(0);

    ieee_floatt inf = one;
    inf /= zero;

    REQUIRE(inf.is_infinity());
    REQUIRE_FALSE(inf.is_normal());
  }

  SECTION("overflow")
  {
    ieee_floatt max(spec);
    max.make_fltmax();
    REQUIRE(max.is_normal());

    ieee_floatt two(spec);
    two.from_integer(2);

    ieee_floatt inf = max;
    inf *= two;

    REQUIRE(inf.is_infinity());
    REQUIRE_FALSE(inf.is_normal());
  }

  SECTION("explicit infinity and NaN")
  {
    ieee_floatt inf(spec);
    inf.make_plus_infinity();
    REQUIRE_FALSE(inf.is_normal());

    inf.make_minus_infinity();
    REQUIRE_FALSE(inf.is_normal());

    ieee_floatt nan(spec);
    nan.make_NaN();
    REQUIRE_FALSE(nan.is_normal());
  }

  SECTION("subnormal stays subnormal")
  {
    ieee_floatt min_normal(spec);
    min_normal.make_fltmin();
    REQUIRE(min_normal.is_normal());

    ieee_floatt two(spec);
    two.from_integer(2);

    ieee_floatt subnormal = min_normal;
    subnormal /= two;

    REQUIRE(subnormal.is_finite());
    REQUIRE_FALSE(subnormal.is_normal());
  }
}

TEST_CASE("ieee float division rounds correctly", "[core][util][ieee_floatt]")
{
  // #10172: quotients just above a halfway point, and subnormal dividends.
  constexpr double inf = std::numeric_limits<double>::infinity();
  constexpr double max = std::numeric_limits<double>::max();
  struct row
  {
    double a, b, even, away, plus_inf, minus_inf, zero;
  };
  const row rows[] = {
    {0x1p+0,
     0x1.fffffffffffffp+52,
     0x1.0000000000001p-53,
     0x1.0000000000001p-53,
     0x1.0000000000001p-53,
     0x1p-53,
     0x1p-53},
    {-0x1p+0,
     0x1.fffffffffffffp+52,
     -0x1.0000000000001p-53,
     -0x1.0000000000001p-53,
     -0x1p-53,
     -0x1.0000000000001p-53,
     -0x1p-53},
    {0x0.0000000000001p-1022,
     0x1.fffffffffffffp+0,
     0x0.0000000000001p-1022,
     0x0.0000000000001p-1022,
     0x0.0000000000001p-1022,
     0x0p+0,
     0x0p+0},
    {0x0.0000000000003p-1022,
     0x1.56e1fc2f8f359p-997,
     0x1.1eb2d66005835p-76,
     0x1.1eb2d66005835p-76,
     0x1.1eb2d66005835p-76,
     0x1.1eb2d66005834p-76,
     0x1.1eb2d66005834p-76},
    {0x1p+0,
     0x1.8p+1,
     0x1.5555555555555p-2,
     0x1.5555555555555p-2,
     0x1.5555555555556p-2,
     0x1.5555555555555p-2,
     0x1.5555555555555p-2},
    {0x1.8p+1, 0x1.8p+0, 0x1p+1, 0x1p+1, 0x1p+1, 0x1p+1, 0x1p+1},
    {0x0.0000000000003p-1022,
     0x1p+1,
     0x0.0000000000002p-1022,
     0x0.0000000000002p-1022,
     0x0.0000000000002p-1022,
     0x0.0000000000001p-1022,
     0x0.0000000000001p-1022},
    {0x1p-1000,
     0x0.0000000000003p-1022,
     0x1.5555555555555p+72,
     0x1.5555555555555p+72,
     0x1.5555555555556p+72,
     0x1.5555555555555p+72,
     0x1.5555555555555p+72},
    {max, 0x1p-1, inf, inf, inf, max, max},
    {-max, 0x1p-1, -inf, -inf, -max, -inf, -max},
  };

  const auto divide = [](double a, double b, ieee_floatt::rounding_modet mode) {
    ieee_floatt x(ieee_float_spect(52, 11)), y(ieee_float_spect(52, 11));
    x.from_double(a);
    y.from_double(b);
    x.rounding_mode = mode;
    x /= y;
    return x.to_double();
  };

  for (const row &r : rows)
  {
    CAPTURE(r.a, r.b);
    CHECK(divide(r.a, r.b, ieee_floatt::ROUND_TO_EVEN) == r.even);
    CHECK(divide(r.a, r.b, ieee_floatt::ROUND_TO_AWAY) == r.away);
    CHECK(divide(r.a, r.b, ieee_floatt::ROUND_TO_PLUS_INF) == r.plus_inf);
    CHECK(divide(r.a, r.b, ieee_floatt::ROUND_TO_MINUS_INF) == r.minus_inf);
    CHECK(divide(r.a, r.b, ieee_floatt::ROUND_TO_ZERO) == r.zero);
  }
}
