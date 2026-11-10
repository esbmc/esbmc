#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
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

TEST_CASE(
  "ieee float division is correctly rounded",
  "[core][util][ieee_floatt]")
{
  // #10172: a quotient just above a halfway point must not round as a tie.
  const std::pair<int, ieee_floatt::rounding_modet> modes[] = {
    {FE_TONEAREST, ieee_floatt::ROUND_TO_EVEN},
    {FE_TOWARDZERO, ieee_floatt::ROUND_TO_ZERO},
    {FE_UPWARD, ieee_floatt::ROUND_TO_PLUS_INF},
    {FE_DOWNWARD, ieee_floatt::ROUND_TO_MINUS_INF}};

  std::vector<std::pair<double, double>> operands;

  // k / (2^53 - c) has a long run of zeros after its 54th bit, so a quotient
  // truncated there lands on a halfway point; also with subnormal dividends.
  for (int k = 1; k <= 64; ++k)
    for (int c = 1; c <= 64; ++c)
    {
      const double b = 9007199254740992.0 - c;
      operands.emplace_back(k, b);
      operands.emplace_back(std::ldexp(k, -1074), std::ldexp(b, -100));
    }

  std::mt19937_64 rng(10172);
  for (unsigned i = 0; i < 20000; ++i)
  {
    // Every fourth operand is subnormal.
    uint64_t bits[2];
    for (uint64_t &b : bits)
    {
      b = rng();
      if (rng() % 4 == 0)
        b &= 0x800fffffffffffffULL;
    }
    double a, b;
    std::memcpy(&a, &bits[0], sizeof a);
    std::memcpy(&b, &bits[1], sizeof b);
    if (!std::isnan(a) && !std::isnan(b))
      operands.emplace_back(a, b);
  }

  for (const auto &[a, b] : operands)
  {
    for (const auto &[host_mode, mode] : modes)
    {
      std::fesetround(host_mode);
      volatile double va = a, vb = b;
      const double expected = va / vb;
      std::fesetround(FE_TONEAREST);

      ieee_floatt x, y;
      x.from_double(a);
      y.from_double(b);
      x.rounding_mode = mode;
      x /= y;

      CAPTURE(a, b, host_mode, expected, x.to_double());
      if (std::isnan(expected))
        REQUIRE(std::isnan(x.to_double()));
      else
      {
        REQUIRE(x.to_double() == expected);
        REQUIRE(std::signbit(x.to_double()) == std::signbit(expected));
      }
    }
  }
}
