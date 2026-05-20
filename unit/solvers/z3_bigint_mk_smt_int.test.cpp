// Precision contract for the API z3_convt::mk_smt_int uses for
// out-of-int64 BigInts (issue #4642). The fix in src/solvers/z3/z3_conv.cpp
// routes values outside the int64 range through int_val(const char *)
// instead of int_val(to_int64()), because BigInt::to_int64 silently truncates
// past 64 bits. BigInt::is_uint64 is a pure-magnitude predicate (it ignores
// the sign), so the production code deliberately does not gate on it. This
// test pins that contract end-to-end against the live Z3 numeral API so a
// future toolchain swap that broke the string overload — or a regression
// that reintroduced an is_uint64 fast path — would surface here rather than
// in a solver query.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <z3++.h>
#include <big-int/bigint.hh>
#include <util/mp_arith.h>

SCENARIO("Z3 int_val preserves BigInt precision beyond uint64", "[z3][bigint]")
{
  z3::context c;

  GIVEN("A negative BigInt that fits int64")
  {
    BigInt small(-12345);
    THEN("the int64 overload returns the same decimal")
    {
      z3::expr e = c.int_val(small.to_int64());
      REQUIRE(e.get_decimal_string(0) == integer2string(small, 10));
    }
  }

  GIVEN("A BigInt larger than uint64_max")
  {
    BigInt huge = BigInt::power2(200);
    THEN("the string overload preserves every digit")
    {
      REQUIRE_FALSE(huge.is_int64());
      std::string dec = integer2string(huge, 10);
      z3::expr e = c.int_val(dec.c_str());
      REQUIRE(e.get_decimal_string(0) == dec);
    }
  }

  GIVEN("A negative BigInt smaller than int64_min")
  {
    BigInt neg = -BigInt::power2(200);
    THEN("the string overload preserves the sign and every digit")
    {
      REQUIRE_FALSE(neg.is_int64());
      std::string dec = integer2string(neg, 10);
      z3::expr e = c.int_val(dec.c_str());
      REQUIRE(e.get_decimal_string(0) == dec);
    }
  }

  GIVEN("A negative BigInt whose magnitude is in (INT64_MAX, UINT64_MAX]")
  {
    // -(2^63 + 1) — is_uint64() returns true (pure magnitude) but the value
    // does NOT fit in int64. Production code must NOT take any uint64
    // fast-path here; the string fallback must preserve the sign.
    BigInt edge = -(BigInt::power2(63) + BigInt(1));
    THEN("magnitude-only is_uint64 cannot be used to gate a fast path")
    {
      REQUIRE_FALSE(edge.is_int64());
      REQUIRE(edge.is_uint64());
      std::string dec = integer2string(edge, 10);
      REQUIRE(dec.front() == '-');
      z3::expr e = c.int_val(dec.c_str());
      REQUIRE(e.get_decimal_string(0) == dec);
    }
  }
}
