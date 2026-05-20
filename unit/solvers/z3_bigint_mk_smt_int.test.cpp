// Precision contract for z3_convt::mk_smt_int on out-of-int64 BigInts
// (issue #4642). The fix in src/solvers/z3/z3_conv.cpp routes values outside
// the int64 range through int_val(const char *) instead of
// int_val(to_int64()), because BigInt::to_int64 silently truncates past 64
// bits. BigInt::is_uint64 is a pure-magnitude predicate (it ignores the
// sign), so the production code deliberately does not gate on it.
//
// The first SCENARIO pins the underlying Z3 numeral API surface that the
// fix relies on. The second SCENARIO exercises z3_convt::mk_smt_int
// end-to-end — instantiating ESBMC's full Z3 backend and reading back the
// resulting numeral — so a regression that reintroduced an is_uint64
// fast-path (silent sign loss) or a to_int64 truncation (silent value
// loss) inside ESBMC's overload would surface here rather than in a solver
// query downstream.
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <memory>
#include <z3++.h>
#include <big-int/bigint.hh>
#include <util/context.h>
#include <util/mp_arith.h>
#include <util/namespace.h>
#include <util/options.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/solve.h>
#include <z3_conv.h>

// solver_creator declared in src/solvers/solve.h; the Z3 backend defines it
// in src/solvers/z3/z3_conv.cpp but does not publish a header — declare it
// here so the test can call it without taking an explicit cross-module
// header dependency.
extern solver_creator create_new_z3_solver;

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

SCENARIO(
  "z3_convt::mk_smt_int preserves BigInt precision and sign",
  "[z3][bigint]")
{
  // Drive ESBMC's overload directly so a regression that reintroduces a
  // to_int64 / is_uint64 fast path inside z3_convt::mk_smt_int surfaces
  // here, not at the next bignum-aware caller.
  contextt ctx;
  namespacet ns(ctx);
  optionst options;
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  fp_convt *fp_api = nullptr;
  std::unique_ptr<smt_convt> solver{
    create_new_z3_solver(options, ns, &tuple_api, &array_api, &fp_api)};
  REQUIRE(solver != nullptr);

  auto numeral_string = [](smt_astt ast) {
    z3::expr e = to_solver_smt_ast<z3_smt_ast>(ast)->a;
    return std::string(e.get_decimal_string(0));
  };

  GIVEN("A BigInt that fits int64")
  {
    BigInt v(-12345);
    THEN("mk_smt_int reproduces the decimal representation")
    {
      REQUIRE(numeral_string(solver->mk_smt_int(v)) == integer2string(v, 10));
    }
  }

  GIVEN("A BigInt larger than uint64_max")
  {
    BigInt huge = BigInt::power2(200);
    THEN("mk_smt_int preserves every digit via the string path")
    {
      REQUIRE_FALSE(huge.is_int64());
      REQUIRE(
        numeral_string(solver->mk_smt_int(huge)) == integer2string(huge, 10));
    }
  }

  GIVEN("A negative BigInt with magnitude in (INT64_MAX, UINT64_MAX]")
  {
    // The regression this case pins: a previous draft of the fix included
    // an `is_uint64` fast-path that silently flipped this value's sign.
    BigInt edge = -(BigInt::power2(63) + BigInt(1));
    THEN("mk_smt_int preserves the sign")
    {
      REQUIRE_FALSE(edge.is_int64());
      REQUIRE(edge.is_uint64());
      const std::string out = numeral_string(solver->mk_smt_int(edge));
      REQUIRE(out.front() == '-');
      REQUIRE(out == integer2string(edge, 10));
    }
  }

  GIVEN("A negative BigInt smaller than int64_min")
  {
    BigInt neg = -BigInt::power2(200);
    THEN("mk_smt_int preserves the sign and every digit")
    {
      REQUIRE_FALSE(neg.is_int64());
      REQUIRE(
        numeral_string(solver->mk_smt_int(neg)) == integer2string(neg, 10));
    }
  }
}
