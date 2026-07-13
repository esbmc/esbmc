// H-A6 — constant_int2t::as_ulong / as_long bounds (R2), verified against the
// *actual* irep2 implementation (src/irep2/irep2_expr.cpp), not a model.
//
// as_ulong()/as_long() forward to BigInt::to_uint64()/to_int64(), which shift
// every digit into a 64-bit accumulator and therefore SILENTLY TRUNCATE a
// value whose magnitude exceeds 64 bits (finding R2 in
// docs/irep2-verification-plan.md). This suite:
//   * pins the correct round-trip contract for the full in-range boundary set
//     on the real constant_int2t (would catch a regression in the conversion
//     itself), and
//   * documents the out-of-range behaviour both ways: in an asserts-on build
//     the new is_uint64()/is_int64() guards fire (the value's precondition is
//     shown false); in NDEBUG the guards are compiled out and the raw
//     truncation is exhibited — the residual release-mode gap R2 records.

#define CATCH_CONFIG_MAIN // Catch provides main() for this test executable
#include <catch2/catch.hpp>

#include <cstdint>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/format_constant.h>

namespace
{
constant_int2t as_uint(const BigInt &v)
{
  return constant_int2t(get_uint_type(64), v);
}

constant_int2t as_int(const BigInt &v)
{
  return constant_int2t(get_int_type(64), v);
}
} // namespace

TEST_CASE(
  "constant_int2t::as_ulong round-trips every in-range value",
  "[core][irep2][const_int]")
{
  config.ansi_c.word_size = 64;

  REQUIRE(as_uint(BigInt(0u)).as_ulong() == 0u);
  REQUIRE(as_uint(BigInt(1u)).as_ulong() == 1u);
  REQUIRE(as_uint(BigInt(0x7fffffffu)).as_ulong() == 0x7fffffffu);
  REQUIRE(
    as_uint(BigInt(static_cast<BigInt::ullong_t>(INT64_MAX))).as_ulong() ==
    static_cast<uint64_t>(INT64_MAX));
  // Upper boundary: the largest value as_ulong may legitimately return.
  REQUIRE(
    as_uint(BigInt(static_cast<BigInt::ullong_t>(UINT64_MAX))).as_ulong() ==
    UINT64_MAX);
}

TEST_CASE(
  "constant_int2t::as_long round-trips the full signed range",
  "[core][irep2][const_int]")
{
  config.ansi_c.word_size = 64;

  REQUIRE(as_int(BigInt(0)).as_long() == 0);
  REQUIRE(as_int(BigInt(1)).as_long() == 1);
  REQUIRE(as_int(BigInt(-1)).as_long() == -1);
  REQUIRE(
    as_int(BigInt(static_cast<BigInt::llong_t>(INT64_MAX))).as_long() ==
    INT64_MAX);
  REQUIRE(
    as_int(BigInt(static_cast<BigInt::llong_t>(INT64_MIN))).as_long() ==
    INT64_MIN);
}

// R2: a magnitude just past the 64-bit boundary. The guards added to
// as_ulong/as_long are is_uint64()/is_int64(); this pins that they correctly
// classify the out-of-range value as NOT convertible, which is exactly the
// precondition the (asserts-on) guard checks.
TEST_CASE(
  "constant_int2t rejects out-of-range magnitudes (R2 guard precondition)",
  "[core][irep2][const_int]")
{
  config.ansi_c.word_size = 64;

  const BigInt two_pow_64 =
    BigInt(static_cast<BigInt::ullong_t>(UINT64_MAX)) + BigInt(1); // 2^64
  const constant_int2t &big = as_uint(BigInt(two_pow_64));

  REQUIRE_FALSE(big.value.is_uint64()); // as_ulong()'s guard would fire
  REQUIRE_FALSE(big.value.is_int64());  // as_long()'s guard would fire

#ifdef NDEBUG
  // Guards compiled out: exhibit the raw truncation R2 records — 2^64 wraps
  // to 0 in the 64-bit accumulator. (In an asserts-on build the guard aborts
  // instead, so this call is only exercised under NDEBUG.)
  REQUIRE(big.as_ulong() == 0u);
#endif
}

// format_constantt previously printed unsigned/signed constants via
// as_ulong()/as_long(), truncating anything wider than 64 bits (and, once
// the R2 guards are added, aborting in asserts-on builds). It now formats the
// full-width value straight from the BigInt. This pins that a 128-bit
// constant prints its exact value rather than the low-64-bit truncation.
TEST_CASE(
  "format_constantt prints wide constants without truncation (R2)",
  "[core][irep2][const_int]")
{
  config.ansi_c.word_size = 64;
  format_constantt fmt;

  // In-range values are unaffected.
  REQUIRE(fmt(constant_int2tc(get_uint_type(64), BigInt(42u))) == "42");
  REQUIRE(fmt(constant_int2tc(get_int_type(64), BigInt(-42))) == "-42");

  // 2^64 in a 128-bit unsigned constant must print in full, not as "0".
  const BigInt two_pow_64 =
    BigInt(static_cast<BigInt::ullong_t>(UINT64_MAX)) + BigInt(1);
  REQUIRE(
    fmt(constant_int2tc(get_uint_type(128), two_pow_64)) ==
    "18446744073709551616");
}
