/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tests for to_integer(const expr2tc&, BigInt&) — the IREP2 twin of the
 * legacy to_integer(const exprt&, BigInt&). Contract (mirrors the legacy
 * one): returns false on success with the folded integer in int_value, true
 * on failure (not a foldable integer constant).
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/migrate.h>

TEST_CASE(
  "to_integer(expr2tc) extracts bare integer constants",
  "[util][arith]")
{
  BigInt v;

  expr2tc u = constant_int2tc(get_uint32_type(), BigInt(42));
  REQUIRE_FALSE(to_integer(u, v));
  REQUIRE(v == 42);

  expr2tc s = constant_int2tc(get_int32_type(), BigInt(-7));
  REQUIRE_FALSE(to_integer(s, v));
  REQUIRE(v == -7);
}

TEST_CASE("to_integer(expr2tc) maps booleans to 1/0", "[util][arith]")
{
  BigInt v;

  expr2tc t = gen_true_expr();
  REQUIRE_FALSE(to_integer(t, v));
  REQUIRE(v == 1);

  expr2tc f = gen_false_expr();
  REQUIRE_FALSE(to_integer(f, v));
  REQUIRE(v == 0);
}

TEST_CASE(
  "to_integer(expr2tc) applies the cast for a typecast of a constant",
  "[util][arith]")
{
  BigInt v;

  // (uint8_t)300 == 44 (300 mod 256) — the cast MUST be applied; reading the
  // operand's raw value (300) would be wrong.
  expr2tc wide = constant_int2tc(get_uint32_type(), BigInt(300));
  expr2tc narrowed = typecast2tc(get_uint8_type(), wide);
  REQUIRE_FALSE(to_integer(narrowed, v));
  REQUIRE(v == 44);
}

TEST_CASE("to_integer(expr2tc) fails on non-constants and nil", "[util][arith]")
{
  BigInt v;

  expr2tc sym = symbol2tc(get_uint32_type(), "x");
  REQUIRE(to_integer(sym, v));

  expr2tc nil;
  REQUIRE(is_nil_expr(nil));
  REQUIRE(to_integer(nil, v));
}

TEST_CASE(
  "to_integer(expr2tc) agrees with the legacy to_integer on constants",
  "[util][arith]")
{
  for (const BigInt &value : {BigInt(0), BigInt(1), BigInt(255), BigInt(-128)})
  {
    const typet lt = signedbv_typet(32);
    exprt legacy = from_integer(value, lt);
    REQUIRE(legacy.is_constant());

    expr2tc migrated;
    migrate_expr(legacy, migrated);

    BigInt vl, v2;
    bool fl = to_integer(legacy, vl);
    bool f2 = to_integer(migrated, v2);
    REQUIRE(fl == f2);
    REQUIRE_FALSE(fl);
    REQUIRE(vl == v2);
    REQUIRE(v2 == value);
  }
}
