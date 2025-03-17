#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>

#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/namespace.h>
#include <util/simplify_expr.h>
#include <util/std_expr.h>
#include <util/std_types.h>

TEST_CASE("for a typecast to bool...", "[unit][util][simplify_expr]")
{
  type2tc int_type = signedbv_type2tc(32);
  type2tc bool_type = bool_type2tc();

  SECTION("cast of int 0 to bool should simplify to false")
  {
    expr2tc op = typecast2tc(bool_type, constant_int2tc(int_type, BigInt(0)));
    expr2tc simp = op.simplify();
    REQUIRE(simp == gen_false_expr());
  }
  SECTION("cast of int 1 to bool should simplify to true")
  {
    expr2tc op = typecast2tc(bool_type, constant_int2tc(int_type, BigInt(1)));
    expr2tc simp = op.simplify();
    REQUIRE(simp == gen_true_expr());
  }
  SECTION("cast of int nondet to bool should simplify to nondet != 0")
  {
    expr2tc op = typecast2tc(bool_type, symbol2tc(int_type, "nondet"));
    expr2tc simp = op.simplify();
    REQUIRE(
      simp ==
      not2tc(equality2tc(
        symbol2tc(int_type, "nondet"), constant_int2tc(int_type, BigInt(0)))));
  }
  SECTION("cast of int nondet to bool to bool should simplify to nondet != 0")
  {
    expr2tc op = typecast2tc(
      bool_type, typecast2tc(bool_type, symbol2tc(int_type, "nondet")));
    expr2tc simp = op.simplify();
    REQUIRE(
      simp ==
      not2tc(equality2tc(
        symbol2tc(int_type, "nondet"), constant_int2tc(int_type, BigInt(0)))));
  }
  SECTION("cast of bool nondet to int to bool should simplify to nondet")
  {
    expr2tc op = typecast2tc(
      bool_type, typecast2tc(int_type, symbol2tc(bool_type, "nondet")));
    expr2tc simp = op.simplify();
    REQUIRE(simp == symbol2tc(bool_type, "nondet"));
  }
}