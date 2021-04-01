/*******************************************************************\
 Module: Expressions Green Normal Form unit tests
 Author: Rafael Sá Menezes
 Date: April 2020
 Test Plan:
 - Check if trivial cases are ok
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cache/algorithms/expr_green_normal_form.h>
#include "cache_test_utils.h"

TEST_CASE("normal form tests", "[caching]")
{
  SECTION("less_should_become_lessthanequal")
  {
    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc b = create_unsigned_32_symbol_expr("b");

    // * < *
    expr2tc lesser = create_lesser_relation(a, b);
    auto actual = lesser->expr_id;

    // * <= *
    auto expected = expr2t::expr_ids::lessthanequal_id;

    REQUIRE(actual != expected);
    expr_green_normal_form algorithm(lesser);
    algorithm.run();

    actual = lesser->expr_id;
    REQUIRE(actual == expected);
  }

  SECTION("lessthanequal_should_become_lessthanequal")
  {
    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc b = create_unsigned_32_symbol_expr("b");

    // * <= *
    expr2tc lesser = create_lessthanequal_relation(a, b);
    auto actual = lesser->expr_id;

    // * <= *
    auto expected = expr2t::expr_ids::lessthanequal_id;

    REQUIRE(actual == expected);
    expr_green_normal_form algorithm(lesser);
    algorithm.run();

    actual = lesser->expr_id;
    REQUIRE(actual == expected);
  }

  SECTION("greaterthan_should_become_lessthanequal")
  {
    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc b = create_unsigned_32_symbol_expr("b");

    // * > *
    expr2tc lesser = create_greater_relation(a, b);
    auto actual = lesser->expr_id;

    // * <= *
    auto expected = expr2t::expr_ids::lessthanequal_id;

    REQUIRE(actual != expected);
    expr_green_normal_form algorithm(lesser);
    algorithm.run();

    actual = lesser->expr_id;
    REQUIRE(actual == expected);
  }

  SECTION("greaterthanequal_should_become_lessthanequal")
  {
    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc b = create_unsigned_32_symbol_expr("b");

    // * >=*
    expr2tc lesser = create_greaterthanequal_relation(a, b);
    auto actual = lesser->expr_id;

    // * <= *
    auto expected = expr2t::expr_ids::lessthanequal_id;

    REQUIRE(actual != expected);
    expr_green_normal_form algorithm(lesser);
    algorithm.run();

    actual = lesser->expr_id;
    REQUIRE(actual == expected);
  }

  SECTION("equal_should_become_equal")
  {
    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc b = create_unsigned_32_symbol_expr("b");

    // * == *
    expr2tc lesser = create_equality_relation(a, b);
    auto actual = lesser->expr_id;

    // *== *
    auto expected = expr2t::expr_ids::equality_id;

    REQUIRE(actual == expected);
    expr_green_normal_form algorithm(lesser);
    algorithm.run();

    actual = lesser->expr_id;
    REQUIRE(actual == expected);
  }
}
