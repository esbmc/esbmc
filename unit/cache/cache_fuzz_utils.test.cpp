/*******************************************************************\
Module: Cache fuzz utilities unit testing
Author: Rafael Sá Menezes
Date: April 2020
Test Plan:
 - Check if trivial cases are ok
 - Check if expressions of size 5-6 are created ok
\*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "cache_fuzz_utils.h"

// ******************** TESTS ********************

/*
 * Check if trivial cases are ok
 *
 * - Mostly this will check if errors are throw and if basic expressions are
 *   created correcly
 */

TEST_CASE("expr generator tests", "[caching, fuzz]")
{
  SECTION("minimum_expression_ok")
  {
    const char *expression = "00a%a";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(fuzzer.is_expr_valid());
  }

  SECTION("length_expression_not_ok")
  {
    const char *expression = "00a%";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(!fuzzer.is_expr_valid());
  }

  SECTION("lhs_expression_not_ok")
  {
    const char *expression = "00%a";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(!fuzzer.is_expr_valid());
  }

  SECTION("rhs_expression_not_ok")
  {
    const char *expression = "00aa%";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(!fuzzer.is_expr_valid());
  }

  SECTION("no_rhs_expression_not_ok")
  {
    const char *expression = "00aaa";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(!fuzzer.is_expr_valid());
  }

  SECTION("expression_from_fuzzer_1_ok")
  {
    const char *expression = "|+|+%\x0a";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(fuzzer.is_expr_valid());
  }

  SECTION("expression_from_fuzzer_2_ok")
  {
    const char *expression = "0x6a0xa0xfb0x250xff0xfb\nj\x0a\xfb%\xff\xfb";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(fuzzer.is_expr_valid());
  }

  SECTION("five_same_symbol")
  {
    // a == a
    const char *expression = "00a%a";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(fuzzer.is_expr_valid());

    symbol2tc a = create_unsigned_32_symbol_expr("a");

    expr2tc result = fuzzer.convert_input_to_expression();

    REQUIRE(result->expr_id == expr2t::expr_ids::equality_id);

    equality2tc equality = result;

    REQUIRE(is_symbols_equal(equality->side_1, a));
    REQUIRE(is_symbols_equal(equality->side_2, a));
  }

  SECTION("five_different_symbol")
  {
    // a == c
    const char *expression = "00a%c";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(fuzzer.is_expr_valid());

    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc c = create_unsigned_32_symbol_expr("c");

    expr2tc result = fuzzer.convert_input_to_expression();

    REQUIRE(result->expr_id == expr2t::expr_ids::equality_id);

    equality2tc equality = result;

    REQUIRE(is_symbols_equal(equality->side_1, a));
    REQUIRE(is_symbols_equal(equality->side_2, c));
  }

  SECTION("six_lhs_different_symbol")
  {
    // a + b == c
    const char *expression = "00ab%c";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(fuzzer.is_expr_valid());

    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc b = create_unsigned_32_symbol_expr("b");
    symbol2tc c = create_unsigned_32_symbol_expr("c");

    expr2tc result = fuzzer.convert_input_to_expression();

    REQUIRE(result->expr_id == expr2t::expr_ids::equality_id);

    equality2tc equality = result;
    add2tc add = equality->side_1;

    REQUIRE(is_symbols_equal(add->side_1, a));
    REQUIRE(is_symbols_equal(add->side_2, b));
    REQUIRE(is_symbols_equal(equality->side_2, c));
  }

  SECTION("six_rhs_different_symbol")
  {
    // b == a + c
    const char *expression = "00b%ac";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(fuzzer.is_expr_valid());

    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc b = create_unsigned_32_symbol_expr("b");
    symbol2tc c = create_unsigned_32_symbol_expr("c");

    expr2tc result = fuzzer.convert_input_to_expression();

    REQUIRE(result->expr_id == expr2t::expr_ids::equality_id);

    equality2tc equality = result;
    add2tc add = equality->side_2;

    REQUIRE(is_symbols_equal(equality->side_1, b));
    REQUIRE(is_symbols_equal(add->side_1, a));
    REQUIRE(is_symbols_equal(add->side_2, c));
  }

  SECTION("seven_different_symbol")
  {
    // (b + a) + c == d
    const char *expression = "00bac%d";
    std::string input(expression);
    expr_generator_fuzzer fuzzer(input);
    REQUIRE(fuzzer.is_expr_valid());

    symbol2tc a = create_unsigned_32_symbol_expr("a");
    symbol2tc b = create_unsigned_32_symbol_expr("b");
    symbol2tc c = create_unsigned_32_symbol_expr("c");
    symbol2tc d = create_unsigned_32_symbol_expr("d");

    expr2tc result = fuzzer.convert_input_to_expression();

    REQUIRE(result->expr_id == expr2t::expr_ids::equality_id);

    equality2tc equality = result;
    add2tc add = equality->side_1;
    add2tc inner_add = add->side_1;

    REQUIRE(is_symbols_equal(inner_add->side_1, b));
    REQUIRE(is_symbols_equal(inner_add->side_2, a));
    REQUIRE(is_symbols_equal(add->side_2, c));
    REQUIRE(is_symbols_equal(equality->side_2, d));
  }
}
