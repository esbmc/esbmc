/*******************************************************************\
 Module: Expressions Green Normal Form unit tests

 Author: Rafael Sá Menezes

 Date: April 2020

 Test Plan:

 - Check if trivial cases are ok
\*******************************************************************/

#define BOOST_TEST_MODULE "Expr Green Normal Form"

#include <cache/algorithms/expr_green_normal_form.h>
#include <boost/test/included/unit_test.hpp>
#include "cache_test_utils.h"

BOOST_AUTO_TEST_SUITE(operators)

BOOST_AUTO_TEST_CASE(less_should_become_lessthanequal)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");

  // * < *
  expr2tc lesser = create_lesser_relation(a, b);
  auto actual = lesser->expr_id;

  // * <= *
  auto expected = expr2t::expr_ids::lessthanequal_id;

  BOOST_TEST(actual != expected);
  expr_green_normal_form algorithm(lesser);
  algorithm.run();

  actual = lesser->expr_id;
  BOOST_TEST(actual == expected);
}

BOOST_AUTO_TEST_CASE(lessthanequal_should_become_lessthanequal)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");

  // * <= *
  expr2tc lesser = create_lessthanequal_relation(a, b);
  auto actual = lesser->expr_id;

  // * <= *
  auto expected = expr2t::expr_ids::lessthanequal_id;

  BOOST_TEST(actual == expected);
  expr_green_normal_form algorithm(lesser);
  algorithm.run();

  actual = lesser->expr_id;
  BOOST_TEST(actual == expected);
}

BOOST_AUTO_TEST_CASE(greaterthan_should_become_lessthanequal)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");

  // * > *
  expr2tc lesser = create_greater_relation(a, b);
  auto actual = lesser->expr_id;

  // * <= *
  auto expected = expr2t::expr_ids::lessthanequal_id;

  BOOST_TEST(actual != expected);
  expr_green_normal_form algorithm(lesser);
  algorithm.run();

  actual = lesser->expr_id;
  BOOST_TEST(actual == expected);
}

BOOST_AUTO_TEST_CASE(greaterthanequal_should_become_lessthanequal)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");

  // * >=*
  expr2tc lesser = create_greaterthanequal_relation(a, b);
  auto actual = lesser->expr_id;

  // * <= *
  auto expected = expr2t::expr_ids::lessthanequal_id;

  BOOST_TEST(actual != expected);
  expr_green_normal_form algorithm(lesser);
  algorithm.run();

  actual = lesser->expr_id;
  BOOST_TEST(actual == expected);
}

BOOST_AUTO_TEST_CASE(equal_should_become_equal)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");

  // * == *
  expr2tc lesser = create_equality_relation(a, b);
  auto actual = lesser->expr_id;

  // *== *
  auto expected = expr2t::expr_ids::equality_id;

  BOOST_TEST(actual == expected);
  expr_green_normal_form algorithm(lesser);
  algorithm.run();

  actual = lesser->expr_id;
  BOOST_TEST(actual == expected);
}

BOOST_AUTO_TEST_SUITE_END()
