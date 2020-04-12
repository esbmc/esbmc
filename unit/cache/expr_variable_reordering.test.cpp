/*******************************************************************\
 Module: Expressions Variable Reordering unit tests

 Author: Rafael Sá Menezes

 Date: March 2020

 Test Plan:

 - Check if trivial cases are ok
 - Check symbols and symbols with the same operator
 - Multiple operators (not working)
\*******************************************************************/

#define BOOST_TEST_MODULE "Expr Variable Reordering"

#include <cache/algorithms/expr_variable_reordering.h>
#include <boost/test/included/unit_test.hpp>
#include "cache_test_utils.h"
// ******************** TESTS ********************

// ** Check if trivial cases are ok
// expressions which does not contain a symbol/value or only one at max
BOOST_AUTO_TEST_SUITE(trivial)

BOOST_AUTO_TEST_CASE(unsigned_expr_should_not_change_value)
{
  constant_int2tc life_expr = create_unsigned_32_value_expr(42);
  auto crc = life_expr->crc();

  expr_variable_reordering algorithm(life_expr);
  algorithm.run();

  BOOST_CHECK(life_expr->value.compare(42) == 0);
  BOOST_CHECK(life_expr->crc() == crc);
}

BOOST_AUTO_TEST_CASE(signed_expr_should_not_change_value)
{
  int meaning_of_death = ~42;
  constant_int2tc death_expr = create_signed_32_value_expr(meaning_of_death);
  auto crc = death_expr->crc();

  expr_variable_reordering algorithm(death_expr);
  algorithm.run();

  BOOST_CHECK(death_expr->value.compare(meaning_of_death) == 0);
  BOOST_CHECK(death_expr->crc() == crc);
}

BOOST_AUTO_TEST_CASE(symbol_expr_should_not_change_value)
{
  symbol2tc x = create_unsigned_32_symbol_expr("X");
  auto crc = x->crc();

  expr_variable_reordering algorithm(x);
  algorithm.run();

  BOOST_CHECK(x->get_symbol_name() == "X");
  BOOST_CHECK(x->crc() == crc);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(same_expr)

BOOST_AUTO_TEST_CASE(a_add_b_should_become_a_add_b)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");

  // a + b => a + b
  add2tc add = create_unsigned_32_add_expr(a, b);
  auto crc = add->crc();

  // Check if object is created as expected
  BOOST_TEST(is_symbols_equal(add->side_1, a));
  BOOST_TEST(is_symbols_equal(add->side_2, b));

  expr_variable_reordering algorithm(add);
  algorithm.run();

  std::shared_ptr<arith_2ops> arith;
  arith = std::dynamic_pointer_cast<arith_2ops>(add);

  // Check if object is reordered correctly
  BOOST_TEST(is_symbols_equal(a, arith->side_1));
  BOOST_TEST(is_symbols_equal(b, arith->side_2));

  BOOST_CHECK(add->crc() == crc);
}

BOOST_AUTO_TEST_CASE(b_add_a_should_become_a_add_b)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");

  // b + a => a + b
  add2tc add = create_unsigned_32_add_expr(b, a);
  auto crc = add->crc();

  // Check if object is created as expected
  BOOST_TEST(is_symbols_equal(add->side_1, b));
  BOOST_TEST(is_symbols_equal(add->side_2, a));

  expr_variable_reordering algorithm(add);
  algorithm.run();

  // Check if object is reordered correctly
  BOOST_TEST(is_symbols_equal(add->side_1, a));
  BOOST_TEST(is_symbols_equal(add->side_2, b));

  BOOST_CHECK(add->crc() != crc);
}

BOOST_AUTO_TEST_CASE(a_add_value_should_become_a_add_value)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  constant_int2tc value = create_unsigned_32_value_expr(42);

  // a + 42 => a + 42
  add2tc add = create_unsigned_32_add_expr(a, value);

  // Check if object is created as expected
  BOOST_TEST(is_symbols_equal(add->side_1, a));
  BOOST_TEST(is_unsigned_equal(add->side_2, value));

  auto crc = add->crc();

  expr_variable_reordering algorithm(add);
  algorithm.run();

  // Check if object is reordered correctly
  BOOST_TEST(is_symbols_equal(add->side_1, a));
  BOOST_TEST(is_unsigned_equal(add->side_2, value));
  BOOST_CHECK(add->crc() == crc);
}

BOOST_AUTO_TEST_CASE(value_add_a_should_become_a_add_value)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  constant_int2tc value = create_unsigned_32_value_expr(42);

  // 42 + a => a + 42
  add2tc add = create_unsigned_32_add_expr(value, a);

  // Check if object is created as expected
  BOOST_TEST(is_unsigned_equal(add->side_1, value));
  BOOST_TEST(is_symbols_equal(add->side_2, a));

  auto crc = add->crc();

  expr_variable_reordering algorithm(add);
  algorithm.run();

  // Check if object is reordered correctly
  BOOST_TEST(is_symbols_equal(add->side_1, a));
  BOOST_TEST(is_unsigned_equal(add->side_2, value));
  BOOST_CHECK(add->crc() != crc);
}

BOOST_AUTO_TEST_CASE(equality_1_check)
{
  init_test_values();

  // ((y + x) + 7) == 9
  auto actual = equality_1();

  // ((x + y) + 7) == 9
  auto expected = equality_1_ordered();

  BOOST_TEST(actual->crc() != expected->crc());

  expr_variable_reordering algorithm(actual);
  algorithm.run();

  BOOST_TEST(actual->crc() == expected->crc());
}

BOOST_AUTO_TEST_CASE(equality_2_check)
{
  init_test_values();

  // (1 + x) == 0
  auto actual = equality_2();

  // (x + 1) == 0
  auto expected = equality_2_ordered();

  BOOST_TEST(actual->crc() != expected->crc());

  expr_variable_reordering algorithm(actual);
  algorithm.run();

  BOOST_TEST(actual->crc() == expected->crc());
}

BOOST_AUTO_TEST_CASE(equality_3_check)
{
  init_test_values();

  // (y + 4) == 8
  auto actual = equality_3();

  // (y + 4) == 8
  auto expected = equality_3_ordered();

  BOOST_TEST(actual->crc() == expected->crc());

  expr_variable_reordering algorithm(actual);
  algorithm.run();

  BOOST_TEST(actual->crc() == expected->crc());
}

BOOST_AUTO_TEST_CASE(equality_4_check)
{
  init_test_values();

  // (x + 0) == 0
  auto actual = equality_4();

  // (x + 0) == 0
  auto expected = equality_4_ordered();

  BOOST_TEST(actual->crc() == expected->crc());

  expr_variable_reordering algorithm(actual);
  algorithm.run();

  BOOST_TEST(actual->crc() == expected->crc());
}

BOOST_AUTO_TEST_SUITE_END()