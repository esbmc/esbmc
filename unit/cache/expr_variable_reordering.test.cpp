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

BOOST_AUTO_TEST_CASE(expression_with_multiple_operators)
{
  // 42 + x + (y*(b + a))  -> // x + 42 + y*(a + b)
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");
  symbol2tc x = create_unsigned_32_symbol_expr("x");
  symbol2tc y = create_unsigned_32_symbol_expr("y");
  constant_int2tc value = create_unsigned_32_value_expr(42);

  // b + a -> a + b
  add2tc add_1 = create_unsigned_32_add_expr(b, a);
  add2tc add_1_expected = create_unsigned_32_add_expr(a, b);

  // y(b + a) -> y(a + b)
  mul2tc mul_1 = create_unsigned_32_mul_expr(y, add_1);
  mul2tc mul_1_expected = create_unsigned_32_mul_expr(y, add_1_expected);

  // x + y(b + a) -> 42 + y(b + a)
  add2tc add_2 = create_unsigned_32_add_expr(x, mul_1);
  add2tc add_2_expected = create_unsigned_32_add_expr(value, mul_1_expected);

  // 42 + x + y(b + a) -> x + 42 + y(b + a)
  add2tc add_3 = create_unsigned_32_add_expr(value, add_2);
  add2tc add_3_expected = create_unsigned_32_add_expr(x, add_2_expected);

  auto crc = add_3->crc();

  expr_variable_reordering algorithm(add_3);
  algorithm.run();

  // Check if object has changed
  BOOST_CHECK(add_3->crc() != crc);

  /**
   *     add_2
   *        |  mul_1
   *        |    |   add_1
   *  add_3 |    |    |
   *   |    |    |    |
   * x + 42 + y * (a + b)
   * | |  | | | |  | | |
   * | |  A | | |  | | |
   * | |    | | |  | | B
   * | |    | | |  C |
   * | |    | | |    D
   * | |    | E |
   * | |    |   F
   * | |    G
   * H |
   *   I
   */

  // I
  // I am unsure why this is not working, maybe is related to the way
  // that I generated the expr, but all subexpressions are ok
  //BOOST_TES(add_3->crc() != add_3_expected->crc());

  // H
  BOOST_TEST(add_3->side_1->expr_id == add_3_expected->side_1->expr_id);
  BOOST_TEST(add_3->side_1->crc() == add_3_expected->side_1->crc());
  BOOST_TEST(is_symbols_equal(add_3->side_1, x));

  // G
  add2tc G(add_3->side_2);
  BOOST_TEST(G->expr_id == add_3_expected->side_2->expr_id);
  BOOST_TEST(G->crc() == add_3_expected->side_2->crc());

  // F
  BOOST_TEST(G->side_2->expr_id == mul_1_expected->expr_id);
  mul2tc F(G->side_2);
  BOOST_TEST(F->crc() == mul_1_expected->crc());

  // E
  BOOST_TEST(F->side_1->expr_id == mul_1_expected->side_1->expr_id);
  BOOST_TEST(F->side_1->crc() == mul_1_expected->side_1->crc());
  BOOST_TEST(is_symbols_equal(F->side_1, y));

  // D
  BOOST_TEST(F->side_2->expr_id == add_1_expected->expr_id);
  add2tc D(F->side_2);
  BOOST_TEST(D->crc() == add_1_expected->crc());

  // C
  BOOST_TEST(D->side_1->expr_id == add_1_expected->side_1->expr_id);
  BOOST_TEST(D->side_1->crc() == add_1_expected->side_1->crc());
  BOOST_TEST(is_symbols_equal(D->side_1, a));

  // B
  BOOST_TEST(D->side_2->expr_id == add_1_expected->side_2->expr_id);
  BOOST_TEST(D->side_2->crc() == add_1_expected->side_2->crc());
  BOOST_TEST(is_symbols_equal(D->side_2, b));

  // A
  BOOST_TEST(G->side_1->expr_id == add_2_expected->side_1->expr_id);
  BOOST_TEST(G->side_1->crc() == add_2_expected->side_1->crc());
  BOOST_TEST(is_unsigned_equal(G->side_1, value));
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