/*******************************************************************
 Module: Green Normal Form unit tests

 Author: Rafael Sá Menezes

 Date: March 2019

 Test Plan:
 \*******************************************************************/

#define BOOST_TEST_MODULE "SSA Step Algorithm"

#include <cache/green/green_storage.h>
#include <boost/test/included/unit_test.hpp>
#include "cache_test_utils.h"
namespace utf = boost::unit_test;

namespace
{
}

// ******************** TESTS ********************

BOOST_AUTO_TEST_SUITE(green_storage_unordered_map)

BOOST_AUTO_TEST_CASE(expression_true_saved_ok)
{
  init_test_values();
  expr2tc expr = equality_1_ordered();
  green_storage gs;

  std::set<expr_hash> items;
  items.insert(green_storage::hash(*expr));
  gs.add(items);
  BOOST_CHECK(gs.get(items));
}

BOOST_AUTO_TEST_CASE(expression_default_ok)
{
  init_test_values();
  expr2tc expr = equality_1_ordered();
  green_storage gs;
  std::set<expr_hash> items;
  items.insert(green_storage::hash(*expr));
  BOOST_CHECK(!gs.get(items));
}

BOOST_AUTO_TEST_CASE(expression_multiple_false_true_ok)
{
  init_test_values();
  expr2tc expr1 = equality_1();
  expr2tc expr2 = equality_2();
  green_storage gs;

  std::set<expr_hash> items_1;
  items_1.insert(expr1->crc());

  std::set<expr_hash> items_2;
  items_2.insert(expr2->crc());

  gs.add(items_1);

  BOOST_CHECK(gs.get(items_1));
  BOOST_CHECK(!gs.get(items_2));
}

BOOST_AUTO_TEST_CASE(expression_manually_built_ok)
{
  init_test_values();
  // ((x + y) + -2) == 0
  expr2tc expr1 = equality_1_green_normal();

  symbol2tc Y = create_unsigned_32_symbol_expr("Y");
  symbol2tc X = create_unsigned_32_symbol_expr("X");
  constant_int2tc minus_two = create_signed_32_value_expr(-2);
  constant_int2tc zero = create_signed_32_value_expr(0);
  add2tc add_1 = create_signed_32_add_expr(X, Y);
  add2tc add_2 = create_signed_32_add_expr(add_1, minus_two);
  expr2tc expr2 = create_equality_relation(add_2, zero);

  std::set<expr_hash> items_1;
  items_1.insert(expr1->crc());

  std::set<expr_hash> items_2;
  items_2.insert(expr2->crc());

  green_storage gs;
  gs.add(items_1);
  gs.add(items_2);

  BOOST_CHECK(gs.get(items_1));
  BOOST_CHECK(gs.get(items_2));
}

BOOST_AUTO_TEST_CASE(expression_manually_built_inner_expr)
{
  /*
   * This is a big test, it will:
   *
   * 1. Add (X < 0) AND (X>0) AS UNSAT
   * 2. Check whether (X < 0) AND (X>0) is UNSAT
   * 3. Check whether X < 0 is unknown
   * 4. Check whether X > 0 is unknown
   * 5. Check whether (X < 0) AND (X>0) AND (X == 0) is UNSAT
  */

  symbol2tc X = create_unsigned_32_symbol_expr("X");
  constant_int2tc zero = create_signed_32_value_expr(0);

  // expr1 -> X < 0
  expr2tc expr1 = create_lesser_relation(X, zero);
  // expr2 -> X > 0
  expr2tc expr2 = create_greater_relation(X, zero);
  // expr3 -> X == 0
  expr2tc expr3 = create_equality_relation(X, zero);

  // X < 0 AND X > 0
  std::set<expr_hash> items_1;
  items_1.insert(green_storage::hash(*expr1));
  items_1.insert(green_storage::hash(*expr2));
  // X < 0
  std::set<expr_hash> items_2;
  items_2.insert(green_storage::hash(*expr1));
  // X > 0
  std::set<expr_hash> items_3;
  items_3.insert(green_storage::hash(*expr2));
  // X < 0 AND X > 0 AND X == 0
  std::set<expr_hash> items_4;
  items_4.insert(green_storage::hash(*expr1));
  items_4.insert(green_storage::hash(*expr2));
  items_4.insert(green_storage::hash(*expr3));

  green_storage gs;
  // 1. Add (X < 0) AND (X>0) AS UNSAT
  gs.add(items_1);

  // 2. Check if (X < 0) AND (X>0) is UNSAT
  BOOST_CHECK(gs.get(items_1));
  // 3. Check if X < 0 is unknown
  BOOST_CHECK(!gs.get(items_2));
  // 4. Check if X > 0 is unknown
  BOOST_CHECK(!gs.get(items_3));
  // 5. Check if (X < 0) AND (X>0) AND (X == 0) is UNSAT
  BOOST_CHECK(gs.get(items_4));
}

BOOST_AUTO_TEST_SUITE_END()