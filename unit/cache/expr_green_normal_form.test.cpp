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

// ******************** TESTS ********************

// ** Check if trivial cases are ok
// expressions which does not contain a symbol/value or only one at max
BOOST_AUTO_TEST_SUITE(trivial)

BOOST_AUTO_TEST_CASE(equality_1)
{
  init_test_values();
  expr2tc actual = equality_1_ordered();
  expr2tc expected = equality_1_green_normal();

  BOOST_TEST(actual->crc() != expected->crc());
  expr_green_normal_form algorithm(actual);
  algorithm.run();

  is_equality_1_equivalent(actual, expected);
  BOOST_TEST(actual->crc() == expected->crc());
}

BOOST_AUTO_TEST_CASE(equality_2)
{
  init_test_values();
  expr2tc actual = equality_2_ordered();
  expr2tc expected = equality_2_green_normal();

  BOOST_TEST(actual->crc() == expected->crc());
  expr_green_normal_form algorithm(actual);
  algorithm.run();

  BOOST_TEST(actual->crc() == expected->crc());
}

BOOST_AUTO_TEST_CASE(equality_3)
{
  init_test_values();
  expr2tc actual = equality_3_ordered();
  expr2tc expected = equality_3_green_normal();

  BOOST_TEST(actual->crc() != expected->crc());
  expr_green_normal_form algorithm(actual);
  algorithm.run();

  BOOST_TEST(actual->crc() == expected->crc());
}

BOOST_AUTO_TEST_CASE(equality_4)
{
  init_test_values();
  expr2tc actual = equality_4_ordered();
  expr2tc expected = equality_4_green_normal();

  BOOST_TEST(actual->crc() == expected->crc());
  expr_green_normal_form algorithm(actual);
  algorithm.run();

  BOOST_TEST(actual->crc() == expected->crc());
}

BOOST_AUTO_TEST_SUITE_END()