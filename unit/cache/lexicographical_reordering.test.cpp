/*******************************************************************
 Module: Lexicographical_reordering unit tests

 Author: Rafael Sá Menezes

 Date: March 2019

 Test Plan:
   - Assignments
   - Assumptions
   - Assertives
 \*******************************************************************/

#define BOOST_TEST_MODULE "SSA Step Algorithm"

#include <cache/algorithms/lexicographical_reordering.h>
#include <boost/test/included/unit_test.hpp>
#include "ssa_step_utils.h"
namespace utf = boost::unit_test;

// ******************** TESTS ********************

// ** Assignment
// Check whether assignment are being reordered ok

BOOST_AUTO_TEST_SUITE(lexicographical_reordering_assignments)

BOOST_AUTO_TEST_CASE(a_add_b_should_become_a_add_b)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");
  add2tc add = create_unsigned_32_add_expr(a,b);

  symex_target_equationt::SSA_stepst ssa_steps;
  create_assignment(ssa_steps, add);

  lexicographical_reordering t(ssa_steps);
  t.run();
  expr2tc &rhs = ssa_steps.front().rhs;
  std::shared_ptr<arith_2ops> arith;
  arith = std::dynamic_pointer_cast<arith_2ops>(rhs);
  // TEST if after applying the algorithm side_1 still is 'a'
  //      and side_2 is still b
  BOOST_TEST(is_symbols_equal(a, arith->side_1));
  BOOST_TEST(is_symbols_equal(b, arith->side_2));
}

BOOST_AUTO_TEST_CASE(a_add_value_should_become_a_add_value)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  constant_int2tc b = create_unsigned_32_value_expr(42);
  add2tc add = create_unsigned_32_add_expr(a,b);

  symex_target_equationt::SSA_stepst ssa_steps;
  create_assignment(ssa_steps, add);

  lexicographical_reordering t(ssa_steps);
  t.run();
  expr2tc &rhs = ssa_steps.front().rhs;
  std::shared_ptr<arith_2ops> arith;
  arith = std::dynamic_pointer_cast<arith_2ops>(rhs);
  // TEST if after applying the algorithm side_1 still is 'a'
  //      and side_2 is still b
  BOOST_TEST(is_symbols_equal(a, arith->side_1));
  BOOST_TEST(is_unsigned_equal(b, arith->side_2));
}

BOOST_AUTO_TEST_CASE(value_add_a_should_become_a_add_value)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  constant_int2tc b = create_unsigned_32_value_expr(42);
  add2tc add = create_unsigned_32_add_expr(b,a);

  symex_target_equationt::SSA_stepst ssa_steps;
  create_assignment(ssa_steps, add);

  lexicographical_reordering t(ssa_steps);
  t.run();
  expr2tc &rhs = ssa_steps.front().rhs;
  std::shared_ptr<arith_2ops> arith;
  arith = std::dynamic_pointer_cast<arith_2ops>(rhs);
  // TEST if after applying the algorithm side_1 still is 'a'
  //      and side_2 is still b
  BOOST_TEST(is_symbols_equal(a, arith->side_1));
  BOOST_TEST(is_unsigned_equal(b, arith->side_2));
}

BOOST_AUTO_TEST_CASE(b_add_a_should_become_a_add_b)
{
  symbol2tc a = create_unsigned_32_symbol_expr("a");
  symbol2tc b = create_unsigned_32_symbol_expr("b");
  add2tc add = create_unsigned_32_add_expr(b,a);

  symex_target_equationt::SSA_stepst ssa_steps;
  create_assignment(ssa_steps, add);

  lexicographical_reordering t(ssa_steps);
  t.run();
  expr2tc &rhs = ssa_steps.front().rhs;
  std::shared_ptr<arith_2ops> arith;
  arith = std::dynamic_pointer_cast<arith_2ops>(rhs);
  // TEST if after applying the algorithm side_1 still is 'a'
  //      and side_2 is still b
  BOOST_TEST(is_symbols_equal(a, arith->side_1));
  BOOST_TEST(is_symbols_equal(b, arith->side_2));
}

BOOST_AUTO_TEST_SUITE_END()