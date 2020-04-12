/*******************************************************************
 Module: Green Normal Form unit tests

 Author: Rafael Sá Menezes

 Date: March 2019

 Test Plan:
   - A + B + C + ... + k == y -> A + B + C + ... + (k-y) == 0
   - A + B + C + ... + k != y -> A + B + C + ... + (k-y) != 0
   - A + B + C + ... + k <= y -> A + B + C + (k-y) <= 0
   - A + B + C + ... + k < y -> A + B + C + (k-y+1) <= 0
   - A + B + C + ... + k > y -> -(A + B + C + (k+y-1)) <= 0
   - A + B + C + ... + k >= y -> -(A + B + C + (k+y)) <= 0
 \*******************************************************************/

#define BOOST_TEST_MODULE "SSA Step Algorithm"

#include <cache/algorithms/green_normal_form.h>
#include <boost/test/included/unit_test.hpp>
#include "ssa_step_utils.h"
namespace utf = boost::unit_test;

namespace
{
}

// ******************** TESTS ********************

// ** A + B + C + ... + k == y -> A + B + C + ... + (k-y) == 0
// Check whether equalities with value are being canonized ok

BOOST_AUTO_TEST_SUITE(green_normal_form_equality)

BOOST_AUTO_TEST_CASE(equality_1_test)
{
  init_test_values();
  expr2tc expr = equality_1_ordered();
  expr->hash() crypto_hash

    symex_target_equationt::SSA_stepst ssa_steps;
  create_assumption(ssa_steps, expr);

  green_normal_form t(ssa_steps);
  t.run();

  expr2tc &actual = ssa_steps.front().cond;
  expr2tc expected = equality_1_green_normal();
  is_equality_1_equivalent(actual, expected);
}

BOOST_AUTO_TEST_SUITE_END()