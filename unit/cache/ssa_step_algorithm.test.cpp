/*******************************************************************
 Module: SSA Step Algorithm unit tests

 Author: Rafael Sá Menezes

 Date: January 2019

 Test Plan:
   - Constructors
   - Base Methods
   - Integrity/Atomicity
 \*******************************************************************/

#define BOOST_TEST_MODULE "SSA Step Algorithm"

#include <cache/ssa_step_algorithm.h>
#include <esbmc/esbmc_parseoptions.h>
#include <boost/test/included/unit_test.hpp>
namespace utf = boost::unit_test;

void generate_ssa_steps_1(symex_target_equationt::SSA_stepst &output)
{
  symex_target_equationt::SSA_stept step1;
  step1.type = goto_trace_stept::ASSERT;
  output.push_back(step1);
}

// ******************** TESTS ********************

// ** Helpers
// Check whether the helpers objects are ok

BOOST_AUTO_TEST_SUITE(helpers)
BOOST_AUTO_TEST_CASE(generate_ssa_steps_1_size)
{
  symex_target_equationt::SSA_stepst ssa_steps;
  generate_ssa_steps_1(ssa_steps);
  size_t actual = ssa_steps.size();
  size_t expected = 1;
  BOOST_TEST(expected == actual);
}

BOOST_AUTO_TEST_SUITE_END()
