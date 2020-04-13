/*******************************************************************\
 Module: Expressions Green Normal Form fuzz tests

 Author: Rafael Sá Menezes

 Date: April 2020

 Fuzz Plan:

 1 - Generate input
 2 - Generate expr from input, adapting it to the expected form
 3 - Copy the original expr and injects the correct form
 4 - Apply normal form in 2
 5 - Compare crc from 3 with 4
\*******************************************************************/

#include <cache/algorithms/expr_green_normal_form.h>
#include <cstdlib>
#include "cache_fuzz_utils.h"

extern "C" int LLVMFuzzerTestOneInput(const char *Data, size_t Size)
{
  std::string input(Data);
  expr_generator_fuzzer fuzzer(input);
  if(fuzzer.is_expr_valid())
  {
    // 1 - Generate input
    expr2tc original = fuzzer.convert_input_to_expression();
    const auto original_crc = original->crc();
    expr2tc expected = fuzzer.convert_input_to_expression();

    expr2tc actual = fuzzer.convert_input_to_expression();

    /*
     * 2 - Generate expr from input, adapting it to the expected form
     *
     * First set the RHS to the last value of the input, since the original
     * RHS will be discarded anyway.
     *
     * Then we add the LHS of the expression with a value, which in this case
     * will be the summation of all values of input
     */

    const int rhs_value = input[Size - 1];

    auto actual_relation = std::dynamic_pointer_cast<equality2t>(actual);
    assert(actual_relation); // TODO: Add support to other relations

    actual_relation->side_2 = create_signed_32_value_expr(rhs_value);

    int lhs_value = 0;

    /**
     * @warning This could overlap back to zero if Size is a max value,
     *          by default, libFuzzer has a 100 as the dafault limit,
     *          be careful when overriding it
     */
    for(size_t i = 0; i < Size; i++)
    {
      lhs_value += Data[i];
    }

    auto lhs_value_expr = create_signed_32_value_expr(lhs_value);
    auto add_lhs_actual =
      create_signed_32_add_expr(actual_relation->side_1, lhs_value_expr);
    actual_relation->side_1 = add_lhs_actual;

    // Assert that this did not manipulate the original expression in any way
    assert(original->crc() == original_crc);
    // Assert that the actual indeed changed
    assert(original->crc() != actual->crc());

    /*
     * Copy the original expr and injects the correct form
     *
     * First set the RHS to 0
     *
     * Then we add the LHS of the expression with a value, which will be
     * lhs_value - rhs_value
     */

    auto expected_relation = std::dynamic_pointer_cast<equality2t>(expected);
    assert(expected_relation); // TODO: Add support to other relations
    expected_relation->side_2 = create_signed_32_value_expr(0);

    auto expected_value_expr =
      create_signed_32_value_expr(lhs_value - rhs_value);
    auto add_lhs_expected =
      create_signed_32_add_expr(expected_relation->side_1, expected_value_expr);
    expected_relation->side_1 = add_lhs_expected;

    // Just a minor check to check if expected is indeed different that actual
    if(rhs_value != 0)
      assert(actual->crc() != expected->crc());

    /**
     * 4 - Apply normal form in 2
     */

    expr_green_normal_form f(actual);
    f.run();

    // Check if expected == actual
    assert(actual->crc() == expected->crc());
  }

  return 0;
}