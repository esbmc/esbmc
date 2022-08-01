/*******************************************************************
 Module: Goto Programs algorithms unit test

 Author: Rafael Sá Menezes

 Date: May 2021

 Test Plan:
   - Bounded loop unroller.
 \*******************************************************************/

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include "../testing-utils/goto_factory.h"
#include <goto-programs/loop_unroll.h>

// ** Bounded loop unroller
// Check whether the object converts a loop properly

SCENARIO("the loop unroller detects bounded loops", "[algorithms]")
{
  GIVEN("An empty goto-functions")
  {
    std::istringstream empty("");
    auto goto_function = goto_factory::get_goto_functions(empty);
    unsigned functions = 0;
    Forall_goto_functions(it, goto_function)
    {
      functions++;
    }
    REQUIRE(functions == 0);
  }
  GIVEN("A loopless goto-functions")
  {
    std::istringstream program(
      "int main() {"
      "int a = nondet_int();"
      "return a;"
      "}");
    auto goto_functions = goto_factory::get_goto_functions(program);

    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 0);
  }
  GIVEN("An unbounded loop")
  {
    std::istringstream program(
      "int main() {"
      "while(1) __ESBMC_assert(1,\"\");"
      "return 0;"
      "}");
    auto goto_functions = goto_factory::get_goto_functions(program);
    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 1);
    REQUIRE(unwind_loops.get_number_of_bounded_loops() == 0);
  }
  GIVEN("A bounded crescent-for loop without control-flow")
  {
    std::istringstream program(
      "int main() { "
      "  int a; "
      "  for(int i = 0; i < 5; i++) a = i; "
      "  return 0; "
      "}");
    auto goto_functions = goto_factory::get_goto_functions(program);
    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 1);
    REQUIRE(unwind_loops.get_number_of_bounded_loops() == 1);
  }
  GIVEN("A bounded incremental-for loop with control-flow")
  {
    std::istringstream program(
      "int main() { "
      "  int a; "
      "  for(int i = 0; i < 5; i++) "
      "    if(i == 2) a = 3;"
      "  return 0; "
      "}");
    auto goto_functions = goto_factory::get_goto_functions(program);
    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 1);
    REQUIRE(unwind_loops.get_number_of_bounded_loops() == 1);
  }
  GIVEN("A bounded incremental-for loop with inner-loop")
  {
    std::istringstream program(
      "int main() { "
      "  int a; "
      "  for(int i = 0; i < 5; i++) "
      "    for(int j = 0; j < 4; j++) a = 4;"
      "  return 0; "
      "}");
    auto goto_functions = goto_factory::get_goto_functions(program);
    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 2);
    REQUIRE(unwind_loops.get_number_of_bounded_loops() == 2);
  }
}
