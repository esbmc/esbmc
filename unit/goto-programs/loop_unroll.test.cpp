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
    program P = goto_factory::get_goto_functions(empty);
    auto &goto_function = P.functions;
    unsigned functions = 0;
    Forall_goto_functions (it, goto_function)
    {
      functions++;
    }
    REQUIRE(functions == 0);
  }
  GIVEN("A loopless goto-functions")
  {
    std::istringstream src(
      "int main() {"
      "int a = nondet_int();"
      "return a;"
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;

    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 0);
  }
  GIVEN("An unbounded loop")
  {
    std::istringstream src(
      "int main() {"
      "while(1) __ESBMC_assert(1,\"\");"
      "return 0;"
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;
    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 1);
    REQUIRE(unwind_loops.get_number_of_bounded_loops() == 0);
  }
  GIVEN("A bounded crescent-for loop without control-flow")
  {
    std::istringstream src(
      "int main() { "
      "  int a; "
      "  for(int i = 0; i < 5; i++) a = i; "
      "  return 0; "
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;
    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 1);
    REQUIRE(unwind_loops.get_number_of_bounded_loops() == 1);
  }
  GIVEN("A bounded incremental-for loop with control-flow")
  {
    std::istringstream src(
      "int main() { "
      "  int a; "
      "  for(int i = 0; i < 5; i++) "
      "    if(i == 2) a = 3;"
      "  return 0; "
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;
    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 1);
    REQUIRE(unwind_loops.get_number_of_bounded_loops() == 1);
  }
  GIVEN("A bounded incremental-for loop with inner-loop")
  {
    std::istringstream src(
      "int main() { "
      "  int a; "
      "  for(int i = 0; i < 5; i++) "
      "    for(int j = 0; j < 4; j++) a = 4;"
      "  return 0; "
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;
    bounded_loop_unroller unwind_loops;
    unwind_loops.run(goto_functions);

    REQUIRE(unwind_loops.get_number_of_functions() > 0);
    REQUIRE(unwind_loops.get_number_of_loops() == 2);
    REQUIRE(unwind_loops.get_number_of_bounded_loops() == 2);
  }
}

// ** Intrinsic unroller
// Check whether apply_intrinsic_unroller correctly annotates loops
// that are preceded by __ESBMC_unroll(N) calls.

static unsigned count_pragma_unroll_instructions(goto_functionst &goto_functions)
{
  unsigned count = 0;
  Forall_goto_functions(fit, goto_functions)
  {
    forall_goto_program_instructions(iit, fit->second.body)
    {
      if(iit->pragma_unroll_count > 0)
        count++;
    }
  }
  return count;
}

static unsigned max_pragma_unroll_count(goto_functionst &goto_functions)
{
  unsigned max_count = 0;
  Forall_goto_functions(fit, goto_functions)
  {
    forall_goto_program_instructions(iit, fit->second.body)
    {
      if(iit->pragma_unroll_count > max_count)
        max_count = iit->pragma_unroll_count;
    }
  }
  return max_count;
}

SCENARIO(
  "the intrinsic unroller detects __ESBMC_unroll calls",
  "[algorithms]")
{
  GIVEN("An empty goto-functions")
  {
    std::istringstream empty("");
    program P = goto_factory::get_goto_functions(empty);
    auto &goto_functions = P.functions;

    apply_intrinsic_unroller unroller;
    unroller.run(goto_functions);

    REQUIRE(count_pragma_unroll_instructions(goto_functions) == 0);
  }

  GIVEN("A loopless goto-functions")
  {
    std::istringstream src(
      "int main() {"
      "  int a = 42;"
      "  return a;"
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;

    apply_intrinsic_unroller unroller;
    unroller.run(goto_functions);

    REQUIRE(count_pragma_unroll_instructions(goto_functions) == 0);
  }

  GIVEN("A for loop without __ESBMC_unroll")
  {
    std::istringstream src(
      "int main() {"
      "  int a = 0;"
      "  for(int i = 0; i < 5; i++) a = i;"
      "  return 0;"
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;

    apply_intrinsic_unroller unroller;
    unroller.run(goto_functions);

    REQUIRE(count_pragma_unroll_instructions(goto_functions) == 0);
  }

  GIVEN("A for loop preceded by __ESBMC_unroll(5)")
  {
    std::istringstream src(
      "void __ESBMC_unroll(int N);"
      "int main() {"
      "  int a = 0;"
      "  __ESBMC_unroll(5);"
      "  for(int i = 0; i < 5; i++) a = i;"
      "  return 0;"
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;

    apply_intrinsic_unroller unroller;
    unroller.run(goto_functions);

    // pragma_unroll_count = 5 + 1 = 6
    REQUIRE(count_pragma_unroll_instructions(goto_functions) == 1);
    REQUIRE(max_pragma_unroll_count(goto_functions) == 6);
  }

  GIVEN("A while loop preceded by __ESBMC_unroll(10)")
  {
    std::istringstream src(
      "void __ESBMC_unroll(int N);"
      "int main() {"
      "  int i = 0;"
      "  __ESBMC_unroll(10);"
      "  while(i < 10) i++;"
      "  return 0;"
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;

    apply_intrinsic_unroller unroller;
    unroller.run(goto_functions);

    // pragma_unroll_count = 10 + 1 = 11
    REQUIRE(count_pragma_unroll_instructions(goto_functions) == 1);
    REQUIRE(max_pragma_unroll_count(goto_functions) == 11);
  }

  GIVEN("Two functions each with __ESBMC_unroll before a loop")
  {
    std::istringstream src(
      "void __ESBMC_unroll(int N);"
      "int foo() {"
      "  int i = 0;"
      "  __ESBMC_unroll(10);"
      "  while(i < 10) i++;"
      "  return i;"
      "}"
      "int main() {"
      "  __ESBMC_unroll(5);"
      "  for(int i = 0; i < 5; i++) ;"
      "  foo();"
      "  return 0;"
      "}");
    program P = goto_factory::get_goto_functions(src);
    auto &goto_functions = P.functions;

    apply_intrinsic_unroller unroller;
    unroller.run(goto_functions);

    // Two loops annotated: pragma_unroll_count 6 and 11
    REQUIRE(count_pragma_unroll_instructions(goto_functions) == 2);
    REQUIRE(max_pragma_unroll_count(goto_functions) == 11);
  }
}
