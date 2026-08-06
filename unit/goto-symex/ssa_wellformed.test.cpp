/*******************************************************************
 Module: SSA well-formedness of equations produced by the real engine

 Tier B of docs/roadmap/goto-symex-verification-plan.md (H-B1, milestones
 M1/M4). The checks themselves live in `ssa_validator.h` so the other Tier-B
 tests can assert them over the equations they already build (§7.2); this file
 exercises them over programs chosen to cover the control-flow shapes.

 Unlike a Tier-A harness, nothing here is transcribed: `goto_factory` parses a
 real C program, a real `reachability_treet` drives the real `goto_symext` over
 it, and the assertions are made against the `symex_target_equationt` that comes
 out.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <algorithm>
#include <string>

#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
#include <util/symtab/namespace.h>

#include "ssa_validator.h"
#include "../testing-utils/goto_factory.h"

namespace
{
using symex_ssa::is_ssa_symbol;
using symex_ssa::validate;
using symex_ssa::violationst;

/** Run the real engine over `src` and return the equation it produced. */
std::shared_ptr<symex_target_equationt> symex_equation(std::string src)
{
  program prog =
    goto_factory::get_goto_functions(src, goto_factory::Architecture::BIT_64);
  REQUIRE(prog.functions.function_map.size() > 0);

  namespacet ns(prog.context);
  cmdlinet cmd = goto_factory::get_default_cmdline("test.c");
  optionst opts = goto_factory::get_default_options(cmd);
  opts.set_option("unwind", "4");

  reachability_treet rt(
    prog.functions,
    ns,
    opts,
    std::make_shared<symex_target_equationt>(ns),
    prog.context);
  rt.setup_for_new_explore();

  goto_symext::symex_resultt result = rt.get_next_formula();
  auto eq = std::dynamic_pointer_cast<symex_target_equationt>(result.target);
  REQUIRE(eq != nullptr);
  REQUIRE(eq->SSA_steps.size() > 0);
  return eq;
}

/** Every property at once, so a new program costs one line. */
void require_well_formed(const std::string &src)
{
  symex_ssa::require_well_formed(*symex_equation(src));
}
} // namespace

TEST_CASE("straight-line assignments produce well-formed SSA", "[symex][ssa]")
{
  require_well_formed(R"(
int main(void)
{
  int x = 1;
  int y = x + 2;
  x = y * 3;
  y = x - y;
  return x + y;
}
)");
}

TEST_CASE("a branch and its join produce well-formed SSA", "[symex][ssa]")
{
  require_well_formed(R"(
int nondet_int(void);
int main(void)
{
  int x = 0, y = 0;
  if (nondet_int() > 0)
  {
    x = 1;
    y = x + 1;
  }
  else
  {
    x = 2;
  }
  return x + y;
}
)");
}

TEST_CASE(
  "nested branches and a shared join produce well-formed SSA",
  "[symex][ssa]")
{
  require_well_formed(R"(
int nondet_int(void);
int main(void)
{
  int x = 0;
  if (nondet_int() > 0)
  {
    if (nondet_int() > 1)
      x = 1;
    else
      x = 2;
  }
  else if (nondet_int() < -1)
    x = 3;
  return x;
}
)");
}

TEST_CASE("an unwound loop produces well-formed SSA", "[symex][ssa]")
{
  require_well_formed(R"(
int main(void)
{
  int sum = 0;
  for (int i = 0; i < 3; i++)
    sum += i;
  return sum;
}
)");
}

TEST_CASE(
  "function calls and recursion produce well-formed SSA",
  "[symex][ssa]")
{
  require_well_formed(R"(
int add(int a, int b) { return a + b; }
int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
int main(void)
{
  int x = add(2, 3);
  return add(x, fact(3));
}
)");
}

TEST_CASE("the validator is discriminating", "[symex][ssa]")
{
  // The three checks are only worth running if a violation would be seen.
  // Re-defining an SSA name in an equation the engine produced must be caught;
  // this is the check R5 says `check_for_duplicate_assigns` never performs.
  auto eq = symex_equation(R"(
int main(void)
{
  int x = 1;
  x = x + 1;
  return x;
}
)");

  REQUIRE(validate(*eq).duplicate_definitions.empty());

  auto assignment = std::find_if(
    eq->SSA_steps.begin(), eq->SSA_steps.end(), [](const auto &step) {
      return step.is_assignment() && is_ssa_symbol(step.lhs);
    });
  REQUIRE(assignment != eq->SSA_steps.end());

  eq->SSA_steps.push_back(*assignment);
  const violationst bad = validate(*eq);
  CHECK(bad.duplicate_definitions.size() == 1);
  CHECK(bad.non_monotonic.size() == 1);
}

TEST_CASE("the shipped I10 detector reports a duplicate", "[symex][ssa]")
{
  // Otherwise the repaired detector is pinned only by a KNOWNBUG whose input
  // is slated to be fixed (R14), leaving its ability to detect unguarded.
  auto eq = symex_equation(R"(
int main(void)
{
  int x = 1;
  x = x + 1;
  return x;
}
)");

  REQUIRE(eq->check_for_duplicate_assigns());

  auto assignment = std::find_if(
    eq->SSA_steps.begin(), eq->SSA_steps.end(), [](const auto &step) {
      return step.is_assignment() && is_ssa_symbol(step.lhs);
    });
  REQUIRE(assignment != eq->SSA_steps.end());

  eq->SSA_steps.push_back(*assignment);
  CHECK_FALSE(eq->check_for_duplicate_assigns());
}
