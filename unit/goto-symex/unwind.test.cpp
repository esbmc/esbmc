/*******************************************************************
 Module: Unwind bounding on the real engine

 Tier B of docs/roadmap/goto-symex-verification-plan.md (H-A5, M2).

 `get_unwind` decides *when* a loop stops unwinding and `loop_bound_exceeded`
 decides *what the truncation means* (symex_goto.cpp:497,525). Both are P1: a
 bound applied to the wrong loop silently verifies less than the user asked
 for, and a truncation that fails to strengthen the state guard would let the
 code after the loop run on a path the loop never reached.

 §11.3 and R12 forbid pairing `--no-unwinding-assertions` with a reachability
 claim, so none of these cases makes one: every assertion here is a count or a
 presence check over the equation symex produced.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <string>

#include <goto-programs/goto_functions.h>
#include <goto-symex/reachability_tree.h>
#include <irep2/irep2_expr.h>
#include <util/symtab/namespace.h>

#include "../testing-utils/goto_factory.h"

namespace
{
/** A loop whose condition constant-folds, run under an explicit bound. */
const char *const long_loop = R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  int i = 0;
  while (i < 100)
  {
    x = nondet_int();
    i = i + 1;
  }
  int after = nondet_int();
  return x + after;
}
)";

/** The same shape, but exhausted after three iterations. */
const char *const short_loop = R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  int i = 0;
  while (i < 3)
  {
    x = nondet_int();
    i = i + 1;
  }
  return x;
}
)";

class engine
{
public:
  explicit engine(std::string src)
    : source(std::move(src)),
      prog(goto_factory::get_goto_functions(
        source,
        goto_factory::Architecture::BIT_64)),
      ns(prog.context),
      opts(goto_factory::get_default_options(
        goto_factory::get_default_cmdline("test.c"))),
      rt(
        prog.functions,
        ns,
        opts,
        std::make_shared<symex_target_equationt>(ns),
        prog.context)
  {
  }

  /** Options must be set before run(): goto_symext reads them in its ctor. */
  engine &with(const std::string &key, const std::string &value)
  {
    opts.set_option(key, value);
    return *this;
  }

  /** The id `--unwindset` keys on, read from the program rather than assumed. */
  unsigned main_loop_number() const
  {
    for (const auto &[name, func] : prog.functions.function_map)
    {
      if (name.as_string().find("@F@main") == std::string::npos)
        continue;
      for (const auto &insn : func.body.instructions)
        if (insn.is_backwards_goto())
          return insn.loop_number;
    }
    FAIL("main has no loop");
    return 0;
  }

  std::shared_ptr<symex_target_equationt> run()
  {
    rt.setup_for_new_explore();
    auto eq = std::dynamic_pointer_cast<symex_target_equationt>(
      rt.get_next_formula().target);
    REQUIRE(eq != nullptr);
    return eq;
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
};

size_t assignments_to(const symex_target_equationt &eq, const char *var)
{
  size_t found = 0;
  for (const auto &step : eq.SSA_steps)
    if (
      step.is_assignment() && is_symbol2t(step.lhs) &&
      to_symbol2t(step.lhs).thename.as_string().find(var) != std::string::npos)
      ++found;
  return found;
}

/** Body executions: `x` is written exactly once per iteration, plus its init. */
size_t iterations(const symex_target_equationt &eq)
{
  const size_t writes = assignments_to(eq, "@main@x");
  REQUIRE(writes >= 1);
  return writes - 1;
}

size_t unwinding_assertions(const symex_target_equationt &eq)
{
  size_t found = 0;
  for (const auto &step : eq.SSA_steps)
    if (
      step.is_assert() &&
      step.comment.as_string().rfind("unwinding assertion loop", 0) == 0)
      ++found;
  return found;
}

size_t assumptions(const symex_target_equationt &eq)
{
  size_t found = 0;
  for (const auto &step : eq.SSA_steps)
    if (step.is_assume())
      ++found;
  return found;
}
} // namespace

TEST_CASE("the global bound stops the loop and claims the truncation")
{
  engine e(long_loop);
  e.with("unwind", "5");

  auto eq = e.run();
  REQUIRE(iterations(*eq) == 5);
  REQUIRE(unwinding_assertions(*eq) == 1);
}

TEST_CASE("a function-specific bound overrides the global one")
{
  // A5.1, first half: unwind_func_set is consulted before max_unwind is used.
  engine e(long_loop);
  e.with("unwind", "5").with("unwindsetname", "main:0:3");

  auto eq = e.run();
  REQUIRE(iterations(*eq) == 3);
  REQUIRE(unwinding_assertions(*eq) == 1);
}

TEST_CASE("a loop-specific bound overrides the function-specific one")
{
  // A5.1, second half. All three bounds are distinct, so a precedence swap
  // lands on a different count rather than an equal one.
  engine e(long_loop);
  const std::string id = std::to_string(e.main_loop_number());
  e.with("unwind", "5")
    .with("unwindsetname", "main:0:3")
    .with("unwindset", id + ":2");

  auto eq = e.run();
  REQUIRE(iterations(*eq) == 2);
  REQUIRE(unwinding_assertions(*eq) == 1);
}

TEST_CASE("a zero bound means unbounded, not zero iterations")
{
  // A5.2: `this_loop_max_unwind != 0` gates the whole comparison, so a
  // loop-specific 0 is the documented way to exempt one loop from --unwind.
  engine e(short_loop);
  const std::string id = std::to_string(e.main_loop_number());
  e.with("unwind", "2").with("unwindset", id + ":0");

  auto eq = e.run();
  REQUIRE(iterations(*eq) == 3);
  REQUIRE(unwinding_assertions(*eq) == 0);
}

TEST_CASE("the same loop truncates when the exemption is dropped")
{
  // Non-vacuity twin of the case above: three iterations there is the loop
  // running to exhaustion, not the bound being ignored for another reason.
  engine e(short_loop);
  e.with("unwind", "2");

  auto eq = e.run();
  REQUIRE(iterations(*eq) == 2);
  REQUIRE(unwinding_assertions(*eq) == 1);
}

// Case names must not start with `--`: Catch2 registers them as CLI tokens.
TEST_CASE("no-unwinding-assertions trades the claim for an assumption")
{
  // A5.3: exactly one of the three arms of loop_bound_exceeded is taken. The
  // assumption arm is reachable *only* under no_unwinding_assertions, so the
  // claim disappearing and an assumption appearing must happen together.
  engine base(long_loop);
  base.with("unwind", "3");
  auto claimed = base.run();

  engine assumed_e(long_loop);
  assumed_e.with("unwind", "3").with("no-unwinding-assertions", "1");
  auto assumed = assumed_e.run();

  REQUIRE(unwinding_assertions(*claimed) == 1);
  REQUIRE(unwinding_assertions(*assumed) == 0);
  REQUIRE(assumptions(*assumed) == assumptions(*claimed) + 1);
  REQUIRE(iterations(*assumed) == iterations(*claimed));
}

TEST_CASE("partial-loops takes neither arm")
{
  // A5.3, third arm: loop_bound_exceeded returns before emitting anything.
  engine base(long_loop);
  base.with("unwind", "3");
  auto claimed = base.run();

  engine partial_e(long_loop);
  partial_e.with("unwind", "3").with("partial-loops", "1");
  auto partial = partial_e.run();

  REQUIRE(unwinding_assertions(*partial) == 0);
  REQUIRE(assumptions(*partial) == assumptions(*claimed));
  REQUIRE(iterations(*partial) == iterations(*claimed));
}

TEST_CASE("truncation strengthens the state guard by the negated condition")
{
  // A5.4. `i < 100` is still true at the bound, so the ¬cond that
  // loop_bound_exceeded adds is false and everything after the loop becomes
  // unreachable. Under --partial-loops that guard is never added, and the
  // post-loop code is emitted — which is the whole reason partial loops are
  // unsound for reachability. The difference *is* the guard strengthening.
  engine base(long_loop);
  base.with("unwind", "3");
  auto claimed = base.run();

  engine partial_e(long_loop);
  partial_e.with("unwind", "3").with("partial-loops", "1");
  auto partial = partial_e.run();

  REQUIRE(assignments_to(*claimed, "@main@after") == 0);
  REQUIRE(assignments_to(*partial, "@main@after") == 1);
}
