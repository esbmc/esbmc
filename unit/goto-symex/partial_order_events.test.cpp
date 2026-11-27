/*******************************************************************
 Module: --partial-order event steps (#10990)

 The events symex records under --partial-order carry no constraint of their
 own: the memory-model pass orders them. These tests pin how the equation
 stores them and that the step passes route them to run_on_event.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sstream>
#include <vector>

#include <goto-symex/equation/ssa_step_algorithm.h>
#include <goto-symex/equation/symex_target_equation.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

namespace
{
using stept = symex_targett::step_typet;

const std::vector<stept> event_types{
  symex_targett::SHARED_READ,
  symex_targett::SHARED_WRITE,
  symex_targett::SPAWN,
  symex_targett::JOIN,
  symex_targett::ATOMIC_BEGIN,
  symex_targett::ATOMIC_END};

class event_counter : public ssa_step_algorithm
{
public:
  event_counter() : ssa_step_algorithm(false)
  {
  }

  bool run(symex_target_equationt::SSA_stepst &steps) override
  {
    for (auto &step : steps)
      run_on_step(step);
    return true;
  }

  BigInt ignored() const override
  {
    return 0;
  }

  void run_on_event(symex_target_equationt::SSA_stept &step) override
  {
    seen.push_back(step.type);
  }

  void run_on_skip(symex_target_equationt::SSA_stept &) override
  {
    ++skips;
  }

  std::vector<stept> seen;
  unsigned skips = 0;
};

expr2tc shared_x()
{
  return symbol2tc(get_int32_type(), "c:@x");
}
} // namespace

TEST_CASE("event() records each event type", "[partial-order]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);

  for (stept type : event_types)
    eq.event(gen_true_expr(), type, shared_x(), symex_targett::sourcet());

  REQUIRE(eq.SSA_steps.size() == event_types.size());
  auto it = eq.SSA_steps.begin();
  for (stept type : event_types)
  {
    CHECK(it->type == type);
    CHECK(it->is_event());
    CHECK(it->lhs == shared_x());
    CHECK(is_true(it->guard));
    ++it;
  }
}

TEST_CASE("only event types are events", "[partial-order]")
{
  symex_target_equationt::SSA_stept step;
  for (stept type :
       {symex_targett::ASSIGNMENT,
        symex_targett::ASSUME,
        symex_targett::ASSERT,
        symex_targett::OUTPUT,
        symex_targett::SKIP,
        symex_targett::RENUMBER,
        symex_targett::BREANCHING})
  {
    step.type = type;
    CHECK_FALSE(step.is_event());
  }
}

TEST_CASE("ssa_step_algorithm routes events to run_on_event", "[partial-order]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);

  for (stept type : event_types)
    eq.event(gen_true_expr(), type, expr2tc(), symex_targett::sourcet());
  eq.SSA_steps.emplace_back();
  eq.SSA_steps.back().type = symex_targett::SKIP;

  event_counter counter;
  counter.run(eq.SSA_steps);

  CHECK(counter.seen == event_types);
  CHECK(counter.skips == 1);
}

TEST_CASE("an event step prints its type and symbol", "[partial-order]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);

  eq.event(
    gen_true_expr(),
    symex_targett::SHARED_WRITE,
    shared_x(),
    symex_targett::sourcet());
  eq.event(
    gen_true_expr(),
    symex_targett::ATOMIC_BEGIN,
    expr2tc(),
    symex_targett::sourcet());
  eq.event(
    gen_true_expr(),
    symex_targett::JOIN,
    constant_int2tc(get_uint32_type(), BigInt(2)),
    symex_targett::sourcet());

  std::ostringstream out;
  eq.output(out);
  CHECK(out.str().find("SHARED_WRITE") != std::string::npos);
  CHECK(out.str().find("ATOMIC_BEGIN") != std::string::npos);
  CHECK(out.str().find("JOIN 2") != std::string::npos);
}

TEST_CASE("an update stores and prints the value it updates", "[partial-order]")
{
  contextt ctx;
  namespacet ns(ctx);
  symex_target_equationt eq(ns);

  eq.event(
    gen_true_expr(),
    symex_targett::SHARED_WRITE,
    shared_x(),
    symex_targett::sourcet(),
    shared_x());
  CHECK(eq.SSA_steps.back().rhs == shared_x());

  std::ostringstream out;
  eq.output(out);
  CHECK(out.str().find(" updating ") != std::string::npos);
}
