/*******************************************************************
 Module: Path merging on the real engine

 Tier B of docs/roadmap/goto-symex-verification-plan.md (H-A2 / H-A3, M2).

 §4.3 ranks the merge machinery P0: a lost path is a missed bug with no
 diagnostic anywhere. Two of its invariants are unenforced in the shipped
 binary, and both are visible in the equation symex produces:

   I8  (H-A2) `phi_function` emits an `ite` for a variable exactly when its L2
       index differs between the two states, and the emitted definition is
       fresh — it must not alias either value it selects between.
   I6  (H-A3, finding R2) every `merge_statet` pushed at a join is consumed by
       exactly one `merge_gotos`. `pop_frame` asserts the frame's
       `merge_state_map` is empty; that assert is a no-op under NDEBUG, so in
       the shipped binary a frame popped with pending merges drops those paths
       silently.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <set>
#include <string>

#include <goto-symex/reachability_tree.h>
#include <irep2/irep2_expr.h>
#include <util/symtab/namespace.h>

#include "ssa_validator.h"
#include "../testing-utils/goto_factory.h"

namespace
{
using symex_ssa::is_ssa_symbol;

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
    opts.set_option("unwind", "4");
    rt.setup_for_new_explore();
  }

  std::shared_ptr<symex_target_equationt> run()
  {
    auto eq = std::dynamic_pointer_cast<symex_target_equationt>(
      rt.get_next_formula().target);
    REQUIRE(eq != nullptr);
    // §7.2: every equation a Tier-B test builds is another I1/I10/P11 sample.
    symex_ssa::require_well_formed(*eq);
    return eq;
  }

  /** Merge snapshots still pending on every live frame (I6 / R2). */
  size_t pending_merges()
  {
    const auto &stack = rt.get_cur_state().get_active_state().call_stack;
    // Guard against a vacuous zero: no frames would mean nothing was examined.
    REQUIRE(!stack.empty());
    size_t pending = 0;
    for (const auto &frame : stack)
      pending += frame.merge_state_map.size();
    return pending;
  }

private:
  std::string source;
  program prog;
  namespacet ns;
  optionst opts;
  reachability_treet rt;
};

bool defines(const symex_target_equationt::SSA_stept &step, const char *var)
{
  return step.is_assignment() && is_ssa_symbol(step.lhs) &&
         to_symbol2t(step.lhs).thename.as_string().find(var) !=
           std::string::npos;
}

/** Assignments to `var` whose rhs is an ite — phi_function's output shape. */
std::vector<const symex_target_equationt::SSA_stept *>
phis_for(const symex_target_equationt &eq, const char *var)
{
  std::vector<const symex_target_equationt::SSA_stept *> found;
  for (const auto &step : eq.SSA_steps)
    if (defines(step, var) && !is_nil_expr(step.rhs) && is_if2t(step.rhs))
      found.push_back(&step);
  return found;
}

/** Highest L2 index defined for `var` strictly before `step`. */
unsigned highest_index_before(
  const symex_target_equationt &eq,
  const char *var,
  const symex_target_equationt::SSA_stept *step)
{
  unsigned highest = 0;
  for (const auto &s : eq.SSA_steps)
  {
    if (&s == step)
      break;
    if (defines(s, var))
      highest = std::max(highest, to_symbol2t(s.lhs).level2_num);
  }
  return highest;
}
} // namespace

TEST_CASE("a two-armed branch merges into one fresh phi", "[symex][merge]")
{
  // Nondet arms so constant propagation cannot fold the selection away.
  engine e(R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  if (nondet_int() > 0)
    x = nondet_int();
  else
    x = nondet_int();
  return x;
}
)");

  auto eq = e.run();
  const auto phis = phis_for(*eq, "@main@x");
  REQUIRE(phis.size() == 1);

  // I8 freshness: the phi must define a *new* SSA name, not reuse either of
  // the values it selects between — reuse would alias two distinct values.
  const unsigned phi_index = to_symbol2t(phis[0]->lhs).level2_num;
  REQUIRE(phi_index > highest_index_before(*eq, "@main@x", phis[0]));

  REQUIRE(e.pending_merges() == 0);
}

TEST_CASE("a one-armed branch still merges the untaken value", "[symex][merge]")
{
  // The lost-behaviour direction: if the else path's value were not one of the
  // ite arms, the branch-not-taken case would vanish from the formula.
  engine e(R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  if (nondet_int() > 0)
    x = nondet_int();
  return x;
}
)");

  auto eq = e.run();
  const auto phis = phis_for(*eq, "@main@x");
  REQUIRE(phis.size() == 1);

  const if2t &sel = to_if2t(phis[0]->rhs);
  REQUIRE(sel.true_value != sel.false_value);
  REQUIRE(e.pending_merges() == 0);
}

TEST_CASE("an untouched variable gets no phi", "[symex][merge]")
{
  // I8's filter chain: phi_function skips variables whose L2 index is equal in
  // both states. A spurious phi here would be harmless but signals the
  // "changed?" test is not doing its job.
  engine e(R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  int untouched = nondet_int();
  if (nondet_int() > 0)
    x = nondet_int();
  else
    x = nondet_int();
  return x + untouched;
}
)");

  auto eq = e.run();
  REQUIRE(phis_for(*eq, "@main@x").size() == 1);
  REQUIRE(phis_for(*eq, "@main@untouched").empty());
  REQUIRE(e.pending_merges() == 0);
}

TEST_CASE("nested branches merge at each join", "[symex][merge]")
{
  engine e(R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  if (nondet_int() > 0)
  {
    if (nondet_int() > 1)
      x = nondet_int();
    else
      x = nondet_int();
  }
  else
    x = nondet_int();
  return x;
}
)");

  auto eq = e.run();
  // One join per branch construct.
  REQUIRE(phis_for(*eq, "@main@x").size() == 2);
  REQUIRE(e.pending_merges() == 0);
}

TEST_CASE(
  "branches inside a called function leave no pending merge",
  "[symex][merge]")
{
  // R2 is about a frame popped while its merge_state_map is non-empty. That
  // needs a branch *inside a callee*, so the join and the pop belong to the
  // same frame.
  engine e(R"(
int nondet_int(void);
int pick(int a)
{
  int r = a;
  if (nondet_int() > 0)
    r = nondet_int();
  else if (nondet_int() < 0)
    r = nondet_int();
  return r;
}
int main(void) { return pick(1) + pick(2); }
)");

  auto eq = e.run();
  REQUIRE(!phis_for(*eq, "@pick@r").empty());
  REQUIRE(e.pending_merges() == 0);
}

TEST_CASE("an early return leaves no pending merge", "[symex][merge]")
{
  // An early return jumps past the join, which is the shape most likely to
  // orphan a snapshot in framet::merge_state_map.
  engine e(R"(
int nondet_int(void);
int guarded(int a)
{
  if (a < 0)
    return 0;
  if (nondet_int() > 0)
    return a + 1;
  return a + 2;
}
int main(void) { return guarded(nondet_int()); }
)");

  auto eq = e.run();
  REQUIRE(eq->SSA_steps.size() > 0);
  REQUIRE(e.pending_merges() == 0);
}

TEST_CASE(
  "a branch inside an unwound loop leaves no pending merge",
  "[symex][merge]")
{
  engine e(R"(
int nondet_int(void);
int main(void)
{
  int x = nondet_int();
  for (int i = 0; i < 3; i++)
    if (nondet_int() > 0)
      x = nondet_int();
  return x;
}
)");

  auto eq = e.run();
  REQUIRE(phis_for(*eq, "@main@x").size() >= 3);
  REQUIRE(e.pending_merges() == 0);
}
