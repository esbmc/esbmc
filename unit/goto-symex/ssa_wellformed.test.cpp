/*******************************************************************
 Module: SSA well-formedness of equations produced by the real engine

 Tier B of docs/roadmap/goto-symex-verification-plan.md (H-B1, milestone M1).

 Unlike a Tier-A harness, nothing here is transcribed: `goto_factory` parses a
 real C program, a real `reachability_treet` drives the real `goto_symext` over
 it, and the assertions are made against the `symex_target_equationt` that comes
 out. The properties are the ones §4.2 lists as unenforced in the shipped
 binary:

   I1  (P8)  per (base_name, L1, thread), the L2 index of each definition
             strictly increases in equation order.
   I10 (P11) no two assignment steps define the same SSA name. This is the
             invariant `check_for_duplicate_assigns` was written for and never
             enforces — it logs and returns (R5).
   P11       no step reads an SSA name before the step that defines it.

 \*******************************************************************/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <map>
#include <set>
#include <sstream>
#include <string>

#include <goto-symex/reachability_tree.h>
#include <goto-symex/symex_target_equation.h>
#include <irep2/irep2_expr.h>
#include <util/symtab/namespace.h>

#include "../testing-utils/goto_factory.h"

namespace
{
/** The identity of one SSA definition: everything but the L2 counter. */
struct l2_keyt
{
  std::string base_name;
  unsigned l1_num;
  unsigned thread_num;

  bool operator<(const l2_keyt &o) const
  {
    return std::tie(base_name, l1_num, thread_num) <
           std::tie(o.base_name, o.l1_num, o.thread_num);
  }
};

bool is_ssa_symbol(const expr2tc &e)
{
  if (!is_symbol2t(e))
    return false;
  const symbol_renaming_level lev = to_symbol2t(e).rlevel;
  return lev == symbol_renaming_level::level2 ||
         lev == symbol_renaming_level::level2_global;
}

l2_keyt key_of(const expr2tc &e)
{
  const symbol2t &s = to_symbol2t(e);
  return {s.thename.as_string(), s.level1_num, s.thread_num};
}

void collect_ssa_reads(const expr2tc &e, std::set<std::string> &out)
{
  if (is_nil_expr(e))
    return;
  if (is_ssa_symbol(e))
    out.insert(to_symbol2t(e).get_symbol_name());
  e->foreach_operand(
    [&out](const expr2tc &sub) { collect_ssa_reads(sub, out); });
}

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

struct violationst
{
  std::vector<std::string> duplicate_definitions; // I10
  std::vector<std::string> non_monotonic;         // I1
  std::vector<std::string> use_before_def;        // P11
};

violationst validate(const symex_target_equationt &eq)
{
  violationst bad;

  // Names defined anywhere in the equation. A read of a name that is never
  // defined is a free symbol (nondet, argument, uninitialised global) and is
  // not a use-before-def; only a read that *precedes* its definition is.
  std::set<std::string> ever_defined;
  for (const auto &step : eq.SSA_steps)
    if (step.is_assignment() && is_ssa_symbol(step.lhs))
      ever_defined.insert(to_symbol2t(step.lhs).get_symbol_name());

  std::set<std::string> defined;
  std::map<l2_keyt, unsigned> highest_index;

  for (const auto &step : eq.SSA_steps)
  {
    // An assignment step's `cond` is equality2tc(lhs, rhs)
    // (symex_target_equationt::assignment), so reading it here would report
    // every definition as a use of itself. Its rhs carries the actual reads.
    std::set<std::string> reads;
    collect_ssa_reads(step.guard, reads);
    if (step.is_assignment())
      collect_ssa_reads(step.rhs, reads);
    else
      collect_ssa_reads(step.cond, reads);

    for (const std::string &name : reads)
      if (ever_defined.count(name) && !defined.count(name))
        bad.use_before_def.push_back(name);

    if (!step.is_assignment() || !is_ssa_symbol(step.lhs))
      continue;

    const symbol2t &lhs = to_symbol2t(step.lhs);
    const std::string name = lhs.get_symbol_name();

    if (!defined.insert(name).second)
      bad.duplicate_definitions.push_back(name);

    auto [it, fresh] = highest_index.emplace(key_of(step.lhs), lhs.level2_num);
    if (!fresh)
    {
      if (lhs.level2_num <= it->second)
        bad.non_monotonic.push_back(name);
      it->second = lhs.level2_num;
    }
  }

  return bad;
}

/** Every property at once, so a new program costs one line. */
void require_well_formed(const std::string &src)
{
  const violationst bad = validate(*symex_equation(src));
  CHECK(bad.duplicate_definitions.empty());
  CHECK(bad.non_monotonic.empty());
  CHECK(bad.use_before_def.empty());
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
