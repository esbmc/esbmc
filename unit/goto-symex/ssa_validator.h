/*******************************************************************
 Module: SSA well-formedness validator, shared across the Tier-B tests

 H-B1 of docs/roadmap/goto-symex-verification-plan.md, §7.2. Any test driving
 the real engine already holds an equation, so checking it costs one line and
 turns that test into another sample of the properties §4.2 lists as unenforced
 in the shipped binary:

   I1  (P8)  per (base_name, L1, thread), the L2 index of each definition
             strictly increases in equation order.
   I10 (P11) no two assignment steps define the same SSA name.
   P11       no step reads an SSA name before the step that defines it.

 \*******************************************************************/

#pragma once

// Include this *after* the consumer's own `#define CATCH_CONFIG_MAIN`:
// including it first swallows the consumer's later include via Catch's own
// guard, and the test binary fails to link with a missing `main`.
#include <catch2/catch.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <goto-symex/symex_target_equation.h>
#include <irep2/irep2_expr.h>

namespace symex_ssa
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

inline bool is_ssa_symbol(const expr2tc &e)
{
  if (!is_symbol2t(e))
    return false;
  const symbol_renaming_level lev = to_symbol2t(e).rlevel;
  return lev == symbol_renaming_level::level2 ||
         lev == symbol_renaming_level::level2_global;
}

inline l2_keyt key_of(const expr2tc &e)
{
  const symbol2t &s = to_symbol2t(e);
  return {s.thename.as_string(), s.level1_num, s.thread_num};
}

inline void collect_ssa_reads(const expr2tc &e, std::set<std::string> &out)
{
  if (is_nil_expr(e))
    return;
  if (is_ssa_symbol(e))
    out.insert(to_symbol2t(e).get_symbol_name());
  e->foreach_operand(
    [&out](const expr2tc &sub) { collect_ssa_reads(sub, out); });
}

struct violationst
{
  std::vector<std::string> duplicate_definitions; // I10
  std::vector<std::string> non_monotonic;         // I1
  std::vector<std::string> use_before_def;        // P11
};

inline violationst validate(const symex_target_equationt &eq)
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

inline std::string describe(const std::vector<std::string> &names)
{
  std::string out;
  for (const std::string &name : names)
    out += (out.empty() ? "" : ", ") + name;
  return out;
}

/** Assert I1, I10 and P11 over `eq`, naming the offenders on failure.
 *  Uses REQUIRE/CHECK, so call it from inside a Catch2 test case. */
inline void require_well_formed(const symex_target_equationt &eq)
{
  // All three properties hold vacuously over an equation with no SSA
  // definition, so guard on that rather than on the step count.
  const auto definitions = std::count_if(
    eq.SSA_steps.begin(), eq.SSA_steps.end(), [](const auto &step) {
      return step.is_assignment() && is_ssa_symbol(step.lhs);
    });
  REQUIRE(definitions > 0);

  const violationst bad = validate(eq);
  INFO("duplicate SSA definitions: " << describe(bad.duplicate_definitions));
  INFO("non-monotonic L2 indices: " << describe(bad.non_monotonic));
  INFO("SSA names read before definition: " << describe(bad.use_before_def));
  CHECK(bad.duplicate_definitions.empty());
  CHECK(bad.non_monotonic.empty());
  CHECK(bad.use_before_def.empty());
}
} // namespace symex_ssa
