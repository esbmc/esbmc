#include <util/ssa/cache.h>
#include <util/message/message.h>
#include <algorithm>
#include <unordered_set>

namespace
{
using conjunct_sett = std::unordered_set<expr2tc, irep2_hash>;

void add_conjuncts(const expr2tc &e, conjunct_sett &conjuncts)
{
  // Guards are long left-leaning and-chains, so no recursion.
  std::vector<expr2tc> todo{e};
  while (!todo.empty())
  {
    const expr2tc next = todo.back();
    todo.pop_back();
    if (is_and2t(next))
    {
      todo.push_back(to_and2t(next).side_1);
      todo.push_back(to_and2t(next).side_2);
    }
    else if (!is_true(next))
      conjuncts.insert(next);
  }
}

/// Splits `a1 => (a2 => ... => c)`, the form symex gives a guarded claim,
/// into c and the conjuncts of a1 && a2 && ...
expr2tc split_claim(const expr2tc &cond, conjunct_sett &antecedents)
{
  expr2tc claim = cond;
  while (is_implies2t(claim))
  {
    add_conjuncts(to_implies2t(claim).side_1, antecedents);
    claim = to_implies2t(claim).side_2;
  }
  return claim;
}
} // namespace

void assertion_cache::run_on_assert(symex_target_equationt::SSA_stept &step)
{
  ++total;
  conjunct_sett guard;
  auto &seen = db[split_claim(step.cond, guard)];

  // `g' => c` implies `g => c` when g has every conjunct of g'. The later
  // claim also sits under at least the assumptions of the earlier one.
  const auto implied = [&](const std::vector<expr2tc> &weaker) {
    if (
      subsume_guards ? weaker.size() > guard.size()
                     : weaker.size() != guard.size())
      return false;
    return std::all_of(weaker.begin(), weaker.end(), [&](const expr2tc &c) {
      return guard.count(c) != 0;
    });
  };

  if (std::any_of(seen.begin(), seen.end(), implied))
  {
    ++hits;
    step.ignore = true;
    return;
  }
  seen.emplace_back(guard.begin(), guard.end());
}

bool assertion_cache::run(symex_target_equationt::SSA_stepst &eq)
{
  fine_timet algorithm_start = current_time();
  for (auto &step : eq)
    run_on_step(step);
  fine_timet algorithm_stop = current_time();
  log_status(
    "Caching time: {}s (removed {} assertions)",
    time2string(algorithm_stop - algorithm_start),
    hits);
  return true;
}
