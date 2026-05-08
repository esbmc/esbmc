#include <goto-programs/k_path_spanning.h>

#include <algorithm>

namespace
{
// Structural total order on atoms: deep `expr2tc::operator<` (which
// recurses into `expr2t`) followed by polarity. Using structural rather
// than pointer-identity equality keeps subsumption sound under any future
// expr2tc copy or canonicalisation, at the cost of one extra deep compare
// per atom — atom-set sizes are bounded by k_path_n ≤ 30 so the marginal
// cost is small relative to the O(n²) outer loop.
bool atom_lt(
  const k_path_spanning_sett::atom_t &a,
  const k_path_spanning_sett::atom_t &b)
{
  if (a.first < b.first)
    return true;
  if (b.first < a.first)
    return false;
  return a.second < b.second;
}

// True iff `small` is a proper subset of `big` as multisets — every
// occurrence of an atom in `small` is matched by an occurrence in `big`,
// and `big` has at least one atom not matched.
bool is_proper_multiset_subset(
  const k_path_spanning_sett::atom_set_t &small,
  const k_path_spanning_sett::atom_set_t &big)
{
  if (small.size() >= big.size())
    return false;
  // Both vectors are pre-sorted by atom_lt.
  return std::includes(
    big.begin(), big.end(), small.begin(), small.end(), atom_lt);
}
} // namespace

void k_path_spanning_sett::add_goal(
  atom_set_t atoms,
  std::string claim_msg,
  std::string claim_loc)
{
  std::sort(atoms.begin(), atoms.end(), atom_lt);
  goals_.push_back(
    {std::move(atoms), std::move(claim_msg), std::move(claim_loc), true});
}

void k_path_spanning_sett::finalize()
{
  // Worst-case O(n²) subsumption check, n = total emitted goals across
  // every function. We sort by atom-set size ascending so the inner loop
  // can skip same-size pairs — a multiset can only be a proper subset of
  // a strictly larger one — and start from the first strictly-larger
  // index, which halves the comparisons in the dense case where many
  // goals share a size. Instrumentation runs once per build, so even at
  // the default per-function cap of k_path_max_goals=10000 the cost is
  // amortised.
  std::sort(goals_.begin(), goals_.end(), [](const goal_t &a, const goal_t &b) {
    return a.atoms.size() < b.atoms.size();
  });

  for (size_t i = 0; i < goals_.size(); ++i)
  {
    for (size_t j = i + 1; j < goals_.size(); ++j)
    {
      if (goals_[j].atoms.size() == goals_[i].atoms.size())
        continue;
      if (is_proper_multiset_subset(goals_[i].atoms, goals_[j].atoms))
      {
        goals_[i].maximal = false;
        break;
      }
    }
  }

  // Build the redundant-claims set. A (msg, loc) pair is redundant iff
  // every emission at that pair is non-maximal — if any emission at the
  // pair is maximal, the claim is feasible.
  std::set<std::pair<std::string, std::string>> feasible_claims;
  for (const auto &g : goals_)
    if (g.maximal)
      feasible_claims.emplace(g.msg, g.loc);

  redundant_claims_.clear();
  spanning_size_ = 0;
  for (const auto &g : goals_)
  {
    if (g.maximal)
      ++spanning_size_;
    else if (feasible_claims.count({g.msg, g.loc}) == 0)
      redundant_claims_.emplace(g.msg, g.loc);
  }
}

bool k_path_spanning_sett::is_redundant(
  const std::string &msg,
  const std::string &loc) const
{
  return redundant_claims_.count({msg, loc}) > 0;
}

void k_path_spanning_sett::clear()
{
  goals_.clear();
  redundant_claims_.clear();
  spanning_size_ = 0;
}
