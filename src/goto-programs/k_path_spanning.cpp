#include <goto-programs/k_path_spanning.h>

#include <algorithm>

namespace
{
// Structural total order on atoms: deep `expr2tc::operator<` (which
// recurses into `expr2t`) followed by polarity. Using structural rather
// than pointer-identity equality keeps subsumption sound under any future
// expr2tc copy or canonicalisation, at the cost of one extra deep compare
// per atom — atom-multiset sizes are bounded by k_path_n ≤ 30 so the
// marginal cost is small relative to the O(n²) outer loop.
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

// Bloom-style 64-bit signature: each atom contributes one bit chosen by
// `crc(expr) ^ polarity` modulo 64. Idempotent under multiset duplication
// (setting the same bit twice is a no-op), which matches the `small ⊂ big`
// check we want: every bit set in `small.sig` must also be set in
// `big.sig`. Used as a cheap necessary condition before the deep
// std::includes call in is_proper_multiset_subset.
uint64_t atom_signature(const k_path_spanning_sett::atom_multiset_t &atoms)
{
  uint64_t sig = 0;
  for (const auto &[e, pol] : atoms)
  {
    // expr2tc::crc() is a deep structural hash — equal expressions get
    // equal CRCs regardless of pointer identity, so two structurally
    // equal atoms always set the same bit.
    const std::size_t h = e.crc() ^ (pol ? 0x9E3779B97F4A7C15ULL : 0);
    sig |= uint64_t(1) << (h % 64);
  }
  return sig;
}

// True iff `small` is a proper subset of `big` as multisets — every
// occurrence of an atom in `small` is matched by an occurrence in `big`,
// and `big` has at least one atom not matched.
//
// Both vectors must be pre-sorted by atom_lt; the size guard short-circuits
// equal- and over-sized cases (a multiset cannot be a proper subset of
// something its own size or smaller).
bool is_proper_multiset_subset(
  const k_path_spanning_sett::atom_multiset_t &small,
  const k_path_spanning_sett::atom_multiset_t &big)
{
  if (small.size() >= big.size())
    return false;
  return std::includes(
    big.begin(), big.end(), small.begin(), small.end(), atom_lt);
}
} // namespace

void k_path_spanning_sett::add_goal(
  atom_multiset_t atoms,
  std::string claim_msg,
  std::string claim_loc)
{
  std::sort(atoms.begin(), atoms.end(), atom_lt);
  const uint64_t sig = atom_signature(atoms);
  goals_.push_back(
    {std::move(atoms), sig, std::move(claim_msg), std::move(claim_loc), true});
}

void k_path_spanning_sett::finalize()
{
  // Worst-case O(n²) subsumption check, n = total emitted goals across
  // every function. Two layers of pre-filtering keep the deep multiset
  // compare from running on the bulk of pairs:
  //
  //   1. Sort by atom-multiset size ascending. For each `i`, the inner
  //      loop only scans `j > i`, and skips same-size pairs — a multiset
  //      can only be a proper subset of a strictly larger one.
  //
  //   2. Bloom-style signature check `(small.sig & big.sig) == small.sig`.
  //      A bit set in `small.sig` that is missing from `big.sig` proves
  //      some atom in `small` cannot appear in `big`, so subset is
  //      impossible. False positives fall through to the structural
  //      `std::includes` check, so soundness is preserved.
  //
  // Instrumentation runs once per build; even at the default per-function
  // cap of k_path_max_goals=10000 the cost is amortised.
  std::sort(goals_.begin(), goals_.end(), [](const goal_t &a, const goal_t &b) {
    return a.atoms.size() < b.atoms.size();
  });

  for (size_t i = 0; i < goals_.size(); ++i)
  {
    const uint64_t small_sig = goals_[i].signature;
    for (size_t j = i + 1; j < goals_.size(); ++j)
    {
      if (goals_[j].atoms.size() == goals_[i].atoms.size())
        continue;
      // Necessary-condition pre-filter: every bit in small.sig must be
      // present in big.sig. Cheap (one AND + one compare) and skips the
      // structural compare for the common case where the goals share no
      // atoms in common.
      if ((small_sig & goals_[j].signature) != small_sig)
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
