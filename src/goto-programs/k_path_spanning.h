#pragma once

#include <irep2/irep2.h>

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

/*
k-path spanning-set scoring (Phase 2, see GitHub issue #4335).

Marré & Bertolino (IEEE TSE 29(11), 2003) define the *spanning set* of a
coverage criterion as a minimal subset of test requirements such that
covering every requirement in the subset implies covering all the original
requirements. Equivalently: the maximal elements of the subsumption order,
where requirement g subsumes requirement g' iff every test that covers g
also covers g'.

For k-path coverage, witnesses are conjunctions of branch-direction atoms.
Goal g (atoms A) subsumes goal g' (atoms A') iff A' ⊂ A — any test that
satisfies all of A automatically satisfies the smaller A'. So the spanning
set is the set of goals whose atom multisets are not a proper subset of any
other emitted goal's atom multiset. Atom equality is structural (deep
expr2tc compare), not pointer-identity, so the analysis stays sound under
future expr2tc copies, canonicalisation, or hash-consing changes.

Phase-1 reports `reached / total_emitted`, which over-counts redundancy:
the depth-(n-1) goal {a} is structurally redundant when a depth-n goal
{a, b} is also emitted, but it inflates the denominator. Spanning-set
scoring switches the denominator to |maximal_set|, tightening the lower
bound without ever inflating coverage past the true value.

This module is a pure structural analysis over the emitted goal list — no
SMT call, no value-flow analysis. Phase 2 PR2 (semantic comparison-domain
contradictions) feeds in by skipping infeasible goals at instrumentation
time, which transitively shrinks the spanning set.
*/
class k_path_spanning_sett
{
public:
  // An atom is a (stored_guard, polarity) pair. Polarity true means the
  // guard appears positively in the witness; false means negated.
  using atom_t = std::pair<expr2tc, bool>;

  // Multiset of atoms in a goal's witness conjunction. Duplicates are
  // preserved (the contradiction filter in goto_coverage drops opposite-
  // polarity duplicates before this point, but identical-polarity
  // duplicates can still appear when the same guard repeats in the
  // sliding window — see test 8). Named *_multiset_t to make the
  // duplicate-preserving semantics explicit at the call site.
  using atom_multiset_t = std::vector<atom_t>;

  // Record a goal that has been emitted as an instrumented assertion. The
  // (claim_msg, claim_loc) pair is the same key used in
  // goto_coveraget::all_claims and the JSON report.
  void
  add_goal(atom_multiset_t atoms, std::string claim_msg, std::string claim_loc);

  // Compute the spanning set. After this call, spanning_size() returns
  // |maximal_set| and is_redundant(msg, loc) returns true iff every goal
  // record with that (msg, loc) is non-maximal (subsumed by another goal).
  void finalize();

  // Number of emitted goals — denominator under Phase-1 scoring.
  size_t total() const
  {
    return goals_.size();
  }

  // Number of maximal goals — denominator under spanning-set scoring.
  size_t spanning_size() const
  {
    return spanning_size_;
  }

  // True iff the (claim_msg, claim_loc) pair is non-maximal in every
  // emission. A claim is "feasible" otherwise.
  bool is_redundant(const std::string &msg, const std::string &loc) const;

  // Reset for a fresh run (e.g., second invocation in a session).
  void clear();

private:
  struct goal_t
  {
    // Sorted by atom_lt — a deep structural ordering (expr2tc::operator<
    // followed by polarity). Sorting is required for std::includes-based
    // multiset subset checks in is_proper_multiset_subset.
    atom_multiset_t atoms;
    // Bloom-style 64-bit signature: one bit set per atom hash modulo 64.
    // Used as a fast necessary condition before the deep multiset compare:
    // for `small ⊂ big`, every bit in small.signature must appear in
    // big.signature, so (small.signature & big.signature) == small.signature
    // is implied. False positives are possible (two distinct atoms can
    // collide on the same bucket); a false signature match always falls
    // back to the structural check, so soundness is preserved.
    uint64_t signature = 0;
    std::string msg;
    std::string loc;
    bool maximal = true;
  };

  std::vector<goal_t> goals_;
  std::set<std::pair<std::string, std::string>> redundant_claims_;
  size_t spanning_size_ = 0;
};
