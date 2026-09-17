#ifndef ESBMC_INVARIANT_MINING_H
#define ESBMC_INVARIANT_MINING_H

#include <esbmc/kind_invariants.h>
#include <vector>

/// Candidates true at every sampled state of their loop: affine relations
/// between two integer variables at least one of which the loop writes
/// (generalised over the variables it does not), bounds on the written
/// variables and on the differences between them, and parity. The samples come
/// from one or a few executions, so most relations they suggest are
/// coincidences; the probe that proves candidates is what removes them.
std::vector<kind_candidatet>
mine_candidates(const std::vector<loop_head_samplet> &samples);

/// Bounds on a variable the loop writes, or on the difference of two, that
/// hold at every one of @p samples of @p cti's loop but not at @p cti, the
/// state an inductive-step counterexample starts that loop from.
std::vector<kind_candidatet> separate_counterexample(
  const std::vector<loop_head_samplet> &samples,
  const loop_head_samplet &cti);

/// Whether @p candidate evaluates to false at a sample of its loop. A
/// candidate mentioning a variable a sample has no value for is not refuted
/// by that sample.
bool refuted_by_samples(
  const kind_candidatet &candidate,
  const std::vector<loop_head_samplet> &samples);

#endif
