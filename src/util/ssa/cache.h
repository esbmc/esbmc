#pragma once

#include <unordered_set>

#include <goto-symex/equation/ssa_step_algorithm.h>
#include <util/base/time_stopping.h>
#include <util/ssa/cache_defs.h>

/**
 * @Brief This class stores all asserts conditions and guards
 *        for a given SSA, and ignores an assertion that an earlier
 *        one implies: the same claim under the same guard conjuncts or,
 *        with subsume_guards, under a subset of them.
 *        Currently, this works for the base case check only.
 */
class assertion_cache : public ssa_step_algorithm
{
public:
  assertion_cache(assert_db &db, bool subsume_guards)
    : ssa_step_algorithm(true), db(db), subsume_guards(subsume_guards)
  {
  }

  bool run(symex_target_equationt::SSA_stepst &) override;

  void run_on_assert(symex_target_equationt::SSA_stept &) override;
  virtual BigInt ignored() const override
  {
    return hits;
  }

protected:
  assert_db &db;

private:
  const bool subsume_guards;
  BigInt hits = 0;
  BigInt total = 0;
};
