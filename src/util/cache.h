#pragma once

#include <unordered_set>

#include <util/algorithms.h>
#include <util/time_stopping.h>
#include <util/crypto_hash.h>
#include <util/cache_defs.h>

/**
 * @Brief This class stores all asserts conditions and guards
 *        for a given SSA, if some pair was already added
 *        then it will assume that it is UNSAT.
 *        Currently, this works for the base case check only.    
 */
class assertion_cache : public ssa_step_algorithm
{
public:
  assertion_cache(assert_db &db) : ssa_step_algorithm(true), db(db)
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
  BigInt hits = 0;
  BigInt total = 0;
};
