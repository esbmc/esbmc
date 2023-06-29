#pragma once

#include <unordered_set>

#include <util/algorithms.h>
#include <util/time_stopping.h>
#include <util/crypto_hash.h>
#include <util/cache_defs.h>

/**
 * @Brief This class stores all asserts conditions and guards
 *        for a given SSA, if some pair was already added
 *        then it will convert it to be a trivial case,
 *        e.g. GUARD => COND ----> GUARD => 1
 */
class assertion_cache : public ssa_step_algorithm
{
public:
  assertion_cache(assert_db &db, bool trivial_value)
    : ssa_step_algorithm(true), db(db), trivial_value(trivial_value)
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
  /// value to be set for  the COND
  bool trivial_value;

private:
  BigInt hits = 0;
  BigInt total = 0;
};
