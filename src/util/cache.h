#pragma once

#include "irep2/irep2.h"
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
  assertion_cache(assert_db &db, bool trivial_value, bool match_subexpressions)
    : ssa_step_algorithm(true),
      db(db),
      trivial_value(trivial_value),
      match_subexpressions(match_subexpressions)
  {
  }

  bool run(symex_target_equationt::SSA_stepst &) override;

  void run_on_assert(symex_target_equationt::SSA_stept &) override;
  virtual BigInt ignored() const override
  {
    return hits + sub_hits;
  }

  /**
   * @brief recursively explore an expression for relations,
   *        if any operands is in the cache, then replace then with
   *        `trivial_value`
   *
   * Example: If some expression `cond1` is cached, then `cond1 && cond2`
              can be replaced into `trivial_value && cond2`
   */
  bool try_matching_sub_expression(expr2tc &e, const expr2tc &guards);

  inline bool is_in_cache(const expr2tc &e, const expr2tc &guards) const
  {
    return db.count(std::make_pair(e, guards)) != 0;
  }

  bool check_and_add_in_cache(expr2tc &cond, const expr2tc &guards);

protected:
  assert_db &db;
  /// value to be set for  the COND
  bool trivial_value;
  bool match_subexpressions;

private:
  BigInt hits = 0;
  BigInt sub_hits = 0;
};
