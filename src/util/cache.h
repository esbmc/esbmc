#pragma once

#include <set>
#include <util/algorithms.h>
#include <util/time_stopping.h>

/**
 * @brief This class stores all asserts conditions and guards
 *        for a given SSA, if some pair was already added
 *        then it will convert it to be a trivial case,
 *        e.g. GUARD => COND ----> GUARD => 1
 */
class assertion_cache : public ssa_step_algorithm
{
public:
  assertion_cache(std::set<assert_pair> &assert_set, bool trivial_value)
    : ssa_step_algorithm(true),
      assert_set(assert_set),
      trivial_value(trivial_value)
  {
  }

  bool run(symex_target_equationt::SSA_stepst &eq) override
  {
    fine_timet algorithm_start = current_time();
    for(auto &step : eq)
      run_on_step(step);
    fine_timet algorithm_stop = current_time();
    log_status(
      "Caching time: {}s (removed {} assignments)",
      time2string(algorithm_stop - algorithm_start),
      hits);
    return true;
  }

  void run_on_assert(symex_target_equationt::SSA_stept &) override;
  virtual BigInt ignored() const override
  {
    return hits;
  }

protected:
  std::set<assert_pair> &assert_set;
  /// value to be set for  the COND
  bool trivial_value;

private:
  BigInt hits = 0;
};
