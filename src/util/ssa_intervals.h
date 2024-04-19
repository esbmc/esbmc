#pragma once

#include "goto-programs/abstract-interpretation/interval_domain.h"
#include <goto-symex/symex_target_equation.h>
#include <util/time_stopping.h>
#include <util/algorithms.h>

class ssa_intervals : public ssa_step_algorithm
{
public:
  ssa_intervals() : ssa_step_algorithm(true)
  {
  }

  interval_domaint intervals;

  bool run(symex_target_equationt::SSA_stepst &eq) override
  {
    intervals.make_top();
    optimized = 0;
    fine_timet algorithm_start = current_time();
    for (auto &step : eq)
      run_on_step(step);
    fine_timet algorithm_stop = current_time();
    log_status(
      "SSA interval analysis time: {}s (optimized {} asserts)",
      time2string(algorithm_stop - algorithm_start),
      optimized);

    intervals.dump();
    return true;
  }

  BigInt ignored() const override
  {
    return optimized;
  }

protected:
  BigInt optimized = 0;

  void run_on_assert(symex_target_equationt::SSA_stept &SSA_step) override;
  void run_on_assume(symex_target_equationt::SSA_stept &SSA_step) override;
  void run_on_assignment(symex_target_equationt::SSA_stept &SSA_step) override;
};
