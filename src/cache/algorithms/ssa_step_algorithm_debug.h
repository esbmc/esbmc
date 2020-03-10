//
// Created by rafaelsa on 10/03/2020.
//

#ifndef ESBMC_SSA_STEP_ALGORITHM_DEBUG_H
#define ESBMC_SSA_STEP_ALGORITHM_DEBUG_H

#include <cache/ssa_step_algorithm.h>

class ssa_step_algorithm_debug : public ssa_step_algorithm
{
public:
  explicit ssa_step_algorithm_debug(symex_target_equationt::SSA_stepst &steps)
    : ssa_step_algorithm(steps)
  {
  }

protected:
  void run_on_assignment(symex_target_equationt::SSA_stept &step) override;
  void run_on_assume(symex_target_equationt::SSA_stept &step) override;
  void run_on_assert(symex_target_equationt::SSA_stept &step) override;
  void run_on_output(symex_target_equationt::SSA_stept &step) override;
  void run_on_skip(symex_target_equationt::SSA_stept &step) override;
  void run_on_renumber(symex_target_equationt::SSA_stept &step) override;
};

#endif //ESBMC_SSA_STEP_ALGORITHM_DEBUG_H
