//
// Created by rafaelsa on 10/03/2020.
//

#ifndef ESBMC_LEXICOGRAPHICAL_REORDERING_H
#define ESBMC_LEXICOGRAPHICAL_REORDERING_H

#include <cache/ssa_step_algorithm.h>

/**
 * This algorithm reorders variables in assertions based on lexicographic order
 */
class lexicographical_reordering : public ssa_step_algorithm
{
public:
  lexicographical_reordering(symex_target_equationt::SSA_stepst &steps);

protected:
  void run_on_assignment(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_assume(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_assert(symex_target_equationt::SSA_stept &step) override;
  void run_on_output(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_skip(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_renumber(symex_target_equationt::SSA_stept &step) override
  {
  }
};

#endif //ESBMC_LEXICOGRAPHICAL_REORDERING_H
