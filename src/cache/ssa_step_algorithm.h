//
// Created by rafaelsa on 04/02/2020.
//

#ifndef ESBMC_SSA_STEP_ALGORITHM_H
#define ESBMC_SSA_STEP_ALGORITHM_H

/** @file ssa_step_algorithm.h
 * This file will define methods and classes to help manipulate
 * SSA_steps for algorithms
 */
#include <iostream>
#include <goto-symex/symex_target_equation.h>

/**
 *  A generic class to represent algorithms to run in SSA steps
 *  The concept is to take change the SSA steps based on the algorithm
 *
 */
class ssa_step_algorithm
{
public:
  explicit ssa_step_algorithm(symex_target_equationt::SSA_stepst &steps)
    : steps(steps)
  {
  }

  void run();

protected:
  virtual void run_on_assignment(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_assume(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_assert(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_output(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_skip(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_renumber(symex_target_equationt::SSA_stept &step) = 0;

  symex_target_equationt::SSA_stepst &steps;
};
#endif //ESBMC_SSA_STEP_ALGORITHM_H
