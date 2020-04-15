/*******************************************************************\
 Module: SSA Step Algorithm

 Author: Rafael Sá Menezes

 Date: April 2020
\*******************************************************************/

#ifndef ESBMC_SSA_STEP_ALGORITHM_H
#define ESBMC_SSA_STEP_ALGORITHM_H

#include <iostream>
#include <goto-symex/symex_target_equation.h>

/**
 *  A generic class to represent algorithms to run in SSA steps
 *  The concept is to take change the SSA steps based on the algorithm
 */
class ssa_step_algorithm
{
public:
  explicit ssa_step_algorithm(symex_target_equationt::SSA_stepst &steps)
    : steps(steps)
  {
  }

  void run()
  {
    typedef goto_trace_stept::typet ssa_type;
    typedef symex_target_equationt::SSA_stept ssa_step;
    typedef std::function<void(ssa_step &)> ssa_function;

    static std::map<ssa_type, ssa_function> run_on_function;
    static bool map_initialized = false;

    // Simple optimization so we do not use switch-case
    if(!map_initialized)
    {
      run_on_function[ssa_type::ASSIGNMENT] = [this](ssa_step &step) {
        run_on_assignment(step);
      };
      run_on_function[ssa_type::ASSUME] = [this](ssa_step &step) {
        run_on_assume(step);
      };
      run_on_function[ssa_type::ASSERT] = [this](ssa_step &step) {
        run_on_assert(step);
      };
      run_on_function[ssa_type::OUTPUT] = [this](ssa_step &step) {
        run_on_output(step);
      };
      run_on_function[ssa_type::RENUMBER] = [this](ssa_step &step) {
        run_on_renumber(step);
      };
      run_on_function[ssa_type::SKIP] = [this](ssa_step &step) {
        run_on_skip(step);
      };

      map_initialized = true;
    }
    for(auto &step : steps)
    {
      if(parse_step(step))
      {
        run_on_function[step.type](step);
      }
    }
  }

protected:
  virtual void run_on_assignment(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_assume(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_assert(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_output(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_skip(symex_target_equationt::SSA_stept &step) = 0;
  virtual void run_on_renumber(symex_target_equationt::SSA_stept &step) = 0;

  symex_target_equationt::SSA_stepst &steps;
  virtual bool parse_step(symex_target_equationt::SSA_stept &stept)
  {
    return !stept.ignore;
  }
};

class ssa_step_algorithm_hidden : public ssa_step_algorithm
{
public:
  explicit ssa_step_algorithm_hidden(symex_target_equationt::SSA_stepst &steps)
    : ssa_step_algorithm(steps)
  {
  }

protected:
  virtual bool parse_step(symex_target_equationt::SSA_stept &stept) override
  {
    return !stept.ignore || stept.hidden;
  };
};

#endif //ESBMC_SSA_STEP_ALGORITHM_H
