#pragma once

#include <goto-symex/equation/symex_target_equation.h>
#include <util/base/algorithm.h>

/**
 * @brief Base interface for ssa-step algorithms
 */
class ssa_step_algorithm : public algorithm<symex_target_equationt::SSA_stepst>
{
public:
  explicit ssa_step_algorithm(bool sideffect) : algorithm(sideffect)
  {
  }

  /// How many steps were ignored after this algorithm
  virtual BigInt ignored() const = 0;

  void run_on_step(symex_target_equationt::SSA_stept &);

  virtual void run_on_assignment(symex_target_equationt::SSA_stept &)
  {
  }
  virtual void run_on_assume(symex_target_equationt::SSA_stept &)
  {
  }
  virtual void run_on_assert(symex_target_equationt::SSA_stept &)
  {
  }
  virtual void run_on_output(symex_target_equationt::SSA_stept &)
  {
  }
  virtual void run_on_skip(symex_target_equationt::SSA_stept &)
  {
  }
  virtual void run_on_renumber(symex_target_equationt::SSA_stept &)
  {
  }
  virtual void run_on_branching(symex_target_equationt::SSA_stept &)
  {
  }
};
