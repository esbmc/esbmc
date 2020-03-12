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
  explicit lexicographical_reordering(symex_target_equationt::SSA_stepst &steps)
    : ssa_step_algorithm(steps)
  {
  }

  static bool should_swap(expr2tc &side1, expr2tc &side2);

private:
  void process_expr(expr2tc &rhs);
  void run_on_relation(expr2tc &expr);
  void run_on_arith(expr2tc &expr);
  void run_on_symbol(expr2tc &expr){};
  void run_on_value(expr2tc &expr){};

protected:
  void run_on_assignment(symex_target_equationt::SSA_stept &step) override;
  void run_on_assume(symex_target_equationt::SSA_stept &step) override;
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
