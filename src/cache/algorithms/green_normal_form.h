//
// Created by rafaelsa on 13/03/2020.
//

#ifndef ESBMC_GREEN_NORMAL_FORM_H
#define ESBMC_GREEN_NORMAL_FORM_H

#include <cache/ssa_step_algorithm.h>
/**
 * This algorithm changes relations based on green heuristic
 *
 * After execution, the formula will be:
 *
 * A + B + C + ... + k OP 0, where:
 *
 * - [A,B,C,...] Are variables
 * - k is an integer
 * - OP belongs to { =, !=, <= }
 *
 * Example:
 * A + B + C + 7 < 0 -> A + B + C + 8 <= 0
 * A + B + C + 7 > 0 -> -A -B -C -6  <= 0
 * A + B + C + 7 >= 0 -> -A -B -C -7  <= 0
 *
 * NOTES:
 *
 * - This assumes that lexicographical_reordering was applied.
 * - The substituition rules works only for *Integers*!
 */

class green_normal_form: public ssa_step_algorithm
{
public:
  green_normal_form(symex_target_equationt::SSA_stepst &steps)
    : ssa_step_algorithm(steps)
  {
  }

private:
  /**
   * This should convert inequalities of the form <, >, >= into <=
   * @param inequality to be converted
   */
  void convert_inequality(expr2tc &inequality);

  /**
   * Checks whether the operator of the relation is not <=, = or !=
   * @param relation
   * @return
   */
  bool is_operator_correct(expr2tc &relation);

protected:
  void run_on_assignment(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_assume(symex_target_equationt::SSA_stept &step) override
  {
  }
  void run_on_assert(symex_target_equationt::SSA_stept &step) override
  {
  }
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

#endif //ESBMC_GREEN_NORMAL_FORM_H
