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
 * Rules for changing relations:
 * A + B + C + ... + k = y -> A + B + C + ... + (k-y) = 0
 * A + B + C + ... + k != y -> A + B + C + ... + (k-y) != 0
 * A + B + C + ... + k <= y -> A + B + C + (k-y) <= 0
 * A + B + C + ... + k < y -> A + B + C + (k-y+1) <= 0
 * A + B + C + ... + k > y -> -(A + B + C + (k+y-1)) <= 0
 * A + B + C + ... + k >= y -> -(A + B + C + (k+y)) <= 0
 *
 * NOTES:
 *
 * - This assumes that lexicographical_reordering was applied.
 * - The substituition rules works only for *Integers*!
 */

namespace
{
const std::array<expr2t::expr_ids, 6> relation_expr_names = {
  expr2t::expr_ids::equality_id,
  expr2t::expr_ids::notequal_id,
  expr2t::expr_ids::lessthan_id,
  expr2t::expr_ids::greaterthan_id,
  expr2t::expr_ids::lessthanequal_id,
  expr2t::expr_ids::greaterthanequal_id};

const std::array<expr2t::expr_ids, 3> valid_relation_names = {
  expr2t::expr_ids::equality_id,
  expr2t::expr_ids::notequal_id,
  expr2t::expr_ids::lessthanequal_id};

} // namespace
class green_normal_form : public ssa_step_algorithm
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

  /**
   * Checks whether the RHS side of the expression is 0
   * @param relation
   * @return
   */
  bool is_normal_form(expr2tc &relation)
  {
    return false;
  }

    /**
   * Checks whether the expression contains only Integers
   * @param relation
   * @return
   */
  bool is_integer_expr(expr2tc &relation);

  void process_expr(expr2tc &rhs);

  /**
   * This should convert equality of the form != into =
   * @param equality to be converted
   */
  void convert_equality(expr2tc &equality);

  void convert_to_normal_form(expr2tc &equality);

  void set_rightest_value_of_lhs_relation(expr2tc &equality, BigInt value);

protected:
  void run_on_assignment(symex_target_equationt::SSA_stept &step) override
  {
  }
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

#endif //ESBMC_GREEN_NORMAL_FORM_H
