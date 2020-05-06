/*******************************************************************\
 Module: Expressions Green Normal Form

 Author: Rafael Sá Menezes

 Date: April 2020
\*******************************************************************/

#ifndef ESBMC_EXPR_GREEN_NORMAL_FORM_H
#define ESBMC_EXPR_GREEN_NORMAL_FORM_H

#include <cache/expr_algorithm.h>

/**
 * This algorithm changes relations based on green heuristic
 *
 * Input:
 *
 * A + B + C + ... + k OP c
 *
 * After execution, the formula will be:
 *
 * A + B + C + ... + k OP 0, where:
 *
 * - [A,B,C,...] Are symbols
 * - k,c are integers
 * - OP belongs to { =, !=, <= }
 *
 * Rules for changing relations:
 * A + B + C + ... + k == y --> A + B + C + ... + (k-y) == 0
 * A + B + C + ... + k != y --> A + B + C + ... + (k-y) != 0
 *
 * A + B + C + ... + k <  y --> A + B + C + ... + (k-y+1) <= 0
 * A + B + C + ... + k <= y --> A + B + C + ... + (k-y)   <= 0
 *
 * A + B + C + ... + k >  y --> -(A + B + C + ... + (k+y-1)) <= 0
 * A + B + C + ... + k >= y --> -(A + B + C + ... + (k+y))   <= 0
 *
 * NOTES:
 * - This assumes that expr_variable_reordering was applied.
 * - The substitution rules work only for Integers. Because the
 *   rule to convert from < to <= (and similar) is to add 1 to LHS.
 */
class expr_green_normal_form : public expr_algorithm
{
private:
  /**
   * This should convert inequalities from the form <, >, >= into <=
   * @param inequality to be converted
   */
  expr2tc convert_inequality(expr2tc &inequality);

public:
  explicit expr_green_normal_form(expr2tc &expr) : expr_algorithm(expr)
  {
  }

  void run() override;
};

#endif //ESBMC_EXPR_GREEN_NORMAL_FORM_H
