#include "irep2/irep2_expr.h"
#include "irep2/irep2_utils.h"
#include <util/ssa_intervals.h>

void ssa_intervals::run_on_assert(symex_target_equationt::SSA_stept &SSA_step)
{
  // GUARD => COND
  if (is_constant_bool2t(SSA_step.cond))
    return;

  assert(is_implies2t(SSA_step.cond));
  expr2tc &cond = to_implies2t(SSA_step.cond).side_2;
  assert(is_bool_type(cond->type));
  const auto result =
    intervals.get_interval<interval_domaint::integer_intervalt>(cond);

  if (!result.contains(0))
    cond = gen_true_expr();
  else if (result.singleton())
    cond = gen_false_expr();

  if (simplify(SSA_step.cond))
    optimized = optimized + 1;
  intervals.assume(SSA_step.cond);
}

void ssa_intervals::run_on_assume(symex_target_equationt::SSA_stept &SSA_step)
{
  // GUARD => COND
  if (is_constant_bool2t(SSA_step.cond))
    return;

  assert(is_implies2t(SSA_step.cond));
  expr2tc &cond = to_implies2t(SSA_step.cond).side_2;
  assert(is_bool_type(cond->type));
  const auto result =
    intervals.get_interval<interval_domaint::integer_intervalt>(cond);

  if (!result.contains(0))
    cond = gen_true_expr();
  else if (result.singleton())
    cond = gen_false_expr();

  if (simplify(SSA_step.cond))
    optimized = optimized + 1;
  
  intervals.assume(SSA_step.cond);
}

void ssa_intervals::run_on_assignment(symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));
  if (!(is_signedbv_type(SSA_step.lhs) || is_unsignedbv_type(SSA_step.rhs)))
    return;
  intervals.apply_assignment<interval_domaint::integer_intervalt>(SSA_step.lhs, SSA_step.rhs, false);
}
