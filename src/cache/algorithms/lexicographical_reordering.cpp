//
// Created by rafaelsa on 10/03/2020.
//

#include "lexicographical_reordering.h"

lexicographical_reordering::lexicographical_reordering(
  symex_target_equationt::SSA_stepst &steps)
  : ssa_step_algorithm(steps)
{
}

void lexicographical_reordering::run_on_assert(
  symex_target_equationt::SSA_stept &step)
{
  expr2tc &cond = step.cond;
  std::string &comment = step.comment;

  // First assert irep should begin with an implies
  assert(cond->expr_id == expr2t::expr_ids::implies_id);

  // LHS only holds the guard which is not useful. So we parse RHS
  expr2tc &rhs = step.rhs;
}
