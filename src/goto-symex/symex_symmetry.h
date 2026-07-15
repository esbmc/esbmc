#ifndef CPROVER_GOTO_SYMEX_SYMEX_SYMMETRY_H
#define CPROVER_GOTO_SYMEX_SYMEX_SYMMETRY_H

#include <goto-symex/symex_target_equation.h>

class smt_convt;

/// Recognises `lhs = ite(cond, t, e)` max/min idioms in `equation` and
/// asserts the redundant bounds they imply (`lhs >= t && lhs >= e` for a
/// max, `<=` for a min) into `smt_conv`. This hands the solver the fold
/// bound directly instead of forcing it to case-split to re-derive it --
/// the motivating case (discussion #5998 / Z3 #10125) is a running max/min
/// folded over free values.
///
/// Each bound is a tautology implied by the ite's own definition, so it
/// cannot change satisfiability, provided the comparison is over a total
/// order -- floatbv is excluded because NaN comparisons break it.
/// Recognition is a bounded syntactic check with no solver queries.
void assert_symmetry_breaking(
  const symex_target_equationt &equation,
  smt_convt &smt_conv);

#endif
