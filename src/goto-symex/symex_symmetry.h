#ifndef CPROVER_GOTO_SYMEX_SYMEX_SYMMETRY_H
#define CPROVER_GOTO_SYMEX_SYMEX_SYMMETRY_H

#include <goto-symex/symex_target_equation.h>

class smt_convt;

/// Recognises `lhs = ite(cond, t, e)` assignments in `equation` that encode
/// a max/min idiom (cond compares t and e, choosing the greater/lesser) and
/// asserts the redundant bounds this implies -- `lhs >= t && lhs >= e` for a
/// max, `lhs <= t && lhs <= e` for a min -- directly into `smt_conv`.
///
/// This targets the class of formula that motivated ESBMC discussion #5998
/// / Z3 issue #10125: a running max/min folded over a sequence of free
/// values (e.g. an uninitialised array), where later assertions compare an
/// individual element against the fold result. Z3's search has to
/// case-split to re-derive that the fold result bounds every element it
/// folded over; asserting the bound directly, once per fold step, gives it
/// that fact without the case-split.
///
/// Each injected bound is a tautology implied by the ite's own definition
/// (already asserted elsewhere in the equation) -- it adds no information,
/// so it can never change satisfiability, PROVIDED the comparison is over a
/// total order. floatbv is excluded: under IEEE-754, any relation involving
/// a NaN operand is false, so the bound does not hold when either branch
/// may be NaN. Recognition is otherwise a bounded, step-length-independent
/// syntactic check (at most one `not` peel and one symbol-lookup hop per
/// operand, to see through ESBMC's guard-then-merge phi lowering) with no
/// solver queries, so cost is O(number of ite assignments), not O(chain
/// length).
void assert_symmetry_breaking(
  const symex_target_equationt &equation,
  smt_convt &smt_conv);

#endif
