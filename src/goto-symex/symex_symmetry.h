#ifndef CPROVER_GOTO_SYMEX_SYMEX_SYMMETRY_H
#define CPROVER_GOTO_SYMEX_SYMEX_SYMMETRY_H

#include <goto-symex/symex_target_equation.h>
#include <util/algorithms.h>

/// SSA transformation pass that recognises `lhs = ite(cond, t, e)` max/min
/// idioms in the equation and injects the redundant bounds they imply
/// (`t <= lhs && e <= lhs` for a max, `<=` reversed for a min) as extra
/// ASSUME steps. This hands the solver the fold bound directly instead of
/// forcing it to case-split to re-derive it -- the motivating case
/// (discussion #5998 / Z3 #10125) is a running max/min folded over free
/// values. For a chain of such ites, a direct leaf-to-final bound is also
/// injected, so a property against the chain's final result doesn't force
/// the solver to chain the per-step bounds transitively.
///
/// Each bound is a tautology implied by the ite's own definition, so it
/// cannot change satisfiability, provided the comparison is over a total
/// order -- floatbv is excluded because NaN comparisons break it.
/// Recognition is a bounded syntactic check with no solver queries. Running
/// this as its own pass (alongside slicing) keeps the optimisation out of
/// `symex_target_equationt::convert`, which stays pure formula creation.
class symmetry_breakingt : public ssa_step_algorithm
{
public:
  symmetry_breakingt() : ssa_step_algorithm(true)
  {
  }

  bool run(symex_target_equationt::SSA_stepst &steps) override;

  BigInt ignored() const override
  {
    return 0;
  }
};

#endif
