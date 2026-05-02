#ifndef CPROVER_GOTO_SYMEX_SSA_REASSOCIATE_H
#define CPROVER_GOTO_SYMEX_SSA_REASSOCIATE_H

#include <util/algorithms.h>

/// SSA-level reassociation pass: walks every step in the symex equation and
/// reassociates add/sub chains in its expressions.
///
/// Catches reassoc opportunities exposed by symex value substitution and
/// constant propagation that the GOTO-level pass can't see — for example,
/// after a function is inlined and its arguments substituted, an
/// `(x + 5) - 5` shape may appear that wasn't present in the static program.
///
/// Runs before slicing so the slicer sees the canonicalized form.
class ssa_reassociate : public ssa_step_algorithm
{
public:
  ssa_reassociate() : ssa_step_algorithm(true)
  {
  }

  bool run(symex_target_equationt::SSA_stepst &steps) override;

  BigInt ignored() const override
  {
    return 0;
  }
};

#endif
