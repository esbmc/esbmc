#include <goto-symex/ssa_reassociate.h>

#include <util/expr_reassociate.h>

bool ssa_reassociate::run(symex_target_equationt::SSA_stepst &steps)
{
  for (auto &step : steps)
  {
    // The fields populated depend on the step type, but reassociate_arith
    // is a no-op on nil exprs, so blindly calling it on every field is
    // safe and avoids an extra dispatch.
    reassociate_arith(step.guard);
    reassociate_arith(step.lhs);
    reassociate_arith(step.rhs);
    reassociate_arith(step.cond);
    for (auto &arg : step.output_args)
      reassociate_arith(arg);
  }
  return true;
}
