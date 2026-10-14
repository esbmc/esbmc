/* Soundness test: sqrt with a definitely negative operand must not make
 * the SMT formula UNSAT.
 *
 * The quadratic axiom  s >= 0 ∧ s² = operand  has no real solution when
 * operand < 0.  If the axiom is asserted unconditionally, the formula
 * becomes UNSAT for this path, and ESBMC would report VERIFICATION
 * SUCCESSFUL — incorrectly discarding the counterexample.
 *
 * With the guarded axiom  (operand >= 0) → (s >= 0 ∧ s² = operand),
 * the formula remains SAT when operand < 0.  The observable result is an
 * unconstrained real fallback, so the path is not eliminated and the
 * assertion remains falsifiable.  Therefore VERIFICATION FAILED is the
 * correct result.
 *
 * IEEE 754 note: sqrt(x < 0) = NaN.  NaN > 0.0 is false, so the
 * assertion would indeed fail for this input.  The --ir-ieee encoding
 * does not model NaN directly; instead, the result is unconstrained
 * (see comment in smt_conv.cpp ieee_sqrt_id handler). */

#include <math.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x < -1.0); /* operand definitely negative */
  double s = sqrt(x);
  __ESBMC_assert(s > 0.0, "falsifiable: NaN is not > 0");
  return 0;
}
