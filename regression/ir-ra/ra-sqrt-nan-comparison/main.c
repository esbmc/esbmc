/* Soundness test: sqrt(x < 0) with a comparison that would be trivially
 * VERIFICATION SUCCESSFUL if the result were pinned near zero.
 *
 * IEEE 754: sqrt(x < 0) = NaN.  NaN < 100.0 is false, so the assertion
 * should fail (VERIFICATION FAILED).
 *
 * Under --ir-ieee, the result for negative operands is the unconstrained
 * fresh real sqrt_nan::.  Without the ITE split introduced in the
 * ieee_sqrt_id handler, the enclosure constraints (applied to the inner
 * sqrt_pos:: symbol) would pin the observable result near zero, making
 * "s >= 100.0" unsatisfiable and yielding a wrong VERIFICATION SUCCESSFUL.
 *
 * With the ITE split, the observable result for operand < 0 is sqrt_nan
 * (unconstrained), so "s >= 100.0" is satisfiable and ESBMC correctly
 * reports VERIFICATION FAILED. */

#include <math.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x < -1.0); /* operand definitely negative */
  double s = sqrt(x);
  __ESBMC_assert(s < 100.0, "NaN < 100.0 is false; must be falsifiable");
  return 0;
}
