/* Regression test: FMA interval lifting under --ir-ieee (RNE, double).
 *
 * PURPOSE
 * -------
 * Verifies that encode_ieee_fma takes the interval-lifted path when
 * --ir-ieee is active and the rounding mode is known (RNE).
 *
 * PROOF SHAPE
 * -----------
 * x in [1, 2],  y in [3, 4],  z in [5, 6]
 * Multiplication hull: min/max of {1*3, 1*4, 2*3, 2*4} = [3, 8]
 * FMA hull: [3+5, 8+6] = [8, 14]
 * After RNE enclosure: [8 - B(8), 14 + B(14)]  (tight, eps ~ 1e-16)
 * The assertion r > 100 is always false (r <= ~14), so a counterexample
 * exists: VERIFICATION FAILED.
 *
 * The formula patterns confirm the IR-IEEE interval-lifted path was taken:
 *   ra_lo::  -- RNE lower enclosure variable declared
 *   ra_hi::  -- RNE upper enclosure variable declared
 *   (*       -- multiplication products in hull computation
 *   5551115123125783  -- Z3 numerator for eps_rel = 2^-53 (double)
 */
#include <assert.h>
#include <math.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = __VERIFIER_nondet_double();
  __ESBMC_assume(x >= 1.0 && x <= 2.0);
  __ESBMC_assume(y >= 3.0 && y <= 4.0);
  __ESBMC_assume(z >= 5.0 && z <= 6.0);

  double r = fma(x, y, z);

  /* Maximum possible is fma(2,4,6) = 14; assertion is always false. */
  assert(r > 100.0);
  return 0;
}
