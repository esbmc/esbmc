/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_div --
 * both operands fresh (zero-regression sentinel), double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a RDN ieee_div are fresh nondet
 * variables, the point-interval fallback applies and the formula uses the
 * RDN directed enclosure with double-precision directed constants.
 * The upper endpoint is exact; only the lower is widened.
 *
 * PROOF SHAPE (B_dir, RDN, double precision)
 * ------------------------------------------
 * Both x and y are fresh.
 *   iv(x) = {x_smt, x_smt},  iv(y) = {y_smt, y_smt}  (point fallback)
 *   All four endpoint quotients collapse to x_smt / y_smt = real_z.
 *   lo_r = hi_r = real_z
 * EbRDN([R,R]):
 *   ra_lo_dn::0 = real_z - B_dir(real_z)   (widened lower)
 *   ra_hi_dn::0 = real_z                    (exact upper)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_dn::0   -- RDN tight path taken
 *   ra_hi_dn::0   -- RDN tight path taken
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 3; /* ROUND_TO_MINUS_INF */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x / y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x / y exactly. */
  assert(z != x / y);
  return 0;
}
