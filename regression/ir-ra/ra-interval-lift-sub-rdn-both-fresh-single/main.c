/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_sub --
 * both operands fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-sub-rdn-both-fresh but exercises the
 * single-precision (float) path. Verifies that when both operands are fresh
 * nondet variables, the point-interval fallback collapses to the single-step
 * RDN formula with single-precision epsilon constants.
 *
 * PROOF SHAPE (B_dir, RDN, single precision)
 * ------------------------------------------
 * Both x and y are fresh; lo_r == hi_r == real_z, producing:
 *   ra_lo_dn = real_z - B_dir(real_z)   [eps_rel_dir = 2^-23]
 *   ra_hi_dn = real_z          (exact upper: RDN never rounds above true value)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_dn::       -- RDN tight path taken (single precision)
 *   ra_hi_dn::       -- RDN tight path taken
 *   (ite             -- |r| absolute value present in B_dir computation
 *   8388608          -- Z3 denominator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 3; /* ROUND_TO_MINUS_INF */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x - y; /* RDN sub: both fresh -> point fallback */

  /* Always false in real/integer encoding. */
  assert(z != x - y);
  return 0;
}
