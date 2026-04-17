/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_mul --
 * one tracked, one fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-mul-rdn-one-fresh but exercises the
 * single-precision (float) path.
 *
 * PROOF SHAPE (B_dir, RDN, single precision)
 * ------------------------------------------
 * Second mul:  w = z * x  (z tracked, x fresh -> mixed path)
 *   lo_r = min(ra_lo_dn::0 * x_smt, ra_hi_dn::0 * x_smt)
 *   hi_r = max(ra_lo_dn::0 * x_smt, ra_hi_dn::0 * x_smt)
 *   ra_lo_dn::1 = lo_r - B_dir(lo_r)  [single eps]
 *   ra_hi_dn::1 = hi_r                (exact upper)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_dn::0   -- first mul's lower bound declared
 *   ra_lo_dn::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo_dn::0|  -- tracked endpoint in hull product
 *   (ite           -- nested ITE present
 *   8388608        -- Z3 numerator for eps_rel_dir = 2^-23 (single)
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
  float z = x * y; /* first RDN mul: both fresh -> point fallback; stored */
  float w = z * x; /* second RDN mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
