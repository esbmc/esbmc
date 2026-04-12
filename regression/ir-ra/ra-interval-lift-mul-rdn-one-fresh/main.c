/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_mul --
 * one tracked, one fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RDN ieee_mul: when one operand of a
 * second RDN ieee_mul is tracked in ir_ra_interval_map and the other is a
 * fresh nondet variable, the tracked operand uses its stored interval while
 * the fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_dir, RDN, double precision)
 * ------------------------------------------
 * First mul:  z = x * y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_dn::0, ra_hi_dn::0}
 *
 * Second mul:  w = z * x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_dn::0, ra_hi_dn::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   p1 = ra_lo_dn::0 * x_smt
 *   p3 = ra_hi_dn::0 * x_smt
 *   lo_r = min(p1,p3),  hi_r = max(p1,p3)
 *   ra_lo_dn::1 = lo_r - B_dir(lo_r)  (widened lower)
 *   ra_hi_dn::1 = hi_r                (exact upper)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_dn::0   -- first mul's lower bound declared
 *   ra_lo_dn::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo_dn::0|  -- tracked endpoint in hull product
 *   (ite           -- nested ITE for min/max hull and |r| in B_dir
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
  double z = x * y; /* first RDN mul: both fresh -> point fallback; stored */
  double w = z * x; /* second RDN mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
