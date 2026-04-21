/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_sub --
 * both operands tracked, double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a second RDN ieee_sub were themselves
 * results of a prior tracked RDN ieee_sub, ir_ra_interval_map lookup fires
 * for both operands and the interval-lifted RDN subtraction path is taken.
 *
 * PROOF SHAPE (B_dir, RDN, double precision)
 * ------------------------------------------
 * First sub:  z = x - y   (both fresh -> point-interval fallback)
 *   iv(x) = {x, x},  iv(y) = {y, y}
 *   L_R1 = x - y = real_z,   U_R1 = x - y = real_z
 *   ra_lo_dn::0 = real_z - B_dir(real_z)
 *   ra_hi_dn::0 = real_z          (exact: RDN never rounds above)
 *   stored: ir_ra_interval_map[real_z] = {ra_lo_dn::0, ra_hi_dn::0}
 *
 * Second sub:  w = z - z  (both operands are z -> both tracked)
 *   iv(z) = {ra_lo_dn::0, ra_hi_dn::0}  (found in map, same entry twice)
 *   L_R2 = ra_lo_dn::0 - ra_hi_dn::0
 *   U_R2 = ra_hi_dn::0 - ra_lo_dn::0
 *   ra_lo_dn::1 = L_R2 - B_dir(L_R2)
 *   ra_hi_dn::1 = U_R2          (exact upper bound for second sub)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_dn::0   -- first subtraction's interval lower bound declared
 *   ra_lo_dn::1   -- second subtraction's lifted lower bound declared
 *   (- ra_lo_dn::0 ra_hi_dn::0)  -- L_R2 subterm confirming tracked lookup
 *   (ite           -- absolute value present in B_dir computation
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 3; /* ROUND_TO_MINUS_INF */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x - y;   /* first RDN sub: both fresh -> point fallback; stored */
  double w = z - z;   /* second RDN sub: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z - z exactly. */
  assert(w != z - z);
  return 0;
}
