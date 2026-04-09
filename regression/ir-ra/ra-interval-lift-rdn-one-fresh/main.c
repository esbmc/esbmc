/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_add --
 * one tracked, one fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RDN ieee_add: when one operand of a
 * second RDN ieee_add is tracked in ir_ra_interval_map and the other is a
 * fresh nondet variable, the tracked operand uses its stored interval while
 * the fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_dir, RDN, double precision)
 * ------------------------------------------
 * First add:  z = x + y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_dn::0, ra_hi_dn::0}
 *
 * Second add:  w = z + x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_dn::0, ra_hi_dn::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   L_R2 = ra_lo_dn::0 + x_smt
 *   U_R2 = ra_hi_dn::0 + x_smt
 *   ra_lo_dn::1 = L_R2 - B_dir(L_R2)
 *   ra_hi_dn::1 = U_R2          (exact upper bound for RDN)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_dn::0   -- first addition's interval lower bound declared
 *   ra_lo_dn::1   -- second addition's mixed-path lower bound declared
 *   (+ ra_lo_dn::0 ...  -- L_R2 prefix confirming tracked lookup
 *   (ite           -- absolute value present in B_dir computation
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
  double z = x + y;   /* first RDN add: both fresh -> point fallback; stored */
  double w = z + x;   /* second RDN add: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z + x exactly. */
  assert(w != z + x);
  return 0;
}
