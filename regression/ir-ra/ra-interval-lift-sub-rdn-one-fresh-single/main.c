/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_sub --
 * one tracked, one fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-sub-rdn-one-fresh but exercises the
 * single-precision (float) path. Verifies the mixed lookup path: when one
 * operand of a second RDN ieee_sub is tracked and the other is a fresh
 * nondet variable, the tracked operand uses its stored interval while the
 * fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_dir, RDN, single precision)
 * ------------------------------------------
 * First sub:  z = x - y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_dn::0, ra_hi_dn::0}
 *
 * Second sub:  w = z - x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_dn::0, ra_hi_dn::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   L_R2 = ra_lo_dn::0 - x_smt
 *   U_R2 = ra_hi_dn::0 - x_smt
 *   ra_lo_dn::1 = L_R2 - B_dir(L_R2)   [single eps: 2^-23]
 *   ra_hi_dn::1 = U_R2          (exact upper bound for RDN)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_dn::0   -- first subtraction's interval lower bound declared
 *   ra_lo_dn::1   -- second subtraction's mixed-path lower bound declared
 *   (- ra_lo_dn::0 ...  -- L_R2 prefix confirming tracked lookup
 *   (ite           -- absolute value present in B_dir computation
 *   8388608        -- Z3 denominator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 3; /* ROUND_TO_MINUS_INF */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x - y;   /* first RDN sub: both fresh -> point fallback; stored */
  float w = z - x;   /* second RDN sub: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z - x exactly. */
  assert(w != z - x);
  return 0;
}
