/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_add --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rdn-both-tracked but exercises the single-precision
 * (float) path. Verifies that the get_single_eps_up() /
 * get_single_min_subnormal() branch in apply_ieee754_rdn_enclosure is reached
 * when both operands of a second RDN ieee_add are tracked.
 *
 * PROOF SHAPE (B_dir, RDN, single precision)
 * ------------------------------------------
 * Second add:  w = z + z  (both operands tracked -> full lift)
 *   L_R2 = ra_lo_dn::0 + ra_lo_dn::0
 *   U_R2 = ra_hi_dn::0 + ra_hi_dn::0
 *   ra_lo_dn::1 = L_R2 - B_dir(L_R2)   [single eps: 2^-23]
 *   ra_hi_dn::1 = U_R2          (exact upper: RDN never rounds above)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_dn::0   -- first addition's interval lower bound declared
 *   ra_lo_dn::1   -- second addition's lifted lower bound declared
 *   (+ ra_lo_dn::0 ra_lo_dn::0)  -- L_R2 subterm confirming tracked lookup
 *   (ite           -- absolute value present in B_dir computation
 *   8388608        -- Z3 denominator for eps_rel_dir = 2^-23 (single)
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
  float z = x + y;   /* first RDN add: both fresh -> point fallback; stored */
  float w = z + z;   /* second RDN add: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z + z exactly. */
  assert(w != z + z);
  return 0;
}
