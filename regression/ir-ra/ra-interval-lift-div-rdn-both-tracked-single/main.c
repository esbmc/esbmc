/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_div --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Verifies proof-aligned compositional interval lifting for single-precision
 * RDN ieee_div when both operands are tracked. The admissibility-guarded
 * four-endpoint hull uses tracked-over-tracked endpoint quotients.
 *
 * PROOF SHAPE (B_dir, RDN, single precision)
 * -------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_dn::0, ra_hi_dn::0}
 *
 * Second div:  w = z / z  (both operands tracked)
 *   iv(z) = {ra_lo_dn::0, ra_hi_dn::0} for numerator AND denominator
 *   Full hull includes q2 = ra_lo_dn::0 / ra_hi_dn::0 (cross-product)
 *   ra_lo_dn::1 = lo_r - B_dir(lo_r)   (widened lower, single-precision)
 *   ra_hi_dn::1 = hi_r                  (exact upper)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_dn::0   -- first div's lower bound declared
 *   ra_lo_dn::1   -- second div's lifted lower bound declared
 *   (/ |smt_conv::ra_lo_dn::0| |smt_conv::ra_hi_dn::0|  -- q2 cross-product
 *   8388608  -- Z3 denominator for eps_rel_dir = 2^-23 (single)
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
  float z = x / y; /* first RDN div: both fresh -> point fallback; stored */
  float w = z / z; /* second RDN div: both operands tracked */

  /* Always false in real/integer encoding: w == z / z exactly. */
  assert(w != z / z);
  return 0;
}
