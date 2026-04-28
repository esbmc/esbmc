/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_div --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Verifies proof-aligned compositional interval lifting for single-precision
 * RNA ieee_div when both operands are tracked. The admissibility-guarded
 * four-endpoint hull formula uses tracked-over-tracked endpoint quotients.
 *
 * PROOF SHAPE (B_near, RNA, single precision)
 * --------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second div:  w = z / z  (both operands tracked)
 *   iv(z) = {ra_lo_aw::0, ra_hi_aw::0} for numerator AND denominator
 *   admissible = (ra_lo_aw::0 > 0 || ra_hi_aw::0 < 0)
 *   Full hull includes q2 = ra_lo_aw::0 / ra_hi_aw::0 (cross-product)
 *   ra_lo_aw::1, ra_hi_aw::1 pinned via single-precision RNA enclosure
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- first div's lower bound declared
 *   ra_lo_aw::1   -- second div's lifted lower bound declared
 *   (/ |smt_conv::ra_lo_aw::0| |smt_conv::ra_hi_aw::0|  -- q2 cross-product
 *   5960464477539063  -- Z3 numerator for eps_rel_near = 2^-24 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x / y; /* first RNA div: both fresh -> point fallback; stored */
  float w = z / z; /* second RNA div: both operands tracked */

  /* Always false in real/integer encoding: w == z / z exactly. */
  assert(w != z / z);
  return 0;
}
