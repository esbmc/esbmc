/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_div --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Verifies proof-aligned compositional interval lifting for single-precision
 * RUP ieee_div when both operands are tracked. The admissibility-guarded
 * four-endpoint hull uses tracked-over-tracked endpoint quotients.
 *
 * PROOF SHAPE (B_dir, RUP, single precision)
 * -------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_up::0, ra_hi_up::0}
 *
 * Second div:  w = z / z  (both operands tracked)
 *   iv(z) = {ra_lo_up::0, ra_hi_up::0} for numerator AND denominator
 *   Full hull includes q2 = ra_lo_up::0 / ra_hi_up::0 (cross-product)
 *   ra_lo_up::1 = lo_r               (exact lower)
 *   ra_hi_up::1 = hi_r + B_dir(hi_r) (widened upper, single-precision)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_up::0   -- first div's lower bound declared
 *   ra_lo_up::1   -- second div's lifted lower bound declared
 *   (/ |smt_conv::ra_lo_up::0| |smt_conv::ra_hi_up::0|  -- q2 cross-product
 *   8388608  -- Z3 denominator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x / y; /* first RUP div: both fresh -> point fallback; stored */
  float w = z / z; /* second RUP div: both operands tracked */

  /* Always false in real/integer encoding: w == z / z exactly. */
  assert(w != z / z);
  return 0;
}
