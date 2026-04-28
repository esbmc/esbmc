/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_div --
 * numerator tracked, denominator fresh, single precision.
 *
 * PURPOSE
 * -------
 * Verifies the single-precision mixed lookup path for RUP ieee_div: tracked
 * numerator endpoints appear in the hull quotients and the single-precision
 * directed constant is used.
 *
 * PROOF SHAPE (B_dir, RUP, single precision)
 * -------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_up::0, ra_hi_up::0}
 *
 * Second div:  w = z / x  (z tracked, x fresh)
 *   Hull quotients: ra_lo_up::0 / x_smt, ra_hi_up::0 / x_smt
 *   ra_lo_up::1 = lo_r               (exact lower)
 *   ra_hi_up::1 = hi_r + B_dir(hi_r) (widened upper, single-precision)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_up::0   -- first div's lower bound declared
 *   ra_lo_up::1   -- second div's mixed-path lower bound declared
 *   (/ |smt_conv::ra_lo_up::0|  -- tracked endpoint in hull quotient
 *   (ite           -- ITE for hull sort / admissibility guard
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
  float w = z / x; /* second RUP div: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z / x exactly. */
  assert(w != z / x);
  return 0;
}
