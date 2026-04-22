/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_div --
 * numerator tracked, denominator fresh, single precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for single-precision RTZ ieee_div: when
 * the numerator is tracked and the denominator is fresh, the tracked
 * numerator endpoints appear in the hull quotients.
 *
 * PROOF SHAPE (B_dir, RTZ, single precision)
 * -------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_tz::0, ra_hi_tz::0}
 *
 * Second div:  w = z / x  (z tracked as numerator, x fresh as denominator)
 *   iv(z) = {ra_lo_tz::0, ra_hi_tz::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   Hull quotients: ra_lo_tz::0 / x_smt, ra_hi_tz::0 / x_smt
 *   RTZ enclosure applied sign-sensitively with eps_rel_dir = 2^-23.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::0   -- first div's lower bound declared
 *   ra_lo_tz::1   -- second div's mixed-path lower bound declared
 *   (/ |smt_conv::ra_lo_tz::0|  -- tracked endpoint in hull quotient
 *   (ite           -- ITE for hull sort / admissibility guard
 *   8388608  -- Z3 denominator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 4; /* ROUND_TO_ZERO */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x / y; /* first RTZ div: both fresh -> point fallback; stored */
  float w = z / x; /* second RTZ div: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z / x exactly. */
  assert(w != z / x);
  return 0;
}
