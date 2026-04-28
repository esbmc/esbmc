/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_mul --
 * one tracked, one fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RTZ ieee_mul: when one operand of a
 * second RTZ ieee_mul is tracked in ir_ra_interval_map and the other is a
 * fresh nondet variable, the tracked operand uses its stored interval while
 * the fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_dir, RTZ, double precision)
 * ------------------------------------------
 * First mul:  z = x * y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_tz::0, ra_hi_tz::0}
 *
 * Second mul:  w = z * x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_tz::0, ra_hi_tz::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   lo_r = min(ra_lo_tz::0 * x_smt, ra_hi_tz::0 * x_smt)
 *   hi_r = max(ra_lo_tz::0 * x_smt, ra_hi_tz::0 * x_smt)
 *   ra_lo_tz::1, ra_hi_tz::1 pinned via RTZ three-way ITE
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::0   -- first mul's lower bound declared
 *   ra_lo_tz::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo_tz::0|  -- tracked endpoint in hull product
 *   (ite           -- nested ITE for hull and RTZ three-way sign check
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 4; /* ROUND_TO_ZERO */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* first RTZ mul: both fresh -> point fallback; stored */
  double w = z * x; /* second RTZ mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
