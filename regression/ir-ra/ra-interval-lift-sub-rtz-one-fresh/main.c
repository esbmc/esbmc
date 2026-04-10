/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_sub --
 * one tracked, one fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RTZ ieee_sub: when one operand of a
 * second RTZ ieee_sub is tracked in ir_ra_interval_map and the other is a
 * fresh nondet variable, the tracked operand uses its stored interval while
 * the fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_dir, RTZ, double precision)
 * ------------------------------------------
 * First sub:  z = x - y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_tz::0, ra_hi_tz::0}
 *
 * Second sub:  w = z - x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_tz::0, ra_hi_tz::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   lo_r = ra_lo_tz::0 - x_smt
 *   hi_r = ra_hi_tz::0 - x_smt
 *   RTZ three-way ITE on sign of [lo_r, hi_r]:
 *   ra_lo_tz::1 and ra_hi_tz::1 pinned accordingly
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::0   -- first subtraction's interval lower bound declared
 *   ra_lo_tz::1   -- second subtraction's mixed-path lower bound declared
 *   (- ra_lo_tz::0 ...  -- lo_r prefix confirming tracked lookup
 *   (ite           -- sign-conditional ITEs present
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
  double z = x - y;   /* first RTZ sub: both fresh -> point fallback; stored */
  double w = z - x;   /* second RTZ sub: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z - x exactly. */
  assert(w != z - x);
  return 0;
}
