/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_sub --
 * one tracked, one fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-sub-rtz-one-fresh but exercises the
 * single-precision (float) path. Verifies the mixed lookup path: when one
 * operand of a second RTZ ieee_sub is tracked and the other is a fresh
 * nondet variable, the tracked operand uses its stored interval while the
 * fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_dir, RTZ, single precision)
 * ------------------------------------------
 * First sub:  z = x - y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_tz::0, ra_hi_tz::0}
 *
 * Second sub:  w = z - x  (z tracked, x fresh -> mixed path)
 *   lo_r = ra_lo_tz::0 - x_smt
 *   hi_r = ra_hi_tz::0 - x_smt
 *   RTZ three-way ITE on sign of [lo_r, hi_r]:  [eps_rel_dir = 2^-23]
 *   ra_lo_tz::1 and ra_hi_tz::1 pinned accordingly
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::0   -- first subtraction's interval lower bound declared
 *   ra_lo_tz::1   -- second subtraction's mixed-path lower bound declared
 *   (- ra_lo_tz::0 ...  -- lo_r prefix confirming tracked lookup
 *   (ite           -- sign-conditional ITEs present
 *   8388608        -- Z3 denominator for eps_rel_dir = 2^-23 (single)
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
  float z = x - y;   /* first RTZ sub: both fresh -> point fallback; stored */
  float w = z - x;   /* second RTZ sub: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z - x exactly. */
  assert(w != z - x);
  return 0;
}
