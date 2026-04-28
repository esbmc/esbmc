/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_sub --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-sub-rtz-both-tracked but exercises the
 * single-precision (float) path. Verifies that the get_single_eps_up() /
 * get_single_min_subnormal() branch in apply_ieee754_rtz_enclosure is
 * reached when both operands of a second RTZ ieee_sub are tracked.
 * Also exercises the crossing-zero conservative fallback (see both-tracked
 * double variant for the proof shape).
 *
 * PROOF SHAPE (B_dir, RTZ, single precision)
 * ------------------------------------------
 * Second sub: w = z - z (both tracked, hull crosses zero)
 *   B_dir_max = eps_rel_dir * max(|lo_r|, |hi_r|) + eps_abs  [eps = 2^-23]
 *   ra_lo_tz::1 = lo_r - B_dir_max
 *   ra_hi_tz::1 = hi_r + B_dir_max
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::0   -- first subtraction's interval lower bound declared
 *   ra_lo_tz::1   -- second subtraction's lifted lower bound declared
 *   (- ra_lo_tz::0 ra_hi_tz::0)  -- lo_r subterm confirming tracked lookup
 *   (ite           -- sign-conditional/max ITEs present
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
  float w = z - z;   /* second RTZ sub: both operands tracked -> crossing-zero */

  /* Always false in real/integer encoding: w == z - z exactly. */
  assert(w != z - z);
  return 0;
}
