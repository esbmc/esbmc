/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_add --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rtz-both-tracked but exercises the single-precision
 * (float) path. Verifies that the get_single_eps_up() /
 * get_single_min_subnormal() branch in apply_ieee754_rtz_enclosure is reached
 * when both operands of a second RTZ ieee_add are tracked.
 *
 * PROOF SHAPE (B_dir, RTZ, single precision)
 * ------------------------------------------
 * Second add:  w = z + z  (both operands tracked)
 *   lo_r = ra_lo_tz::0 + ra_lo_tz::0
 *   hi_r = ra_hi_tz::0 + ra_hi_tz::0
 *   RTZ three-way ITE on sign of [lo_r, hi_r]  [eps_rel_dir = 2^-23]
 *   ra_lo_tz::1 and ra_hi_tz::1 pinned accordingly
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::0   -- first addition's interval lower bound declared
 *   ra_lo_tz::1   -- second addition's lifted lower bound declared
 *   (+ ra_lo_tz::0 ra_lo_tz::0)  -- lo_r subterm confirming tracked lookup
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
  float z = x + y;   /* first RTZ add: both fresh -> point fallback; stored */
  float w = z + z;   /* second RTZ add: both operands tracked */

  /* Always false in real/integer encoding: w == z + z exactly. */
  assert(w != z + z);
  return 0;
}
