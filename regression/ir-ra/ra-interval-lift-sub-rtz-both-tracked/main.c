/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_sub --
 * both operands tracked, double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a second RTZ ieee_sub were themselves
 * results of a prior tracked RTZ ieee_sub, ir_ra_interval_map lookup fires
 * for both operands and the interval-lifted RTZ subtraction path is taken.
 *
 * This test also exercises the crossing-zero conservative fallback.
 * For the second sub w = z - z:
 *   iv(z) = {ra_lo_tz::0, ra_hi_tz::0}
 *   lo_r = ra_lo_tz::0 - ra_hi_tz::0  (<= 0, since ra_lo <= ra_hi)
 *   hi_r = ra_hi_tz::0 - ra_lo_tz::0  (>= 0, since ra_lo <= ra_hi)
 * So the hull crosses zero -> conservative B_dir_max bound fires.
 *
 * PROOF SHAPE (B_dir, RTZ, double precision)
 * ------------------------------------------
 * First sub:  z = x - y   (both fresh -> point-interval fallback)
 *   lo_r1 = hi_r1 = real_z (point hull)
 *   RTZ sign-ITE on real_z:  ra_lo_tz::0, ra_hi_tz::0
 *   stored: ir_ra_interval_map[real_z] = {ra_lo_tz::0, ra_hi_tz::0}
 *
 * Second sub:  w = z - z  (both operands are z -> both tracked)
 *   lo_r = ra_lo_tz::0 - ra_hi_tz::0  (crosses zero)
 *   hi_r = ra_hi_tz::0 - ra_lo_tz::0  (crosses zero)
 *   B_dir_max = eps_rel_dir * max(|lo_r|, |hi_r|) + eps_abs
 *   ra_lo_tz::1 = lo_r - B_dir_max
 *   ra_hi_tz::1 = hi_r + B_dir_max
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::0   -- first subtraction's interval lower bound declared
 *   ra_lo_tz::1   -- second subtraction's lifted lower bound declared
 *   (- ra_lo_tz::0 ra_hi_tz::0)  -- lo_r subterm confirming tracked lookup
 *   (ite           -- sign-conditional/max ITEs present
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
  double w = z - z;   /* second RTZ sub: both operands tracked -> crossing-zero */

  /* Always false in real/integer encoding: w == z - z exactly. */
  assert(w != z - z);
  return 0;
}
