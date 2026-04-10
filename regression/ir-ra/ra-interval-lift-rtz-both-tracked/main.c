/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_add --
 * both operands tracked, double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a second RTZ ieee_add were themselves
 * results of a prior tracked RTZ ieee_add, ir_ra_interval_map lookup fires
 * for both operands and the interval-lifted RTZ path is taken.
 *
 * This test also exercises the crossing-zero conservative fallback.
 * For the second add w = z + z:
 *   iv(z) = {ra_lo_tz::0, ra_hi_tz::0}
 *   lo_r = ra_lo_tz::0 + ra_lo_tz::0  (= 2 * ra_lo_tz::0)
 *   hi_r = ra_hi_tz::0 + ra_hi_tz::0  (= 2 * ra_hi_tz::0)
 * Since ra_lo_tz::0 <= ra_hi_tz::0 and both can be negative or positive
 * depending on the value of z, the hull can cross zero.
 *
 * PROOF SHAPE (B_dir, RTZ, double precision)
 * ------------------------------------------
 * First add:  z = x + y   (both fresh -> point-interval fallback)
 *   lo_r1 = hi_r1 = real_z (point hull)
 *   RTZ sign-ITE on real_z:  ra_lo_tz::0, ra_hi_tz::0
 *   stored: ir_ra_interval_map[real_z] = {ra_lo_tz::0, ra_hi_tz::0}
 *
 * Second add:  w = z + z  (both operands are z -> both tracked)
 *   lo_r = ra_lo_tz::0 + ra_lo_tz::0
 *   hi_r = ra_hi_tz::0 + ra_hi_tz::0
 *   RTZ three-way ITE on sign of [lo_r, hi_r]:
 *   ra_lo_tz::1 and ra_hi_tz::1 pinned accordingly
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::0   -- first addition's interval lower bound declared
 *   ra_lo_tz::1   -- second addition's lifted lower bound declared
 *   (+ ra_lo_tz::0 ra_lo_tz::0)  -- lo_r subterm confirming tracked lookup
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
  double z = x + y;   /* first RTZ add: both fresh -> point fallback; stored */
  double w = z + z;   /* second RTZ add: both operands tracked */

  /* Always false in real/integer encoding: w == z + z exactly. */
  assert(w != z + z);
  return 0;
}
