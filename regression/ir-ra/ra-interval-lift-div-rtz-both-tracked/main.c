/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_div --
 * both operands tracked, double precision.
 *
 * PURPOSE
 * -------
 * Verifies proof-aligned compositional interval lifting for RTZ ieee_div
 * when both the numerator and denominator are prior tracked results.
 * The four-endpoint hull uses tracked-over-tracked endpoint quotients in
 * the admissible branch. RTZ enclosure is sign-sensitive: positive hull
 * widens lower only, negative hull widens upper only, zero-crossing hull
 * uses symmetric B_dir_max conservative fallback.
 *
 * PROOF SHAPE (B_dir, RTZ, double precision)
 * ------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_tz::0, ra_hi_tz::0}
 *
 * Second div:  w = z / z  (both operands tracked)
 *   iv(z) = {ra_lo_tz::0, ra_hi_tz::0} for numerator AND denominator
 *   admissible = (ra_lo_tz::0 > 0 || ra_hi_tz::0 < 0)
 *   If admissible: four-endpoint hull
 *     q1 = ra_lo_tz::0 / ra_lo_tz::0
 *     q2 = ra_lo_tz::0 / ra_hi_tz::0
 *     q3 = ra_hi_tz::0 / ra_lo_tz::0
 *     q4 = ra_hi_tz::0 / ra_hi_tz::0
 *   RTZ enclosure applied to [lo_r, hi_r] with sign-sensitive ITE.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::0   -- first div's lower bound declared
 *   ra_lo_tz::1   -- second div's lifted lower bound declared
 *   (/ |smt_conv::ra_lo_tz::0| |smt_conv::ra_lo_tz::0|  -- q1
 *   (/ |smt_conv::ra_lo_tz::0| |smt_conv::ra_hi_tz::0|  -- q2
 *   (< 0.0 |smt_conv::ra_lo_tz::0|  -- admissibility guard
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
  double z = x / y; /* first RTZ div: both fresh -> point fallback; stored */
  double w = z / z; /* second RTZ div: both operands tracked */

  /* Always false in real/integer encoding: w == z / z exactly. */
  assert(w != z / z);
  return 0;
}
