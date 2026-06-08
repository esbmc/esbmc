/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_div --
 * both operands fresh (zero-regression sentinel), double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a RTZ ieee_div are fresh nondet
 * variables, the point-interval fallback applies and the formula uses the
 * RTZ sign-sensitive enclosure with double-precision directed constants.
 * RTZ is sign-dependent: positive hull widens lower, negative hull widens
 * upper, zero-crossing hull uses symmetric B_dir_max.
 *
 * PROOF SHAPE (B_dir, RTZ, double precision)
 * ------------------------------------------
 * Both x and y are fresh.
 *   iv(x) = {x_smt, x_smt},  iv(y) = {y_smt, y_smt}  (point fallback)
 *   All four endpoint quotients collapse to x_smt / y_smt = real_z.
 *   lo_r = hi_r = real_z
 * EbRTZ([R,R]): sign-sensitive ITE on lo_nonneg / hi_nonpos.
 *   ra_lo_tz::0 and ra_hi_tz::0 set per sign case.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::0   -- RTZ path taken
 *   ra_hi_tz::0   -- RTZ path taken
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
  double z = x / y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x / y exactly. */
  assert(z != x / y);
  return 0;
}
