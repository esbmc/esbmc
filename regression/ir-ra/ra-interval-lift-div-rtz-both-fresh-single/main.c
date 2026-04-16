/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_div --
 * both operands fresh, single precision.
 *
 * PURPOSE
 * -------
 * Verifies the single-precision RTZ ieee_div path with point-interval
 * fallback and the single-precision directed enclosure constant.
 *
 * PROOF SHAPE (B_dir, RTZ, single precision)
 * -------------------------------------------
 * Both x and y are fresh.
 *   lo_r = hi_r = real_z
 * EbRTZ([R,R]): sign-sensitive ITE on lo_nonneg / hi_nonpos.
 *   ra_lo_tz::0 and ra_hi_tz::0 set per sign case.
 * with eps_rel_dir = 2^-23 (single precision).
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::0   -- RTZ path taken
 *   ra_hi_tz::0   -- RTZ path taken
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
  float z = x / y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x / y exactly. */
  assert(z != x / y);
  return 0;
}
