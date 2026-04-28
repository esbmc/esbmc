/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_mul --
 * both operands fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-mul-rtz-both-fresh but exercises the
 * single-precision (float) path.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::   -- RTZ tight path taken
 *   ra_hi_tz::   -- RTZ tight path taken
 *   (ite          -- three-way sign ITE present
 *   8388608       -- Z3 numerator for eps_rel_dir = 2^-23 (single)
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
  float z = x * y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x * y exactly. */
  assert(z != x * y);
  return 0;
}
