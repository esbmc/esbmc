/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_mul --
 * one tracked, one fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-mul-rtz-one-fresh but exercises the
 * single-precision (float) path.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::0   -- first mul's lower bound declared
 *   ra_lo_tz::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo_tz::0|  -- tracked endpoint in hull product
 *   (ite           -- nested ITE present
 *   8388608        -- Z3 numerator for eps_rel_dir = 2^-23 (single)
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
  float z = x * y; /* first RTZ mul: both fresh -> point fallback; stored */
  float w = z * x; /* second RTZ mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
