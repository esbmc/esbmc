/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_mul --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-mul-rtz-both-tracked but exercises the
 * single-precision (float) path.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::0   -- first mul's lower bound declared
 *   ra_lo_tz::1   -- second mul's lifted lower bound declared
 *   (* |smt_conv::ra_lo_tz::0|  -- endpoint product in hull
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
  float w = z * z; /* second RTZ mul: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z * z exactly. */
  assert(w != z * z);
  return 0;
}
