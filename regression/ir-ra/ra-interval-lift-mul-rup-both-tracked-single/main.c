/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_mul --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-mul-rup-both-tracked but exercises the
 * single-precision (float) path.
 *
 * PROOF SHAPE (B_dir, RUP, single precision)
 * ------------------------------------------
 * Second mul:  w = z * z  (both tracked)
 *   lo_r = min(p1,p2,p3,p4), hi_r = max(p1,p2,p3,p4)
 *   ra_lo_up::1 = lo_r,  ra_hi_up::1 = hi_r + B_dir(hi_r)  [single eps]
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_up::0   -- first mul's lower bound declared
 *   ra_lo_up::1   -- second mul's lifted lower bound declared
 *   (* |smt_conv::ra_lo_up::0|  -- endpoint product in hull
 *   (ite           -- nested ITE present
 *   8388608        -- Z3 numerator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x * y; /* first RUP mul: both fresh -> point fallback; stored */
  float w = z * z; /* second RUP mul: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z * z exactly. */
  assert(w != z * z);
  return 0;
}
