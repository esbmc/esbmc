/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_sub --
 * both operands fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-sub-rup-both-fresh but exercises the
 * single-precision (float) path. Verifies that when both operands are fresh
 * nondet variables, the point-interval fallback collapses to the single-step
 * RUP formula with single-precision epsilon constants.
 *
 * PROOF SHAPE (B_dir, RUP, single precision)
 * ------------------------------------------
 * Both x and y are fresh; lo_r == hi_r == real_z, producing:
 *   ra_lo_up = real_z          (exact lower: RUP never rounds below true value)
 *   ra_hi_up = real_z + B_dir(real_z)   [eps_rel_dir = 2^-23]
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_up::       -- RUP tight path taken (single precision)
 *   ra_hi_up::       -- RUP tight path taken
 *   (ite             -- |r| absolute value present in B_dir computation
 *   8388608          -- Z3 denominator for eps_rel_dir = 2^-23 (single)
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
  float z = x - y; /* RUP sub: both fresh -> point fallback */

  /* Always false in real/integer encoding. */
  assert(z != x - y);
  return 0;
}
