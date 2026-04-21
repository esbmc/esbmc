/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_sub --
 * both operands fresh, single precision (zero-regression sentinel).
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rna-sub-both-fresh but exercises the
 * single-precision (float) path. Verifies that when both operands are fresh
 * nondet variables, the point-interval fallback collapses to the single-step
 * RNA formula with single-precision epsilon constants.
 *
 * PROOF SHAPE (point-interval fallback, single precision)
 * -------------------------------------------------------
 * Both x and y are fresh; lo_r == hi_r == real_z, producing:
 *   ra_lo_aw = real_z - B_near(real_z)   [eps_rel = 2^-24]
 *   ra_hi_aw = real_z + B_near(real_z)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::       -- RNA tight path taken
 *   ra_hi_aw::       -- RNA tight path taken
 *   (ite             -- |r| absolute value present
 *   5960464477539063 -- Z3 numerator for eps_rel = 2^-24 (single)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x - y;   /* both operands fresh -> point fallback -> single-step equivalent */

  /* Always false in real/integer encoding: z == x - y exactly. */
  assert(z != x - y);
  return 0;
}
