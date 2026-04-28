/* Regression test: RNE interval lifting for ieee_mul -- both operands fresh,
 * single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-mul-both-fresh but exercises the single-precision
 * (float) path. Verifies point-interval fallback for fresh operands.
 *
 * PROOF SHAPE (point-interval fallback, single precision)
 * -------------------------------------------------------
 * Both x and y are fresh (no prior tracked RNE mul).
 *   iv(x) = {x_smt, x_smt}  (point fallback)
 *   iv(y) = {y_smt, y_smt}  (point fallback)
 *   lo_r = hi_r = x_smt * y_smt = real_z
 * The helper receives lo_r == hi_r, producing tight RNE enclosure.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::          -- tight RNE path taken
 *   ra_hi::          -- tight RNE path taken
 *   (ite             -- |r| absolute value present
 *   5960464477539063 -- Z3 numerator for eps_rel = 2^-24 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x * y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x * y exactly. */
  assert(z != x * y);
  return 0;
}
