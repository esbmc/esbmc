/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_mul --
 * both operands fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rna-mul-both-fresh but exercises the
 * single-precision (float) path. Verifies point-interval fallback for fresh
 * operands under RNA.
 *
 * PROOF SHAPE (point-interval fallback, single precision RNA)
 * -----------------------------------------------------------
 * Both x and y are fresh (no prior tracked RNA mul).
 *   lo_r = hi_r = x_smt * y_smt = real_z
 * The helper receives lo_r == hi_r, producing tight RNA enclosure.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::   -- RNA tight path taken
 *   ra_hi_aw::   -- RNA tight path taken
 *   (ite          -- |r| absolute value present
 *   5960464477539063  -- Z3 numerator for eps_rel = 2^-24 (single, same as RNE)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x * y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x * y exactly. */
  assert(z != x * y);
  return 0;
}
