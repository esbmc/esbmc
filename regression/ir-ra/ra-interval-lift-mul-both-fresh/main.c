/* Regression test: RNE interval lifting for ieee_mul -- both operands fresh
 * (zero-regression sentinel).
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of an ieee_mul are fresh nondet variables
 * (not in ir_ra_interval_map), the point-interval fallback is applied to
 * both, and the resulting formula uses the RNE enclosure over the
 * degenerate hull lo_r = hi_r = real_z.
 *
 * PROOF SHAPE (point-interval fallback, collapses to single-step)
 * ---------------------------------------------------------------
 * Both x and y are fresh (no prior tracked RNE mul).
 *   iv(x) = {x_smt, x_smt}  (point fallback)
 *   iv(y) = {y_smt, y_smt}  (point fallback)
 *   p1=p2=p3=p4 = x_smt * y_smt = real_z
 *   lo_r = hi_r = real_z
 * The helper receives lo_r == hi_r, producing:
 *   ra_lo = real_z - B_near(real_z)
 *   ra_hi = real_z + B_near(real_z)
 * This is structurally equivalent to the pre-lifting single-step formula.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::          -- tight RNE path taken
 *   ra_hi::          -- tight RNE path taken
 *   (ite             -- |r| absolute value present
 *   5551115123125783 -- Z3 numerator for eps_rel = 2^-53 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* both fresh -> point fallback -> single-step equivalent */

  /* Always false in real/integer encoding: z == x * y exactly. */
  assert(z != x * y);
  return 0;
}
