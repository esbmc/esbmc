/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_mul --
 * both operands fresh (zero-regression sentinel).
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a RUP ieee_mul are fresh nondet
 * variables (not in ir_ra_interval_map), the point-interval fallback applies
 * to both, and the resulting formula uses the RUP enclosure over the
 * degenerate hull lo_r = hi_r = real_z.
 *
 * PROOF SHAPE (point-interval fallback, collapses to single-step RUP)
 * -------------------------------------------------------------------
 * Both x and y are fresh (no prior tracked RUP mul).
 *   iv(x) = {x_smt, x_smt},  iv(y) = {y_smt, y_smt}  (point fallback)
 *   p1=p2=p3=p4 = x_smt * y_smt = real_z
 *   lo_r = hi_r = real_z
 * EbRUP([R,R]) = [R, R + B_dir(R)]:
 *   ra_lo_up = real_z          (exact lower: RUP never rounds below)
 *   ra_hi_up = real_z + B_dir(real_z)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_up::   -- RUP tight path taken
 *   ra_hi_up::   -- RUP tight path taken
 *   (ite          -- |r| absolute value in B_dir computation
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x * y exactly. */
  assert(z != x * y);
  return 0;
}
