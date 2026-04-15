/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_div --
 * both operands fresh (zero-regression sentinel), double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a RNA ieee_div are fresh nondet
 * variables, the point-interval fallback applies and the formula uses the
 * RNA enclosure with the nearest-mode double-precision constants.
 *
 * PROOF SHAPE (B_near, RNA, double precision)
 * -------------------------------------------
 * Both x and y are fresh.
 *   iv(x) = {x_smt, x_smt}  (point fallback)
 *   iv(y) = {y_smt, y_smt}  (point fallback)
 *   All four endpoint quotients collapse to x_smt / y_smt = real_z.
 *   lo_r = hi_r = real_z
 * Eb_away([R,R]) applies the RNA nearest enclosure.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- RNA tight path taken
 *   ra_hi_aw::0   -- RNA tight path taken
 *   5551115123125783  -- Z3 numerator for eps_rel_near = 2^-53 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x / y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x / y exactly. */
  assert(z != x / y);
  return 0;
}
