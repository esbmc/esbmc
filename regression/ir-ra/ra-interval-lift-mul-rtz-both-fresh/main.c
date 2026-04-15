/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_mul --
 * both operands fresh (zero-regression sentinel).
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a RTZ ieee_mul are fresh nondet
 * variables (not in ir_ra_interval_map), the point-interval fallback applies
 * to both, and the resulting formula uses the RTZ enclosure over the
 * degenerate hull lo_r = hi_r = real_z.
 *
 * PROOF SHAPE (point-interval fallback, collapses to single-step RTZ)
 * -------------------------------------------------------------------
 * Both x and y are fresh (no prior tracked RTZ mul).
 *   iv(x) = {x_smt, x_smt},  iv(y) = {y_smt, y_smt}  (point fallback)
 *   p1=p2=p3=p4 = x_smt * y_smt = real_z
 *   lo_r = hi_r = real_z
 * EbRTZ([R,R]) applies the sign-sensitive three-way ITE.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_tz::   -- RTZ tight path taken
 *   ra_hi_tz::   -- RTZ tight path taken
 *   (ite          -- three-way sign ITE present
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 4; /* ROUND_TO_ZERO */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x * y exactly. */
  assert(z != x * y);
  return 0;
}
