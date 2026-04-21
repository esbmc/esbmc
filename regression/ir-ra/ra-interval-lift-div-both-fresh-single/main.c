/* Regression test: RNE (ROUND_TO_EVEN) interval lifting for ieee_div --
 * both operands fresh (zero-regression sentinel), single precision.
 *
 * PURPOSE
 * -------
 * Verifies the single-precision path for RNE ieee_div: when both operands
 * are fresh nondet variables, the point-interval fallback applies and the
 * formula uses the single-precision RNE enclosure constants.
 *
 * PROOF SHAPE (point-interval fallback, collapses to single-step RNE)
 * -------------------------------------------------------------------
 * Both x and y are fresh.
 *   iv(x) = {x_smt, x_smt}  (point fallback)
 *   denominator = y_smt      (point)
 *   d_lo = d_hi = x_smt / y_smt = real_z
 *   lo_r = hi_r = real_z
 * Eb_near([R,R]) applies with eps_rel_near = 2^-24 (single precision).
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0     -- RNE tight path taken
 *   ra_hi::0     -- RNE tight path taken
 *   5960464477539063  -- Z3 numerator for eps_rel_near = 2^-24 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x / y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x / y exactly. */
  assert(z != x / y);
  return 0;
}
