/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_div --
 * both operands fresh, single precision.
 *
 * PURPOSE
 * -------
 * Verifies the single-precision RNA ieee_div path with point-interval
 * fallback and the single-precision nearest-mode enclosure constants.
 *
 * PROOF SHAPE (B_near, RNA, single precision)
 * --------------------------------------------
 * Both x and y are fresh.
 *   iv(x) = {x_smt, x_smt},  iv(y) = {y_smt, y_smt}  (point fallback)
 *   lo_r = hi_r = real_z
 * Eb_away([R,R]) applies with eps_rel_near = 2^-24 (single precision).
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- RNA tight path taken
 *   ra_hi_aw::0   -- RNA tight path taken
 *   5960464477539063  -- Z3 numerator for eps_rel_near = 2^-24 (single)
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
  float z = x / y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x / y exactly. */
  assert(z != x / y);
  return 0;
}
