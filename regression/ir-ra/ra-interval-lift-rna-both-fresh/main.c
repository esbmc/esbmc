/* Regression test: RNA (ROUND_TO_AWAY) interval lifting -- both fresh
 * (zero-regression sentinel).
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of an RNA ieee_add are fresh nondet
 * variables (not in ir_ra_interval_map), the point-interval fallback
 * applies to both, and the resulting formula is structurally equivalent to
 * the pre-lifting single-step RNA path.
 *
 * PROOF SHAPE (point-interval fallback, collapses to single-step RNA)
 * -------------------------------------------------------------------
 * Both x and y are fresh (no prior tracked RNA add).
 *   iv(x) = {x_smt, x_smt}  (point fallback)
 *   iv(y) = {y_smt, y_smt}  (point fallback)
 *   L_R = x + y = real_z,   U_R = x + y = real_z
 * The helper receives lo_r == hi_r == real_z, producing:
 *   ra_lo_aw = real_z - B_near(real_z)
 *   ra_hi_aw = real_z + B_near(real_z)
 * This is identical in structure to the pre-lifting single-step RNA formula.
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::    -- RNA tight path taken (same naming as single-step RNA)
 *   ra_hi_aw::    -- RNA tight path taken
 *   (ite           -- |r| absolute value present
 *   5551115123125783  -- Z3 numerator for eps_rel = 2^-53 (double, same as RNE)
 *   ^VERIFICATION FAILED$  -- run completed
 *
 * These are the same patterns as ra-rounding-away-tight-bounds, confirming
 * zero regression from the point-interval fallback on single RNA additions.
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y;   /* both fresh -> point fallback -> single-step equivalent */

  /* Always false in real/integer encoding: z == x + y exactly. */
  assert(z != x + y);
  return 0;
}
