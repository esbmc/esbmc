/* Regression test: RNE (ROUND_TO_EVEN) interval lifting for ieee_div --
 * both operands fresh (zero-regression sentinel), double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a RNE ieee_div are fresh nondet
 * variables (not in ir_ra_interval_map), the point-interval fallback applies
 * to the numerator, the denominator is used directly, and the resulting
 * formula uses the RNE enclosure over the degenerate hull lo_r = hi_r = real_z.
 *
 * PROOF SHAPE (point-interval fallback, collapses to single-step RNE)
 * -------------------------------------------------------------------
 * Both x and y are fresh (no prior tracked RNE div).
 *   iv(x) = {x_smt, x_smt}  (point fallback for numerator)
 *   denominator = y_smt      (point, always)
 *   d_lo = d_hi = x_smt / y_smt = real_z
 *   lo_r = hi_r = real_z
 * Eb_near([R,R]) applies the symmetric RNE enclosure.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0     -- RNE tight path taken
 *   ra_hi::0     -- RNE tight path taken
 *   5960464477539063  -- unused (double uses 5551115123125783)
 *   5551115123125783  -- Z3 numerator for eps_rel_near = 2^-53 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x / y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x / y exactly. */
  assert(z != x / y);
  return 0;
}
