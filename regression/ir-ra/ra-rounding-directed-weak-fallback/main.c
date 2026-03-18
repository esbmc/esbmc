/* Regression test: directed rounding mode uses weak enclosure fallback under --ir-ra.
 *
 * PURPOSE
 * -------
 * Verifies that a floating-point addition performed under a directed rounding
 * mode (FE_UPWARD) takes the weak unconstrained enclosure path, not the
 * theorem-driven tight path that is valid only for round-to-nearest.
 *
 * MECHANISM
 * ---------
 * fesetround(FE_UPWARD) writes 2 (ROUND_TO_PLUS_INF) to __ESBMC_rounding_mode.
 * ESBMC symex propagates this concrete value into the rounding_mode operand of
 * the ieee_add2t IR node.  In smt_conv::apply_ieee754_semantics, the guard:
 *
 *   if (!is_nearest_rounding_mode(rounding_mode))
 *
 * fires (value 2 != ROUND_TO_EVEN == 0), so only the three weak containment
 * assertions are emitted:
 *   (assert (<= |ra_lo| result))
 *   (assert (<= result  |ra_hi|))
 *   (assert (<= |ra_lo| |ra_hi|))
 *
 * WHAT IS CHECKED
 * ---------------
 *   ra_lo / ra_hi symbols are declared  -- weak path was taken
 *   ^VERIFICATION FAILED$               -- run completed
 *
 * NOTE: the tight-path epsilon numerator (5551115123125783 for double) would
 * be absent from the SMT output because the tight path was not taken.  This
 * cannot be expressed as a negative pattern in the current testing harness, so
 * absence is documented here rather than machine-checked.
 */
#include <assert.h>
#include <fenv.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  fesetround(FE_UPWARD);
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* rounding_mode == ROUND_TO_PLUS_INF -> weak fallback */

  /* Always false in real/integer encoding: z == x+y exactly.
   * Gives a deterministic VERIFICATION FAILED to confirm the run completed. */
  assert(z != x + y);
  return 0;
}
