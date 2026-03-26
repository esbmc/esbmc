/* Regression test: directed rounding mode uses weak enclosure fallback under --ir-ra.
 *
 * PURPOSE
 * -------
 * Verifies that a floating-point addition performed under ROUND_TO_AWAY takes
 * the weak unconstrained enclosure path, not any theorem-driven tight path.
 *
 * MECHANISM
 * ---------
 * ROUND_TO_AWAY (value 1 in ieee_floatt) has no standard fesetround() constant
 * in C, so __ESBMC_rounding_mode is written directly.  ESBMC symex propagates
 * this concrete value into the rounding_mode operand of the ieee_add2t IR node.
 * In smt_conv::apply_ieee754_semantics, none of the guards fire:
 *   is_nearest_rounding_mode  (value 1 != ROUND_TO_EVEN == 0)
 *   is_round_to_plus_inf      (value 1 != ROUND_TO_PLUS_INF == 2)
 *   is_round_to_minus_inf     (value 1 != ROUND_TO_MINUS_INF == 3)
 *   is_round_to_zero          (value 1 != ROUND_TO_ZERO == 4)
 * so only the three weak containment assertions are emitted:
 *   (assert (<= |ra_lo| result))
 *   (assert (<= result  |ra_hi|))
 *   (assert (<= |ra_lo| |ra_hi|))
 *
 * WHAT IS CHECKED
 * ---------------
 *   ra_lo_weak / ra_hi_weak symbols are declared  -- weak path was taken
 *   ^VERIFICATION FAILED$               -- run completed
 *
 * NOTE: the tight-path epsilon numerator (5551115123125783 for double) would
 * be absent from the SMT output because the tight path was not taken.  This
 * cannot be expressed as a negative pattern in the current testing harness, so
 * absence is documented here rather than machine-checked.
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY -- no standard fesetround constant */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* rounding_mode == ROUND_TO_AWAY -> weak fallback */

  /* Always false in real/integer encoding: z == x+y exactly.
   * Gives a deterministic VERIFICATION FAILED to confirm the run completed. */
  assert(z != x + y);
  return 0;
}
