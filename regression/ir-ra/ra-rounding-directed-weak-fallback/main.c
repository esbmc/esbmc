/* Regression test: symbolic rounding mode uses weak enclosure fallback under --ir-ra.
 *
 * PURPOSE
 * -------
 * Verifies that a floating-point addition performed under a symbolic (non-
 * constant) rounding mode takes the weak unconstrained enclosure path.  All
 * five concrete rounding modes (ROUND_TO_EVEN, ROUND_TO_AWAY,
 * ROUND_TO_PLUS_INF, ROUND_TO_MINUS_INF, ROUND_TO_ZERO) now have dedicated
 * tight paths; this test exercises the remaining catch-all: a rounding_mode
 * that is not a compile-time constant.
 *
 * MECHANISM
 * ---------
 * __ESBMC_rounding_mode is assigned __VERIFIER_nondet_int(), which ESBMC
 * symex cannot constant-fold.  The rounding_mode operand of the ieee_add2t
 * IR node therefore remains symbolic.  In smt_conv::apply_ieee754_semantics,
 * is_constant_int2t returns false for every guard, so only the three weak
 * containment assertions are emitted:
 *   (assert (<= |ra_lo| result))
 *   (assert (<= result  |ra_hi|))
 *   (assert (<= |ra_lo| |ra_hi|))
 *
 * WHAT IS CHECKED
 * ---------------
 *   ra_lo_weak / ra_hi_weak symbols are declared  -- weak path was taken
 *   ^VERIFICATION FAILED$               -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern int __VERIFIER_nondet_int(void);
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode =
    __VERIFIER_nondet_int(); /* symbolic mode -> weak fallback */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y;

  /* Always false in real/integer encoding: z == x+y exactly.
   * Gives a deterministic VERIFICATION FAILED to confirm the run completed. */
  assert(z != x + y);
  return 0;
}
