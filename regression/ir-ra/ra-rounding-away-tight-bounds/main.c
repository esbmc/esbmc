/* Regression test: ROUND_TO_AWAY uses tight symmetric enclosure under --ir-ra.
 *
 * PURPOSE
 * -------
 * Verifies that __ESBMC_rounding_mode = 1 (ROUND_TO_AWAY) causes
 * apply_ieee754_semantics to take the tight ROUND_TO_AWAY path, not the weak
 * fallback.
 *
 * PROOF SHAPE (B_near, symmetric)
 * --------------------------------
 * ROUND_TO_AWAY is a nearest-rounding mode (round to nearest, ties away from
 * zero). Like ROUND_TO_EVEN, it uses the same nearest-mode linear enclosure:
 *   |fl_RTA(r) - r| <= eps_rel * |r| + eps_abs
 *
 * The enclosure is symmetric -- identical in shape to the ROUND_TO_EVEN path:
 *   ra_lo = r - B(r)
 *   ra_hi = r + B(r)
 * where B(r) = eps_rel * |r| + eps_abs
 * and eps_rel = 2^-53 (nearest-mode relative constant for double, same as
 * ROUND_TO_EVEN).
 *
 * HOW ROUND_TO_AWAY IS TRIGGERED
 * --------------------------------
 * ROUND_TO_AWAY (value 1 in ieee_floatt) has no standard fesetround() constant
 * in C, and ESBMC's fenv.c does not map any FE_* to value 1.
 * __ESBMC_rounding_mode is therefore written directly. ESBMC symex propagates
 * this concrete value into the rounding_mode operand of the ieee_add2t IR node,
 * and is_round_to_away fires in apply_ieee754_semantics.
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::          -- tight round-to-away path taken (not weak fallback)
 *   ra_hi_aw::          -- tight round-to-away path taken
 *   \(ite               -- |r| absolute value present
 *   5551115123125783    -- Z3 rational numerator for eps_rel = 2^-53 (double)
 *                          Same numerator as ROUND_TO_EVEN: expected and correct,
 *                          because ROUND_TO_AWAY uses the same nearest-mode
 *                          relative constant and symmetric enclosure.
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY -- no standard fesetround constant */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* rounding_mode == ROUND_TO_AWAY -> tight aw path */

  /* Always false in real/integer encoding: z == x+y exactly. */
  assert(z != x + y);
  return 0;
}
