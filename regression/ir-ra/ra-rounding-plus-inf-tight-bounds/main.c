/* Regression test: ROUND_TO_PLUS_INF uses tight asymmetric enclosure under --ir-ra.
 *
 * PURPOSE
 * -------
 * Verifies that fesetround(FE_UPWARD) causes apply_ieee754_semantics to take
 * the tight ROUND_TO_PLUS_INF path, not the weak fallback.
 *
 * PROOF SHAPE (B_dir)
 * -------------------
 * For round-toward-+inf, fl_RUP(r) >= r always. The enclosure is asymmetric:
 *   fl_RUP(r) in [r,  r + B_dir(r)]
 * where B_dir(r) = eps_rel_dir * |r| + eps_abs
 * and eps_rel_dir = 2^-52 (full machine epsilon for double, DBL_EPSILON).
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_up::  -- tight round-up path taken (not weak fallback)
 *   ra_hi_up::  -- tight round-up path taken
 *   \(ite       -- |r| absolute value present (upper bound needs it)
 *   22204460492503131  -- Z3 rational numerator for eps_rel_dir = 2^-52
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>
#include <fenv.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  fesetround(FE_UPWARD);
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* rounding_mode == ROUND_TO_PLUS_INF -> tight up path */

  /* Always false in real/integer encoding: z == x+y exactly. */
  assert(z != x + y);
  return 0;
}
