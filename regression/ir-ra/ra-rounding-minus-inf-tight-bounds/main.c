/* Regression test: ROUND_TO_MINUS_INF uses tight asymmetric enclosure under --ir-ieee.
 *
 * PURPOSE
 * -------
 * Verifies that fesetround(FE_DOWNWARD) causes apply_ieee754_semantics to take
 * the tight ROUND_TO_MINUS_INF path, not the weak fallback.
 *
 * PROOF SHAPE (B_dir)
 * -------------------
 * For round-toward-minus-inf, fl_RDN(r) <= r always. The enclosure is asymmetric:
 *   fl_RDN(r) in [r - B_dir(r),  r]
 * where B_dir(r) = eps_rel_dir * |r| + eps_abs
 * and eps_rel_dir = 2^-52 (full machine epsilon for double, DBL_EPSILON).
 * This is the same directed-mode constant used for ROUND_TO_PLUS_INF; only
 * the bound shape is mirrored (ra_hi is now the exact side, ra_lo the loose side).
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_dn::  -- tight round-down path taken (not weak fallback)
 *   ra_hi_dn::  -- tight round-down path taken
 *   \(ite       -- |r| absolute value present (lower bound B_dir needs it)
 *   22204460492503131  -- Z3 rational numerator for eps_rel_dir = 2^-52
 *                         (same as ROUND_TO_PLUS_INF; this is expected and correct)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>
#include <fenv.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  fesetround(FE_DOWNWARD);
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* rounding_mode == ROUND_TO_MINUS_INF -> tight down path */

  /* Always false in real/integer encoding: z == x+y exactly. */
  assert(z != x + y);
  return 0;
}
