/* Regression test: ROUND_TO_ZERO uses tight asymmetric enclosure under --ir-ra.
 *
 * PURPOSE
 * -------
 * Verifies that fesetround(FE_TOWARDZERO) causes apply_ieee754_semantics to
 * take the tight ROUND_TO_ZERO path, not the weak fallback.
 *
 * PROOF SHAPE (B_dir, sign-dependent)
 * -------------------------------------
 * ROUND_TO_ZERO truncates toward zero: it rounds down for r >= 0 and rounds
 * up for r < 0.  The enclosure is therefore sign-dependent:
 *
 *   r >= 0:  fl_RTZ(r) in [r - B_dir(r),  r]   (truncate-down shape)
 *   r <  0:  fl_RTZ(r) in [r,  r + B_dir(r)]   (truncate-up shape)
 *
 * Unified as:
 *   ra_lo = ite(r >= 0,  r - B_dir(r),  r)
 *   ra_hi = ite(r >= 0,  r,              r + B_dir(r))
 *
 * where B_dir(r) = eps_rel_dir * |r| + eps_abs
 * and eps_rel_dir = 2^-52 (full machine epsilon for double, DBL_EPSILON).
 * This is the same directed-mode constant used for ROUND_TO_PLUS_INF and
 * ROUND_TO_MINUS_INF; only the bound shape differs (sign-conditional here).
 *
 * MECHANISM
 * ---------
 * fesetround(FE_TOWARDZERO) writes 4 (ROUND_TO_ZERO) to __ESBMC_rounding_mode.
 * ESBMC symex propagates this concrete value into the rounding_mode operand of
 * the ieee_add2t IR node.  In smt_conv::apply_ieee754_semantics, the guard
 * is_round_to_zero fires and emits the ra_lo_tz:: / ra_hi_tz:: tight path.
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::          -- tight round-toward-zero path taken (not weak)
 *   ra_hi_tz::          -- tight round-toward-zero path taken
 *   \(ite               -- sign-conditional ITE present in formula
 *   22204460492503131   -- Z3 rational numerator for eps_rel_dir = 2^-52
 *                          (same as ROUND_TO_PLUS_INF / ROUND_TO_MINUS_INF)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>
#include <fenv.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  fesetround(FE_TOWARDZERO);
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* rounding_mode == ROUND_TO_ZERO -> tight tz path */

  /* Always false in real/integer encoding: z == x+y exactly. */
  assert(z != x + y);
  return 0;
}
