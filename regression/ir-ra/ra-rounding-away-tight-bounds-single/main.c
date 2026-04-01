/* Regression test: ROUND_TO_AWAY tight enclosure under --ir-ieee, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-rounding-away-tight-bounds/ but exercises the IEEE single-
 * precision (float) path. Verifies that the get_single_eps_rel() /
 * get_single_min_subnormal() branch in apply_ieee754_semantics is reached and
 * emits ra_lo_aw:: / ra_hi_aw:: symbols, not the weak fallback.
 *
 * PROOF SHAPE (B_near, single precision)
 * --------------------------------------
 * ROUND_TO_AWAY uses the same symmetric nearest-mode enclosure as the double
 * case:
 *   ra_lo = r - B(r)
 *   ra_hi = r + B(r)
 * where B(r) = eps_rel * |r| + eps_abs
 * and eps_rel = 2^-24 (nearest-mode relative constant for single, same as
 * ROUND_TO_EVEN).
 *
 * EPSILON IN SMT OUTPUT
 * ---------------------
 * get_single_eps_rel() passes "5.960464477539063e-08" to mk_smt_real.
 * This decimal does not simplify to a small rational, so Z3 retains the
 * large numerator 5960464477539063 in its output -- same numerator as the
 * single-precision ROUND_TO_EVEN test. This is expected and correct.
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::          -- tight round-to-away path taken
 *   ra_hi_aw::          -- tight round-to-away path taken
 *   \(ite               -- |r| absolute value present
 *   5960464477539063    -- Z3 numerator for eps_rel = 2^-24 (single)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY -- no standard fesetround constant */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x + y; /* rounding_mode == ROUND_TO_AWAY -> tight aw path */

  /* Always false in real/integer encoding: z == x+y exactly. */
  assert(z != x + y);
  return 0;
}
