/* Regression test: ROUND_TO_ZERO tight enclosure under --ir-ra, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-rounding-zero-tight-bounds/ but exercises the IEEE single-
 * precision (float) path.  Verifies that the get_single_eps_up() /
 * get_single_min_subnormal() branch in apply_ieee754_semantics is reached and
 * emits ra_lo_tz:: / ra_hi_tz:: symbols, not the weak fallback.
 *
 * PROOF SHAPE (B_dir, single precision)
 * --------------------------------------
 * Same sign-dependent enclosure as the double case:
 *   r >= 0:  fl_RTZ(r) in [r - B_dir(r),  r]
 *   r <  0:  fl_RTZ(r) in [r,  r + B_dir(r)]
 * where eps_rel_dir = 2^-23 (full machine epsilon for single, FLT_EPSILON).
 *
 * EPSILON IN SMT OUTPUT
 * ---------------------
 * get_single_eps_up() passes "1.1920928955078125e-07" to mk_smt_real.
 * This decimal is exactly 2^-23 = 1/8388608, so Z3 simplifies it to the
 * rational (/ 1 8388608) rather than keeping a large numerator.  The pattern
 * 8388608 (= 2^23) is matched in test.desc as the single-precision indicator,
 * distinguishing this path from the double-precision path (numerator
 * 22204460492503131).
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::      -- tight round-toward-zero path taken (not weak)
 *   ra_hi_tz::      -- tight round-toward-zero path taken
 *   \(ite           -- sign-conditional ITE present in formula
 *   8388608         -- Z3 denominator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>
#include <fenv.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  fesetround(FE_TOWARDZERO);
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x + y; /* rounding_mode == ROUND_TO_ZERO -> tight tz path */

  /* Always false in real/integer encoding: z == x+y exactly. */
  assert(z != x + y);
  return 0;
}
