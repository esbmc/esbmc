/* Regression test: RNA (ROUND_TO_AWAY) does NOT use the interval-lifted path.
 *
 * PURPOSE
 * -------
 * The interval-lifted ieee_add path is guarded by is_nearest_rounding_mode(),
 * which checks for ROUND_TO_EVEN (value 0) only, not for ROUND_TO_AWAY
 * (value 1).  This test verifies that chained additions under RNA continue
 * to use the existing single-step apply_ieee754_semantics path, producing
 * ra_lo_aw:: / ra_hi_aw:: symbols -- not ra_lo:: / ra_hi:: from the
 * interval-lifting helper.
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::   -- RNA single-step path taken (not interval path)
 *   ra_hi_aw::   -- RNA single-step path taken
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY -- bypasses interval-lift guard */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y;   /* RNA add: must NOT enter interval-lifting path */
  double w = z + x;   /* RNA add: both single-step, ra_lo_aw:: expected */

  /* Always false in real/integer encoding. */
  assert(w != z + x);
  return 0;
}
