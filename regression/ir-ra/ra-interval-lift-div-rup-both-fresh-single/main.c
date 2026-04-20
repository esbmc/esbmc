/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_div --
 * both operands fresh, single precision.
 *
 * PURPOSE
 * -------
 * Verifies the single-precision RUP ieee_div path with point-interval
 * fallback and the single-precision directed enclosure constant.
 *
 * PROOF SHAPE (B_dir, RUP, single precision)
 * -------------------------------------------
 * Both x and y are fresh.
 *   lo_r = hi_r = real_z
 * EbRUP([R,R]):
 *   ra_lo_up::0 = real_z                    (exact lower)
 *   ra_hi_up::0 = real_z + B_dir(real_z)   (widened upper)
 * with eps_rel_dir = 2^-23 (single precision).
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_up::0   -- RUP tight path taken
 *   ra_hi_up::0   -- RUP tight path taken
 *   8388608  -- Z3 denominator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x / y; /* both fresh -> point fallback */

  /* Always false in real/integer encoding: z == x / y exactly. */
  assert(z != x / y);
  return 0;
}
