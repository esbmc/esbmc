/* Regression test: FMA interval lifting under --ir-ieee (RNA / ROUND_TO_AWAY).
 *
 * Verifies that encode_ieee_fma takes the RNA interval-lifted path when
 * __ESBMC_rounding_mode == 1 (ROUND_TO_AWAY).
 *
 * x in [1, 2],  y in [3, 4],  z in [5, 6]
 * FMA hull [8, 14]; RNA enclosure adds symmetric B_near bounds.
 * Assertion r > 100 is always false: VERIFICATION FAILED.
 *
 * Pattern ra_lo_aw:: / ra_hi_aw:: confirms the RNA tight path was taken.
 */
#include <assert.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = __VERIFIER_nondet_double();
  __ESBMC_assume(x >= 1.0 && x <= 2.0);
  __ESBMC_assume(y >= 3.0 && y <= 4.0);
  __ESBMC_assume(z >= 5.0 && z <= 6.0);

  double r = fma(x, y, z);

  assert(r > 100.0);
  return 0;
}
