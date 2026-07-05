/* Regression test: FMA interval-lifted path produces a sound result
 * under --ir-ieee with nondet bounded inputs (CORE, no formula-shape check).
 *
 * x in [1, 2],  y in [3, 4],  z in [5, 6]
 * Exact FMA range: x*y + z in [8, 14].
 * After RNE enclosure the interval is [8 - eps, 14 + eps] where
 * eps ~ 2^-53 * 14 + 2^-1074 << 1, so the lower bound stays well above 7.
 *
 * assert(r > 7.0) must hold: VERIFICATION SUCCESSFUL confirms that the
 * interval-lifted FMA path does not over-approximate into a false alarm
 * on bounded nondet inputs. */

#include <assert.h>
#include <math.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = __VERIFIER_nondet_double();
  __ESBMC_assume(x >= 1.0 && x <= 2.0);
  __ESBMC_assume(y >= 3.0 && y <= 4.0);
  __ESBMC_assume(z >= 5.0 && z <= 6.0);

  double r = fma(x, y, z);

  assert(r > 7.0);
  return 0;
}
