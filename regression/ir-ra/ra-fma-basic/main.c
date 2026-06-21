/* Regression test: fma(2.0, 3.0, 4.0) == 10.0 under --ir-ieee.
 *
 * FMA is fused: the exact real value 2*3+4 = 10 is computed in one step.
 * Under --ir-ieee the interval for the result is [10 - eps, 10 + eps],
 * which still contains only 10.0 for double precision.
 * The assertion must hold: VERIFICATION SUCCESSFUL. */

#include <math.h>

int main(void)
{
  double r = fma(2.0, 3.0, 4.0);
  __ESBMC_assert(r == 10.0, "fma(2,3,4) must equal 10");
  return 0;
}
