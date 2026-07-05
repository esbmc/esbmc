/* Regression: ieee_sqrt integer-encoding path under --ir-ieee.
 *
 * sqrt(4.0) is exactly 2.0 in IEEE 754, so sqrt(4.0) == 2.0 must hold.
 * Under --ir-ieee, the quadratic axiom  s * s = 4.0 ∧ s >= 0  pins s = 2.0,
 * and the RNE enclosure adds a bound of ±eps around 2.0, keeping the
 * assertion reachable only for the value 2.0.
 *
 * The assertion s != 2.0 must be unsatisfiable: VERIFICATION SUCCESSFUL. */

#include <math.h>

int main(void)
{
  double s = sqrt(4.0);
  __ESBMC_assert(s == 2.0, "sqrt(4.0) must equal 2.0");
  return 0;
}
