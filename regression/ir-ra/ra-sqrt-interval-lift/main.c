/* Regression: interval lifting for ieee_sqrt under --ir-ieee.
 *
 * x is nondet in [1, 4], so sqrt(x) is in roughly [1, 2].
 * Under --ir-ieee, the integer-encoding path introduces fresh real symbols:
 *   ra_sqrt::0    -- exact sqrt, pinned by  s * s = x  and  s >= 0
 *   ra_sqrt_lo::0 -- sqrt of the hull lower bound
 *   ra_sqrt_hi::0 -- sqrt of the hull upper bound
 * followed by the RNE enclosure ra_lo::0 / ra_hi::0.
 *
 * The assertion z > 100.0 is clearly falsifiable (z <= 2.0 + eps for
 * x <= 4.0), so VERIFICATION FAILED is expected.
 * Z3's nlsat finds the witness x=1, z=1 trivially. */

#include <math.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x >= 1.0 && x <= 4.0);

  double z = sqrt(x);

  __ESBMC_assert(z > 100.0, "should be falsifiable");
  return 0;
}
