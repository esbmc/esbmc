#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double z = __VERIFIER_nondet_double();
  /* fma(DBL_MAX, 2.0, -inf): both factors are finite, so the intermediate
   * product is an exact finite real in the FMA model even though it would
   * overflow to +inf in a standalone ieee_mul.  Adding -inf to a finite
   * exact product is not an invalid operation; the result is -inf, not NaN. */
  __ESBMC_assume(z < -DBL_MAX);
  double r = fma(DBL_MAX, 2.0, z);

  __ESBMC_assert(r == r, "fma(DBL_MAX, 2.0, -inf) should not be NaN");
  return 0;
}
