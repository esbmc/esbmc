#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double z = __VERIFIER_nondet_double();
  /* fma(DBL_MAX, 2.0, -inf): both factors finite; no invalid operation.
   * The result is not NaN, so asserting r != r (self-inequality) must fail. */
  __ESBMC_assume(z < -DBL_MAX);
  double r = fma(DBL_MAX, 2.0, z);

  __ESBMC_assert(r != r, "fma(DBL_MAX, 2.0, -inf) != itself should not hold (not NaN)");
  return 0;
}
