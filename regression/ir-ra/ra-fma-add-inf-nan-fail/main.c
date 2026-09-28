#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = __VERIFIER_nondet_double();
  /* fma(+inf, +inf, -inf): product is +inf; adding -inf is invalid (NaN) */
  __ESBMC_assume(x > DBL_MAX);
  __ESBMC_assume(y > DBL_MAX);
  __ESBMC_assume(z < -DBL_MAX);
  double r = fma(x, y, z);

  __ESBMC_assert(r > -100.0, "fma(+inf, +inf, -inf) > -100.0 should not hold (NaN)");
  return 0;
}
