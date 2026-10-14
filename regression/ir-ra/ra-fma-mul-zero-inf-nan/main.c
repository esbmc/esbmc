#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double y = __VERIFIER_nondet_double();
  double z = __VERIFIER_nondet_double();
  /* fma(0, +inf, z): 0 * +inf is an invalid operation producing NaN */
  __ESBMC_assume(y > DBL_MAX);
  double r = fma(0.0, y, z);

  __ESBMC_assert(!(r > -100.0), "fma(0, +inf, z) > -100.0 is false (NaN)");
  return 0;
}
