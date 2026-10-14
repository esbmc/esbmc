#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  double z = __VERIFIER_nondet_double();
  /* fma(+inf, 0, z): +inf * 0 is an invalid operation producing NaN */
  __ESBMC_assume(x > DBL_MAX);
  double r = fma(x, 0.0, z);

  __ESBMC_assert(!(r > -100.0), "fma(+inf, 0, z) > -100.0 is false (NaN)");
  return 0;
}
