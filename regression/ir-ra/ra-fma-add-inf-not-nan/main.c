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
  /* fma(+inf, +inf, +inf) = +inf: same-sign infinite addend, not NaN */
  __ESBMC_assume(x > DBL_MAX);
  __ESBMC_assume(y > DBL_MAX);
  __ESBMC_assume(z > DBL_MAX);
  double r = fma(x, y, z);

  __ESBMC_assert(r > DBL_MAX, "fma(+inf, +inf, +inf) should be +inf");
  return 0;
}
