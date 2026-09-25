#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  /* fmod(+inf, finite): infinite dividend is an invalid operation (NaN) */
  __ESBMC_assume(x > DBL_MAX);
  __ESBMC_assume(y > 0.0 && y < DBL_MAX);
  double r = fmod(x, y);

  __ESBMC_assert(!(r > -100.0), "fmod(+inf, finite) > -100.0 is false (NaN)");
  return 0;
}
