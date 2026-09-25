#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  /* fmod(x, 0): zero divisor is an invalid operation (NaN) */
  __ESBMC_assume(x > 0.0 && x < DBL_MAX);
  double r = fmod(x, 0.0);

  __ESBMC_assert(r > -100.0, "fmod(finite, 0) > -100.0 should not hold (NaN)");
  return 0;
}
