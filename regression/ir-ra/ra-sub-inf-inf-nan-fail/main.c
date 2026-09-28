#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  /* both symbolic +inf */
  __ESBMC_assume(x > DBL_MAX);
  __ESBMC_assume(y > DBL_MAX);
  double z = x - y;

  __ESBMC_assert(z > -100.0, "+inf - (+inf) > -100.0 should not hold (NaN)");
  return 0;
}
