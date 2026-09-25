#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  /* x is symbolic +inf, y is symbolic -inf */
  __ESBMC_assume(x > DBL_MAX);
  __ESBMC_assume(y < -DBL_MAX);
  double z = x + y;

  __ESBMC_assert(!(z > -100.0), "+inf + (-inf) > -100.0 is false (NaN)");
  return 0;
}
