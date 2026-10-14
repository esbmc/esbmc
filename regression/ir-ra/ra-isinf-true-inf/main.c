#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0;
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  __ESBMC_assume(x == 1.0);
  __ESBMC_assume(y == 0.0);
  double z = x / y;
  __ESBMC_assert(isinf(z), "isinf(1.0/0.0) is true");
  return 0;
}
