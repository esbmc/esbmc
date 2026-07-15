#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0;
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x == 1.0);
  __ESBMC_assert(isnormal(x), "isnormal(1.0) is true");
  return 0;
}
