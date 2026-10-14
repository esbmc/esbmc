#include <math.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x >= 1.0 && x <= 4.0);
  double z = sqrt(x);
  __ESBMC_assert(z > 100.0, "should be falsifiable");
  return 0;
}
