#include <float.h>



static inline double fabs_custom(double x)
{
  return x < 0.0 ? -x : x;
}

int main(void)
{
  double a = nondet_double();
  double b = nondet_double();
  __ESBMC_assume(a >= -1000.0 && a <= 1000.0);
  __ESBMC_assume(b >= -1000.0 && b <= 1000.0);
  double diff = a - b;

  __ESBMC_assert(fabs_custom(diff) <= 1.0, "diff should stay tiny");
  return 0;
}

