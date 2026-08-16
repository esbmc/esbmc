#include <math.h>

int main()
{
  double x = nondet_double();

  __ESBMC_assume(isunordered(x, 0.0));

  __ESBMC_assert(isgreaterequal(x, 0.0), "unordered compares greater-equal");

  return 0;
}
