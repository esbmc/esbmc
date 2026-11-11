#include <math.h>

int main()
{
  double x = nondet_double();
  __ESBMC_assume(x == 2.0);
  __ESBMC_assert(pow(x, 2.0) == 4.0, "false: replaced pow is unknown");
  return 0;
}
