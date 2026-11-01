#include <math.h>

static int g = 5;

int main()
{
  double x = nondet_double(), y = nondet_double();
  double a = pow(x, y), b = pow(x, y);
  __ESBMC_assert(a == b || (a != a && b != b), "deterministic");
  __ESBMC_assert(g == 5, "statics untouched");
  return 0;
}
