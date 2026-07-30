#include <math.h>

_Bool P(double x, double y, double z)
{
  double val = sin(y) + x + y;
  return fmod(val, 3.14) < 1.0;
}

int main()
{
  double x, y, z;
  __ESBMC_assert(
    __ESBMC_forall(
      &x, __ESBMC_forall(&y, __ESBMC_exists(&z, P(x, 0.5, z)))),
    "nested quantifiers over libm calls");
}
