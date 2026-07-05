#include <math.h>

int main(void)
{
  double s = sqrt(-1.0);
  double t = s + 1.0;
  __ESBMC_assert(!(t < 100.0), "NaN + 1.0 < 100.0 is false");
  __ESBMC_assert(t != t, "NaN + 1.0 != itself");
  return 0;
}
