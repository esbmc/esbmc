#include <math.h>

int main(void)
{
  double s = sqrt(-1.0);
  double t = s * 2.0;
  __ESBMC_assert(!(t < 100.0), "NaN * 2.0 < 100.0 is false");
  __ESBMC_assert(t != t, "NaN * 2.0 != itself");
  return 0;
}
