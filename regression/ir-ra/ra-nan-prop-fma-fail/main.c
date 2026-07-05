#include <math.h>

int main(void)
{
  double s = sqrt(-1.0);
  double t = fma(s, 2.0, 1.0);
  __ESBMC_assert(t == t, "fma(NaN, 2.0, 1.0) == itself");
  return 0;
}
