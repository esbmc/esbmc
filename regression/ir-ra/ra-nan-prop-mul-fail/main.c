#include <math.h>

int main(void)
{
  double s = sqrt(-1.0);
  double t = s * 2.0;
  __ESBMC_assert(t == t, "NaN * 2.0 == itself");
  return 0;
}
