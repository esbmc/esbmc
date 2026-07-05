#include <math.h>

int main(void)
{
  double s = sqrt(-1.0);
  double t = sqrt(s);
  __ESBMC_assert(t == t, "sqrt(NaN) == itself");
  return 0;
}
