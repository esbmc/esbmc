#include <math.h>

int main(void)
{
  double s = sqrt(4.0);
  __ESBMC_assert(s == 2.0, "sqrt(4.0) == 2.0");
  __ESBMC_assert(s == s, "positive sqrt result equals itself");
  return 0;
}
