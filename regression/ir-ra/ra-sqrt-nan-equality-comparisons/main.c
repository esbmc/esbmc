#include <math.h>

int main(void)
{
  double s = sqrt(-1.0);
  __ESBMC_assert(!(s == s), "NaN == NaN is false");
  __ESBMC_assert(s != s, "NaN != NaN is true");
  return 0;
}
