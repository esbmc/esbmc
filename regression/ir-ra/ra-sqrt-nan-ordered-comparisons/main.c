#include <math.h>

int main(void)
{
  double s = sqrt(-1.0);
  __ESBMC_assert(!(s < 100.0), "NaN < 100.0 is false");
  __ESBMC_assert(!(s <= 100.0), "NaN <= 100.0 is false");
  __ESBMC_assert(!(s > -100.0), "NaN > -100.0 is false");
  __ESBMC_assert(!(s >= -100.0), "NaN >= -100.0 is false");
  return 0;
}
