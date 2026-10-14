#include <math.h>

int main(void)
{
  /* The operational model defines asin outside [-1, 1] as NaN. */
  double r = asin(2.0);
  __ESBMC_assert(r != r, "asin(2.0) must be NaN");
  return 0;
}
