#include <math.h>

int main(void)
{
  /* The operational model defines remainder(±inf, y) as NaN. */
  double r = remainder(INFINITY, 1.0);
  __ESBMC_assert(r != r, "remainder(INFINITY, 1.0) must be NaN");
  return 0;
}
