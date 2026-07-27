#include <math.h>

int main(void)
{
  /* The operational model defines log of a negative argument as NaN. */
  double r = log(-1.0);
  __ESBMC_assert(r != r, "log(-1.0) must be NaN");
  return 0;
}
