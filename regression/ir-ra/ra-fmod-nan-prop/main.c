#include <math.h>

int main(void)
{
  /* fmod(NaN, 1.0): NaN operand propagates through fmod */
  double s = sqrt(-1.0);
  double r = fmod(s, 1.0);

  __ESBMC_assert(!(r > -100.0), "fmod(NaN, 1.0) > -100.0 is false (NaN propagated)");
  return 0;
}
