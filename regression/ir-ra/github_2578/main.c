// #2578: a nondet float could take a value between 0 and the smallest
// subnormal, where f + g flushes to 0, so h >= f was reported violated.
#include <assert.h>
#include <math.h>
#include <fenv.h>

int main(void)
{
  float f;
  float g;

  __ESBMC_assume(!isnan(f));
  __ESBMC_assume(!isnan(g));

  if (f > g)
    fesetround(FE_UPWARD);

  if (f < g)
    fesetround(FE_DOWNWARD);

  if (!isinf(f) && g > 0.0f)
  {
    float h = f + g;
    assert(h >= f);
  }

  return 0;
}
