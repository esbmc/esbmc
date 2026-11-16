#include <assert.h>
#include <float.h>
#include <math.h>

int main(void)
{
  volatile double big = DBL_MAX, two = 2.0, minf = -INFINITY, inf = INFINITY;
  assert(fma(big, two, minf) == -INFINITY);
  assert(fma(big, -two, inf) == INFINITY);

  volatile float bigf = FLT_MAX, twof = 2.0f, minff = -INFINITY;
  assert(fmaf(bigf, twof, minff) == -INFINITY);

  assert(isnan(fma(inf, two, minf)));
  assert(isnan(fma(minf, -two, minf)));
  assert(fma(inf, -two, minf) == -INFINITY);
  return 0;
}
