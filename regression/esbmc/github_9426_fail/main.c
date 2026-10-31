#include <assert.h>
#include <float.h>
#include <math.h>

int main(void)
{
  volatile double big = DBL_MAX, two = 2.0, minf = -INFINITY;
  double r = fma(big, two, minf);
  assert(isnan(r));
  return 0;
}
