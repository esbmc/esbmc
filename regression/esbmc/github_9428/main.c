#include <assert.h>
#include <math.h>

int main(void)
{
  volatile double mz = -0.0, pz = 0.0, minf = -INFINITY, h = 0.5, big = 1e10;
  assert(signbit(asinh(mz)) && asinh(mz) == 0.0);
  assert(!signbit(asinh(pz)));
  assert(asinh(minf) == -INFINITY && asinh(-minf) == INFINITY);
  assert(asinh(-h) == -asinh(h));
  assert(asinh(-big) < -23.7189 && asinh(-big) > -23.719);
  assert(asinh(h) > 0.4812 && asinh(h) < 0.4813);
  return 0;
}
