#include <assert.h>
#include <math.h>

int main(void)
{
  volatile double mz = -0.0, pz = 0.0, minf = -INFINITY, h = 0.5, big = 1e10;
  double r_mz = asinh(mz);
  assert(signbit(r_mz) && r_mz == 0.0);
  assert(!signbit(asinh(pz)));
  assert(asinh(minf) == -INFINITY && asinh(-minf) == INFINITY);
  double r_h = asinh(h);
  assert(asinh(-h) == -r_h);
  assert(r_h > 0.4812 && r_h < 0.4813);
  double r_big = asinh(-big);
  assert(r_big < -23.7189 && r_big > -23.719);
  return 0;
}
