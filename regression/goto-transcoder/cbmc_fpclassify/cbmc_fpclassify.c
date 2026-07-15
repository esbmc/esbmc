#include <assert.h>
#include <math.h>
int main()
{
  double n = 1.0, z = 0.0, inf = 1.0 / 0.0, nan = 0.0 / 0.0;
  double sub = 0x1p-1030; // subnormal double
  assert(fpclassify(n) == FP_NORMAL);
  assert(fpclassify(z) == FP_ZERO);
  assert(fpclassify(inf) == FP_INFINITE);
  assert(fpclassify(nan) == FP_NAN);
  assert(fpclassify(sub) == FP_SUBNORMAL);
  float f = 2.0f;
  long double l = 3.0L;
  assert(fpclassify(f) == FP_NORMAL);
  assert(fpclassify(l) == FP_NORMAL);
  return 0;
}
