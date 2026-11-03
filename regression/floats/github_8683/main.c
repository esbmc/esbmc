/* github #8683: hypot without undue overflow or underflow, C11 7.12.7.3 */
#include <assert.h>
#include <float.h>
#include <math.h>

int main(void)
{
  double big = hypot(1e200, 1e200);
  assert(big > 1.414e200 && big < 1.415e200);
  double small = hypot(-1e-200, 1e-200);
  assert(small > 1.414e-200 && small < 1.415e-200);
  assert(hypot(3.0, -4.0) == 5.0);
  assert(hypot(0.0, -0.0) == 0.0 && !signbit(hypot(0.0, -0.0)));
  assert(isinf(hypot(INFINITY, NAN)));
  assert(isinf(hypot(NAN, -INFINITY)));
  assert(isnan(hypot(NAN, 1.0)));
  assert(isnan(hypot(NAN, 1e300)));
  assert(hypot(0x1p-1074, 0.0) == 0x1p-1074);
  assert(hypot(DBL_MAX, 0.0) == DBL_MAX);
  assert(isinf(hypot(1.7e308, 1.7e308)));
  assert(hypot(0x1.8p600, 0.0) == 0x1.8p600);
  return 0;
}
