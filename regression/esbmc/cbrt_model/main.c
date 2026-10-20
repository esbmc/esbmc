#include <assert.h>
#include <math.h>

int main()
{
  assert(cbrt(27.0) == 3.0);
  assert(cbrt(-8.0) == -2.0);
  assert(cbrt(0x1p-1074) == 0x1p-358);
  assert(signbit(cbrt(-0.0)));
  assert(isinf(cbrt(-INFINITY)) && cbrt(-INFINITY) < 0);
  assert(isnan(cbrt(NAN)));
  assert(cbrtf(64.0f) == 4.0f);
  assert(__builtin_cbrt(1e-9) == 1e-3);
  return 0;
}
