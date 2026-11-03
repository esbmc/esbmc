#include <assert.h>
#include <math.h>

int main()
{
  assert(cbrtl(27.0L) == 3.0L);
  assert(cbrtl(-8.0L) == -2.0L);
  assert(cbrtl(0x1p-12000L) == 0x1p-4000L);
  assert(cbrtl(27 * 0x1p15000L) == 3 * 0x1p5000L);
  assert(signbit(cbrtl(-0.0L)));
  assert(isinf(cbrtl(-INFINITY)) && cbrtl(-INFINITY) < 0);
  assert(isnan(cbrtl(NAN)));
  assert(__builtin_cbrtl(1e-9L) > 0.000999L && __builtin_cbrtl(1e-9L) < 0.001001L);
  return 0;
}
