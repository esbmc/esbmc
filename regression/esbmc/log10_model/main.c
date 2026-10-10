#include <assert.h>
#include <math.h>

int main()
{
  assert(log10(1000.0) == 3.0);
  assert(log10(1e22) == 22.0);
  assert(log10(0.001) == -3.0);
  assert(log10(1.0) == 0.0);
  assert(log10(0x1p-1074) == -323.30621534311581);
  assert(isinf(log10(0.0)) && log10(-0.0) < 0);
  assert(isnan(log10(-1.0)));
  assert(isinf(log10(INFINITY)));
  assert(__builtin_log10(100.0) == 2.0);
  return 0;
}
