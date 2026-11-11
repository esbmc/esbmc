#include <assert.h>
#include <math.h>

int main()
{
  assert(log(M_E) == 1.0);
  assert(log(2.0) == M_LN2);
  assert(log(1.0) == 0.0);
  assert(log(0x1p-1074) == -744.44007192138122);
  assert(isinf(log(0.0)) && log(-0.0) < 0);
  assert(isnan(log(-1.0)));
  assert(isinf(log(INFINITY)));
  assert(log2(8.0) == 3.0);
  assert(log2(0x1p-1074) == -1074.0);
  assert(log2(10.0) == 3.3219280948873622);
  assert(log1p(M_E - 1.0) == 1.0);
  assert(log1p(0.125) == 0.11778303565638346);
  assert(log1p(-1.0) < 0 && isinf(log1p(-1.0)));
  assert(isnan(log1p(-2.0)));
  assert(__builtin_log(M_E) == 1.0 && __builtin_log2(0.25) == -2.0);
  return 0;
}
