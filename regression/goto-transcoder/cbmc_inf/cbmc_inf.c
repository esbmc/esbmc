#include <assert.h>
int main()
{
  double d = __builtin_huge_val();
  float f = __builtin_huge_valf();
  long double l = __builtin_huge_vall();
  float fi = __builtin_inff();
  long double li = __builtin_infl();
  // +Inf is the only value with x == x + 1; the x > 0 guard pins the sign.
  assert(d > 0.0 && d == d + 1.0);
  assert(f > 0.0f && f == f + 1.0f);
  assert(l > 0.0L && l == l + 1.0L);
  assert(fi > 0.0f && fi == fi + 1.0f);
  assert(li > 0.0L && li == li + 1.0L);
  return 0;
}
