#include <assert.h>
int main()
{
  double d = __builtin_nan("");
  float f = __builtin_nanf("");
  assert(d != d); // a NaN is the only value that compares unequal to itself
  assert(f != f);
  return 0;
}
