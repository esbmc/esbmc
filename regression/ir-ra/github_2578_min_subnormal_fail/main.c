// #2578: the smallest subnormal itself remains a possible nondet value.
#include <assert.h>

int main(void)
{
  float f;
  double d;
  if (f > 0.0f)
    assert(f >= 0x1p-148f);
  if (d < 0.0)
    assert(d <= -0x1p-1073);
  return 0;
}
