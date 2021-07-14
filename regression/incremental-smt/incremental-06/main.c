#include <assert.h>

int main()
{
  _Bool x, y;
  int a, b, f;
  x = nondet_bool();
  y = nondet_bool();
  a = ((2 * x) - (3 * y));
  if(a < 0)
    a = 0;
  assert(a <= 1);
  b = (x + (4 * y));
  if(b < 0)
    b = 0;
  assert(b <= 4);
  f = ((3 * x) + y);
  if(f < 0)
    f = 0;
  assert(f <= 2);
  return 0;
}
