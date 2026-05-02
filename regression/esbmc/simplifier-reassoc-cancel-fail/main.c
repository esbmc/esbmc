#include <assert.h>

int nondet_int(void);

int main()
{
  int x = nondet_int();
  int y = nondet_int();
  int w = nondet_int();

  // (x + y) + w - x - y reassociates to w; w == w + 1 must fail.
  int a = (x + y) + w - x - y;
  assert(a == w + 1);

  return 0;
}
