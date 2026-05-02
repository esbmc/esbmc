#include <assert.h>

int nondet_int(void);

int main()
{
  int x = nondet_int();
  int y = nondet_int();
  int w = nondet_int();

  // X + (-X) cancellation across a chain.
  // (x + y) + w - x - y should reassociate down to w.
  int a = (x + y) + w - x - y;
  assert(a == w);

  // Constant cancels constant; variables cancel pairwise.
  // 5 + x - 5 + y - x reassociates to y.
  int b = 5 + x - 5 + y - x;
  assert(b == y);

  // Triple cancellation.
  // x - x + y - y + 3 reassociates to 3.
  int c = x - x + y - y + 3;
  assert(c == 3);

  return 0;
}
