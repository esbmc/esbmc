#include <assert.h>

int nondet_int(void);

int main()
{
  int x = nondet_int();
  int y = nondet_int();

  // Constant fold across an add/sub chain.
  // 2 - x - 10*2 - y*4 should reassociate to -18 - x - 4*y.
  int z = 2 - x - 10 * 2 - y * 4;
  assert(z == -18 - x - 4 * y);

  // Mixed signs on the constants.
  int a = (5 - x) + 3;
  assert(a == 8 - x);

  int b = (x - 5) + 3;
  assert(b == x - 2);

  int c = 3 - (5 - x);
  assert(c == x - 2);

  int d = 3 - (x - 5);
  assert(d == 8 - x);

  return 0;
}
