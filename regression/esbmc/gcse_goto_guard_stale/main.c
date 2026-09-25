#include <assert.h>

// The guard `a + b < 0` makes `a + b` available again after `a = 5` without
// assigning its CSE symbol, so the symbol must not be reused (#7992).
int main()
{
  int a = 1, b = 2;
  int x = a + b;
  a = 5;
  if (a + b < 0)
    goto L;
  int y = a + b;
  assert(y == 7);
L:
  return x;
}
