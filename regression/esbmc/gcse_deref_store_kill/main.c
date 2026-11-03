#include <assert.h>

int nondet_int();

// The store through p writes x, which the points-to analysis resolves; `*p + 1`
// must be recomputed after it.
int main()
{
  int x = nondet_int();
  int *p = &x;
  int a = *p + 1;
  *p = 3;
  int b = *p + 1;
  assert(b == 4);
}
