#include <assert.h>

int nondet_int();

// Reusing the stale `a[j] + 1` made y equal x.
int main()
{
  int a[2] = {0, 0};
  int i = nondet_int() & 1;
  int j = nondet_int() & 1;
  __ESBMC_assume(i == j);
  int x = a[j] + 1;
  a[i] = 5;
  int y = a[j] + 1;
  assert(y == x);
}
