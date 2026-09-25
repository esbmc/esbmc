#include <assert.h>

int nondet_int();

// `a[i] = 5` writes `a[j]` when i == j, so `a[j] + 1` must be recomputed.
int main()
{
  int a[2] = {0, 0};
  int i = nondet_int() & 1;
  int j = nondet_int() & 1;
  __ESBMC_assume(i == j);
  int x = a[j] + 1;
  a[i] = 5;
  int y = a[j] + 1;
  assert(y == 6);
}
