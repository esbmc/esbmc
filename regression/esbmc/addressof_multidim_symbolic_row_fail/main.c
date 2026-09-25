// Anti-vacuity twin: with a symbolic row index the two addresses coincide only
// when i is 0, so an unconditional equality has to be refuted.
#include <assert.h>

int main(void)
{
  int a[2][3] = {{1, 2, 3}, {4, 5, 6}};
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i <= 1);

  int *p = &a[0][1];

  assert(p == &a[i][1]);
  return 0;
}
