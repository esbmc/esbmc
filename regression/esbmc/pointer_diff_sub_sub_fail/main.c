// p - (p - j) folds to j in ptrdiff_t.
#include <assert.h>
#include <stddef.h>
_Bool nondet_bool(void);
long nondet_long(void);
int nondet_int(void);
int main(void)
{
  int a[8];
  int *p = &a[4];
  int j = nondet_int();
  __ESBMC_assume(j >= -3 && j <= 4);
  long k = nondet_long();
  ptrdiff_t n = k;
  if (nondet_bool())
    n = p - (p - j);
  assert(n != j);
  return 0;
}
