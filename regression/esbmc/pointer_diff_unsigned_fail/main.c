// (a + j) - a folds to j in ptrdiff_t for an unsigned offset.
#include <assert.h>
#include <stddef.h>
_Bool nondet_bool(void);
long nondet_long(void);
unsigned nondet_uint(void);
int main(void)
{
  int a[8];
  int *p = &a[4];
  unsigned j = nondet_uint();
  __ESBMC_assume(j < 8);
  long k = nondet_long();
  ptrdiff_t n = k;
  if (nondet_bool())
    n = (a + j) - a;
  assert(n != (long)j);
  return 0;
}
