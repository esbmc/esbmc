#include <assert.h>
void f(void *restrict a, void *restrict b)
{
  __ESBMC_assume(a != 0);
  __ESBMC_assume(b != 0);
  assert(a != b);
}
