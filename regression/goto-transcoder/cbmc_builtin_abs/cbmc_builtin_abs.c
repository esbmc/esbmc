#include <assert.h>
int main()
{
  int x;
  __CPROVER_assume(x == -12);
  /* exercises a __builtin_ spelling (vs the libc name in cbmc_abs) */
  assert(__builtin_abs(x) == 12);
  return 0;
}
