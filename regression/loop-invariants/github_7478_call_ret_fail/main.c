/* Regression: GitHub #7478 -- the write is the return of a call, `*p = dec(*p)`,
 * not an ASSIGN.  Its target is still a dereference the havoc cannot cover. */
#include <assert.h>

static int dec(int c)
{
  return c - 1;
}

int main()
{
  int cnt = 4;
  int *p = &cnt;
  __ESBMC_loop_invariant(cnt >= 0);
  while (*p)
  {
    *p = dec(*p);
  }
  assert(0);
  return 0;
}
