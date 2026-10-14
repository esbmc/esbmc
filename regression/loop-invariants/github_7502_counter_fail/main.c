/* Regression: GitHub #7502 -- the loop writes `cnt` only through `p`, so the
 * havoc over named symbols never touched it and the schema decided this
 * assertion on the value the loop had overwritten.  `cnt` is 0 at exit. */
#include <assert.h>

int main()
{
  int cnt = 4;
  int *p = &cnt;
  int i;
  __ESBMC_loop_invariant(i >= 0 && i <= 4);
  for (i = 0; i < 4; i++)
  {
    (*p)--;
  }
  assert(cnt == 4);
  return 0;
}
