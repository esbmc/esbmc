/* Regression: GitHub #7478 -- declining the loop must not drop the base case.
 * `cnt` is 4 on entry, so the annotated invariant is false there and has to be
 * reported even though the havoc cannot cover the loop's pointer write. */
#include <assert.h>

int main()
{
  int cnt = 4;
  int *p = &cnt;
  __ESBMC_loop_invariant(cnt < 0);
  while (*p)
  {
    (*p)--;
  }
  return 0;
}
