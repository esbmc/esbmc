/* Regression: GitHub #7502 -- the passing half.  The counter is havoc'd and the
 * invariant carries it, so the loop still leaves `i == 4`. */
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
  assert(i == 4);
  return 0;
}
