/* Regression: GitHub #7502 -- the write is through the callee's own parameter,
 * which the caller cannot name, so the schema declines rather than claim a
 * proof it cannot make. */
#include <assert.h>

static void dec(int *x)
{
  (*x)--;
}

int main()
{
  int cnt = 4;
  __ESBMC_loop_invariant(cnt >= 0);
  while (cnt > 0)
  {
    dec(&cnt);
  }
  assert(0);
  return 0;
}
