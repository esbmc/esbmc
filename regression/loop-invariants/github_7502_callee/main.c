/* Regression: GitHub #7502 -- the write is through the callee's own parameter,
 * which the caller cannot name. The call's argument can be named, so the havoc
 * reaches the pointee through that: `cnt` is 0 at exit, not 4. */
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
  assert(cnt == 0);
  return 0;
}
