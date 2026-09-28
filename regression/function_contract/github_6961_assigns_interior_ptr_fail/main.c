/* `&b[2]` is not the decay a compiler produces, but it names a place inside b
 * that the callee writes forward from. The array is the only object to widen
 * to, so the whole of b is havocked: more than the callee writes, which loses
 * the caller information rather than granting it any. */
#include <assert.h>
#define N 4

void clr(int *p)
{
  __ESBMC_assigns(p);
  __ESBMC_ensures(p[0] == 0);

  for (int i = 0; i < 2; i++)
    p[i] = 0;
}

int main(void)
{
  int b[N];
  b[3] = 7;
  clr(&b[2]);
  assert(b[3] == 7);
  return 0;
}
