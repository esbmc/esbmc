/* The callee writes through a function pointer, whose target the loop's
 * summary cannot see, so the loop is left to the unwinder. */
#include <assert.h>

int x;

void inc(int *q)
{
  *q = *q + 1;
}

void (*fp)(int *) = inc;

void call(int *q)
{
  fp(q);
}

int main()
{
  int i;
  __ESBMC_loop_invariant(i >= 0 && i <= 10);
  for (i = 0; i < 10; i++)
    call(&x);
  assert(x == 0);
  return 0;
}
