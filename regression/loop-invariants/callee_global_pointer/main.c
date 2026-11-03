/* The callee writes through a global pointer, not through anything the call
 * hands it, so the loop invariant cannot havoc the write and the loop is left
 * to the unwinder. */
#include <assert.h>

int x;
int *gp = &x;

void inc(void)
{
  *gp = *gp + 1;
}

int main()
{
  int i;
  __ESBMC_loop_invariant(i >= 0 && i <= 10);
  for (i = 0; i < 10; i++)
    inc();
  assert(x == 10);
  return 0;
}
