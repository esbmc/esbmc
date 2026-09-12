/* A loop that modifies nothing has no counter to summarise. The pass must skip
 * it without disturbing the affine loop ahead of it, which is still
 * summarised. The degenerate loop is kept unreachable so the test does not
 * depend on an unwinding bound. */
#include <assert.h>

int main(void)
{
  unsigned int n, z;
  __ESBMC_assume(n <= 3);
  __ESBMC_assume(z == 0);

  unsigned int i = 0;
  unsigned int s = 0;

  while (i < n)
  {
    s = s + 1;
    i = i + 1;
  }

  assert(s == n);

  if (z)
  {
    while (1)
    {
    }
  }

  return 0;
}
