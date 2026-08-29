#include <assert.h>

int main()
{
  unsigned n, i, s = 0;
  __ESBMC_assume(n <= 3);

  for (i = 0; i < n; i++)
    s += 1;

  assert(s == n);
  return 0;
}
