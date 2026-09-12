/* The guard counts on `a[0]`, an index expression rather than a plain symbol.
 * The closed form is built over the counter symbol, so the recogniser
 * declines. */
#include <assert.h>

int main(void)
{
  unsigned int a[2];
  unsigned int n;
  __ESBMC_assume(n <= 3);

  a[0] = 0;

  while (a[0] < n)
    a[0] = a[0] + 1;

  assert(a[0] == n);
  return 0;
}
