/* Guard is `i != n`, not `<` or `<=`. split_bound handles only the two
 * ordering shapes, so the recogniser declines. Declining must stay silent and
 * must leave the verdict alone. */
#include <assert.h>

int main(void)
{
  unsigned int n;
  __ESBMC_assume(n <= 4);

  unsigned int i = 0;
  unsigned int s = 0;

  while (i != n)
  {
    s = s + 1;
    i = i + 1;
  }

  assert(s == n);
  return 0;
}
