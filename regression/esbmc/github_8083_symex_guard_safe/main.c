#include <string.h>
#include <assert.h>

/* As github_8083_symex_guard, with a bound that holds. */
int main()
{
  int n = 1, m = nondet_int();
  memcpy(&n, &m, sizeof n);
  __ESBMC_assume(n >= 0 && n <= 1);
  int i = 0;
  while (i < n)
    ++i;
  assert(i <= 1);
  return 0;
}
