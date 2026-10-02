#include <stdio.h>
#include <assert.h>

/* As github_8083_symex_guard, through scanf. */
int main()
{
  int n = 1;
  scanf("%d", &n);
  __ESBMC_assume(n >= 0 && n <= 2);
  int i = 0;
  while (i < n)
    ++i;
  assert(i <= 1);
  return 0;
}
