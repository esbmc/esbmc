#include <string.h>
#include <assert.h>

/* Symex's own interval domain, on by default, must see memcpy's write too:
   with n stuck at 1 it pruned the second loop iteration. */
int main()
{
  int n = 1, m = nondet_int();
  memcpy(&n, &m, sizeof n);
  __ESBMC_assume(n >= 0 && n <= 2);
  int i = 0;
  while (i < n)
    ++i;
  assert(i <= 1);
  return 0;
}
