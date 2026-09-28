#include <assert.h>

int nondet_int();

int main()
{
  int n = nondet_int();
  __ESBMC_assume(n == 3);

  int i = 0;
  while (i < n)
    i++;

  assert(i == n);
  return 0;
}
