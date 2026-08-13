#include <assert.h>

int nondet_int();

int main()
{
  int n = nondet_int();
  __ESBMC_assume(n == 5);

  int i = 0;
  while (i < n)
    i++;

  assert(i != 5);
  return 0;
}
