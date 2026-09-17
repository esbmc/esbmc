#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 100);
  unsigned x = 0;
  for (unsigned i = 0; i < n; i++)
  {
    if (x == 3)
      x = 0;
    else
      x++;
    if (i == 99)
      x = 9;
  }
  assert(x < 4);
  return 0;
}
