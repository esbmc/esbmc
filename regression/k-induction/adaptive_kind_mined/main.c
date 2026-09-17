#include <assert.h>

unsigned nondet_uint();
_Bool nondet_bool();

int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 1000);
  unsigned i = 0, s = 2;
  while (i < n)
  {
    if (nondet_bool())
    {
      i++;
      s += 3;
    }
    else
    {
      s += 3;
      i++;
    }
  }
  assert(s == 3 * n + 2);
  return 0;
}
