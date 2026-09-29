#include <assert.h>
int main()
{
  unsigned n = nondet_uint();
  unsigned x = 0;
  __ESBMC_assume(n < 5);
  for (unsigned i = 0; i < n; ++i)
    ++x;
  __ESBMC_assume(x > 100);
  assert(x == 7);
  return 0;
}
