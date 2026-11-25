#include <assert.h>

unsigned nondet_uint(void);

int main()
{
  unsigned x = nondet_uint();
  __ESBMC_assume(x < 100);
  for (unsigned i = 0; i < 40; ++i)
    assert(x != i + 1000);
  assert(x != 7);
  return 0;
}
