/* Both copies --loop-invariant makes of assert(n != 7) are proved. */
#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  unsigned i = 0;
  __ESBMC_assume(n != 7);
  __ESBMC_loop_invariant(i <= 10);
  while (i < 10)
  {
    assert(n != 7);
    assert(i < 10);
    i++;
  }
  return 0;
}
