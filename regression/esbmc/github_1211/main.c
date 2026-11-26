#include <assert.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  int y = nondet_int();
  int z = nondet_int();
  __ESBMC_assume(x < 5);
  assert(x != 5);
  if (y > 0)
  {
    assert(x != 5);
    if (z > 0)
      assert(x != 5);
  }
  return 0;
}
