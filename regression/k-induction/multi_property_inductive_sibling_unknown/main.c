/* #7900 §7 question 2: x > 0 is inductive on its own, y != ... is not. */
#include <assert.h>
int nondet_int(void);
int main(void)
{
  int x = nondet_int(), y = 0;
  __ESBMC_assume(x > 0);
  while (nondet_int())
  {
    assert(x > 0);
    assert(y != 3);
    y = 1 - y;
  }
  return 0;
}
