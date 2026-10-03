#include <assert.h>

/* #8103: the wrapped interval domain must not build an interval from a float
 * operand of a cast or a comparison. */
int main(void)
{
  float f = __VERIFIER_nondet_float();
  __ESBMC_assume(f > 2.5f && f < 3.5f);
  int i = (int)f;
  if (f > 1.0f)
    assert(i == 2);
  return 0;
}
