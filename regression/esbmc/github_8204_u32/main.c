#include <assert.h>
short __VERIFIER_nondet_short(void);

/* #8204 at 32 bits: x is [0, 2^15-1] or [2^32-2^15, 2^32-1], and the first
 * assume keeps [0, 2^15-1]. */
int main()
{
  short v = __VERIFIER_nondet_short();
  unsigned int x = v;
  __ESBMC_assume(x < 0xFFFF8001u);
  __ESBMC_assume(x < 10);
  assert(x < 10 && v >= 0);
  return 0;
}
