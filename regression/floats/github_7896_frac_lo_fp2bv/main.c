/* github #7896: 1 + 2^-10 is representable in binary16 and differs from 1.
 * At 9 significand bits it rounds back to 1 and this fails, so this is what
 * pins the significand from below. --fp2bv lowers to bit-vectors, so the
 * solver never sees an FP sort it could reject instead of answering. */
#include <assert.h>

_Float16 nondet_h(void);

int main(void)
{
  _Float16 x = nondet_h();
  __ESBMC_assume((float)x == 1.0f);
  _Float16 y = x + (_Float16)0x1p-10f;
  assert(y != x);
  return 0;
}
