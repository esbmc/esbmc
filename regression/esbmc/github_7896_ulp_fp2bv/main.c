// #7896: pins the binary16 fraction from below. At 10 fraction bits the gap
// above 1 is 2^-10, so 1 + 2^-10 is representable and differs from 1. At 9 it
// rounds back to 1 and this assertion fails. --fp2bv lowers to bit-vectors, so
// the solver never sees an FP sort it could reject instead.
#include <assert.h>

_Float16 nondet_f16(void);

int main(void)
{
  _Float16 x = nondet_f16();
  __ESBMC_assume((float)x == 1.0f);
  _Float16 y = x + (_Float16)0x1p-10f;
  assert(y != x);
  return 0;
}
