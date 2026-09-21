// #7896: _Float16 is IEEE binary16: 10 fraction bits, so 1 + 2^-11 rounds to 1.
#include <assert.h>

_Float16 nondet_f16(void);

int main(void)
{
  _Float16 x = nondet_f16();
  __ESBMC_assume((float)x == 1.0f);
  _Float16 y = x + (_Float16)0x1p-11f;
  assert(y != x);
  return 0;
}
