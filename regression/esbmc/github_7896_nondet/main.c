// #7896: _Float16 is IEEE binary16: 5 exponent bits, so 128 * 128 is finite.
#include <assert.h>

_Float16 nondet_f16(void);

int main(void)
{
  _Float16 x = nondet_f16();
  __ESBMC_assume((float)x == 128.0f);
  _Float16 y = x * x;
  assert(!__builtin_isinf((float)y));
  assert((float)y == 16384.0f);
  return 0;
}
