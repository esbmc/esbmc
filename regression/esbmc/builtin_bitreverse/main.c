#include <assert.h>

unsigned nondet_uint(void);

int main()
{
  assert(__builtin_bitreverse8(0x01) == 0x80);
  assert(__builtin_bitreverse16(0x1234) == 0x2c48);
  assert(__builtin_bitreverse32(0x12345678u) == 0x1e6a2c48u);
  assert(__builtin_bitreverse64(1) == 1ull << 63);

  unsigned x = nondet_uint();
  assert(__builtin_bitreverse32(__builtin_bitreverse32(x)) == x);
  assert((__builtin_bitreverse32(x) >> 31) == (x & 1));
  return 0;
}
