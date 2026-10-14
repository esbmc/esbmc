#include <assert.h>
int main()
{
  unsigned int x, z;
  unsigned long long y;
  __CPROVER_assume(x == 0xAAAAAAAAu);
  __CPROVER_assume(z == 0x0000000Fu);
  __CPROVER_assume(y == 0x1ULL);
  assert(__builtin_bitreverse32(x) == 0x55555555u);           // alternating bits swap
  assert(__builtin_bitreverse32(z) == 0xF0000000u);           // low nibble -> high nibble
  assert(__builtin_bitreverse64(y) == 0x8000000000000000ULL); // 64-bit MSB
  return 0;
}
