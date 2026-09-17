// #7891: under --fixedbv, __bf16 is 16-bit fixed point with 8 integer bits.
#include <assert.h>

int main(void)
{
  __bf16 r = (__bf16)127.00390625f; // 127 + 2^-8: exact in 8.8, not in bfloat16
  assert((float)r == 127.00390625f);
  return 0;
}
