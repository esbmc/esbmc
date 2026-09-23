// #7891: __bf16 is bfloat16 -- float's 8-bit exponent, 8 significand bits.
#include <assert.h>

int main(void)
{
  // 8 exponent bits: the range is float's, not a narrower 16-bit format's.
  __bf16 big = (__bf16)1.0e38f;
  assert(!__builtin_isinf((float)big));

  // 8 significand bits: 257 needs 9, and ties to even.
  __bf16 r = (__bf16)257.0f;
  assert((float)r == 256.0f);
  return 0;
}
