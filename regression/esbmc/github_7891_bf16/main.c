// #7891: __bf16 is bfloat16 -- float's 8-bit exponent, 8 significand bits.
#include <assert.h>

int main(void)
{
  __bf16 big = (__bf16)1.0e38f; // +inf in IEEE binary16
  assert(!__builtin_isinf((float)big));

  __bf16 r = (__bf16)257.0f; // exact in binary16 and float
  assert((float)r == 256.0f);
  return 0;
}
