// #7891: bfloat16 has 8 significand bits, so 257 is not representable.
#include <assert.h>

int main(void)
{
  __bf16 r = (__bf16)257.0f;
  assert((float)r == 257.0f);
  return 0;
}
