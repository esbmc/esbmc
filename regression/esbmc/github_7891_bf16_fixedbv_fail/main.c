// #7891: under --fixedbv, __bf16 has 8 fraction bits, so 2^-9 is lost.
#include <assert.h>

int main(void)
{
  __bf16 small = (__bf16)0.001953125f;
  assert((float)small == 0.001953125f);
  return 0;
}
