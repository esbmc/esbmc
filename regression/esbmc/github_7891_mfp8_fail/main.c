// #7891: __mfp8 is AArch64's storage-only 8-bit float; two of them are two bytes.
#include <assert.h>
#include <string.h>

int main(void)
{
  unsigned char in[2] = {0x12, 0x5a}, out;
  __mfp8 m[2];
  memcpy(m, in, 2);
  __mfp8 second = m[1];
  memcpy(&out, &second, 1);
  assert(out == 0x12);
  return 0;
}
