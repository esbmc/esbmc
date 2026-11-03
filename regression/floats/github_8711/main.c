/* github #8711: memcpy of bytes into an uninitialised float. */
#include <assert.h>
#include <string.h>
int main(void)
{
  const unsigned char bytes[4] = {0x00, 0x00, 0x80, 0x3f}; /* 1.0f */
  float f;
  memcpy(&f, bytes, sizeof f);
  assert(f == 1.0f);
  return 0;
}
