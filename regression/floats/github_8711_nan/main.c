/* github #8711: memcpy of bytes into a float that holds NaN. */
#include <assert.h>
#include <math.h>
#include <string.h>
int main(void)
{
  const unsigned char bytes[4] = {0x00, 0x00, 0x80, 0x3f}; /* 1.0f */
  float f = NAN;
  memcpy(&f, bytes, sizeof f);
  assert(f == 1.0f);
  return 0;
}
