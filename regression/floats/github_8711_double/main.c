/* github #8711: memcpy into a double from an offset into a byte array. */
#include <assert.h>
#include <string.h>
int main(void)
{
  const unsigned char bytes[10] = {0xaa, 0, 0, 0, 0, 0, 0, 0xf0, 0x3f, 0xbb};
  double d;
  memcpy(&d, bytes + 1, sizeof d);
  assert(d == 1.0);
  return 0;
}
