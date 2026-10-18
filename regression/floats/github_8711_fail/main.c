/* github #8711: the copied bytes are 2.0f. A copy of the whole float does not
 * go through memcpy's byte loop, so --unwind 2 does not cut it short. */
#include <assert.h>
#include <string.h>
int main(void)
{
  const unsigned char bytes[4] = {0x00, 0x00, 0x00, 0x40};
  float f;
  memcpy(&f, bytes, sizeof f);
  assert(f == 1.0f);
  return 0;
}
