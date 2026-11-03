/* github #8711: the copied float is exactly 1.0f, so only the second
 * assertion fails. */
#include <assert.h>
#include <string.h>
int main(void)
{
  const unsigned char bytes[4] = {0x00, 0x00, 0x80, 0x3f};
  float f;
  memcpy(&f, bytes, sizeof f);
  assert(f == 1.0f);
  assert(f != 1.0f);
  return 0;
}
