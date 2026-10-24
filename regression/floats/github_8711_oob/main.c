/* github #8711: a copy into a float that reads past its source */
#include <string.h>
int main(void)
{
  const unsigned char bytes[4] = {0x00, 0x00, 0x80, 0x3f};
  float f;
  memcpy(&f, bytes + 2, sizeof f);
  return 0;
}
