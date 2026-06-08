#include <string.h>
#include <assert.h>

struct quad
{
  char b0;
  char b1;
  char b2;
  char b3;
};

int main()
{
  struct quad q = {0x01, 0x02, 0x03, 0x04};
  unsigned int x;
  memcpy(&x, &q, 4);
  /* big-endian: b0 is MSB, b3 is LSB */
  assert(x == 0x01020304u);
  return 0;
}
