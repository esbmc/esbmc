#include <assert.h>

int main(void)
{
  unsigned char c = 255;
  c <<= 8;
  assert(c == 0);

  unsigned char d = 0x81;
  d >>= 1;
  assert(d == 0x40);

  signed char s = -8;
  s >>= 1;
  assert(s == -4);

  signed char t = 1;
  t <<= 7;

  unsigned short u = 0xffff;
  u <<= 16;
  assert(u == 0);

  unsigned int w = 1u;
  w <<= 31;
  assert(w == 0x80000000u);
  return 0;
}
