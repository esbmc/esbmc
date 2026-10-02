#include <assert.h>

/* A member shorter than its union starts at the union's lowest address, which
   holds the most significant bytes of a wider member on a big-endian target. */
union U
{
  unsigned i;
  unsigned short s;
  unsigned char c;
};

int main()
{
  union U u;
  u.i = 0x01020304u;
  assert(u.s == 0x0102);
  assert(u.c == 1);
  u.s = 0x0506;
  assert(u.i == 0x05060304u);
  union V
  {
    unsigned short s;
    unsigned i;
  } v = {0x0102};
  assert(v.i >> 16 == 0x0102);
  return 0;
}
