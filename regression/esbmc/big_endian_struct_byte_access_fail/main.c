#include <assert.h>

/* The little-endian reading of big_endian_struct_byte_access, which a
   symbolic offset used to give on a big-endian target. */
struct P
{
  unsigned short x, y;
};

int main()
{
  struct P p;
  p.x = nondet_ushort();
  p.y = nondet_ushort();
  unsigned k = nondet_ushort() % 4;
  unsigned short v = k < 2 ? p.x : p.y;
  unsigned char *c = (unsigned char *)&p;
  assert(c[k] == (unsigned char)(k % 2 == 0 ? v : v >> 8));
  return 0;
}
