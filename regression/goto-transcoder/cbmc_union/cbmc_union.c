#include <assert.h>

// A union stores all members at the same address; writing one member and
// reading another reinterprets the bytes. On a little-endian target the
// low-order byte of 0x04030201 is 0x01.
union U
{
  unsigned int i;
  unsigned char b[4];
};

int main()
{
  union U u;
  u.i = 0x04030201;
  assert(u.b[0] == 0x01);
  return 0;
}
