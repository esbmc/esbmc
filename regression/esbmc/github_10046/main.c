/* #10046: a 2-byte store at an odd offset must reach both char members. */
#include <assert.h>
struct S
{
  unsigned char pad, a, b, tail;
};
struct S s;
int main(void)
{
  unsigned char *p = (unsigned char *)&s;
  *(unsigned short *)(p + 1) = 0x0705;
  assert(s.a == 5);
  assert(s.b == 7);
  assert(s.pad == 0 && s.tail == 0);
  assert(*(unsigned short *)(p + 1) == 0x0705);
  return 0;
}
