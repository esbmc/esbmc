/* #10046: an unaligned 2-byte store into a struct of chars reaches both
 * members it covers. */
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
  assert(s.b != 7);
  return 0;
}
