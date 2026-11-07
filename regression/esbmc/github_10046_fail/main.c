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
  assert(s.b == 0);
  return 0;
}
