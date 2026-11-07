#include <assert.h>
struct S
{
  unsigned short a, b;
};
struct S s = {0x0201, 0x0403};
int main(void)
{
  unsigned char *p = (unsigned char *)&s;
  *(unsigned short *)(p + 1) = 0xbbaa;
  assert(s.a == 0xaa01 && s.b == 0x04bb);
  return 0;
}
