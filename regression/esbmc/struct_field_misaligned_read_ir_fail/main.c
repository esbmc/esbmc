#include <assert.h>
struct S
{
  unsigned short a, b;
};
struct S s = {0x0201, 0x0403};
int main(void)
{
  unsigned char *p = (unsigned char *)&s;
  assert(*(unsigned short *)(p + 1) != 0x0302);
  return 0;
}
