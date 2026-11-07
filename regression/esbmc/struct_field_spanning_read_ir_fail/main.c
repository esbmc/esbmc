#include <assert.h>
struct S
{
  unsigned int p, x, y;
};
struct S s = {0, 1, 2};
int main(void)
{
  unsigned char *q = (unsigned char *)&s;
  assert(*(unsigned long long *)(q + 4) != 0x0000000200000001ULL);
  return 0;
}
