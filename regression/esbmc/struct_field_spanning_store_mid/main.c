#include <assert.h>
struct S { unsigned int p, x, y; };
struct S s;
int main(void)
{
  unsigned char *q = (unsigned char *)&s;
  *(unsigned long long *)(q + 4) = 0x0000000200000001ULL;
  assert(s.p == 0);
  assert(s.x == 1);
  assert(s.y == 2);
  return 0;
}
