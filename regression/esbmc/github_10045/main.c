/* #10045: a store spanning two union members must write both. */
#include <assert.h>
union A
{
  unsigned int x;
  unsigned char c[4];
};
union B
{
  unsigned int y;
  unsigned short h[2];
};
struct S
{
  union A a;
  union B b;
};
struct S s;
int main(void)
{
  unsigned char *p = (unsigned char *)&s;
  *(unsigned long long *)p = 0x0000000200000001ULL;
  assert(s.a.x == 1);
  assert(s.b.y == 2);
  assert(*(unsigned long long *)p == 0x0000000200000001ULL);
  return 0;
}
